#!/usr/bin/env python
"""Step-2 MULTIPLE-CHOICE articulation runner. Makes paid API calls (set OPENAI_API_KEY).

For each target rule we build ONE 8-option multiple-choice set (mc.build_option_set): the
TRUE option (the rule's canonical_articulation) + 7 distractors from the rule's
mc_distractor_seeds, every distractor PROGRAMMATICALLY guaranteed to disagree
with the true rule on >= 25% of the 32 shown examples of EACH of the rule's 3
contexts (the PLAN-locked hard check). The option set is identical across the 3
contexts; a seed that fails in any context is replaced GLOBALLY from the pool.

Per rule: 4 option ORDERS x 3 contexts = 12 multiple-choice queries. Each query:
  prompt = the SAME step-1 few-shot block (contexts.sample_context, same seeds)
           + 'Which rule best describes how the labels were assigned? A) ...
           Answer with the single letter of the best option.'
  temperature=0, small max_tokens, logprobs over the letter options.
Parse the chosen letter; per-rule CLAIM = modal choice across the 12 queries;
accuracy = fraction of the 12 that picked the true option.

CONTROL (--control or run both): the SAME 12 queries with the few-shot block
REMOVED (render_mc_no_examples) — a-priori guessability, chance = 1/8.

Cost: advance estimate printed FIRST and gated by --max-cost. Responses logged
per task AS IT COMPLETES; re-running the same config resumes for free via the
client disk cache.

Example:
  python scripts/run_step2_mc.py --model gpt-4.1 --rules contains_digit,mentions_color
  python scripts/run_step2_mc.py --model gpt-4.1 --all --arms examples,no_examples
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation import mc
from icl_articulation.claude_native import (
    claude_complete,
    claude_cost_usd,
    is_claude,
    make_async_anthropic,
    new_meter,
)
from icl_articulation.client import OpenAIClient, first_token_logprobs, response_text
from icl_articulation.contexts import DatasetError
from icl_articulation.prices import cost_usd, price_for
from icl_articulation.prompts import (
    mc_template_hash,
    render_mc_articulation,
    render_mc_no_examples,
)
from icl_articulation.rule_ids import canonical_rule_id
from icl_articulation.runlog import RunLog, start_run
from icl_articulation.stats import wilson_ci

# The 11 confirmed step-1 survivors (the articulation targets).
TARGET_RULES = [
    "passive_voice",
    "food_topic",
    "positive_sentiment",
    "mentions_animal",
    "contains_first_name",
    "second_word_capitalized",
    "physically_impossible",
    "word_count_geq_8",
    "repeated_content_word",
    "contains_digit",
    "mentions_color",
]

# Locked multiple-choice call config. max_tokens small (a single letter, allow a trailing
# token); logprobs over the letter tokens for a confidence read-out.
TEMPERATURE = 0.0
MAX_TOKENS = 2
CLAUDE_MAX_TOKENS = 16  # claude truncates at 2; a single letter + tolerance
TOP_LOGPROBS = 10  # need all 8 letters visible in the top-k when available
API_SEED = 0
PROMPT_TOKEN_OVERHEAD = 10
ORDER_SEED_BASE = 1000  # per-rule order seeds = base + 10*ctx + order_index

ARMS = ("examples", "no_examples")  # main + no-examples control


@dataclass
class Task:
    """One multiple-choice call (rule x context x order x arm)."""

    rule_id: str
    arm: str
    query: mc.MCQuery
    messages: list[dict[str, str]]


# --- answer parsing ------------------------------------------------------------


_LETTER_TOKEN_RE = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")


def parse_letter(text: str, n_options: int) -> str | None:
    """The chosen option letter (A..), case-insensitive.

    A valid choice is a STANDALONE letter token — not embedded in a word — so
    'Answer: C.' -> C (the 'A' in 'Answer' is part of a word and ignored), while
    'A)', '(A)', ' b', 'Option D' all parse. The FIRST standalone option letter
    wins. Returns None if none of A..(n_options) appears as a standalone token
    (a parse failure scores as not-the-true-option)."""
    valid = set(mc.LETTERS[:n_options])
    for m in _LETTER_TOKEN_RE.finditer(text):
        ch = m.group(1).upper()
        if ch in valid:
            return ch
    return None


_ANSWER_CUE_RE = re.compile(
    r"answer|choose|chosen|correct|best option|\boption\b|the answer is|final|select",
    re.IGNORECASE,
)


def answer_letter(text: str, n_options: int) -> str | None:
    """CoT-tolerant option-letter parse — a superset of ``parse_letter``.

    With a larger token budget a model may reason before answering ('Let's
    analyse... the best option is C'), so the FIRST standalone letter (parse_
    letter) can catch a letter mentioned mid-reasoning. This instead takes the
    option letter immediately AFTER the last answer cue ('answer', 'option',
    'choose', ...) and, failing that, the LAST standalone option letter (the
    conclusion). For a bare single-letter answer ('C', 'C.') first == last ==
    the only letter, so this returns exactly what parse_letter does — gpt-4.1's
    max_tokens=2 runs reproduce identically."""
    valid = set(mc.LETTERS[:n_options])
    hits = [(m.start(), m.group(1).upper())
            for m in _LETTER_TOKEN_RE.finditer(text) if m.group(1).upper() in valid]
    if not hits:
        return None
    cues = list(_ANSWER_CUE_RE.finditer(text))
    if cues:
        last_cue_end = cues[-1].end()
        after = [ch for pos, ch in hits if pos >= last_cue_end]
        if after:
            return after[0]
    return hits[-1][1]


def letter_logprobs(
    record: dict[str, Any], n_options: int
) -> tuple[dict[str, float] | None, float | None]:
    """({letter: logprob} over the option letters seen in the top-k, margin).

    Margin = logprob(top letter) - logprob(2nd letter) among the OPTION letters,
    or None when fewer than two option letters appear in the top-k.
    """
    entry = first_token_logprobs(record)
    if entry is None:
        return None, None
    valid = set(mc.LETTERS[:n_options])
    tops = entry.get("top_logprobs") or []
    by_letter: dict[str, float] = {}
    for t in tops:
        tok = t["token"].strip().upper()
        if len(tok) == 1 and tok in valid:
            by_letter[tok] = max(by_letter.get(tok, float("-inf")), t["logprob"])
    if not by_letter:
        return None, None
    ranked = sorted(by_letter.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) >= 2 else None
    return by_letter, margin


def analyze(task: Task, record: dict[str, Any]) -> dict[str, Any]:
    """The per-call row logged to responses.jsonl."""
    text = response_text(record)
    q = task.query
    n_opt = len(q.options)
    # CoT-tolerant parse (identical to parse_letter for bare single-letter
    # answers, so gpt-4.1's max_tokens=2 runs are unchanged; recovers the answer
    # for gpt-4.1-mini, which reasons and was truncated at max_tokens=2).
    chosen = answer_letter(text, n_opt)
    lps, margin = letter_logprobs(record, n_opt)
    chosen_option = None
    if chosen is not None:
        idx = mc.LETTERS.index(chosen)
        if idx < n_opt:
            chosen_option = q.options[idx]
    return {
        "task": "step2-mc",
        "arm": task.arm,
        "rule_id": task.rule_id,
        "context_index": q.context_index,
        "context_seed": q.context_seed,
        "order_index": q.order_index,
        "order_seed": q.order_seed,
        "true_letter": q.true_letter,
        "n_options": n_opt,
        "chosen_letter": chosen,
        "chosen_is_true": (chosen == q.true_letter) if chosen is not None else False,
        "chosen_predicate_key": chosen_option.predicate_key if chosen_option else None,
        "parse_ok": chosen is not None,
        "letter_logprobs": lps,
        "logprob_margin": margin,
        # the lettered options for this query (auditable: which letter held what)
        "options": [
            {"letter": L, "is_true": o.is_true, "text": o.text, "predicate_key": o.predicate_key}
            for L, o in q.lettered()
        ],
        **record,
    }


# --- task construction ----------------------------------------------------------


def build_rule_tasks(
    rule_id: str,
    extract: dict[str, Any],
    data_dir: str | Path,
    context_seeds: list[int],
    arms: list[str],
) -> tuple[list[Task], dict[str, Any]]:
    """Build the option set + 12 queries x len(arms) tasks for one rule.

    Raises mc.MCBuildError (LOUD) if the 8-option set cannot be assembled under
    the >= 25% per-context policy — a paid run must never silently proceed with
    a degenerate option set."""
    contexts = mc.load_contexts(rule_id, data_dir, context_seeds)
    option_set = mc.build_option_set(rule_id, extract, contexts, context_seeds)
    queries = mc.build_queries(option_set, ORDER_SEED_BASE)
    # render once per (arm, query); the few-shot examples are the SAME step-1 block
    examples_by_ctx = [[(it["text"], it["label"]) for it in ctx] for ctx in contexts]

    tasks: list[Task] = []
    for arm in arms:
        for q in queries:
            option_texts = [o.text for o in q.options]
            letters = list(mc.LETTERS[: len(option_texts)])
            if arm == "examples":
                messages = render_mc_articulation(
                    examples_by_ctx[q.context_index], option_texts, letters
                )
            elif arm == "no_examples":
                messages = render_mc_no_examples(option_texts, letters)
            else:
                raise ValueError(f"unknown arm {arm!r}")
            tasks.append(Task(rule_id, arm, q, messages))

    meta = {
        "rule_id": rule_id,
        "context_seeds": context_seeds,
        "n_queries_per_arm": len(queries),
        "options": [
            {"is_true": o.is_true, "text": o.text, "predicate_key": o.predicate_key, "seed": o.seed}
            for o in option_set.options
        ],
        # the load-bearing audit: per-CHOSEN-distractor per-context disagreement
        "disagreement_per_context": option_set.disagreement,
        "min_disagreement": min(
            (min(v) for v in option_set.disagreement.values()), default=None
        ),
        "rejected_seeds": option_set.rejected,
    }
    return tasks, meta


def build_all_tasks(
    rules: list[str],
    extract: dict[str, Any],
    data_dir: str | Path,
    context_seeds: list[int],
    arms: list[str],
) -> tuple[list[Task], dict[str, Any]]:
    tasks: list[Task] = []
    rules_meta: dict[str, Any] = {}
    for rule_id in rules:
        rt, meta = build_rule_tasks(rule_id, extract, data_dir, context_seeds, arms)
        tasks.extend(rt)
        rules_meta[rule_id] = meta
    return tasks, rules_meta


def estimate_cost_usd(model: str, tasks: list[Task], max_tokens: int = MAX_TOKENS) -> float:
    """Advance estimate: chars/4 heuristic on the rendered prompts + completion."""
    total = 0.0
    for task in tasks:
        prompt_chars = sum(len(m["content"]) for m in task.messages)
        total += cost_usd(model, int(prompt_chars / 4) + PROMPT_TOKEN_OVERHEAD, max_tokens)
    return total


# --- execution -------------------------------------------------------------------


async def execute(
    tasks: list[Task],
    run: RunLog,
    model: str,
    concurrency: int,
    cache_dir: str | Path,
    api: Any | None = None,
    max_tokens: int = MAX_TOKENS,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run all tasks; log each row AS IT COMPLETES (crash leaves a valid jsonl).

    Claude goes through the native-Anthropic path (no logprobs, larger
    max_tokens); gpt/mini/deepseek go through OpenAIClient unchanged."""
    if is_claude(model):
        return await _execute_claude(tasks, run, concurrency)

    client = OpenAIClient(concurrency=concurrency, cache_dir=cache_dir, api=api)
    rows: list[dict[str, Any]] = []
    try:

        async def one(task: Task) -> tuple[Task, dict[str, Any]]:
            record = await client.complete(
                model,
                task.messages,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
                seed=API_SEED,
            )
            return task, record

        futures = [asyncio.ensure_future(one(t)) for t in tasks]
        try:
            for fut in asyncio.as_completed(futures):
                task, record = await fut
                row = analyze(task, record)
                run.log_response(row)
                rows.append(row)
        except BaseException:
            for f in futures:
                f.cancel()
            await asyncio.gather(*futures, return_exceptions=True)
            raise
        return rows, client.stats(), client.cost.summary()
    finally:
        await client.aclose()


async def _execute_claude(
    tasks: list[Task], run: RunLog, concurrency: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Native-Anthropic multiple-choice execution (no logprobs; CoT-tolerant letter parse)."""
    ac = make_async_anthropic()
    sem = asyncio.Semaphore(concurrency)
    meter = new_meter()
    rows: list[dict[str, Any]] = []

    async def one(task: Task) -> tuple[Task, dict[str, Any]]:
        async with sem:
            record = await claude_complete(
                ac, task.messages, max_tokens=CLAUDE_MAX_TOKENS, meter=meter
            )
        return task, record

    futures = [asyncio.ensure_future(one(t)) for t in tasks]
    try:
        for fut in asyncio.as_completed(futures):
            task, record = await fut
            row = analyze(task, record)
            run.log_response(row)
            rows.append(row)
    except BaseException:
        for f in futures:
            f.cancel()
        await asyncio.gather(*futures, return_exceptions=True)
        raise
    finally:
        await ac.close()
    # mirror OpenAIClient.stats() keys so the runner's summary print is happy
    stats = {"api_calls": len(rows), "cache_hits": 0, "n_429": 0,
             "retryable_errors": 0, "failures": 0, "rate_429": 0.0,
             "error_rate": 0.0, "provider": "anthropic"}
    cost = {"total_usd": claude_cost_usd(meter), "claude_usd": claude_cost_usd(meter),
            "tokens": dict(meter)}
    return rows, stats, cost


# --- metrics ----------------------------------------------------------------------


def _rule_arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Modal claim + accuracy over one rule's queries in one arm."""
    n = len(rows)
    n_correct = sum(r["chosen_is_true"] for r in rows)
    n_parse_fail = sum(1 for r in rows if not r["parse_ok"])
    # MODAL choice = most common chosen predicate_key across the 12 queries
    # (predicate identity is order-invariant, unlike the shuffled letter).
    keys = [r["chosen_predicate_key"] for r in rows if r["chosen_predicate_key"] is not None]
    modal_key: str | None = None
    modal_count = 0
    if keys:
        modal_key, modal_count = Counter(keys).most_common(1)[0]
    # find the true option's predicate key from any row's options listing
    true_pred_key = None
    if rows:
        true_pred_key = next(
            (o["predicate_key"] for o in rows[0]["options"] if o["is_true"]), None
        )
    return {
        "n_queries": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else None,
        "wilson_ci_95": list(wilson_ci(n_correct, n)) if n else None,
        "n_parse_failures": n_parse_fail,
        "modal_predicate_key": modal_key,
        "modal_count": modal_count,
        "modal_is_true": (modal_key == true_pred_key) if modal_key is not None else False,
        "true_predicate_key": true_pred_key,
        "choice_distribution": dict(Counter(keys)),
    }


def compute_metrics(rows: list[dict[str, Any]], model: str, arms: list[str]) -> dict[str, Any]:
    by_rule_arm: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by_rule_arm.setdefault((r["rule_id"], r["arm"]), []).append(r)

    rules_out: dict[str, Any] = {}
    rule_ids = sorted({r["rule_id"] for r in rows})
    for rule_id in rule_ids:
        rules_out[rule_id] = {
            arm: _rule_arm_metrics(by_rule_arm.get((rule_id, arm), [])) for arm in arms
        }

    aggregate: dict[str, Any] = {}
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        n = len(arm_rows)
        n_correct = sum(r["chosen_is_true"] for r in arm_rows)
        modal_hits = sum(1 for rid in rule_ids if rules_out[rid][arm]["modal_is_true"])
        aggregate[arm] = {
            "n_queries": n,
            "n_correct": n_correct,
            "query_accuracy": n_correct / n if n else None,
            "n_rules": len(rule_ids),
            "n_rules_modal_true": modal_hits,
            "modal_true_rate": modal_hits / len(rule_ids) if rule_ids else None,
            "n_parse_failures": sum(1 for r in arm_rows if not r["parse_ok"]),
            "chance": 1.0 / 8,
        }
    return {"model": model, "arms": arms, "rules": rules_out, "aggregate": aggregate}


# --- CLI ----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="gpt-4.1 or gpt-4.1-mini")
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--rules", help="comma-separated rule_ids")
    sel.add_argument("--all", action="store_true", help="all 11 target rules")
    p.add_argument(
        "--arms",
        default="examples,no_examples",
        help="comma-separated arms: examples (main), no_examples (control)",
    )
    p.add_argument("--run-seed", type=int, default=0, help="context seeds = run_seed+0..n-1")
    p.add_argument("--n-contexts", type=int, default=mc.N_CONTEXTS,
                   help="number of few-shot contexts (queries = n_contexts x n_orders); "
                        "raise to tighten CIs. Ignored for claude logprobs (none).")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                   help="completion budget; 2 = single-letter (gpt-4.1). Raise (e.g. 256) for "
                        "gpt-4.1-mini, which reasons and truncated at 2 (CoT-tolerant parse).")
    p.add_argument("--max-cost", type=float, default=200.0, help="abort if estimate exceeds (USD)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--spec-extract", default="data/spec_extract.json")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--cache-dir", default="cache")
    return p.parse_args(argv)


def resolve_rules(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(TARGET_RULES)
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    if not rules:
        raise DatasetError("--rules: empty rule list")
    unknown = [r for r in rules if canonical_rule_id(r) not in TARGET_RULES]
    if unknown:
        print(
            f"WARNING: {unknown} are not in the 11 confirmed step-1 survivors "
            f"(TARGET_RULES); proceeding anyway",
            file=sys.stderr,
        )
    return rules


def resolve_arms(args: argparse.Namespace) -> list[str]:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise ValueError(f"unknown arm(s) {bad}; choose from {ARMS}")
    if not arms:
        raise ValueError("--arms: empty")
    return arms


def main(argv: list[str] | None = None, api: Any | None = None) -> int:
    args = parse_args(argv)
    prices = price_for(args.model)  # KeyError (loud) if unpriced
    rules = resolve_rules(args)
    arms = resolve_arms(args)
    context_seeds = [args.run_seed + i for i in range(args.n_contexts)]

    extract = mc.load_extract(args.spec_extract)
    # build option sets + tasks up front (LOUD MCBuildError before any API call)
    tasks, rules_meta = build_all_tasks(rules, extract, args.data_dir, context_seeds, arms)

    estimate = estimate_cost_usd(args.model, tasks, args.max_tokens)
    print(
        f"advance cost estimate: ${estimate:.4f} for {len(tasks)} calls to {args.model} "
        f"({len(rules)} rules x {args.n_contexts} contexts x {mc.N_ORDERS} orders x "
        f"{len(arms)} arms); --max-cost gate ${args.max_cost:.2f}"
    )
    if estimate > args.max_cost:
        print(
            f"ABORT: estimate ${estimate:.4f} exceeds --max-cost ${args.max_cost:.2f}",
            file=sys.stderr,
        )
        return 1

    config = {
        "task": "step2-mc",
        "model": args.model,
        "price_per_mtok": prices,
        "rules": rules,
        "arms": arms,
        "run_seed": args.run_seed,
        "context_seeds": context_seeds,
        "n_contexts": args.n_contexts,
        "n_orders": mc.N_ORDERS,
        "n_options": mc.N_OPTIONS,
        "disagreement_floor": mc.DISAGREEMENT_FLOOR,
        "order_seed_base": ORDER_SEED_BASE,
        "k_few_shot": mc.K_FEW_SHOT,
        "temperature": TEMPERATURE,
        "max_tokens": args.max_tokens,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "api_seed": API_SEED,
        "concurrency": args.concurrency,
        "template_hash": mc_template_hash(),
        "expected_total_calls": len(tasks),
        "data_dir": str(args.data_dir),
        "rules_meta": rules_meta,
    }
    run = start_run(
        name=f"step2mc-{args.model}",
        config=config,
        cost_estimate_usd=estimate,
        results_dir=args.results_dir,
    )
    print(f"run dir: {run.run_dir}  (expected total calls: {len(tasks)})")

    t0 = time.monotonic()
    rows, client_stats, cost_summary = asyncio.run(
        execute(tasks, run, args.model, args.concurrency, args.cache_dir, api=api,
                max_tokens=args.max_tokens)
    )
    wall = time.monotonic() - t0

    metrics = compute_metrics(rows, args.model, arms)
    metrics["wall_seconds"] = wall
    metrics["client_stats"] = client_stats
    run.write_metrics(metrics)
    run.finish(
        cost_actual_usd=cost_summary["total_usd"],
        extra={"client_stats": client_stats, "cost": cost_summary},
    )

    for arm in arms:
        agg = metrics["aggregate"][arm]
        print(
            f"[{arm}] query acc {agg['query_accuracy']:.3f} "
            f"({agg['n_correct']}/{agg['n_queries']}), modal-true "
            f"{agg['n_rules_modal_true']}/{agg['n_rules']}, "
            f"parse_fail {agg['n_parse_failures']}, chance {agg['chance']:.3f}"
        )
    print(
        f"done: {wall:.1f}s, actual cost ${cost_summary['total_usd']:.4f} "
        f"(estimate ${estimate:.4f}), cache_hits={client_stats['cache_hits']}"
    )
    for rule_id in sorted(metrics["rules"]):
        bits = []
        for arm in arms:
            m = metrics["rules"][rule_id][arm]
            bits.append(f"{arm}={m['accuracy']:.3f}{'*' if m['modal_is_true'] else ''}")
        print(f"  {rule_id}: " + "  ".join(bits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
