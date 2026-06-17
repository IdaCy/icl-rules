#!/usr/bin/env python
"""Step-2 FREE-FORM articulation runner. Makes paid API calls (set OPENAI_API_KEY).

For each target rule the prompt is the SAME step-1 few-shot block followed by an
instruction to STATE the labeling rule in one sentence ("Step-2 free-form").

Generation grid (locked): {direct, think-then-state} x {2 phrasings} x 3
contexts = 12 generations/rule. 'direct' asks for the one-sentence rule only;
'think-then-state' MAY use chain-of-thought (CoT is allowed in step-2
articulation, UNLIKE no-CoT step-1). Plus a NO-EXAMPLES control: the same 4
(variant x phrasing) requests with the few-shot block removed (a-priori
guessability). Contexts reuse step-1's sampler: k=32, 16/16, distinct bases,
seeds = run_seed + 0,1,2 (so context 0 here == context 0 in step 1).

temperature 0; max_tokens sized per variant (a sentence for 'direct', sentence
+ reasoning room for 'think-then-state'). All calls go through client.py; the
advance cost estimate is printed FIRST and gated by --max-cost. Generations are
logged AS THEY COMPLETE (crash leaves a valid partial responses.jsonl).

GRADING (--grade, default on): a gpt-4.1 judge with the written rubric in
grading.py grades each candidate 2/1/0 against the canonical articulation +
equivalence class from the COMMITTED data/spec_extract.json; an extensional
check (the rule's own groundtruth.label_of on probe items) corroborates surface
rules. Pre-registered metrics: primary = median grade of the DIRECT variant
across contexts; secondary = best variant.

Example:
  python scripts/run_step2_freeform.py --model gpt-4.1-mini \\
      --rules passive_voice,contains_digit --judge-model gpt-4.1
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.claude_native import (
    claude_complete,
    claude_cost_usd,
    is_claude,
    make_async_anthropic,
    new_meter,
)
from icl_articulation.client import OpenAIClient, response_text
from icl_articulation.contexts import DatasetError, load_items, sample_context, select_queries
from icl_articulation.datagen.groundtruth import RULE_PREDICATES
from icl_articulation.grading import (
    extensional_probe,
    gold_for,
    grade_one,
    load_spec_extract,
    rubric_hash,
    summarize_rule,
)
from icl_articulation.prices import cost_usd, price_for
from icl_articulation.prompts import (
    FREEFORM_VARIANTS,
    extract_rule,
    freeform_template_hash,
    render_freeform_articulation,
    render_freeform_no_examples,
)
from icl_articulation.rule_ids import canonical_rule_id
from icl_articulation.runlog import RunLog, start_run

# The 11 confirmed step-1 survivors (the articulation targets).
DEFAULT_TARGETS = [
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

# Locked generation config (PLAN step-2 free-form).
K_FEW_SHOT = 32
N_CONTEXTS = 3
TEMPERATURE = 0.0
API_SEED = 0
# max_tokens per variant: a sentence for direct, sentence + reasoning room for
# think-then-state (CoT allowed here).
MAX_TOKENS = {"direct": 80, "think-then-state": 600}
# claude tends to add a short preamble before the RULE: line and truncates
# at the small openai budgets; give it room (extract_rule still pulls RULE:).
CLAUDE_MAX_TOKENS = {"direct": 256, "think-then-state": 1200}
# grading examples / probes: show the judge the actual training distribution.
N_JUDGE_EXAMPLES = 12  # labeled examples the judge reasons over (from the context)
N_PROBES = 40  # extensional-check probe items (PLAN: 40 probes)
PROMPT_TOKEN_OVERHEAD = 10  # chat formatting overhead per call (chars/4 heuristic)


@dataclass
class GenTask:
    """One free-form generation call (rule x variant x phrasing x context)."""

    rule_id: str
    variant: str  # 'direct' | 'think-then-state'
    phrasing: int
    context_index: int  # 0..N_CONTEXTS-1 with examples; -1 for the no-examples control
    context_seed: int | None  # None for the control
    has_examples: bool
    messages: list[dict[str, str]]
    examples: list[tuple[str, bool]] = field(default_factory=list)  # (text,label) shown


# --- task construction ---------------------------------------------------------


def build_gen_tasks(
    rules: list[str],
    data_dir: str | Path,
    run_seed: int,
    n_contexts: int = N_CONTEXTS,
    variants: list[str] | None = None,
) -> tuple[list[GenTask], dict[str, Any]]:
    """All generation tasks + per-rule context metadata for the run config.

    Per rule: for each context and each (variant, phrasing) in the grid, one
    with-examples task; plus one no-examples control per (variant, phrasing).
    ``n_contexts`` raises the context count to tighten CIs; ``variants`` (a
    subset of FREEFORM_VARIANTS, default all) lets a run restrict to e.g.
    'direct' only (skips the costly think-then-state generations).
    """
    data_dir = Path(data_dir)
    use_variants = list(FREEFORM_VARIANTS) if variants is None else variants
    tasks: list[GenTask] = []
    contexts_meta: dict[str, Any] = {}
    variant_grid = [
        (variant, phrasing)
        for variant in use_variants
        for phrasing in range(len(FREEFORM_VARIANTS[variant]))
    ]
    for rule_id in rules:
        items = load_items(data_dir / rule_id / "items.jsonl")
        contexts_meta[rule_id] = []
        for ctx_index in range(n_contexts):
            seed = run_seed + ctx_index
            context = sample_context(items, k=K_FEW_SHOT, seed=seed)
            examples = [(it["text"], bool(it["label"])) for it in context]
            contexts_meta[rule_id].append(
                {
                    "context_index": ctx_index,
                    "seed": seed,
                    "item_ids": [it["item_id"] for it in context],
                    "base_ids": [it["base_id"] for it in context],
                }
            )
            for variant, phrasing in variant_grid:
                tasks.append(
                    GenTask(
                        rule_id=rule_id,
                        variant=variant,
                        phrasing=phrasing,
                        context_index=ctx_index,
                        context_seed=seed,
                        has_examples=True,
                        messages=render_freeform_articulation(examples, variant, phrasing),
                        examples=examples,
                    )
                )
        # no-examples control: one per (variant, phrasing) — no context
        for variant, phrasing in variant_grid:
            tasks.append(
                GenTask(
                    rule_id=rule_id,
                    variant=variant,
                    phrasing=phrasing,
                    context_index=-1,
                    context_seed=None,
                    has_examples=False,
                    messages=render_freeform_no_examples(variant, phrasing),
                    examples=[],
                )
            )
    return tasks, contexts_meta


def estimate_gen_cost_usd(model: str, tasks: list[GenTask]) -> float:
    """Advance estimate for generation: chars/4 prompt heuristic + the variant's
    max_tokens completion budget per call (worst case — CoT may fill it)."""
    total = 0.0
    for task in tasks:
        prompt_chars = sum(len(m["content"]) for m in task.messages)
        total += cost_usd(
            model,
            int(prompt_chars / 4) + PROMPT_TOKEN_OVERHEAD,
            MAX_TOKENS[task.variant],
        )
    return total


def estimate_grade_cost_usd(judge_model: str, n_generations: int) -> float:
    """Advance estimate for grading: one judge call per generation. Heuristic
    prompt size = rubric + gold + examples (~1200 tokens) + the judge budget."""
    judge_prompt_tokens = 1200
    return n_generations * cost_usd(judge_model, judge_prompt_tokens, 400)


# --- generation ----------------------------------------------------------------


def gen_row(task: GenTask, record: dict[str, Any]) -> dict[str, Any]:
    """The per-generation row logged to responses.jsonl."""
    text = response_text(record)
    candidate = extract_rule(text)
    return {
        "kind": "generation",
        "rule_id": task.rule_id,
        "variant": task.variant,
        "phrasing": task.phrasing,
        "context_index": task.context_index,
        "context_seed": task.context_seed,
        "has_examples": task.has_examples,
        "completion_text": text,
        "candidate": candidate,
        **record,
    }


async def generate(
    tasks: list[GenTask],
    run: RunLog,
    client: OpenAIClient,
    model: str,
    claude_meter: dict[str, int] | None = None,
    concurrency: int = 16,
) -> list[dict[str, Any]]:
    """Run all generation tasks; log each row AS IT COMPLETES (crash-safe).

    Claude generations go through the native-Anthropic path (claude_meter
    accumulates its token cost); gpt/mini go through OpenAIClient unchanged.
    Grading (the judge) always uses ``client`` regardless of the subject."""
    rows: list[dict[str, Any]] = []
    use_claude = is_claude(model)
    ac = None
    if use_claude:
        ac = make_async_anthropic()
        sem = asyncio.Semaphore(concurrency)

    async def one(task: GenTask) -> tuple[GenTask, dict[str, Any]]:
        if use_claude:
            async with sem:
                record = await claude_complete(
                    ac, task.messages,
                    max_tokens=CLAUDE_MAX_TOKENS[task.variant], meter=claude_meter,
                )
            return task, record
        record = await client.complete(
            model,
            task.messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS[task.variant],
            seed=API_SEED,
        )
        return task, record

    futures = [asyncio.ensure_future(one(t)) for t in tasks]
    try:
        for fut in asyncio.as_completed(futures):
            task, record = await fut
            row = gen_row(task, record)
            run.log_response(row)
            rows.append(row)
    except BaseException:
        for f in futures:
            f.cancel()
        await asyncio.gather(*futures, return_exceptions=True)
        raise
    finally:
        if ac is not None:
            await ac.close()
    return rows


# --- grading -------------------------------------------------------------------


def _task_by_key(tasks: list[GenTask]) -> dict[tuple[Any, ...], GenTask]:
    """Index generation tasks by (rule, variant, phrasing, context_index) so a
    logged row can recover the example block the judge should see."""
    return {
        (t.rule_id, t.variant, t.phrasing, t.context_index): t for t in tasks
    }


def _probes_for(rule_id: str, data_dir: str | Path) -> list[dict[str, Any]]:
    """N_PROBES balanced held-out items for the extensional check (best-effort
    corroboration; reused-from-step-3 spirit)."""
    items = load_items(Path(data_dir) / rule_id / "items.jsonl")
    return select_queries(items, "held_out", N_PROBES)


async def grade(
    gen_rows: list[dict[str, Any]],
    gen_tasks: list[GenTask],
    run: RunLog,
    client: OpenAIClient,
    spec_rules: dict[str, Any],
    judge_model: str,
    data_dir: str | Path,
) -> list[dict[str, Any]]:
    """Grade every generation with the LLM judge; corroborate surface rules with
    the extensional check. Log each graded row AS IT COMPLETES."""
    by_key = _task_by_key(gen_tasks)
    golds = {r["rule_id"]: gold_for(spec_rules, r["rule_id"]) for r in gen_rows}
    # extensional check is per-rule (same probe set for all of a rule's rows)
    ext: dict[str, dict[str, Any]] = {}
    for rule_id in {r["rule_id"] for r in gen_rows}:
        entry = RULE_PREDICATES.get(canonical_rule_id(rule_id))
        label_of = entry.label_of if entry is not None else None
        ext[rule_id] = extensional_probe(label_of, _probes_for(rule_id, data_dir))

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        rule_id = row["rule_id"]
        gold = golds[rule_id]
        # examples the judge reasons over: the task's own context block (empty
        # for the control -> the judge still grades a-priori guessability).
        task = by_key[(rule_id, row["variant"], row["phrasing"], row["context_index"])]
        examples = task.examples[:N_JUDGE_EXAMPLES]
        result = await grade_one(
            client,
            row["candidate"],
            gold["canonical_articulation"],
            gold["equivalence_class"],
            examples,
            model=judge_model,
        )
        return {
            "kind": "grade",
            "rule_id": rule_id,
            "variant": row["variant"],
            "phrasing": row["phrasing"],
            "context_index": row["context_index"],
            "has_examples": row["has_examples"],
            "candidate": row["candidate"],
            "grade": result["grade"],
            "extensionally_equivalent": result["extensionally_equivalent"],
            "rationale": result["rationale"],
            "extensional_check": ext[rule_id],
            **result["record"],
        }

    graded: list[dict[str, Any]] = []
    futures = [asyncio.ensure_future(one(r)) for r in gen_rows]
    try:
        for fut in asyncio.as_completed(futures):
            grow = await fut
            run.log_response(grow)
            graded.append(grow)
    except BaseException:
        for f in futures:
            f.cancel()
        await asyncio.gather(*futures, return_exceptions=True)
        raise
    return graded


# --- metrics -------------------------------------------------------------------


def compute_metrics(graded: list[dict[str, Any]], model: str, judge_model: str) -> dict[str, Any]:
    """Per-rule pre-specified metrics + overall grade distribution."""
    by_rule: dict[str, list[dict[str, Any]]] = {}
    for g in graded:
        by_rule.setdefault(g["rule_id"], []).append(g)
    rules_out = {rule_id: summarize_rule(by_rule[rule_id]) for rule_id in sorted(by_rule)}
    with_ex = [g for g in graded if g["has_examples"]]
    return {
        "task": "step2-freeform",
        "model": model,
        "judge_model": judge_model,
        "rules": rules_out,
        "overall": {
            "n_rules": len(rules_out),
            "n_generations": len(with_ex),
            "n_controls": len(graded) - len(with_ex),
            "grade_counts_with_examples": _overall_counts(with_ex),
        },
    }


def _overall_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"0": 0, "1": 0, "2": 0}
    for r in rows:
        counts[str(r["grade"])] += 1
    return counts


# --- execution -----------------------------------------------------------------


async def execute(
    gen_tasks: list[GenTask],
    run: RunLog,
    model: str,
    judge_model: str,
    do_grade: bool,
    spec_rules: dict[str, Any],
    data_dir: str | Path,
    concurrency: int,
    cache_dir: str | Path,
    api: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Generate (+ optionally grade); one OpenAI client for the run (also the
    judge). Claude subjects generate via the native path; its token cost is
    folded into the returned cost summary."""
    client = OpenAIClient(concurrency=concurrency, cache_dir=cache_dir, api=api)
    claude_meter = new_meter() if is_claude(model) else None
    try:
        gen_rows = await generate(
            gen_tasks, run, client, model,
            claude_meter=claude_meter, concurrency=concurrency,
        )
        graded: list[dict[str, Any]] = []
        if do_grade:
            graded = await grade(
                gen_rows, gen_tasks, run, client, spec_rules, judge_model, data_dir
            )
        cost = client.cost.summary()
        if claude_meter is not None:
            claude_usd = claude_cost_usd(claude_meter)
            cost["claude_usd"] = claude_usd
            cost["claude_tokens"] = dict(claude_meter)
            cost["total_usd"] = cost.get("total_usd", 0.0) + claude_usd
        return gen_rows, graded, client.stats(), cost
    finally:
        await client.aclose()


# --- CLI -----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="generation model (gpt-4.1 or gpt-4.1-mini)")
    p.add_argument("--judge-model", default="gpt-4.1", help="LLM judge model (default gpt-4.1)")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--rules", help="comma-separated rule_ids (default: the 11 targets)")
    sel.add_argument("--all-targets", action="store_true", help="all 11 confirmed targets")
    p.add_argument("--run-seed", type=int, default=0, help="context seeds = run_seed+0..n-1")
    p.add_argument("--n-contexts", type=int, default=N_CONTEXTS,
                   help="number of few-shot contexts per rule (raise to tighten CIs)")
    p.add_argument("--variants", default=",".join(FREEFORM_VARIANTS),
                   help="comma-separated subset of {direct,think-then-state}; "
                        "use 'direct' to skip the costly reasoning variant")
    p.add_argument("--max-cost", type=float, default=200.0, help="abort if the estimate exceeds this (USD)")
    p.add_argument("--no-grade", action="store_true", help="generate only; skip the judge")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--spec-extract", default="data/spec_extract.json")
    return p.parse_args(argv)


def resolve_rules(args: argparse.Namespace) -> list[str]:
    if args.rules:
        rules = [r.strip() for r in args.rules.split(",") if r.strip()]
        if not rules:
            raise DatasetError("--rules: empty rule list")
        return rules
    return list(DEFAULT_TARGETS)  # default + --all-targets are both the 11 targets


def main(argv: list[str] | None = None, api: Any | None = None) -> int:
    args = parse_args(argv)
    # price sanity at start: KeyError (loud) if a model is unpriced
    gen_prices = price_for(args.model)
    judge_prices = price_for(args.judge_model)

    rules = resolve_rules(args)
    do_grade = not args.no_grade
    spec_rules = load_spec_extract(args.spec_extract)
    if do_grade:  # fail loud NOW if a target has no gold in the public extract
        for rule_id in rules:
            gold_for(spec_rules, rule_id)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in FREEFORM_VARIANTS]
    if bad:
        raise ValueError(f"unknown variant(s) {bad}; choose from {list(FREEFORM_VARIANTS)}")
    gen_tasks, contexts_meta = build_gen_tasks(
        rules, args.data_dir, args.run_seed, n_contexts=args.n_contexts, variants=variants
    )
    n_with_ex = sum(1 for t in gen_tasks if t.has_examples)
    n_control = len(gen_tasks) - n_with_ex

    # advance estimate FIRST, then the gate — before any run dir or API call
    gen_est = estimate_gen_cost_usd(args.model, gen_tasks)
    grade_est = estimate_grade_cost_usd(args.judge_model, len(gen_tasks)) if do_grade else 0.0
    estimate = gen_est + grade_est
    print(
        f"advance cost estimate: ${estimate:.4f} "
        f"(generation ${gen_est:.4f} for {len(gen_tasks)} calls to {args.model}: "
        f"{len(rules)} rules x {args.n_contexts} contexts x variants[{args.variants}] "
        f"({n_with_ex} with-examples + {n_control} controls); "
        f"grading ${grade_est:.4f} for {len(gen_tasks) if do_grade else 0} {args.judge_model} "
        f"judge calls); --max-cost gate ${args.max_cost:.2f}"
    )
    if estimate > args.max_cost:
        print(
            f"ABORT: estimate ${estimate:.4f} exceeds --max-cost ${args.max_cost:.2f}",
            file=sys.stderr,
        )
        return 1

    config = {
        "task": "step2-freeform",
        "model": args.model,
        "judge_model": args.judge_model if do_grade else None,
        "gen_price_per_mtok": gen_prices,
        "judge_price_per_mtok": judge_prices if do_grade else None,
        "rules": rules,
        "run_seed": args.run_seed,
        "context_seeds": [args.run_seed + i for i in range(N_CONTEXTS)],
        "n_contexts": N_CONTEXTS,
        "k_few_shot": K_FEW_SHOT,
        "variants": {v: len(ph) for v, ph in FREEFORM_VARIANTS.items()},
        "generations_per_rule": n_with_ex // len(rules) if rules else 0,
        "controls_per_rule": n_control // len(rules) if rules else 0,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "api_seed": API_SEED,
        "grading": do_grade,
        "n_judge_examples": N_JUDGE_EXAMPLES,
        "n_probes": N_PROBES,
        "concurrency": args.concurrency,
        "freeform_template_hash": freeform_template_hash(),
        "rubric_hash": rubric_hash() if do_grade else None,
        "expected_generations": len(gen_tasks),
        "data_dir": str(args.data_dir),
        "contexts": contexts_meta,
    }
    run = start_run(
        name=f"step2-freeform-{args.model}",
        config=config,
        cost_estimate_usd=estimate,
        results_dir=args.results_dir,
    )
    print(
        f"run dir: {run.run_dir}  (generations: {len(gen_tasks)}, "
        f"grading: {'on' if do_grade else 'off'})"
    )

    t0 = time.monotonic()
    gen_rows, graded, client_stats, cost_summary = asyncio.run(
        execute(
            gen_tasks,
            run,
            args.model,
            args.judge_model,
            do_grade,
            spec_rules,
            args.data_dir,
            args.concurrency,
            args.cache_dir,
            api=api,
        )
    )
    wall = time.monotonic() - t0

    metrics = compute_metrics(graded, args.model, args.judge_model) if do_grade else {
        "task": "step2-freeform",
        "model": args.model,
        "n_generations": len(gen_rows),
        "grading": False,
    }
    metrics["wall_seconds"] = wall
    metrics["client_stats"] = client_stats
    run.write_metrics(metrics)
    run.finish(
        cost_actual_usd=cost_summary["total_usd"],
        extra={"client_stats": client_stats, "cost": cost_summary},
    )

    print(
        f"done: {len(gen_rows)} generations"
        + (f", {len(graded)} graded" if do_grade else "")
        + f", {wall:.1f}s, actual cost ${cost_summary['total_usd']:.4f} "
        f"(estimate ${estimate:.4f}), 429s={client_stats['n_429']} "
        f"cache_hits={client_stats['cache_hits']}"
    )
    if do_grade:
        for rule_id, rm in metrics["rules"].items():
            print(
                f"  {rule_id}: primary(median direct)={rm['primary_median_direct']} "
                f"secondary(best)={rm['secondary_best_variant']} "
                f"grades={rm['grade_counts']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
