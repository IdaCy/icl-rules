#!/usr/bin/env python
"""Step-1 classification runner. Makes paid API calls (set OPENAI_API_KEY).

Modes (locked):
  pilot         subset of rules, 1 context, 40 held-out items (20/20)
  full          all selected rules x 3 contexts x 120 held-out items (60/60)
  confirmation  survivor rules, 3 contexts, the 100 confirmation items (50/50)
  rule_given    zero-shot baseline: NO examples, canonical rule text as the
                instruction (same answer-format line); needs --rules-file

Every call: temperature=0, max_tokens=2, logprobs=True, top_logprobs=5,
seed=0. Contexts: k=32, 16/16, distinct bases, shuffled with the logged seed
(context seeds = run_seed + 0,1,2). The advance cost estimate is printed FIRST
and gated by --max-cost (default $200). Responses are logged per task AS IT
COMPLETES (crash leaves a valid partial responses.jsonl); re-running the same
config resumes for free via the client's disk cache.

Example:
  python scripts/run_step1.py --mode pilot --model gpt-4.1 --rules all_lowercase,contains_digit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.client import OpenAIClient, first_token_logprobs, response_text
from icl_articulation.contexts import DatasetError, load_items, sample_context, select_queries
from icl_articulation.prices import cost_usd, price_for
from icl_articulation.prompts import (
    render_rule_given,
    render_step1,
    rule_given_template_hash,
    step1_template_hash,
)
from icl_articulation.rule_ids import canonical_rule_id
from icl_articulation.runlog import RunLog, start_run
from icl_articulation.stats import cluster_bootstrap_ci, wilson_ci

# Locked step-1 call config (PLAN "No-CoT enforcement" + smoke-test findings).
K_FEW_SHOT = 32
TEMPERATURE = 0.0
MAX_TOKENS = 2
TOP_LOGPROBS = 5
API_SEED = 0
PROMPT_TOKEN_OVERHEAD = 10  # chat formatting overhead per call (chars/4 heuristic)

MODES: dict[str, dict[str, Any]] = {
    "pilot": {"split": "held_out", "n_items": 40, "n_contexts": 1, "few_shot": True},
    "full": {"split": "held_out", "n_items": 120, "n_contexts": 3, "few_shot": True},
    "confirmation": {"split": "confirmation", "n_items": 100, "n_contexts": 3, "few_shot": True},
    "rule_given": {"split": "held_out", "n_items": 120, "n_contexts": 1, "few_shot": False},
}


@dataclass
class Task:
    """One classification call (rule x context x query item)."""

    rule_id: str
    context_index: int  # 0-based; 0 for rule_given (no context)
    context_seed: int | None  # None for rule_given
    item: dict[str, Any]
    messages: list[dict[str, str]]


# --- answer parsing ------------------------------------------------------------


def parse_label(text: str) -> bool | None:
    """Robust True/False parse of the emitted completion.

    Strip whitespace, then CASE-INSENSITIVE prefix match (review M3, option a):
    'True', 'true', 'TRUE.' all parse — a semantically correct answer is never
    scored incorrect for casing alone. Exact-case formatting is tracked
    separately per row via format_ok(). A parse failure (None) now means
    NEITHER label is recognizable case-insensitively, and still scores
    INCORRECT (pre-specified).

    PREFIX matching is deliberate — do not "fix" it into an exact match:
    max_tokens=2 means the answer token may be followed by one extra token, so
    'True.' / 'True\\n' MUST parse. A word like 'Truely' cannot sneak through:
    BPE tokenizes it as 'True'+'ly', i.e. the first sampled token is still the
    plain label token, and max_tokens=2 makes anything longer unreachable.
    """
    t = text.strip().lower()
    if t.startswith("true"):
        return True
    if t.startswith("false"):
        return False
    return None


def format_ok(text: str) -> bool:
    """Did the completion start with the exact-case 'True'/'False' token
    (after whitespace strip)? Logged per row and as a per-context rate so the
    no-CoT answer-format claim stays auditable under case-insensitive parsing
    (review M3)."""
    t = text.strip()
    return t.startswith("True") or t.startswith("False")


def answer_logprobs(
    record: dict[str, Any], predicted: bool | None
) -> tuple[list[dict[str, Any]] | None, float | None]:
    """(top-5 at the first answer token, logprob margin).

    Margin = log P(chosen) - log P(other) when BOTH labels appear in the top-5
    (tokens matched after strip, so ' True' variants count); else None.
    """
    entry = first_token_logprobs(record)
    if entry is None:
        return None, None
    tops = entry.get("top_logprobs") or []
    by_label: dict[str, float] = {}
    for t in tops:
        tok = t["token"].strip()
        if tok in ("True", "False"):
            # keep the most probable occurrence if tokenization variants repeat
            by_label[tok] = max(by_label.get(tok, float("-inf")), t["logprob"])
    margin = None
    if predicted is not None and "True" in by_label and "False" in by_label:
        chosen, other = ("True", "False") if predicted else ("False", "True")
        margin = by_label[chosen] - by_label[other]
    return tops, margin


def analyze(task: Task, record: dict[str, Any], mode: str) -> dict[str, Any]:
    """The per-call row that gets logged to responses.jsonl."""
    text = response_text(record)
    predicted = parse_label(text)
    tops, margin = answer_logprobs(record, predicted)
    true_label = task.item["label"]
    return {
        "mode": mode,
        "rule_id": task.rule_id,
        "context_index": task.context_index,
        "context_seed": task.context_seed,
        "item_id": task.item["item_id"],
        "base_id": task.item["base_id"],
        "text": task.item["text"],
        "true_label": true_label,
        "predicted": predicted,
        "parse_ok": predicted is not None,
        # exact-case 'True'/'False' formatting (parsing is case-insensitive)
        "format_ok": format_ok(text),
        # parse failures count as INCORRECT (pre-specified)
        "correct": predicted is not None and predicted == true_label,
        "logprob_margin": margin,
        "answer_top_logprobs": tops,
        **record,
    }


# --- task construction ----------------------------------------------------------


def load_rule_texts(path: str | Path) -> dict[str, str]:
    """Canonical articulations for rule_given mode: {rule_id: rule text} json.

    Values may also be objects with a 'canonical_articulation' key (so a dump
    of the spec yaml can be passed directly).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object mapping rule_id -> rule text")
    texts: dict[str, str] = {}
    for rule_id, value in data.items():
        if isinstance(value, dict):
            value = value.get("canonical_articulation")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: no canonical rule text for rule {rule_id!r}")
        texts[rule_id] = value.strip()
    return texts


def build_tasks(
    mode: str,
    rules: list[str],
    data_dir: str | Path,
    run_seed: int,
    rule_texts: dict[str, str] | None = None,
) -> tuple[list[Task], dict[str, Any]]:
    """All call tasks + per-rule context metadata (for the run config)."""
    spec = MODES[mode]
    data_dir = Path(data_dir)
    tasks: list[Task] = []
    contexts_meta: dict[str, Any] = {}
    for rule_id in rules:
        items = load_items(data_dir / rule_id / "items.jsonl")
        queries = select_queries(items, spec["split"], spec["n_items"])
        if not spec["few_shot"]:
            text_key = rule_id if rule_texts and rule_id in rule_texts else canonical_rule_id(rule_id)
            if rule_texts is None or text_key not in rule_texts:
                raise ValueError(f"rule_given mode: no canonical rule text for {rule_id!r}")
            rule_text = rule_texts[text_key]
            contexts_meta[rule_id] = [{"context_index": 0, "rule_text": rule_text}]
            for item in queries:
                tasks.append(
                    Task(rule_id, 0, None, item, render_rule_given(rule_text, item["text"]))
                )
            continue
        contexts_meta[rule_id] = []
        for ctx_index in range(spec["n_contexts"]):
            seed = run_seed + ctx_index
            context = sample_context(items, k=K_FEW_SHOT, seed=seed)
            examples = [(it["text"], it["label"]) for it in context]
            contexts_meta[rule_id].append(
                {
                    "context_index": ctx_index,
                    "seed": seed,
                    "item_ids": [it["item_id"] for it in context],
                    "base_ids": [it["base_id"] for it in context],
                }
            )
            for item in queries:
                tasks.append(
                    Task(rule_id, ctx_index, seed, item, render_step1(examples, item["text"]))
                )
    return tasks, contexts_meta


def estimate_cost_usd(model: str, tasks: list[Task]) -> float:
    """Advance estimate: chars/4 heuristic on the fully rendered prompts
    + max_tokens completion per call."""
    total = 0.0
    for task in tasks:
        prompt_chars = sum(len(m["content"]) for m in task.messages)
        total += cost_usd(model, int(prompt_chars / 4) + PROMPT_TOKEN_OVERHEAD, MAX_TOKENS)
    return total


# --- execution -------------------------------------------------------------------


async def execute(
    tasks: list[Task],
    run: RunLog,
    model: str,
    mode: str,
    concurrency: int,
    cache_dir: str | Path,
    api: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run all tasks; log each row AS IT COMPLETES (never post-gather).

    One OpenAIClient per asyncio.run (created and closed inside this
    coroutine). Returns (rows, client stats, cost summary).
    """
    client = OpenAIClient(concurrency=concurrency, cache_dir=cache_dir, api=api)
    rows: list[dict[str, Any]] = []
    try:

        async def one(task: Task) -> tuple[Task, dict[str, Any]]:
            record = await client.complete(
                model,
                task.messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
                seed=API_SEED,
            )
            return task, record

        futures = [asyncio.ensure_future(one(t)) for t in tasks]
        try:
            for fut in asyncio.as_completed(futures):
                task, record = await fut
                row = analyze(task, record, mode)
                run.log_response(row)  # incremental: crash leaves a valid partial jsonl
                rows.append(row)
        except BaseException:
            for f in futures:
                f.cancel()
            await asyncio.gather(*futures, return_exceptions=True)
            raise
        return rows, client.stats(), client.cost.summary()
    finally:
        await client.aclose()


# --- metrics ----------------------------------------------------------------------


def _per_class(rows: list[dict[str, Any]], label: bool) -> dict[str, Any]:
    sub = [r for r in rows if r["true_label"] is label]
    n_correct = sum(r["correct"] for r in sub)
    return {
        "n": len(sub),
        "n_correct": n_correct,
        "accuracy": n_correct / len(sub) if sub else None,
    }


def _context_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    margins = [r["logprob_margin"] for r in rows if r["logprob_margin"] is not None]
    return {
        "context_index": rows[0]["context_index"],
        "context_seed": rows[0]["context_seed"],
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n,
        "wilson_ci_95": list(wilson_ci(n_correct, n)),
        "n_parse_failures": sum(1 for r in rows if not r["parse_ok"]),
        "format_ok_rate": sum(1 for r in rows if r["format_ok"]) / n,
        "mean_logprob_margin": sum(margins) / len(margins) if margins else None,
        "n_with_margin": len(margins),
        # catches all-one-class degenerate behavior
        "per_class": {"true": _per_class(rows, True), "false": _per_class(rows, False)},
        "predictions": {
            "true": sum(1 for r in rows if r["predicted"] is True),
            "false": sum(1 for r in rows if r["predicted"] is False),
            "parse_failure": sum(1 for r in rows if r["predicted"] is None),
        },
    }


def compute_metrics(rows: list[dict[str, Any]], mode: str, model: str) -> dict[str, Any]:
    """Pre-registered step-1 metrics (PLAN): per-context accuracy + Wilson CI;
    pooled = mean of per-context accuracies + item-level cluster bootstrap CI
    (only when contexts share items, i.e. full/confirmation). Parse failures
    count as incorrect everywhere and are reported."""
    by_rule: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_rule.setdefault(r["rule_id"], []).append(r)

    rules_out: dict[str, Any] = {}
    for rule_id in sorted(by_rule):
        rule_rows = by_rule[rule_id]
        by_ctx: dict[int, list[dict[str, Any]]] = {}
        for r in rule_rows:
            by_ctx.setdefault(r["context_index"], []).append(r)
        ctx_metrics = [_context_metrics(by_ctx[i]) for i in sorted(by_ctx)]
        accs = [c["accuracy"] for c in ctx_metrics]
        pooled: dict[str, Any] = {
            "n_contexts": len(ctx_metrics),
            "n_items": ctx_metrics[0]["n"],
            "mean_accuracy": sum(accs) / len(accs),
            "n_parse_failures": sum(c["n_parse_failures"] for c in ctx_metrics),
            "cluster_bootstrap_ci_95": None,
        }
        item_id_sets = [
            frozenset(r["item_id"] for r in by_ctx[i]) for i in sorted(by_ctx)
        ]
        if len(ctx_metrics) > 1 and len(set(item_id_sets)) == 1:
            # same items under each context -> item-level cluster bootstrap
            item_order = sorted(item_id_sets[0], key=str)
            col = {item_id: j for j, item_id in enumerate(item_order)}
            arr = np.zeros((len(ctx_metrics), len(item_order)))
            for row_i, ctx in enumerate(sorted(by_ctx)):
                for r in by_ctx[ctx]:
                    arr[row_i, col[r["item_id"]]] = float(r["correct"])
            boot = cluster_bootstrap_ci(arr)
            pooled["cluster_bootstrap_ci_95"] = [boot.low, boot.high]
        rules_out[rule_id] = {"contexts": ctx_metrics, "pooled": pooled}

    n_calls = len(rows)
    n_parse_failures = sum(1 for r in rows if not r["parse_ok"])
    return {
        "mode": mode,
        "model": model,
        "rules": rules_out,
        "overall": {
            "n_calls": n_calls,
            "n_correct": sum(r["correct"] for r in rows),
            "n_parse_failures": n_parse_failures,
            "parse_failure_rate": n_parse_failures / n_calls if n_calls else 0.0,
        },
    }


# --- CLI ----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=sorted(MODES))
    p.add_argument("--model", required=True, help="gpt-4.1 or gpt-4.1-mini")
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--rules", help="comma-separated rule_ids (data/<rule_id>/items.jsonl)")
    sel.add_argument("--all", action="store_true", help="every rule dir under --data-dir")
    p.add_argument("--rules-file", help="json {rule_id: canonical rule text} (rule_given mode)")
    p.add_argument("--run-seed", type=int, default=0, help="global seed; context seeds = run_seed+0,1,2")
    p.add_argument("--max-cost", type=float, default=200.0, help="abort if the estimate exceeds this (USD)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--cache-dir", default="cache")
    args = p.parse_args(argv)
    if args.rules_file and args.mode != "rule_given":
        p.error(f"--rules-file is only used by --mode rule_given (got --mode {args.mode})")
    return args


def resolve_rules(args: argparse.Namespace) -> list[str]:
    if args.all:
        data_dir = Path(args.data_dir)
        rules = sorted(d.name for d in data_dir.iterdir() if (d / "items.jsonl").is_file()) \
            if data_dir.is_dir() else []
        if not rules:
            raise DatasetError(f"--all: no rule datasets found under {data_dir}/")
        return rules
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    if not rules:
        raise DatasetError("--rules: empty rule list")
    return rules


def main(argv: list[str] | None = None, api: Any | None = None) -> int:
    args = parse_args(argv)
    # price sanity check at run start: KeyError (loud) if the model is unpriced
    prices = price_for(args.model)

    rules = resolve_rules(args)
    rule_texts = None
    if args.mode == "rule_given":
        if not args.rules_file:
            print("ERROR: --mode rule_given requires --rules-file", file=sys.stderr)
            return 2
        rule_texts = load_rule_texts(args.rules_file)

    tasks, contexts_meta = build_tasks(args.mode, rules, args.data_dir, args.run_seed, rule_texts)
    spec = MODES[args.mode]

    # advance estimate FIRST, then the gate — before any run dir or API call
    estimate = estimate_cost_usd(args.model, tasks)
    print(
        f"advance cost estimate: ${estimate:.4f} for {len(tasks)} calls to {args.model} "
        f"({len(rules)} rules x {spec['n_contexts']} contexts x {spec['n_items']} items, "
        f"mode={args.mode}); --max-cost gate ${args.max_cost:.2f}"
    )
    if estimate > args.max_cost:
        print(
            f"ABORT: estimate ${estimate:.4f} exceeds --max-cost ${args.max_cost:.2f}",
            file=sys.stderr,
        )
        return 1

    template_hash = step1_template_hash() if spec["few_shot"] else rule_given_template_hash()
    config = {
        "task": "step1-classification",
        "mode": args.mode,
        "model": args.model,
        "price_per_mtok": prices,
        "rules": rules,
        "rules_file": args.rules_file,
        "run_seed": args.run_seed,
        "context_seeds": [args.run_seed + i for i in range(spec["n_contexts"])]
        if spec["few_shot"]
        else None,
        "n_contexts": spec["n_contexts"],
        "k_few_shot": K_FEW_SHOT if spec["few_shot"] else 0,
        "query_split": spec["split"],
        "n_items_per_rule": spec["n_items"],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "api_seed": API_SEED,
        "concurrency": args.concurrency,
        "template_hash": template_hash,
        "expected_total_calls": len(tasks),
        "data_dir": str(args.data_dir),
        "contexts": contexts_meta,
    }
    run = start_run(
        name=f"step1-{args.mode}-{args.model}",
        config=config,
        cost_estimate_usd=estimate,
        results_dir=args.results_dir,
    )
    print(f"run dir: {run.run_dir}  (expected total calls: {len(tasks)})")

    t0 = time.monotonic()
    rows, client_stats, cost_summary = asyncio.run(
        execute(tasks, run, args.model, args.mode, args.concurrency, args.cache_dir, api=api)
    )
    wall = time.monotonic() - t0

    metrics = compute_metrics(rows, args.mode, args.model)
    metrics["wall_seconds"] = wall
    metrics["client_stats"] = client_stats
    run.write_metrics(metrics)
    run.finish(
        cost_actual_usd=cost_summary["total_usd"],
        extra={"client_stats": client_stats, "cost": cost_summary},
    )

    overall = metrics["overall"]
    print(
        f"done: {overall['n_correct']}/{overall['n_calls']} correct, "
        f"{overall['n_parse_failures']} parse failures, {wall:.1f}s, "
        f"actual cost ${cost_summary['total_usd']:.4f} (estimate ${estimate:.4f}), "
        f"429s={client_stats['n_429']} cache_hits={client_stats['cache_hits']}"
    )
    for rule_id, rm in metrics["rules"].items():
        accs = ", ".join(f"{c['accuracy']:.3f}" for c in rm["contexts"])
        print(f"  {rule_id}: per-context acc [{accs}]  pooled {rm['pooled']['mean_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
