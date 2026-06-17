#!/usr/bin/env python
"""Step-3 FAITHFULNESS 3-arm runner. Makes paid API calls (set OPENAI_API_KEY).

For each of the 4 target rules and each EDGE/DIVERGENCE probe (step3_probes),
classify the probe with the model under THREE conditions (no-CoT, temperature 0,
max_tokens=2, parse True/False):

  ARM 1  in_context        the SAME step-1 few-shot block (contexts.sample_context
                           with the SAME seeds step 1 used) + the probe. This is
                           the model's BEHAVIOR — what it learned/does.
  ARM 2  self_application  render_rule_given(model's OWN step-2 verbatim
                           articulation) + the probe, NO examples. What the model
                           does when told its OWN stated rule.
  ARM 3  true_rule_given   render_rule_given(canonical_articulation from the
                           COMMITTED data/spec_extract.json) + the probe. Sanity
                           / upper bound.

The headline question (faithfulness.py): on the EMPIRICAL DIVERGENCE subset
(probes where the model's OWN ARM-2 self-application disagrees with the true
label — i.e. the articulated rule and the true rule give different labels, as the
model itself applies its articulation), does ARM-1 behavior track the TRUE rule
(the model acts on what it LEARNED) or its own ARTICULATION (what it SAID)?
agreement(behavior, true) vs agreement(behavior, self-application), with Wilson
CIs. The build-time Probe.is_divergence tag is only a probe-construction hint.

All calls go through client.py; the advance cost estimate is printed FIRST and
gated by --max-cost. Per-arm rows are logged AS THEY COMPLETE (crash leaves a
valid partial responses.jsonl). config + probes + per-arm rows + metrics land in
results/<run_id>/.

Example:
  python scripts/run_step3_faithfulness.py --model gpt-4.1-mini \\
      --rules food_topic,physically_impossible
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.client import OpenAIClient, response_text
from icl_articulation.contexts import DatasetError, load_items, sample_context
from icl_articulation.faithfulness import (
    ARM_IN_CONTEXT,
    ARM_SELF,
    ARM_TRUE,
    ARMS,
    ArmPredictions,
    analyze,
)
from icl_articulation.grading import gold_for, load_spec_extract
from icl_articulation.prices import cost_usd, price_for
from icl_articulation.prompts import (
    render_rule_given,
    render_step1,
    rule_given_template_hash,
    step1_template_hash,
)
from icl_articulation.rule_ids import canonical_rule_id
from icl_articulation.runlog import RunLog, start_run
from icl_articulation.step3_probes import (
    TARGET_RULES,
    Probe,
    articulation_for,
    build_probe_set,
)

# --- locked classification config (mirrors step-1 no-CoT classification) -------
K_FEW_SHOT = 32
TEMPERATURE = 0.0
MAX_TOKENS = 2  # the answer token (+ 1 for a trailing '.'/'\n'); no CoT
API_SEED = 0
DEFAULT_CONTEXT_SEED = 0  # ARM-1 few-shot block seed (== step-1 context 0)
PROMPT_TOKEN_OVERHEAD = 10  # chat formatting overhead per call (chars/4 heuristic)


# --- answer parsing (identical contract to step 1) -----------------------------


def parse_label(text: str) -> bool | None:
    """Robust True/False parse: strip, then case-insensitive prefix match.

    Same contract as scripts/run_step1.parse_label — 'True', 'true', 'TRUE.' all
    parse; a None means neither label is recognizable (counted as unparsed in the
    agreement denominators, never silently scored)."""
    t = text.strip().lower()
    if t.startswith("true"):
        return True
    if t.startswith("false"):
        return False
    return None


# --- task construction ---------------------------------------------------------


@dataclass
class ProbeTask:
    """One classification call: (rule, probe, arm)."""

    rule_id: str
    probe: Probe
    arm: str  # ARM_IN_CONTEXT | ARM_SELF | ARM_TRUE
    messages: list[dict[str, str]]


@dataclass
class RuleBuild:
    """Everything built for one rule before any API call."""

    rule_id: str
    probes: list[Probe]
    articulation: str  # the model's OWN stated rule (arm 2)
    canonical: str  # the canonical articulation (arm 3)
    context_seed: int
    context_item_ids: list[Any]
    tasks: list[ProbeTask] = field(default_factory=list)


def build_rule(
    rule_id: str,
    data_dir: str | Path,
    spec_rules: dict[str, Any],
    model: str,
    context_seed: int,
    n_in_distribution: int,
    articulations: dict[str, str] | None = None,
) -> RuleBuild:
    """Build the probe set + the 3 arms' messages for one rule.

    ARM 1 shares the step-1 few-shot block (sample_context, same k/seed). ARMS 2/3
    are render_rule_given with the model's OWN articulation and the canonical
    articulation respectively (no examples)."""
    items = load_items(Path(data_dir) / rule_id / "items.jsonl")
    context = sample_context(items, k=K_FEW_SHOT, seed=context_seed)
    examples = [(it["text"], bool(it["label"])) for it in context]

    probes = build_probe_set(rule_id, data_dir, n_in_distribution, model=model)
    articulation = _articulation_for_run(rule_id, model, articulations)
    canonical = gold_for(spec_rules, rule_id)["canonical_articulation"]

    tasks: list[ProbeTask] = []
    for probe in probes:
        tasks.append(
            ProbeTask(rule_id, probe, ARM_IN_CONTEXT, render_step1(examples, probe.text))
        )
        tasks.append(
            ProbeTask(rule_id, probe, ARM_SELF, render_rule_given(articulation, probe.text))
        )
        tasks.append(
            ProbeTask(rule_id, probe, ARM_TRUE, render_rule_given(canonical, probe.text))
        )
    return RuleBuild(
        rule_id=rule_id,
        probes=probes,
        articulation=articulation,
        canonical=canonical,
        context_seed=context_seed,
        context_item_ids=[it["item_id"] for it in context],
        tasks=tasks,
    )


def load_articulations_file(path: str | Path) -> dict[str, str]:
    """Load a Step-3 articulation override file.

    Accepted shapes are `{rule_id: text}` and `{"articulations": {rule_id: text}}`.
    Values may also be objects with a `candidate` or `articulation` string.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object")
    data = raw.get("articulations", raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an articulations object")
    out: dict[str, str] = {}
    for rule_id, value in data.items():
        if isinstance(value, dict):
            value = value.get("candidate") or value.get("articulation")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: no articulation string for {rule_id!r}")
        out[str(rule_id)] = value.strip()
    return out


def _articulation_for_run(
    rule_id: str, model: str, articulations: dict[str, str] | None
) -> str:
    if articulations:
        if rule_id in articulations:
            return articulations[rule_id]
        base_rule_id = canonical_rule_id(rule_id)
        if base_rule_id in articulations:
            return articulations[base_rule_id]
        raise DatasetError(
            f"--articulations-file has no entry for {rule_id!r} or {base_rule_id!r}"
        )
    return articulation_for(rule_id, model)


def estimate_cost_usd(model: str, tasks: list[ProbeTask]) -> float:
    """Advance estimate: chars/4 prompt heuristic + MAX_TOKENS completion budget
    per call (one call per (rule, probe, arm))."""
    total = 0.0
    for task in tasks:
        prompt_chars = sum(len(m["content"]) for m in task.messages)
        total += cost_usd(model, int(prompt_chars / 4) + PROMPT_TOKEN_OVERHEAD, MAX_TOKENS)
    return total


# --- classification ------------------------------------------------------------


def classify_row(task: ProbeTask, record: dict[str, Any]) -> dict[str, Any]:
    """The per-arm row logged to responses.jsonl."""
    text = response_text(record)
    predicted = parse_label(text)
    p = task.probe
    return {
        "kind": "classification",
        "rule_id": task.rule_id,
        "arm": task.arm,
        "probe_id": p.probe_id,
        "text": p.text,
        "source": p.source,
        "is_divergence": p.is_divergence,
        "true_label": p.true_label,
        "true_label_source": p.true_label_source,
        "art_label": p.art_label,
        "completion_text": text,
        "predicted": predicted,
        "parse_ok": predicted is not None,
        "correct_vs_true": predicted is not None and predicted == p.true_label,
        **record,
    }


async def run_arms(
    tasks: list[ProbeTask],
    run: RunLog,
    client: OpenAIClient,
    model: str,
) -> list[dict[str, Any]]:
    """Run every (rule, probe, arm) call; log each row AS IT COMPLETES."""
    rows: list[dict[str, Any]] = []

    async def one(task: ProbeTask) -> tuple[ProbeTask, dict[str, Any]]:
        record = await client.complete(
            model,
            task.messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=API_SEED,
        )
        return task, record

    futures = [asyncio.ensure_future(one(t)) for t in tasks]
    try:
        for fut in asyncio.as_completed(futures):
            task, record = await fut
            row = classify_row(task, record)
            run.log_response(row)
            rows.append(row)
    except BaseException:
        for f in futures:
            f.cancel()
        await asyncio.gather(*futures, return_exceptions=True)
        raise
    return rows


# --- metrics -------------------------------------------------------------------


def assemble_predictions(
    builds: dict[str, RuleBuild], rows: list[dict[str, Any]]
) -> dict[str, ArmPredictions]:
    """Group the logged per-arm rows back into per-rule ArmPredictions, aligned
    to each rule's probe order. A missing (probe, arm) row -> None (parse-failure
    semantics; the analysis drops it from the denominator)."""
    # (rule, probe_id, arm) -> predicted
    by_key: dict[tuple[str, str, str], bool | None] = {}
    for r in rows:
        by_key[(r["rule_id"], r["probe_id"], r["arm"])] = r["predicted"]

    per_rule: dict[str, ArmPredictions] = {}
    for rule_id, build in builds.items():
        cols: dict[str, list[bool | None]] = {arm: [] for arm in ARMS}
        for probe in build.probes:
            for arm in ARMS:
                cols[arm].append(by_key.get((rule_id, probe.probe_id, arm)))
        per_rule[rule_id] = ArmPredictions(
            rule_id=rule_id,
            probes=build.probes,
            in_context=cols[ARM_IN_CONTEXT],
            self_application=cols[ARM_SELF],
            true_rule_given=cols[ARM_TRUE],
        )
    return per_rule


def compute_metrics(
    builds: dict[str, RuleBuild],
    rows: list[dict[str, Any]],
    model: str,
    articulations_file: str | None = None,
) -> dict[str, Any]:
    """Per-rule + cross-rule faithfulness metrics (faithfulness.analyze)."""
    per_rule = assemble_predictions(builds, rows)
    analysis = analyze(per_rule)
    parse_fail = sum(1 for r in rows if not r["parse_ok"])
    return {
        "task": "step3-faithfulness",
        "model": model,
        "arms": list(ARMS),
        "n_classifications": len(rows),
        "n_parse_failures": parse_fail,
        "articulations_file": articulations_file,
        "articulation_override": articulations_file is not None,
        "art_label_source": (
            "legacy_static_probe_predicates"
            if articulations_file is not None
            else "legacy_static_articulation_predicates"
        ),
        "art_label_note": (
            "Fresh articulations are applied to ARM 2 prompts. Static art_label "
            "fields remain legacy probe-predicate diagnostics; use fixed_designed "
            "behavior-vs-self metrics for model self-application comparisons."
            if articulations_file is not None
            else None
        ),
        "faithfulness": analysis,
    }


# --- execution -----------------------------------------------------------------


async def execute(
    tasks: list[ProbeTask],
    run: RunLog,
    model: str,
    concurrency: int,
    cache_dir: str | Path,
    api: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    client = OpenAIClient(concurrency=concurrency, cache_dir=cache_dir, api=api)
    try:
        rows = await run_arms(tasks, run, client, model)
        return rows, client.stats(), client.cost.summary()
    finally:
        await client.aclose()


# --- CLI -----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="model (gpt-4.1 or gpt-4.1-mini)")
    p.add_argument("--rules", help="comma-separated rule_ids (default: the 4 targets)")
    p.add_argument(
        "--context-seed",
        type=int,
        default=DEFAULT_CONTEXT_SEED,
        help="ARM-1 few-shot block seed (default 0 == step-1 context 0)",
    )
    p.add_argument(
        "--n-in-distribution",
        type=int,
        default=40,
        help="balanced held-out items per probe set (default 40; >=50 total/rule)",
    )
    p.add_argument("--max-cost", type=float, default=200.0, help="abort if estimate exceeds this (USD)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--spec-extract", default="data/spec_extract.json")
    p.add_argument(
        "--articulations-file",
        help="JSON mapping rule_id -> fresh Step-2 articulation for ARM 2; default uses legacy constants",
    )
    return p.parse_args(argv)


def resolve_rules(args: argparse.Namespace) -> list[str]:
    if args.rules:
        rules = [r.strip() for r in args.rules.split(",") if r.strip()]
        if not rules:
            raise DatasetError("--rules: empty rule list")
        bad = [r for r in rules if canonical_rule_id(r) not in TARGET_RULES]
        if bad:
            raise DatasetError(f"--rules: {bad} are not step-3 targets {list(TARGET_RULES)}")
        return rules
    return list(TARGET_RULES)


def main(argv: list[str] | None = None, api: Any | None = None) -> int:
    args = parse_args(argv)
    prices = price_for(args.model)  # KeyError (loud) if unpriced

    rules = resolve_rules(args)
    spec_rules = load_spec_extract(args.spec_extract)
    articulations = (
        load_articulations_file(args.articulations_file)
        if args.articulations_file
        else None
    )

    builds: dict[str, RuleBuild] = {}
    all_tasks: list[ProbeTask] = []
    for rule_id in rules:
        build = build_rule(
            rule_id,
            args.data_dir,
            spec_rules,
            args.model,
            args.context_seed,
            args.n_in_distribution,
            articulations,
        )
        builds[rule_id] = build
        all_tasks.extend(build.tasks)

    # advance estimate FIRST, then the gate — before any run dir or API call
    estimate = estimate_cost_usd(args.model, all_tasks)
    n_probes = sum(len(b.probes) for b in builds.values())
    print(
        f"advance cost estimate: ${estimate:.4f} for {len(all_tasks)} calls to "
        f"{args.model} ({len(rules)} rules x {n_probes} probes total x "
        f"{len(ARMS)} arms); --max-cost gate ${args.max_cost:.2f}"
    )
    if estimate > args.max_cost:
        print(
            f"ABORT: estimate ${estimate:.4f} exceeds --max-cost ${args.max_cost:.2f}",
            file=sys.stderr,
        )
        return 1

    probes_meta = {
        rule_id: {
            "n_probes": len(b.probes),
            "n_divergence": sum(p.is_divergence for p in b.probes),
            "articulation": b.articulation,
            "canonical_articulation": b.canonical,
            "context_seed": b.context_seed,
            "context_item_ids": b.context_item_ids,
            "probes": [p.to_row() for p in b.probes],
        }
        for rule_id, b in builds.items()
    }
    config = {
        "task": "step3-faithfulness",
        "model": args.model,
        "price_per_mtok": prices,
        "rules": rules,
        "arms": list(ARMS),
        "context_seed": args.context_seed,
        "k_few_shot": K_FEW_SHOT,
        "n_in_distribution": args.n_in_distribution,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "api_seed": API_SEED,
        "concurrency": args.concurrency,
        "step1_template_hash": step1_template_hash(),
        "rule_given_template_hash": rule_given_template_hash(),
        "expected_classifications": len(all_tasks),
        "data_dir": str(args.data_dir),
        "articulations_file": str(args.articulations_file) if args.articulations_file else None,
        "art_label_source": (
            "legacy_static_probe_predicates"
            if args.articulations_file
            else "legacy_static_articulation_predicates"
        ),
        "probes_per_rule": probes_meta,
    }
    run = start_run(
        name=f"step3-faithfulness-{args.model}",
        config=config,
        cost_estimate_usd=estimate,
        results_dir=args.results_dir,
    )
    print(f"run dir: {run.run_dir}  ({len(all_tasks)} classifications across {len(ARMS)} arms)")

    t0 = time.monotonic()
    rows, client_stats, cost_summary = asyncio.run(
        execute(all_tasks, run, args.model, args.concurrency, args.cache_dir, api=api)
    )
    wall = time.monotonic() - t0

    metrics = compute_metrics(
        builds,
        rows,
        args.model,
        str(args.articulations_file) if args.articulations_file else None,
    )
    metrics["wall_seconds"] = wall
    metrics["client_stats"] = client_stats
    run.write_metrics(metrics)
    run.finish(
        cost_actual_usd=cost_summary["total_usd"],
        extra={"client_stats": client_stats, "cost": cost_summary},
    )

    print(
        f"done: {len(rows)} classifications, {wall:.1f}s, actual cost "
        f"${cost_summary['total_usd']:.4f} (estimate ${estimate:.4f}), "
        f"429s={client_stats['n_429']} cache_hits={client_stats['cache_hits']} "
        f"parse_failures={metrics['n_parse_failures']}"
    )
    for row in metrics["faithfulness"]["summary"]:
        gap = row.get("gap_true_minus_self")
        print(
            f"  {row['rule_id']}: faithfulness(all)="
            f"{_fmt(row.get('faithfulness_overall'))} "
            f"faithfulness(emp-div)={_fmt(row.get('faithfulness_divergence'))} "
            f"| emp-div n={row.get('n_divergence')} (proxy n={row.get('n_divergence_proxy')}): "
            f"behavior~true={_fmt(row.get('behavior_tracks_true_rate'))} "
            f"behavior~self={_fmt(row.get('behavior_tracks_self_rate'))} "
            f"gap={_fmt(gap)} [{row.get('interpretation', 'n/a')}]"
        )
    return 0


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
