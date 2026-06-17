#!/usr/bin/env python
"""Re-grade the existing Step-2 free-form articulations with a SECOND, non-subject
judge and report judge agreement (addresses the judge-circularity limitation, M4).

The primary grade uses a gpt-4.1 judge — the SAME family as the subject models, so
a sceptic can argue the grade reflects the model grading itself. This replays each
logged judge prompt (identical rubric + gold + candidate + examples) to a second
judge (default gpt-4o, a non-subject model) and reports, per rule and overall, how
often the two judges agree. High agreement => the grades are not an artefact of
self-grading. The programmatic extensional check (scripts/analyze_confounds.py) is
the judge-independent corroboration; this is the cross-model-judge corroboration.

Makes paid API calls (set OPENAI_API_KEY). Generations are reused from the
primary run dirs; only the second-judge calls are new.

Run:  python scripts/regrade_second_judge.py --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.client import OpenAIClient, response_text
from icl_articulation.grading import parse_judge
from icl_articulation.prices import cost_usd, price_for
from icl_articulation.runlog import start_run

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_PATH = RESULTS / "figures" / "judge_agreement.json"
MODELS = ["gpt-4.1", "gpt-4.1-mini"]
PRIMARY_JUDGE = "gpt-4.1"


def _latest_freeform(model: str) -> Path | None:
    prefix = "step2-freeform-"
    out: list[Path] = []
    for d in RESULTS.glob(prefix + "*"):
        if not (d / "responses.jsonl").is_file():
            continue
        rest = d.name[len(prefix):]
        owner = next((m for m in sorted(MODELS, key=len, reverse=True)
                      if rest.startswith(m + "-")), None)
        if owner == model:
            out.append(d)
    return sorted(out)[-1] if out else None


def _subject_model_for_run(run: Path) -> str:
    config_path = run / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model = config.get("model")
            if isinstance(model, str) and model:
                return model
        except json.JSONDecodeError:
            pass
    prefix = "step2-freeform-"
    if run.name.startswith(prefix):
        rest = run.name[len(prefix):]
        for model in sorted(MODELS + ["deepseek-v4-flash"], key=len, reverse=True):
            if rest.startswith(model + "-"):
                return model
    return run.name


def load_primary_grades_from_run(run: Path) -> list[dict[str, Any]]:
    """Direct, with-examples grade rows from a specific primary run."""
    rows_path = run / "responses.jsonl"
    if not rows_path.is_file():
        raise FileNotFoundError(f"{run}: missing responses.jsonl")
    rows = [json.loads(l) for l in rows_path.open()]
    subject_model = _subject_model_for_run(run)
    out = []
    for r in rows:
        if (r.get("kind") == "grade" and r.get("has_examples")
                and r.get("variant") == "direct"):
            row = dict(r)
            row["_primary_run_dir"] = str(run)
            row["_subject_model"] = subject_model
            out.append(row)
    return out


def load_primary_grades(model: str) -> list[dict[str, Any]]:
    """Direct, with-examples grade rows from the primary run (the metric variant),
    each carrying the judge request to replay and the primary grade to compare."""
    run = _latest_freeform(model)
    if run is None:
        return []
    return load_primary_grades_from_run(run)


async def regrade(
    judge_model: str,
    max_cost: float,
    concurrency: int = 8,
    run_dirs: list[Path] | None = None,
    results_dir: Path = RESULTS,
    cache_dir: Path = REPO / "cache",
) -> dict[str, Any]:
    jobs: list[tuple[str, dict]] = []  # (subject_model, primary_grade_row)
    input_run_dirs: list[str] = []
    if run_dirs:
        for run in run_dirs:
            input_run_dirs.append(str(run))
            for row in load_primary_grades_from_run(run):
                jobs.append((row["_subject_model"], row))
    else:
        for model in MODELS:
            rows = load_primary_grades(model)
            if rows:
                input_run_dirs.append(rows[0]["_primary_run_dir"])
            for row in rows:
                jobs.append((model, row))
    if not jobs:
        raise SystemExit("no primary direct grades found — run step2_freeform first")

    # advance estimate
    est = 0.0
    for _, row in jobs:
        req = row["request"]
        ptoks = sum(len(m["content"]) for m in req["messages"]) // 4
        est += cost_usd(judge_model, ptoks + 10, req.get("max_tokens", 400))
    print(f"advance cost estimate: ${est:.4f} for {len(jobs)} second-judge calls "
          f"({judge_model}); --max-cost ${max_cost:.2f}")
    if est > max_cost:
        raise SystemExit(f"ABORT: estimate ${est:.4f} exceeds --max-cost ${max_cost:.2f}")

    run = start_run(
        name=f"step2-regrade-{judge_model}",
        config={"task": "step2-second-judge", "judge_model": judge_model,
                "primary_judge": PRIMARY_JUDGE, "n_jobs": len(jobs),
                "input_run_dirs": input_run_dirs,
                "price_per_mtok": price_for(judge_model)},
        cost_estimate_usd=est,
        results_dir=str(results_dir),
    )

    client = OpenAIClient(concurrency=concurrency, cache_dir=str(cache_dir))
    rows_out: list[dict[str, Any]] = []
    try:
        async def one(model: str, prow: dict) -> dict:
            req = prow["request"]
            record = await client.complete(
                judge_model, req["messages"],
                temperature=req.get("temperature", 0.0),
                max_tokens=req.get("max_tokens", 400),
                seed=req.get("seed", 0),
            )
            try:
                verdict = parse_judge(response_text(record))
                grade2 = verdict["grade"]
            except Exception as e:  # noqa: BLE001 — a judge parse failure is data
                grade2 = None
                verdict = {"rationale": f"PARSE_FAIL: {e}"}
            return {
                "subject_model": model,
                "primary_run_dir": prow.get("_primary_run_dir"),
                "rule_id": prow["rule_id"],
                "context_index": prow.get("context_index"),
                "phrasing": prow.get("phrasing"),
                "candidate": prow.get("candidate"),
                "primary_grade": prow.get("grade"),
                "second_grade": grade2,
                "second_judge": judge_model,
                "second_rationale": verdict.get("rationale", ""),
                **record,
            }

        futs = [asyncio.ensure_future(one(m, r)) for m, r in jobs]
        for fut in asyncio.as_completed(futs):
            row = await fut
            run.log_response(row)
            rows_out.append(row)
        cost = client.cost.summary()
    finally:
        await client.aclose()

    summary = summarize(rows_out, judge_model)
    run.write_metrics(summary)
    run.finish(cost_actual_usd=cost["total_usd"], extra={"cost": cost})
    print(f"done: {len(rows_out)} regrades, actual cost ${cost['total_usd']:.4f}")
    return {"summary": summary, "run_dir": run.run_dir.name, "rows": rows_out}


def summarize(rows: list[dict], judge_model: str) -> dict[str, Any]:
    comparable = [r for r in rows if r["primary_grade"] is not None and r["second_grade"] is not None]
    n = len(comparable)
    exact = sum(1 for r in comparable if r["primary_grade"] == r["second_grade"])
    within1 = sum(1 for r in comparable if abs(r["primary_grade"] - r["second_grade"]) <= 1)
    per_rule: dict[str, dict] = {}
    for r in comparable:
        d = per_rule.setdefault(r["rule_id"], {"n": 0, "exact": 0,
                                               "primary": Counter(), "second": Counter()})
        d["n"] += 1
        d["exact"] += int(r["primary_grade"] == r["second_grade"])
        d["primary"][r["primary_grade"]] += 1
        d["second"][r["second_grade"]] += 1
    for d in per_rule.values():
        d["exact_agreement"] = d["exact"] / d["n"] if d["n"] else None
        d["primary"] = dict(d["primary"])
        d["second"] = dict(d["second"])
    return {
        "primary_judge": PRIMARY_JUDGE,
        "second_judge": judge_model,
        "n_comparable": n,
        "n_parse_failures": len(rows) - n,
        "exact_agreement": exact / n if n else None,
        "within_one_agreement": within1 / n if n else None,
        "per_rule": per_rule,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge-model", default="gpt-4o", help="non-subject second judge")
    p.add_argument("--max-cost", type=float, default=200.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        help="explicit Step-2 free-form run dir to regrade; repeat for multiple runs",
    )
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--cache-dir", type=Path, default=REPO / "cache")
    p.add_argument("--out-path", type=Path, default=OUT_PATH)
    args = p.parse_args(argv)

    result = asyncio.run(
        regrade(
            args.judge_model,
            args.max_cost,
            args.concurrency,
            args.run_dir,
            args.results_dir,
            args.cache_dir,
        )
    )
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8")
    s = result["summary"]
    print(f"\n=== judge agreement ({s['primary_judge']} vs {s['second_judge']}) ===")
    print(f"exact {s['exact_agreement']:.3f}  within-1 {s['within_one_agreement']:.3f}  "
          f"(n={s['n_comparable']}, parse-fail={s['n_parse_failures']})")
    print(f"wrote {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
