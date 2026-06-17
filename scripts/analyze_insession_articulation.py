#!/usr/bin/env python
"""in-session — analyze the in-session per-item classify-then-articulate runs (NO API).

Pure aggregation over results/insession-*/responses.jsonl, keyed on
(model, experiment) where experiment in {exp1-no-cot, exp2-cot}. Per
rule × model × experiment:
  - in-session classification accuracy (+ Wilson CI, parse-fail rate, per-class),
  - per-TURN-POSITION accuracy curve (does accuracy drift as the model accumulates
    its own answers — a metric the CoT same-session batch could not produce),
  - articulation grade 0/1/2 distribution + median + grade-2 ("names true rule") rate.
Plus the Exp1-vs-Exp2 delta per (model, rule), and the frozen single-call Step-1
accuracy (from metrics_table.csv) as the clean "can it classify" reference.

Writes results/figures/insession_articulation.json + _metrics.csv. The gpt-4.1
prompted-CoT and claude adaptive-thinking arms are kept SEPARATE (cot_mode).
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from icl_articulation.rule_ids import canonical_rule_id
from icl_articulation.stats import wilson_ci

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"


def _runs() -> list[Path]:
    found = list(RESULTS.glob("insession-*")) + list(RESULTS.glob("*/insession-*"))
    return sorted(d for d in found if d.is_dir() and (d / "responses.jsonl").is_file())


def _step1_baseline() -> dict[tuple[str, str], float | None]:
    path = FIGDIR / "metrics_table.csv"
    out: dict[tuple[str, str], float | None] = {}
    if not path.is_file():
        return out
    for r in csv.DictReader(path.open()):
        try:
            out[(r["rule_id"], r["model"])] = float(r.get("step1_heldout_acc", ""))
        except (TypeError, ValueError):
            out[(r["rule_id"], r["model"])] = None
    return out


def _acc(k: int, n: int) -> dict[str, Any]:
    if n == 0:
        return {"rate": None, "k": 0, "n": 0, "ci": [None, None]}
    lo, hi = wilson_ci(k, n)
    return {"rate": k / n, "k": k, "n": n, "ci": [lo, hi]}


def analyze_run(run_dir: Path) -> dict[str, Any]:
    cfg = json.loads((run_dir / "config.json").read_text())
    model, experiment = cfg["model"], cfg.get("experiment", "")
    cot_mode = cfg.get("cot_mode", "")
    rows = [json.loads(l) for l in (run_dir / "responses.jsonl").read_text().splitlines() if l.strip()]
    by_rule: dict[str, dict[str, list]] = {}
    for r in rows:
        by_rule.setdefault(r["rule_id"], {}).setdefault(r["kind"], []).append(r)

    out: dict[str, Any] = {}
    for rule_id, kinds in sorted(by_rule.items()):
        base = canonical_rule_id(rule_id)
        clf = kinds.get("classify", [])
        n_correct = sum(1 for r in clf if r["correct"])
        n_total = len(clf)
        n_parsefail = sum(1 for r in clf if not r["parse_ok"])
        # per-turn-position accuracy (pooled over the 3 contexts)
        by_turn: dict[int, list[bool]] = {}
        for r in clf:
            by_turn.setdefault(r["turn_index"], []).append(r["correct"])
        turn_curve = [{"turn": t, "n": len(v), "accuracy": sum(v) / len(v)}
                      for t, v in sorted(by_turn.items())]
        # per-class (degeneracy guard)
        n_t = sum(1 for r in clf if r["true_label"]); n_tc = sum(1 for r in clf if r["true_label"] and r["correct"])
        n_f = n_total - n_t; n_fc = n_correct - n_tc
        # articulation
        grades = [g["grade"] for g in kinds.get("grade", [])]
        gc = {str(k): grades.count(k) for k in (0, 1, 2)}
        out[base] = {
            "rule_id": base, "model": model, "experiment": experiment, "cot_mode": cot_mode,
            "classification": {**_acc(n_correct, n_total), "n_parse_fail": n_parsefail,
                               "per_class": {"true": _acc(n_tc, n_t), "false": _acc(n_fc, n_f)},
                               "turn_curve": turn_curve},
            "articulation": {"grades": grades, "grade_counts": gc,
                             "median": statistics.median(grades) if grades else None,
                             "grade2_names_true_rate": (grades.count(2) / len(grades)) if grades else None,
                             "candidates": [g["candidate"] for g in kinds.get("grade", [])]},
        }
    return {"model": model, "experiment": experiment, "cot_mode": cot_mode, "run": run_dir.name, "rules": out}


def main() -> None:
    runs = _runs()
    if not runs:
        print("[in-session] no results/insession-* runs found — skipping (no-op).")
        return
    per_run = [analyze_run(d) for d in runs]
    base = _step1_baseline()

    # Exp1-vs-Exp2 join per (model, rule)
    index: dict[tuple[str, str, str], dict] = {}  # (model, experiment, rule) -> block
    models, rules = set(), set()
    for pr in per_run:
        for rule, b in pr["rules"].items():
            index[(pr["model"], pr["experiment"], rule)] = b
            models.add(pr["model"]); rules.add(rule)
    comparison = []
    for m in sorted(models):
        for rule in sorted(rules):
            e1 = index.get((m, "exp1-no-cot", rule)); e2 = index.get((m, "exp2-cot", rule))
            row = {"model": m, "rule_id": rule, "step1_single_call_acc": base.get((rule, m))}
            for tag, e in (("exp1_nocot", e1), ("exp2_cot", e2)):
                if e:
                    row[f"{tag}_classify_acc"] = e["classification"]["rate"]
                    row[f"{tag}_grade_median"] = e["articulation"]["median"]
                    row[f"{tag}_grade2_rate"] = e["articulation"]["grade2_names_true_rate"]
            if e1 and e2:
                row["acc_delta_cot_minus_nocot"] = (
                    (e2["classification"]["rate"] or 0) - (e1["classification"]["rate"] or 0))
                row["grade2_delta_cot_minus_nocot"] = (
                    (e2["articulation"]["grade2_names_true_rate"] or 0)
                    - (e1["articulation"]["grade2_names_true_rate"] or 0))
            comparison.append(row)

    out = {
        "_measurement": "in-session — in-session per-item classify-then-articulate (Exp1 no-CoT / Exp2 with-CoT)",
        "_caveat": ("Exp1 vs Exp2 differ ONLY by CoT (Exp2). gpt-4.1 prompted-CoT and claude "
                    "adaptive-thinking are different interventions and never pooled. In-session "
                    "accuracy uses a GROWING conversation (the model sees its own prior answers) and "
                    "is NOT identical to the base independent-call Step-1 accuracy, shown for reference."),
        "per_run": per_run, "exp1_vs_exp2": comparison,
    }
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / "insession_articulation.json").write_text(json.dumps(out, indent=2) + "\n")

    cols = ["model", "rule_id", "step1_single_call_acc",
            "exp1_nocot_classify_acc", "exp2_cot_classify_acc", "acc_delta_cot_minus_nocot",
            "exp1_nocot_grade_median", "exp2_cot_grade_median",
            "exp1_nocot_grade2_rate", "exp2_cot_grade2_rate", "grade2_delta_cot_minus_nocot"]
    with (FIGDIR / "insession_articulation_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for row in comparison:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    print("[in-session] wrote results/figures/insession_articulation.json + _metrics.csv\n")
    hdr = f"{'rule':<24}{'model':<16}{'step1':>7}{'e1_acc':>8}{'e2_acc':>8}{'e1_g2':>7}{'e2_g2':>7}"
    print(hdr); print("-" * len(hdr))
    for row in comparison:
        def f(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else " na"
        print(f"{row['rule_id']:<24}{row['model']:<16}{f(row.get('step1_single_call_acc')):>7}"
              f"{f(row.get('exp1_nocot_classify_acc')):>8}{f(row.get('exp2_cot_classify_acc')):>8}"
              f"{f(row.get('exp1_nocot_grade2_rate')):>7}{f(row.get('exp2_cot_grade2_rate')):>7}")


if __name__ == "__main__":
    main()
