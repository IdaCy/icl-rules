#!/usr/bin/env python
"""CoT same-session — analyze the CoT same-session classify-then-articulate runs (NO API).

Pure aggregation: the runner already classified, articulated, judged, compiled,
and ran the sandbox (storing result_vs_gold / result_vs_self per compile row), so
this just reads results/cot-same-session-*/responses.jsonl and emits the metrics:

  - CoT in-distribution accuracy (turn-1 labels vs gold) + Wilson CI + parse-fail
    rate, per (model, rule), pooled over contexts;
  - articulation grade 0/1/2 distribution + median + grade-2 ("names true rule") rate;
  - CONSISTENCY: accuracy_vs_self (stated rule reproduces the model's OWN labels)
    and accuracy_vs_gold, + the self-minus-gold gap, + compile/parse/exec buckets.

Compared against the frozen no-CoT baselines from results/figures/metrics_table.csv
(step1 held-out accuracy, free-form median/best grade) for gpt-4.1.

Writes results/figures/cot_same_session.json + cot_same_session_metrics.csv. The
gpt-4.1 prompted-CoT and claude adaptive-thinking arms are kept SEPARATE (cot_mode).
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
    """CoT run dirs at the top level or one subdir deep (e.g. results/cot-same-session/)."""
    found = list(RESULTS.glob("cot-same-session-*")) + list(RESULTS.glob("*/cot-same-session-*"))
    return sorted(d for d in found if d.is_dir() and (d / "responses.jsonl").is_file())


def _load_baselines() -> dict[tuple[str, str], dict[str, Any]]:
    """(rule, model) -> no-CoT baseline numbers from metrics_table.csv."""
    path = FIGDIR / "metrics_table.csv"
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return out
    for r in csv.DictReader(path.open()):
        def f(k: str) -> float | None:
            v = r.get(k, "")
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        out[(r["rule_id"], r["model"])] = {
            "nocot_step1_acc": f("step1_heldout_acc"),
            "nocot_freeform_median_direct": f("freeform_median_direct"),
            "nocot_freeform_best": f("freeform_best"),
        }
    return out


def _acc_block(k: int, n: int) -> dict[str, Any]:
    if n == 0:
        return {"rate": None, "k": 0, "n": 0, "ci": [None, None]}
    lo, hi = wilson_ci(k, n)
    return {"rate": k / n, "k": k, "n": n, "ci": [lo, hi]}


def _worker_acc(result: dict[str, Any] | None) -> float | None:
    if not result or not result.get("ok"):
        return None
    return result.get("accuracy")


def analyze_run(run_dir: Path, baselines: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    cfg = json.loads((run_dir / "config.json").read_text())
    model = cfg["model"]
    cot_mode = cfg.get("cot_mode", "")
    rows = [json.loads(l) for l in (run_dir / "responses.jsonl").read_text().splitlines() if l.strip()]
    by_rule: dict[str, dict[str, list]] = {}
    for r in rows:
        d = by_rule.setdefault(r["rule_id"], {"cot_turn1": [], "cot_turn2": [], "grade": [], "compile": []})
        d.setdefault(r["kind"], []).append(r)

    out_rules: dict[str, Any] = {}
    for rule_id, kinds in sorted(by_rule.items()):
        base = canonical_rule_id(rule_id)
        # (1) CoT classification accuracy (pooled over contexts) + per-context
        n_correct = n_total = n_parse_fail = 0
        per_ctx = []
        for t1 in kinds["cot_turn1"]:
            a = t1["accuracy"]
            n_correct += a["n_correct"]; n_total += a["n"]; n_parse_fail += a["n_parse_fail"]
            per_ctx.append({"context": t1["context_index"], "accuracy": a["accuracy"],
                            "n_parse_fail": a["n_parse_fail"], "finish": t1.get("finish_reason", "")})
        cot_acc = _acc_block(n_correct, n_total)
        # (2) articulation grades
        grades = [g["grade"] for g in kinds["grade"]]
        grade_counts = {str(k): grades.count(k) for k in (0, 1, 2)}
        grade2_rate = (grades.count(2) / len(grades)) if grades else None
        candidates = [g["candidate"] for g in kinds["grade"]]
        # (3) consistency
        vs_self, vs_gold, n_compiled, n_nocode = [], [], 0, 0
        for c in kinds["compile"]:
            if not c.get("code"):
                n_nocode += 1
                continue
            n_compiled += 1
            s = _worker_acc(c.get("result_vs_self")); g = _worker_acc(c.get("result_vs_gold"))
            if s is not None:
                vs_self.append(s)
            if g is not None:
                vs_gold.append(g)
        cons = {
            "n_compile_rows": len(kinds["compile"]), "n_compiled": n_compiled, "n_no_code": n_nocode,
            "accuracy_vs_self": statistics.mean(vs_self) if vs_self else None,
            "accuracy_vs_gold": statistics.mean(vs_gold) if vs_gold else None,
            "self_minus_gold_gap": (statistics.mean(vs_self) - statistics.mean(vs_gold))
                                   if (vs_self and vs_gold) else None,
            "vs_self_values": [round(x, 3) for x in vs_self],
            "vs_gold_values": [round(x, 3) for x in vs_gold],
        }
        out_rules[base] = {
            "rule_id": base, "model": model, "cot_mode": cot_mode,
            "cot_classification": {**cot_acc, "n_parse_fail": n_parse_fail, "per_context": per_ctx},
            "articulation": {"grades": grades, "grade_counts": grade_counts,
                             "median": statistics.median(grades) if grades else None,
                             "grade2_names_true_rate": grade2_rate, "candidates": candidates},
            "consistency": cons,
            "nocot_baseline": baselines.get((base, model), {}),
            "verdict": _verdict(cot_acc["rate"], grade2_rate, cons),
        }
    return {"model": model, "cot_mode": cot_mode, "run": run_dir.name, "rules": out_rules}


def _verdict(cot_acc: float | None, grade2_rate: float | None, cons: dict[str, Any]) -> str:
    """Reading that guards BOTH the post-hoc-rationalisation confound AND the
    compiled-predicate self-compilation confound (the coder cannot compile open-vocab semantic
    rules like 'is a proper noun'/'positive sentiment' into the no-import sandbox,
    so a low vs_self there reflects coder failure, not rationalisation). The
    rationalisation signature requires the compiled rule to WORK (match gold)
    yet NOT match the model's own labels."""
    vs_self = cons.get("accuracy_vs_self")
    vs_gold = cons.get("accuracy_vs_gold")
    if grade2_rate is None:
        return "no articulation graded"
    if grade2_rate < 0.5:
        return "CoT did NOT recover the rule (articulation still a proxy) — dissociation survives reasoning"
    # grade-2 majority. Consistency is only interpretable if the compile worked,
    # i.e. the compiled predicate matches GOLD (vs_gold high). Otherwise the coder
    # could not faithfully compile the (correct) articulation -> uninformative.
    if vs_gold is None or vs_gold < 0.7:
        return "grade-2 articulation; consistency UNINFORMATIVE (rule not cleanly compilable — compiled-predicate coder limit)"
    if vs_self is not None and vs_self < 0.7:
        return ("grade-2, the compiled rule matches gold but NOT the model's own labels "
                "— possible post-hoc RATIONALISATION")
    return "CoT RECOVERS the true rule AND applies it (grade-2, high vs_self & vs_gold)"


def main() -> None:
    runs = _runs()
    if not runs:
        print("[CoT same-session] no results/cot-same-session-* runs found — skipping (no-op).")
        return
    baselines = _load_baselines()
    per_model = [analyze_run(d, baselines) for d in runs]
    out = {
        "_measurement": "CoT same-session — CoT same-session classify-then-articulate diagnostic",
        "_what": ("does allowing chain-of-thought and asking for the rule in the SAME session "
                  "recover the true rule the no-CoT model couldn't articulate?"),
        "_caveat": ("turn-2 is contaminated by turn-1 (the manipulation): a NULL (still grade 0) "
                    "is strong evidence; a POSITIVE is weaker (could be rationalisation), which is "
                    "why accuracy_vs_self is reported alongside the grade. gpt-4.1 prompted-CoT and "
                    "claude adaptive-thinking are DIFFERENT interventions (cot_mode) and never pooled."),
        "per_model": per_model,
    }
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / "cot_same_session.json").write_text(json.dumps(out, indent=2) + "\n")

    # flat CSV
    cols = ["model", "cot_mode", "rule_id", "cot_accuracy", "nocot_step1_acc",
            "grade_median", "grade2_rate", "nocot_freeform_median",
            "accuracy_vs_self", "accuracy_vs_gold", "self_minus_gold_gap", "verdict"]
    with (FIGDIR / "cot_same_session_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for pm in per_model:
            for rule, b in pm["rules"].items():
                base = b["nocot_baseline"]
                w.writerow({
                    "model": pm["model"], "cot_mode": pm["cot_mode"], "rule_id": rule,
                    "cot_accuracy": _r(b["cot_classification"]["rate"]),
                    "nocot_step1_acc": _r(base.get("nocot_step1_acc")),
                    "grade_median": b["articulation"]["median"],
                    "grade2_rate": _r(b["articulation"]["grade2_names_true_rate"]),
                    "nocot_freeform_median": base.get("nocot_freeform_median_direct"),
                    "accuracy_vs_self": _r(b["consistency"]["accuracy_vs_self"]),
                    "accuracy_vs_gold": _r(b["consistency"]["accuracy_vs_gold"]),
                    "self_minus_gold_gap": _r(b["consistency"]["self_minus_gold_gap"]),
                    "verdict": b["verdict"][:60],
                })

    print("[CoT same-session] wrote results/figures/cot_same_session.json + _metrics.csv\n")
    hdr = f"{'rule':<24}{'model':<18}{'cot_acc':>8}{'(noCoT)':>9}{'grade2':>8}{'vs_self':>9}{'vs_gold':>9}"
    print(hdr); print("-" * len(hdr))
    for pm in per_model:
        for rule, b in pm["rules"].items():
            cc = b["cot_classification"]["rate"]; base = b["nocot_baseline"]
            g2 = b["articulation"]["grade2_names_true_rate"]
            cs = b["consistency"]
            print(f"{rule:<24}{pm['model']:<18}{_f(cc):>8}{_f(base.get('nocot_step1_acc')):>9}"
                  f"{_f(g2):>8}{_f(cs['accuracy_vs_self']):>9}{_f(cs['accuracy_vs_gold']):>9}")


def _r(x: Any) -> Any:
    return round(x, 4) if isinstance(x, (int, float)) else x


def _f(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "  na"


if __name__ == "__main__":
    main()
