#!/usr/bin/env python
"""Re-analyse the Step-3 faithfulness runs with the CORRECTED method, from the
raw responses.jsonl (no API calls). Writes results/figures/step3_corrected.json.

Why this exists: the original Step-3 headline conditioned the "divergence" subset
on the model's OWN arm-2 self-application (keep probe iff arm2 != true_label).
That selects on the model's behaviour and silently swaps the probe set, which
manufactured two of the three earlier-reported gaps. The corrected analysis
(faithfulness.corrected_divergence_analysis) instead scores the FIXED, hand-built
designed-divergence probe set against the AUTHOR's stated-rule label, and reports
three outcomes per rule: behaviour tracks the TRUE rule (unfaithful), the STATED
rule (~faithful), or NEITHER (constant / brittle OOD default).

The empirical-conditioning view is preserved in each run's own metrics.json and
re-derived here too (labelled secondary) so the contrast is auditable.

Run:  python scripts/analyze_step3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.faithfulness import ARMS, ArmPredictions, analyze
from icl_articulation.step3_probes import Probe, build_probe_set

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_PATH = RESULTS / "figures" / "step3_corrected.json"
MODELS = ["gpt-4.1", "gpt-4.1-mini"]


def _latest_run(model: str) -> Path | None:
    """Latest step3-faithfulness-<model>-<ts> dir, disambiguating gpt-4.1 from
    gpt-4.1-mini (the former is a prefix of the latter)."""
    prefix = "step3-faithfulness-"
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


def load_arm_predictions(run_dir: Path) -> dict[str, ArmPredictions]:
    """Reconstruct per-rule ArmPredictions from a run's responses.jsonl.

    Rows carry (rule_id, probe_id, arm, predicted) plus the probe's fixed fields
    (text, true_label, art_label, source). Probes are ordered by probe_id so the
    three arm lists stay aligned."""
    rows = [json.loads(line) for line in (run_dir / "responses.jsonl").open()]
    meta_by_rule: dict[str, dict[str, Probe]] = {}

    def meta_for(rule_id: str, text: str) -> Probe | None:
        if rule_id not in meta_by_rule:
            meta_by_rule[rule_id] = {
                p.text: p for p in build_probe_set(rule_id, "data")
            }
        return meta_by_rule[rule_id].get(text)

    # rule_id -> probe_id -> {"probe": Probe, arm: predicted}
    by_rule: dict[str, dict[str, dict]] = {}
    for r in rows:
        rid = r["rule_id"]
        pid = r["probe_id"]
        slot = by_rule.setdefault(rid, {}).setdefault(pid, {})
        if "probe" not in slot:
            meta = meta_for(rid, r["text"])
            slot["probe"] = Probe(
                rule_id=rid,
                probe_id=pid,
                text=r["text"],
                true_label=r["true_label"],
                # Preserve the fixed stated-rule label logged with the historical
                # run. Current probe metadata may repair shallow-predicate bugs
                # (e.g. runner), but re-analysis of old raw runs should not
                # silently change their denominator.
                art_label=r["art_label"],
                source=r["source"],
                true_label_source=r.get("true_label_source", "hand"),
                note=r.get("note") or (meta.note if meta else ""),
                family=r.get("family") or (meta.family if meta else ""),
                clean_status=r.get("clean_status") or (meta.clean_status if meta else ""),
            )
        slot[r["arm"]] = r["predicted"]

    out: dict[str, ArmPredictions] = {}
    for rid, probes in by_rule.items():
        pids = sorted(probes)
        probe_list = [probes[p]["probe"] for p in pids]
        arms = {arm: [probes[p].get(arm) for p in pids] for arm in ARMS}
        out[rid] = ArmPredictions(
            rule_id=rid,
            probes=probe_list,
            in_context=arms["in_context"],
            self_application=arms["self_application"],
            true_rule_given=arms["true_rule_given"],
        )
    return out


def _fmt(x: float | None, nd: int = 2) -> str:
    return "  --" if x is None else f"{x:.{nd}f}"


def main() -> int:
    out: dict[str, dict] = {}
    for model in MODELS:
        run = _latest_run(model)
        if run is None:
            print(f"[skip] no step3-faithfulness run dir for {model}")
            continue
        preds = load_arm_predictions(run)
        result = analyze(preds)
        out[model] = {"run_dir": run.name, "analysis": result}

        print(f"\n=== {model}  ({run.name}) ===")
        print(f"{'rule':26s} {'verdict':52s} {'n_des':>5s} {'n_disc':>6s} "
              f"{'beh~true':>8s} {'binom_p':>8s} {'self~stated':>11s}")
        for row in result["summary"]:
            print(f"{row['rule_id']:26s} {row['corrected_verdict']:52s} "
                  f"{row['corrected_n_designed']:5d} {row['corrected_n_discriminating']:6d} "
                  f"{_fmt(row['corrected_behaviour_tracks_true_rate']):>8s} "
                  f"{_fmt(row['corrected_binom_p'], 4):>8s} "
                  f"{_fmt(row['corrected_self_vs_stated_rate']):>11s}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
