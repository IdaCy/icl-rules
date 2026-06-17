#!/usr/bin/env python
"""Derive the Step-1 survivor set from the full-sweep metrics, with an explicit,
stated threshold, and write it to a committed artifact.

A "survivor" is a rule learnable well enough in-context to be worth the (paid)
Step-2/Step-3 articulation probes and the winner's-curse confirmation re-run. The
pre-specified criterion is a fixed accuracy floor on the Step-1 held-out sweep:

    Step-1 held-out pooled accuracy >= 0.85 for AT LEAST ONE of the two models.

This single number reproduces exactly the 13-rule set that was confirmed
(verified against the step1-confirmation run dirs). Decoupling the survivor flag
from "rule happens to appear in the confirmation run" (the old make_figures
behaviour) is what lets `--mode confirmation --all` not silently mark all 30
rules survivors. make_figures.py reads the artifact this writes.

Run:  python scripts/select_survivors.py            # writes results/figures/survivors.json
      python scripts/select_survivors.py --print-rules   # just the comma list, for --rules
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_PATH = RESULTS / "figures" / "survivors.json"
MODELS = ["gpt-4.1", "gpt-4.1-mini"]
DEFAULT_THRESHOLD = 0.85


def _latest_full_run(model: str, results_dir: Path) -> Path | None:
    """Latest step1-full-<model>-<ts> dir (timestamps sort lexically).

    'gpt-4.1' is a prefix of 'gpt-4.1-mini', so a naive glob on
    'step1-full-gpt-4.1-*' would also match the mini dirs; pick the LONGEST
    model id the dir name actually belongs to (same disambiguation as
    make_figures._model_of_dir)."""
    prefix = "step1-full-"
    candidates: list[Path] = []
    for d in results_dir.glob(prefix + "*"):
        if not (d / "metrics.json").is_file():
            continue
        rest = d.name[len(prefix):]
        owner = next((m for m in sorted(MODELS, key=len, reverse=True)
                      if rest.startswith(m + "-")), None)
        if owner == model:
            candidates.append(d)
    if not candidates:
        return None

    # prefer the run with the MOST rules (the 30-rule sweep), then latest ts, so a
    # single-rule deconfound re-run (word_count_geq_8_v2) does not get picked.
    def _n_rules(d: Path) -> int:
        try:
            return len(json.loads((d / "metrics.json").read_text()).get("rules", {}))
        except Exception:
            return 0
    return sorted(candidates, key=lambda d: (_n_rules(d), d.name))[-1]


def per_rule_heldout_acc(results_dir: Path = RESULTS) -> dict[str, dict[str, float]]:
    """{rule_id: {model: pooled held-out accuracy}} from the full sweeps."""
    out: dict[str, dict[str, float]] = {}
    for model in MODELS:
        run = _latest_full_run(model, results_dir)
        if run is None:
            continue
        metrics = json.loads((run / "metrics.json").read_text())
        for rule_id, block in metrics.get("rules", {}).items():
            acc = block.get("pooled", {}).get("mean_accuracy")
            if acc is not None:
                out.setdefault(rule_id, {})[model] = acc
    if not out:
        raise SystemExit(f"no step1-full-*/metrics.json under {results_dir}/ — run the full sweep first")
    return out


def select(
    accs: dict[str, dict[str, float]], threshold: float = DEFAULT_THRESHOLD
) -> dict:
    per_rule: dict[str, dict] = {}
    survivors: list[str] = []
    for rule_id in sorted(accs):
        by_model = accs[rule_id]
        best = max(by_model.values())
        is_survivor = best >= threshold
        per_rule[rule_id] = {**by_model, "max": best, "is_survivor": is_survivor}
        if is_survivor:
            survivors.append(rule_id)
    return {
        "threshold": {
            "criterion": "step1_heldout_pooled_accuracy >= value for at least one model",
            "value": threshold,
            "models": MODELS,
        },
        "n_survivors": len(survivors),
        "survivors": survivors,
        "per_rule": per_rule,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--print-rules", action="store_true",
                   help="print only the comma-separated survivor list (for run_step1 --rules)")
    args = p.parse_args(argv)

    accs = per_rule_heldout_acc(Path(args.results_dir))
    result = select(accs, args.threshold)

    if args.print_rules:
        print(",".join(result["survivors"]))
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: {result['n_survivors']} survivors "
          f"(threshold {args.threshold} on at least one model)")
    print("  " + ", ".join(result["survivors"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
