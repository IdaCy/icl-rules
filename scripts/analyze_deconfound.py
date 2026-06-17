#!/usr/bin/env python
"""Compare the original word_count_geq_8 against the deconfounded
word_count_geq_8_v2 (external-review P2b). Writes results/figures/deconfound_wc8.json.

Reads the step-1 in-context and rule-given accuracies for BOTH rules from their
run dirs, and recomputes the confound baselines (naive-Bayes bag-of-words, best
single token, char-length, one-sided tokens) from the committed datasets.

Pre-registered readings (decided before the v2 run): if accuracy SURVIVES
deconfounding the model can learn the length rule without a lexical proxy; if it
COLLAPSES, the original high accuracy was proxy learning. The actual outcome is in
between and is reported honestly.

Run:  python scripts/analyze_deconfound.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analyze_confounds import (
    best_single_token, char_length_best_threshold, load_items, naive_bayes_heldout,
)
from icl_articulation.datagen.confound import tokens_missing_from_a_class

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = RESULTS / "figures" / "deconfound_wc8.json"
MODELS = ["gpt-4.1", "gpt-4.1-mini"]


def _run_acc(prefix: str, model: str, rule: str) -> float | None:
    """Pooled accuracy for `rule` from the latest matching run that contains it."""
    pre = f"{prefix}-{model}-"
    dirs = [d for d in RESULTS.glob(pre + "*") if (d / "metrics.json").is_file()
            and not d.name[len(pre):].startswith(tuple(
                m[len(model) + 1:] for m in MODELS if m != model and m.startswith(model)))]
    # simpler/robust model disambiguation:
    out = []
    for d in RESULTS.glob(f"{prefix}-*"):
        if not (d / "metrics.json").is_file():
            continue
        rest = d.name[len(prefix) + 1:]
        owner = next((m for m in sorted(MODELS, key=len, reverse=True)
                      if d.name.startswith(f"{prefix}-{m}-")), None)
        if owner != model:
            continue
        met = json.loads((d / "metrics.json").read_text())
        if rule in met.get("rules", {}):
            out.append((d.name, met["rules"][rule]["pooled"]["mean_accuracy"]))
    return sorted(out)[-1][1] if out else None


def _confound(rule: str) -> dict:
    rows = load_items(rule)
    nb, n = naive_bayes_heldout(rule)
    flagged = tokens_missing_from_a_class(rows)
    return {
        "naive_bayes_bow_heldout": nb,
        "best_single_token": best_single_token(rule),
        "char_length_best_threshold": char_length_best_threshold(rule),
        "n_one_sided_high_freq_tokens": len(flagged),
        "one_sided_examples": [(t["token"], t["true_count"], t["false_count"]) for t in flagged[:5]],
    }


def main() -> int:
    out = {"_note": "Original vs deconfounded word_count_geq_8 (P2b). The v2 dataset "
                    "matches trailing-adjunct rates across classes and has zero "
                    "one-sided tokens; the char-length signal is inherent to a length "
                    "rule and is reported, not removed."}
    for rule, key in [("word_count_geq_8", "original"), ("word_count_geq_8_v2", "deconfounded_v2")]:
        entry = {"confound": _confound(rule), "step1": {}}
        for m in MODELS:
            entry["step1"][m] = {
                "in_context_acc": _run_acc("step1-full", m, rule),
                "rule_given_acc": _run_acc("step1-rule_given", m, rule),
            }
        out[key] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print("=== word_count_geq_8: original vs deconfounded (v2) ===")
    for key in ["original", "deconfounded_v2"]:
        e = out[key]; c = e["confound"]
        print(f"\n{key}:")
        print(f"  confound: NB {c['naive_bayes_bow_heldout']:.3f} | best-token "
              f"{c['best_single_token']['token']!r} {c['best_single_token']['accuracy']:.3f} | "
              f"char-len {c['char_length_best_threshold']:.3f} | one-sided tokens "
              f"{c['n_one_sided_high_freq_tokens']}")
        for m in MODELS:
            s = e["step1"][m]
            ic = s["in_context_acc"]; rg = s["rule_given_acc"]
            print(f"  {m:14s} in-context {('%.3f'%ic) if ic is not None else 'NA':>6s}  "
                  f"rule-given {('%.3f'%rg) if rg is not None else 'NA':>6s}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
