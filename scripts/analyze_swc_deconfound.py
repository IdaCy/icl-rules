#!/usr/bin/env python
"""Compare original second_word_capitalized against second_word_capitalized_v2.

This is a local/no-API companion to scripts/gen_swc_v2.py. It regenerates
results/figures/deconfound_swc.json from the frozen datasets, run metrics, and
articulation-probe responses. It deliberately does not edit old run configs.

Run:
  .venv/bin/python scripts/analyze_swc_deconfound.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / ""))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analyze_confounds import (  # noqa: E402
    best_single_token,
    char_length_best_threshold,
    load_items,
    naive_bayes_heldout,
    swc_vocab_disjointness,
)
from icl_articulation.datagen.confound import tokens_missing_from_a_class  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIG = RESULTS / "figures"
OUT = FIG / "deconfound_swc.json"

MODELS = ["gpt-4.1", "gpt-4.1-mini"]
ORIGINAL = "second_word_capitalized"
V2 = "second_word_capitalized_v2"


def _owner_model(run_name: str, prefix: str) -> str | None:
    if not run_name.startswith(prefix + "-"):
        return None
    for model in sorted(MODELS, key=len, reverse=True):
        if run_name.startswith(f"{prefix}-{model}-"):
            return model
    return None


def _run_dirs(prefix: str, model: str, rule: str) -> list[Path]:
    dirs: list[Path] = []
    for d in RESULTS.glob(prefix + "-*"):
        if not (d / "metrics.json").is_file():
            continue
        if _owner_model(d.name, prefix) != model:
            continue
        metrics = json.loads((d / "metrics.json").read_text())
        if rule in metrics.get("rules", {}):
            dirs.append(d)
    return sorted(dirs, key=lambda p: p.name)


def _rule_block(prefix: str, model: str, rule: str) -> tuple[dict[str, Any], str] | tuple[None, None]:
    dirs = _run_dirs(prefix, model, rule)
    if not dirs:
        return None, None
    d = dirs[-1]
    metrics = json.loads((d / "metrics.json").read_text())
    return metrics["rules"][rule], d.name


def _run_acc(prefix: str, model: str, rule: str) -> dict[str, Any]:
    block, run_dir = _rule_block(prefix, model, rule)
    if block is None:
        return {"run_dir": None, "acc": None, "ci95": None, "predictions": None}
    pooled = block.get("pooled", {})
    ctxs = block.get("contexts", [])
    pred_true = sum(c.get("predictions", {}).get("true", 0) for c in ctxs)
    pred_false = sum(c.get("predictions", {}).get("false", 0) for c in ctxs)
    pred_parse_fail = sum(c.get("predictions", {}).get("parse_failure", 0) for c in ctxs)
    ci = pooled.get("cluster_bootstrap_ci_95")
    if ci is None and len(ctxs) == 1:
        ci = ctxs[0].get("wilson_ci_95")
    return {
        "run_dir": run_dir,
        "acc": pooled.get("mean_accuracy"),
        "ci95": ci,
        "n_contexts": pooled.get("n_contexts"),
        "n_items": pooled.get("n_items"),
        "n_parse_failures": pooled.get("n_parse_failures"),
        "predictions": {
            "true": pred_true,
            "false": pred_false,
            "parse_failure": pred_parse_fail,
        },
    }


def _proper_proxy_acc(rule: str) -> float:
    rows = load_items(rule)
    original_vocab = {
        (r.get("slots_meta", {}).get("proper") or "").lower()
        for r in rows
    } - {""}
    correct = 0
    for r in rows:
        meta = r.get("slots_meta", {})
        if rule == ORIGINAL:
            toks = r["text"].split()
            pred = len(toks) > 1 and toks[1].lower() in original_vocab
        else:
            pred = bool(meta.get("w2_is_proper"))
        if pred == bool(r["label"]):
            correct += 1
    return correct / len(rows)


def _proper_rates(rule: str) -> dict[str, float]:
    rows = load_items(rule)
    original_vocab = {
        (r.get("slots_meta", {}).get("proper") or "").lower()
        for r in rows
    } - {""}
    out: dict[str, float] = {}
    for lab, key in [(True, "true"), (False, "false")]:
        cls = [r for r in rows if bool(r["label"]) is lab]
        if not cls:
            out[key] = 0.0
            continue
        if rule == ORIGINAL:
            out[key] = sum(
                1
                for r in cls
                if len(r["text"].split()) > 1 and r["text"].split()[1].lower() in original_vocab
            ) / len(cls)
        else:
            out[key] = sum(1 for r in cls if r.get("slots_meta", {}).get("w2_is_proper")) / len(cls)
    return out


def _confound(rule: str) -> dict[str, Any]:
    rows = load_items(rule)
    nb, _ = naive_bayes_heldout(rule)
    one_sided = tokens_missing_from_a_class(rows)
    out = {
        "naive_bayes_bow_heldout": nb,
        "proper_noun_proxy_extensional_acc": _proper_proxy_acc(rule),
        "proper_noun_rates_by_label": _proper_rates(rule),
        "best_single_token": best_single_token(rule),
        "char_length_best_threshold": char_length_best_threshold(rule),
        "n_one_sided_high_freq_tokens": len(one_sided),
        "one_sided_examples": [
            (t["token"], t["true_count"], t["false_count"]) for t in one_sided[:8]
        ],
    }
    if rule == ORIGINAL:
        out.update(swc_vocab_disjointness())
    return out


def _articulation_probe_response_path() -> Path | None:
    """Latest frozen SWC-v2 articulation probe run, selected deterministically.

    Earlier versions used the first matching directory. That was harmless while
    only one run existed, but explicit latest-run selection prevents future
    filesystem-order ambiguity if another probe run is added.
    """
    dirs = sorted(
        d for d in RESULTS.glob("step2-freeform-second_word_capitalized_v2-probe-*")
        if (d / "responses.jsonl").is_file()
    )
    return (dirs[-1] / "responses.jsonl") if dirs else None


def _articulation_samples() -> dict[str, list[str]]:
    path = _articulation_probe_response_path()
    if path is None:
        return {}
    out: dict[str, list[str]] = {m: [] for m in MODELS}
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            model = row.get("model")
            art = row.get("articulation")
            if model in out and art and art not in out[model]:
                out[model].append(art)
    return {m: v[:4] for m, v in out.items()}


def main() -> int:
    out: dict[str, Any] = {
        "_note": (
            "Original vs deconfounded second_word_capitalized. v2 uses minimal "
            "casing pairs on a shared common+proper W2 pool, so capitalization "
            "of word 2 is the intended signal while proper-noun-ness and "
            "case-folded bag-of-words drop to chance. Regenerated locally from "
            "datasets, metrics.json files, and articulation-probe responses."
        ),
        "original": {
            "confound": _confound(ORIGINAL),
            "step1": {
                m: {
                    "in_context": _run_acc("step1-full", m, ORIGINAL),
                    "rule_given": _run_acc("step1-rule_given", m, ORIGINAL),
                }
                for m in MODELS
            },
            "articulation_gpt_4_1": (
                "The subject is a specific person, city, country, or month "
                "(a proper noun)."
            ),
            "judge_grade_median_direct_gpt_4_1": 1.0,
        },
        "deconfounded_v2": {
            "confound": _confound(V2),
            "step1": {
                m: {
                    "in_context": _run_acc("step1-full", m, V2),
                    "rule_given": _run_acc("step1-rule_given", m, V2),
                }
                for m in MODELS
            },
            "articulation_samples": _articulation_samples(),
            "articulation_probe_run_dir": (
                _articulation_probe_response_path().parent.name
                if _articulation_probe_response_path() else None
            ),
        },
    }

    g = out["deconfounded_v2"]["step1"]["gpt-4.1"]["in_context"]
    mini = out["deconfounded_v2"]["step1"]["gpt-4.1-mini"]["in_context"]
    out["reading"] = (
        "Near-chance collapse after removing the proper-noun proxy: "
        f"gpt-4.1 acc={g['acc']:.3f}, CI95={g['ci95']}; "
        f"gpt-4.1-mini acc={mini['acc']:.3f}, CI95={mini['ci95']}. "
        "Rule-given remains near-perfect, so this is a learning-from-examples "
        "gap rather than a competence gap."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print("=== second_word_capitalized: original vs deconfounded_v2 ===")
    for key, rule in [("original", ORIGINAL), ("deconfounded_v2", V2)]:
        c = out[key]["confound"]
        print(f"\n{key}:")
        print(
            f"  NB {c['naive_bayes_bow_heldout']:.3f} | proper-proxy "
            f"{c['proper_noun_proxy_extensional_acc']:.3f} | best token "
            f"{c['best_single_token']['token']!r} {c['best_single_token']['accuracy']:.3f} | "
            f"char-len {c['char_length_best_threshold']:.3f} | "
            f"one-sided {c['n_one_sided_high_freq_tokens']}"
        )
        for m in MODELS:
            ic = out[key]["step1"][m]["in_context"]["acc"]
            rg = out[key]["step1"][m]["rule_given"]["acc"]
            print(f"  {m:14s} in-context {ic:.3f}  rule-given {rg:.3f}")

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
