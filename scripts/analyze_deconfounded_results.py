#!/usr/bin/env python
"""Build Deconfounded deconfounded-vs-original analysis sidecars and figures.

This is a no-API post-processing script. It expects the Deconfounded deconfound-sweep outputs to
have been synced locally under ``results/deconfound-sweep`` and the original runs to
remain under ``results``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.stats import wilson_ci

RULES = [
    "mentions_color",
    "mentions_animal",
    "food_topic",
    "positive_sentiment",
    "contains_first_name",
    "starts_with_vowel",
    "last_word_ends_with_vowel",
    "word_count_geq_8",
    "second_word_capitalized",
    "even_word_count",
    "passive_voice",
    "the_appears_twice",
    "first_word_longer_than_last",
    "all_words_longer_than_3",
    "first_two_words_alphabetical",
]
OPENAI_COMPAT_MODELS = ["gpt-4.1", "gpt-4.1-mini", "deepseek-v4-flash"]
ALL_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "deepseek-v4-flash",
    "claude-opus-4-8",
    "Qwen2.5-7B-Instruct",
]
MODEL_FAMILY = {
    "gpt-4.1": "openai",
    "gpt-4.1-mini": "openai",
    "deepseek-v4-flash": "deepseek",
    "claude-opus-4-8": "anthropic",
    "Qwen2.5-7B-Instruct": "open_weights",
}
EXPECTED = {
    "mentions_color": "DROP/moderate if memorization was load-bearing; HOLD if semantic category abstracts.",
    "mentions_animal": "DROP/moderate for memorization; possible HOLD if animal category generalizes.",
    "food_topic": "DROP likely if keyword list was load-bearing; HOLD if topic abstraction survives.",
    "positive_sentiment": "DROP likely if sentiment words were memorized; HOLD if polarity abstraction survives.",
    "contains_first_name": "DROP/moderate expected because proper-token identity was a strong shortcut.",
    "starts_with_vowel": "HOLD if character rule is learned; intrinsic first-letter residual disclosed.",
    "last_word_ends_with_vowel": "HOLD or moderate DROP; intrinsic final-letter signal remains legitimate.",
    "word_count_geq_8": "HOLD/moderate because word-count threshold is directly learnable.",
    "second_word_capitalized": "DROP relative to original proper-token proxy.",
    "even_word_count": "DROP expected; parity without CoT is hard.",
    "passive_voice": "HOLD/DISCLOSE if passive morphology is learned; otherwise DROP.",
    "the_appears_twice": "HOLD likely; token-count residual is rule-proximal.",
    "first_word_longer_than_last": "Moderate DROP; scalar first/last length components remain intended.",
    "all_words_longer_than_3": "HOLD/moderate with measured short-token residual.",
    "first_two_words_alphabetical": "Moderate DROP; vocabulary skew was found.",
}
RESIDUAL_TYPE = {
    "mentions_color": "non_rule_residual_low",
    "mentions_animal": "non_rule_residual_low",
    "food_topic": "semantic_validator_residual_low",
    "positive_sentiment": "semantic_validator_residual_low",
    "contains_first_name": "non_rule_residual_low",
    "starts_with_vowel": "intrinsic_character_rule",
    "last_word_ends_with_vowel": "intrinsic_character_rule",
    "word_count_geq_8": "rule_proximal_scalar",
    "second_word_capitalized": "intrinsic_capitalization_scalar",
    "even_word_count": "residual_measured",
    "passive_voice": "intrinsic_passive_morphology",
    "the_appears_twice": "rule_proximal_token_count",
    "first_word_longer_than_last": "rule_proximal_length_components",
    "all_words_longer_than_3": "rule_proximal_short_token",
    "first_two_words_alphabetical": "rule_proximal_order_components",
}
FIELDNAMES = [
    "canonical_rule_id",
    "rule_id",
    "dataset_version",
    "model_family",
    "model",
    "step1_mode",
    "source_run_dir",
    "acc",
    "ci_lo",
    "ci_hi",
    "n_contexts",
    "n_items",
    "parse_fail_rate",
    "is_changed_rule",
    "expected_rebuild_outcome",
    "observed_delta_vs_original",
    "observed_outcome_label",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canon(rule_id: str) -> str:
    return rule_id.removesuffix("_deconfounded")


def latest_step1_runs(results_dir: Path, nested_dir: Path | None = None) -> list[Path]:
    root = nested_dir if nested_dir is not None else results_dir
    runs = []
    for metrics_path in root.glob("step1-*/metrics.json"):
        config_path = metrics_path.parent / "config.json"
        if config_path.is_file():
            runs.append(metrics_path.parent)
    return runs


def choose_latest_by_model_mode(runs: list[Path], require_deconfounded: bool) -> dict[tuple[str, str], Path]:
    candidates: dict[tuple[str, str], list[tuple[int, str, Path]]] = defaultdict(list)
    for run in runs:
        config = read_json(run / "config.json")
        if config.get("task") != "step1-classification":
            continue
        model = config.get("model")
        mode = config.get("mode")
        rules = config.get("rules") or []
        if model is None or mode is None:
            continue
        has_deconfounded = any(str(r).endswith("_deconfounded") for r in rules)
        if require_deconfounded != has_deconfounded:
            continue
        candidates[(str(model), str(mode))].append((len(rules), run.name, run))
    out = {}
    for key, vals in candidates.items():
        vals.sort()
        out[key] = vals[-1][2]
    return out


def row_from_step1_metric(
    rule_id: str,
    rm: dict[str, Any],
    dataset_version: str,
    model: str,
    mode: str,
    run: Path,
) -> dict[str, Any]:
    pooled = rm["pooled"]
    ci = pooled.get("cluster_bootstrap_ci_95") or [None, None]
    n_items = pooled.get("n_contexts", 0) * pooled.get("n_items", 0)
    n_parse = pooled.get("n_parse_failures", 0)
    return {
        "canonical_rule_id": canon(rule_id),
        "rule_id": rule_id,
        "dataset_version": dataset_version,
        "model_family": MODEL_FAMILY.get(model, "unknown"),
        "model": model,
        "step1_mode": mode,
        "source_run_dir": str(run),
        "acc": pooled.get("mean_accuracy"),
        "ci_lo": ci[0],
        "ci_hi": ci[1],
        "n_contexts": pooled.get("n_contexts"),
        "n_items": n_items,
        "parse_fail_rate": (n_parse / n_items) if n_items else None,
        "is_changed_rule": True,
        "expected_rebuild_outcome": EXPECTED.get(canon(rule_id)),
        "observed_delta_vs_original": None,
        "observed_outcome_label": None,
    }


def load_step1_rows(results_dir: Path, deconfounded_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    original = choose_latest_by_model_mode(latest_step1_runs(results_dir), require_deconfounded=False)
    deconfounded = choose_latest_by_model_mode(latest_step1_runs(results_dir, deconfounded_dir), require_deconfounded=True)

    for (model, mode), run in sorted(original.items()):
        if model not in {"gpt-4.1", "gpt-4.1-mini"}:
            continue
        metrics = read_json(run / "metrics.json")
        for rule in RULES:
            if rule in metrics.get("rules", {}):
                rows.append(row_from_step1_metric(rule, metrics["rules"][rule], "original", model, mode, run))

    for (model, mode), run in sorted(deconfounded.items()):
        if model not in OPENAI_COMPAT_MODELS:
            continue
        metrics = read_json(run / "metrics.json")
        for rule_id, rm in metrics.get("rules", {}).items():
            if canon(rule_id) in RULES:
                rows.append(row_from_step1_metric(rule_id, rm, "deconfounded", model, mode, run))
    return rows


def load_qwen_rows(results_dir: Path, deconfounded_dir: Path) -> list[dict[str, Any]]:
    rows = []
    original = results_dir / "local-qwen-step1-metrics.json"
    if original.is_file():
        rows.extend(qwen_rows_from_metrics(original, "original"))
    qwen_runs = sorted(deconfounded_dir.glob("qwen-step1-*/local-qwen-step1-metrics.json"))
    if qwen_runs:
        rows.extend(qwen_rows_from_metrics(qwen_runs[-1], "deconfounded"))
    return rows


def qwen_rows_from_metrics(path: Path, dataset_version: str) -> list[dict[str, Any]]:
    metrics = read_json(path)
    out = []
    for rule_id, rm in metrics.get("rules", {}).items():
        c = canon(rule_id)
        if c not in RULES:
            continue
        n_items = rm.get("n_contexts", 0) * rm.get("n_items_per_context", 0)
        acc = rm.get("pooled_accuracy")
        if acc is not None and n_items:
            lo, hi = wilson_ci(round(acc * n_items), n_items)
        else:
            lo = hi = None
        out.append({
            "canonical_rule_id": c,
            "rule_id": rule_id,
            "dataset_version": dataset_version,
            "model_family": "open_weights",
            "model": "Qwen2.5-7B-Instruct",
            "step1_mode": "full",
            "source_run_dir": str(path.parent),
            "acc": acc,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_contexts": rm.get("n_contexts"),
            "n_items": n_items,
            "parse_fail_rate": (rm.get("n_parse_failures", 0) / n_items) if n_items else None,
            "is_changed_rule": True,
            "expected_rebuild_outcome": EXPECTED.get(c),
            "observed_delta_vs_original": None,
            "observed_outcome_label": None,
        })
    return out


def load_claude_rows(results_dir: Path, deconfounded_dir: Path) -> list[dict[str, Any]]:
    rows = []
    old = results_dir / "figures" / "cross_family.json"
    if old.is_file():
        obj = read_json(old)
        think_off = (
            obj.get("B1_deconfound_generality", {})
            .get("per_dataset", {})
            .get("think_off", {})
        )
        for rule, rm in think_off.items():
            if rule in RULES:
                n = rm.get("n_parsed") or rm.get("n")
                acc = rm.get("accuracy")
                lo, hi = wilson_ci(round(acc * n), n) if acc is not None and n else (None, None)
                rows.append({
                    "canonical_rule_id": rule,
                    "rule_id": rule,
                    "dataset_version": "original",
                    "model_family": "anthropic",
                    "model": "claude-opus-4-8",
                    "step1_mode": "full",
                    "source_run_dir": str(old),
                    "acc": acc,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_contexts": len(rm.get("per_context_accuracy", {})) or None,
                    "n_items": rm.get("n"),
                    "parse_fail_rate": rm.get("parse_fail_rate"),
                    "is_changed_rule": True,
                    "expected_rebuild_outcome": EXPECTED.get(rule),
                    "observed_delta_vs_original": None,
                    "observed_outcome_label": None,
                })

    cf_dir = results_dir / "cross_family"
    runs = sorted(cf_dir.glob("claude-deconfounded-step1-*.jsonl"))
    if runs:
        rows.extend(claude_rows_from_jsonl(runs[-1]))
    return rows


def claude_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped[row["rule"]].append(row)
    out = []
    for rule_id, group in sorted(grouped.items()):
        c = canon(rule_id)
        n = len(group)
        parsed = [r for r in group if r.get("parse_ok")]
        correct = sum(1 for r in parsed if r.get("predicted") == r.get("true_label"))
        acc = correct / len(parsed) if parsed else None
        lo, hi = wilson_ci(correct, len(parsed)) if parsed else (None, None)
        out.append({
            "canonical_rule_id": c,
            "rule_id": rule_id,
            "dataset_version": "deconfounded",
            "model_family": "anthropic",
            "model": "claude-opus-4-8",
            "step1_mode": "full",
            "source_run_dir": str(path),
            "acc": acc,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_contexts": len({r.get("context_index") for r in group}),
            "n_items": n,
            "parse_fail_rate": (n - len(parsed)) / n if n else None,
            "is_changed_rule": True,
            "expected_rebuild_outcome": EXPECTED.get(c),
            "observed_delta_vs_original": None,
            "observed_outcome_label": None,
        })
    return out


def label_delta(delta: float | None) -> str | None:
    if delta is None:
        return None
    if delta <= -0.15:
        return "DROP"
    if delta < -0.05:
        return "MODERATE_DROP"
    if delta >= 0.05:
        return "GAIN"
    return "HOLD"


def attach_deltas(rows: list[dict[str, Any]]) -> None:
    original = {
        (r["canonical_rule_id"], r["model"], r["step1_mode"]): r
        for r in rows
        if r["dataset_version"] == "original" and r["acc"] is not None
    }
    for r in rows:
        if r["dataset_version"] != "deconfounded" or r["acc"] is None:
            continue
        base = original.get((r["canonical_rule_id"], r["model"], r["step1_mode"]))
        if base and base["acc"] is not None:
            delta = r["acc"] - base["acc"]
            r["observed_delta_vs_original"] = delta
            r["observed_outcome_label"] = label_delta(delta)


def shortcut_delta_rows(
    original_audit: Path,
    deconfounded_audit: Path,
    verify_path: Path,
) -> list[dict[str, Any]]:
    original = read_json(original_audit)["rules"]
    deconfounded = read_json(deconfounded_audit)["rules"]
    verify = read_json(verify_path)["rules"]
    rows = []
    for rule in RULES:
        mrule = f"{rule}_deconfounded"
        ob = original[rule]["shortcut_baselines"]
        mb = deconfounded[mrule]["shortcut_baselines"]
        v = verify[mrule]
        rows.append({
            "canonical_rule_id": rule,
            "original_rule_id": rule,
            "deconfounded_rule_id": mrule,
            "rule_intact_pass": bool(v.get("passes")),
            "n_mismatches": v.get("n_mismatches"),
            "original_bow": ob["word_bow_nb"]["eval_acc"],
            "deconfounded_bow": mb["word_bow_nb"]["eval_acc"],
            "original_char": ob["char_1_3gram_nb"]["eval_acc"],
            "deconfounded_char": mb["char_1_3gram_nb"]["eval_acc"],
            "original_best_token": ob["best_single_token"]["eval_acc"],
            "deconfounded_best_token": mb["best_single_token"]["eval_acc"],
            "original_scalar": ob["best_scalar_threshold"]["eval_acc"],
            "deconfounded_scalar": mb["best_scalar_threshold"]["eval_acc"],
            "original_max": ob["max_eval_acc"],
            "deconfounded_max": mb["max_eval_acc"],
            "delta_max": mb["max_eval_acc"] - ob["max_eval_acc"],
            "target_0_60_pass": mb["max_eval_acc"] <= 0.60,
            "residual_type": RESIDUAL_TYPE.get(rule, "unclassified"),
        })
    return rows


def write_table(rows: list[dict[str, Any]], json_path: Path, csv_path: Path, fields: list[str]) -> None:
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_figures(rows: list[dict[str, Any]], shortcut_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    full = [r for r in rows if r["step1_mode"] == "full"]
    by_key = {(r["canonical_rule_id"], r["model"], r["dataset_version"]): r for r in full}

    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
    for ax, model in zip(axes, ["gpt-4.1", "gpt-4.1-mini"], strict=True):
        y = np.arange(len(RULES))
        ax.set_title(model)
        ax.set_xlim(0.35, 1.02)
        ax.axvline(0.5, color="#999999", lw=0.8, ls=":")
        for i, rule in enumerate(RULES):
            o = by_key.get((rule, model, "original"))
            m = by_key.get((rule, model, "deconfounded"))
            if not o or not m:
                continue
            ax.plot([o["acc"], m["acc"]], [i, i], color="#999999", lw=1)
            ax.scatter([o["acc"]], [i], color="#6f6f6f", s=20, label="original" if i == 0 else None)
            ax.scatter([m["acc"]], [i], color="#1f77b4", s=24, label="Deconfounded" if i == 0 else None)
        ax.grid(axis="x", color="#dddddd", lw=0.5)
        ax.set_xlabel("Step 1 held-out accuracy")
    axes[0].set_yticks(np.arange(len(RULES)), RULES, fontsize=8)
    axes[1].tick_params(axis="y", left=False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_deconfounded_step1_original_vs_deconfounded.png", dpi=200)
    plt.close(fig)
    (figures_dir / "fig_deconfounded_step1_original_vs_deconfounded.caption.txt").write_text(
        "Deconfounded deconfounded Step 1 held-out accuracy versus the original datasets for the two primary OpenAI subjects. Lines connect the same canonical rule before and after deconfounding.\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(RULES))
    orig = [next(r for r in shortcut_rows if r["canonical_rule_id"] == rule)["original_max"] for rule in RULES]
    deconfounded = [next(r for r in shortcut_rows if r["canonical_rule_id"] == rule)["deconfounded_max"] for rule in RULES]
    ax.scatter(orig, y, color="#777777", label="original", s=22)
    ax.scatter(deconfounded, y, color="#d62728", label="Deconfounded", s=24)
    for i, (o, m) in enumerate(zip(orig, deconfounded, strict=True)):
        ax.plot([o, m], [i, i], color="#bbbbbb", lw=1)
    ax.axvline(0.60, color="#333333", lw=0.8, ls="--")
    ax.set_xlim(0.45, 1.02)
    ax.set_yticks(y, RULES, fontsize=8)
    ax.set_xlabel("Best shortcut baseline eval accuracy")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_deconfounded_shortcut_residuals.png", dpi=200)
    plt.close(fig)
    (figures_dir / "fig_deconfounded_shortcut_residuals.caption.txt").write_text(
        "Best generic shortcut baseline on original versus Deconfounded rebuilt datasets. The dashed line marks the pre-registered near-chance target of 0.60; residuals above it are measured and disclosed by type.\n",
        encoding="utf-8",
    )

    models = ALL_MODELS
    delta = np.full((len(RULES), len(models)), np.nan)
    for i, rule in enumerate(RULES):
        for j, model in enumerate(models):
            r = by_key.get((rule, model, "deconfounded"))
            if r and r["observed_delta_vs_original"] is not None:
                delta[i, j] = r["observed_delta_vs_original"]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(delta), cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(np.arange(len(models)), models, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(RULES)), RULES, fontsize=8)
    ax.set_title("Deconfounded minus original Step 1 accuracy")
    for i in range(len(RULES)):
        for j in range(len(models)):
            if not math.isnan(delta[i, j]):
                ax.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="accuracy delta")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_deconfounded_family_delta_heatmap.png", dpi=200)
    plt.close(fig)
    (figures_dir / "fig_deconfounded_family_delta_heatmap.caption.txt").write_text(
        "Deconfounded-minus-original Step 1 accuracy deltas by model family. Gray cells indicate no original-family baseline was run, so no delta is claimed.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--deconfounded-dir", type=Path, default=Path("results/deconfound-sweep"))
    p.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    p.add_argument("--original-audit", type=Path, default=Path("deconfounded_original_audit.json"))
    p.add_argument("--deconfounded-audit", type=Path, default=Path("deconfounded_data_all_rebuilt_audit.json"))
    p.add_argument("--verify", type=Path, default=Path("deconfounded_data_all_rebuilt_verify.json"))
    args = p.parse_args(argv)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(load_step1_rows(args.results_dir, args.deconfounded_dir))
    rows.extend(load_qwen_rows(args.results_dir, args.deconfounded_dir))
    rows.extend(load_claude_rows(args.results_dir, args.deconfounded_dir))
    attach_deltas(rows)

    shortcut_rows = shortcut_delta_rows(args.original_audit, args.deconfounded_audit, args.verify)

    write_table(
        sorted(rows, key=lambda r: (r["canonical_rule_id"], r["model"], r["dataset_version"], r["step1_mode"])),
        args.figures_dir / "deconfounded_metrics_table.json",
        args.figures_dir / "deconfounded_metrics_table.csv",
        FIELDNAMES,
    )
    shortcut_fields = list(shortcut_rows[0].keys())
    write_table(
        shortcut_rows,
        args.figures_dir / "deconfounded_shortcut_delta.json",
        args.figures_dir / "deconfounded_shortcut_delta.csv",
        shortcut_fields,
    )
    make_figures(rows, shortcut_rows, args.figures_dir)
    summary = {
        "n_metric_rows": len(rows),
        "n_shortcut_rows": len(shortcut_rows),
        "models": sorted({r["model"] for r in rows}),
        "rules": RULES,
        "outputs": [
            "deconfounded_metrics_table.json",
            "deconfounded_metrics_table.csv",
            "deconfounded_shortcut_delta.json",
            "deconfounded_shortcut_delta.csv",
            "fig_deconfounded_step1_original_vs_deconfounded.png",
            "fig_deconfounded_shortcut_residuals.png",
            "fig_deconfounded_family_delta_heatmap.png",
        ],
    }
    (args.figures_dir / "deconfounded_analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
