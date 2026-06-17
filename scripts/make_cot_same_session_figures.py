#!/usr/bin/env python
"""CoT same-session figure (NO API, LOCAL): the CoT same-session diagnostic.

Reads results/figures/cot_same_session.json and renders
results/figures/fig_cot_same_session.png (+ caption): per rule × model, three
panels — CoT vs no-CoT classification accuracy; CoT articulation grade-2 rate vs
the no-CoT free-form grade; and the consistency panel (accuracy_vs_self vs
accuracy_vs_gold, the rationalisation guard). No-op if the json is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "results" / "figures"
RULE_ORDER = ["word_count_geq_8", "second_word_capitalized", "physically_impossible",
              "food_topic", "positive_sentiment"]
MODEL_STYLE = {  # model -> (label, color)
    "gpt-4.1": ("gpt-4.1 (prompted CoT)", "#1565c0"),
    "claude-opus-4-8": ("claude-opus-4-8 (thinking)", "#c1432f"),
}


def main() -> None:
    src = FIGDIR / "cot_same_session.json"
    if not src.is_file():
        print("[fig] results/figures/cot_same_session.json absent — skipping (no-op).")
        return
    data = json.loads(src.read_text())
    blocks: dict[tuple[str, str], dict] = {}
    models: list[str] = []
    for pm in data["per_model"]:
        models.append(pm["model"])
        for rule, b in pm["rules"].items():
            blocks[(rule, pm["model"])] = b
    models = [m for m in MODEL_STYLE if m in models] + [m for m in models if m not in MODEL_STYLE]
    rules = [r for r in RULE_ORDER if any((r, m) in blocks for m in models)]

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 5.2))
    x = range(len(rules))
    w = 0.8 / max(1, len(models))

    def bars(ax, getter, title, ylabel, baseline_getter=None, baseline_label=None):
        for mi, model in enumerate(models):
            label, color = MODEL_STYLE.get(model, (model, "#777"))
            xs, ys = [], []
            for gi, rule in enumerate(rules):
                b = blocks.get((rule, model))
                if b is None:
                    continue
                v = getter(b)
                if v is None:
                    continue
                xs.append(gi + (mi - (len(models) - 1) / 2) * w); ys.append(v)
            ax.bar(xs, ys, width=w * 0.9, color=color, edgecolor="black", linewidth=0.4, label=label)
        if baseline_getter is not None:
            for gi, rule in enumerate(rules):
                b = blocks.get((rule, models[0]))
                if b is None:
                    continue
                base = baseline_getter(b)
                if base is None:
                    continue
                ax.hlines(base, gi - 0.4, gi + 0.4, color="black", ls="--", lw=1.4,
                          label=baseline_label if gi == 0 else None)
        ax.set_xticks(list(x)); ax.set_xticklabels([r.replace("_", "\n") for r in rules], fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, 1.05); ax.legend(fontsize=7.5, loc="lower right")

    bars(axA, lambda b: b["cot_classification"]["rate"],
         "CoT in-distribution accuracy", "accuracy (turn-1 vs gold)",
         baseline_getter=lambda b: b["nocot_baseline"].get("nocot_step1_acc"),
         baseline_label="no-CoT step-1 (dashed)")
    bars(axB, lambda b: b["articulation"]["grade2_names_true_rate"],
         "Articulates the TRUE rule (grade-2 rate)", "fraction grade-2",
         baseline_getter=lambda b: (b["nocot_baseline"].get("nocot_freeform_median_direct") or 0) / 2.0,
         baseline_label="no-CoT freeform grade/2 (dashed)")

    # Panel C: consistency markers
    for mi, model in enumerate(models):
        label, color = MODEL_STYLE.get(model, (model, "#777"))
        for gi, rule in enumerate(rules):
            b = blocks.get((rule, model))
            if b is None:
                continue
            cs = b["consistency"]
            xpos = gi + (mi - (len(models) - 1) / 2) * w
            if cs.get("accuracy_vs_gold") is not None:
                axC.scatter([xpos], [cs["accuracy_vs_gold"]], marker="o", color=color, s=55,
                            label=(f"{label}: vs gold" if gi == 0 else None), zorder=3)
            if cs.get("accuracy_vs_self") is not None:
                axC.scatter([xpos], [cs["accuracy_vs_self"]], marker="x", color=color, s=70,
                            label=(f"{label}: vs self" if gi == 0 else None), zorder=3)
            if cs.get("accuracy_vs_gold") is not None and cs.get("accuracy_vs_self") is not None:
                axC.plot([xpos, xpos], [cs["accuracy_vs_gold"], cs["accuracy_vs_self"]],
                         color=color, lw=1, alpha=0.5, zorder=2)
    axC.set_xticks(list(x)); axC.set_xticklabels([r.replace("_", "\n") for r in rules], fontsize=8)
    axC.set_title("Consistency: stated rule vs own labels (×) and gold (○)", fontsize=10)
    axC.set_ylabel("compiled-rule accuracy", fontsize=9); axC.set_ylim(0, 1.05)
    axC.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    out = FIGDIR / "fig_cot_same_session.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    out.with_suffix(".caption.txt").write_text(
        "CoT same-session CoT same-session diagnostic. Per rule × model (gpt-4.1 prompted CoT; "
        "claude-opus-4-8 native thinking — different interventions, not pooled). "
        "Left: CoT in-distribution classification accuracy (bars) vs the no-CoT "
        "step-1 baseline (dashed). Middle: fraction of CoT articulations the judge "
        "grades as the TRUE rule (grade-2) vs the no-CoT free-form grade/2 (dashed). "
        "Right: the consistency guard — the compiled stated rule's accuracy against "
        "the model's OWN turn-1 labels (×, =faithful-to-itself) and against gold (○); "
        "a high grade-2 with a LOW vs-self marker indicates post-hoc rationalisation "
        "rather than genuine articulation.\n")
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
