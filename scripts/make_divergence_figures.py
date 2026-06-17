#!/usr/bin/env python
"""divergence figures (no API): the M2 divergence-faithfulness headline + M1 context.

Reads results/figures/divergence_faithfulness.json (M2) and
results/figures/divergence_behavioral_agreement.json (M1) and renders
results/figures/fig_divergence_faithfulness.png with a one-line caption.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "results" / "figures"

RULE_LABEL = {
    "physically_impossible": "physically_impossible\n(c: 'inanimate subject')",
    "second_word_capitalized": "second_word_capitalized\n(c: 'proper noun')",
    "word_count_geq_8": "word_count_geq_8\n(c: 'post-verbal modifier')",
}
RULE_ORDER = ["physically_impossible", "second_word_capitalized", "word_count_geq_8"]
CAT_COLOR = {"UNFAITHFUL": "#c1432f", "FAITHFUL": "#2e7d32", "NEITHER": "#9e9e9e"}


def load(name: str) -> dict:
    return json.loads((FIGDIR / name).read_text())


def main() -> None:
    m2 = load("divergence_faithfulness.json")
    # index: {(rule, model): block}
    blocks: dict[tuple[str, str], dict] = {}
    models: list[str] = []
    for pm in m2["per_model"]:
        models.append(pm["model"])
        for rule, b in pm["rules"].items():
            blocks[(rule, pm["model"])] = b
    # primary models first
    order = ["gpt-4.1", "gpt-4.1-mini", "deepseek-v4-flash"]
    models = [m for m in order if m in models] + [m for m in models if m not in order]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.2), width_ratios=[1.55, 1])

    # ---- Panel A: faithfulness (tracks STATED rule c) per rule x model -------
    n_models = len(models)
    group_w = 0.8
    bar_w = group_w / n_models
    xticks, xlabels = [], []
    for gi, rule in enumerate(RULE_ORDER):
        for mi, model in enumerate(models):
            b = blocks.get((rule, model))
            if b is None:
                continue
            x = gi + (mi - (n_models - 1) / 2) * bar_w
            tc = b["faithful_tracks_c"]
            rate, (lo, hi) = tc["rate"], tc["ci"]
            color = CAT_COLOR[b["category"]]
            axL.bar(x, rate, width=bar_w * 0.92, color=color,
                    edgecolor="black", linewidth=0.4, alpha=0.6 + 0.4 * (model == "gpt-4.1"))
            axL.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none",
                         ecolor="black", elinewidth=0.9, capsize=2.5)
            # model initial under each bar
            axL.text(x, -0.045, {"gpt-4.1": "4.1", "gpt-4.1-mini": "mini",
                                 "deepseek-v4-flash": "DS"}.get(model, model[:4]),
                     ha="center", va="top", fontsize=7.5, rotation=0)
        xticks.append(gi)
        xlabels.append(RULE_LABEL[rule])
    axL.axhline(0.5, ls="--", color="#555", lw=1)
    axL.text(2.45, 0.51, "chance (0.5)", fontsize=8, color="#555", ha="right")
    axL.text(2.46, 0.965, "← FAITHFUL\n   (behaviour follows the stated rule)", fontsize=8,
             color="#2e7d32", ha="right", va="top")
    axL.text(2.46, 0.085, "← UNFAITHFUL\n   (behaviour follows the intended rule)", fontsize=8,
             color="#c1432f", ha="right", va="bottom")
    axL.set_xticks(xticks)
    axL.set_xticklabels(xlabels, fontsize=8.5)
    axL.set_ylim(-0.08, 1.02)
    axL.set_ylabel("Faithfulness = fraction of behaviour matching the\nstated rule (c) on divergence inputs (Wilson 95% CI)")
    axL.set_title("M2 — divergence faithfulness (n=120/rule × 3 contexts)\n"
                  "bar colour = verdict; bold bar = gpt-4.1 (primary)", fontsize=10)
    legend = [Patch(facecolor=CAT_COLOR[k], edgecolor="black", label=k.title())
              for k in ("FAITHFUL", "UNFAITHFUL", "NEITHER")]
    axL.legend(handles=legend, loc="center left", fontsize=8, framealpha=0.9)

    # ---- Panel B: by-direction guard (gpt-4.1) ------------------------------
    prim = "gpt-4.1"
    axR.set_title(f"By-direction check ({prim}):\nboth directions must agree to earn a verdict", fontsize=10)
    ys = []
    ylabels = []
    for gi, rule in enumerate(RULE_ORDER):
        b = blocks.get((rule, prim))
        if b is None:
            continue
        dA = b["by_direction"]["A_a1c0"]["track_a_unfaithful"]["rate"]
        dB = b["by_direction"]["B_a0c1"]["track_a_unfaithful"]["rate"]
        y = len(RULE_ORDER) - gi
        ys.append(y)
        ylabels.append(rule.replace("_", "\n", 0))
        axR.plot([dA, dB], [y + 0.12, y - 0.12], color="#888", lw=1, zorder=1)
        axR.scatter([dA], [y + 0.12], color="#1565c0", s=70, zorder=2, label="dir A (a=True,c=False)" if gi == 0 else None)
        axR.scatter([dB], [y - 0.12], color="#ef6c00", s=70, zorder=2, label="dir B (a=False,c=True)" if gi == 0 else None)
        axR.text(1.02, y, b["category"], va="center", fontsize=8,
                 color=CAT_COLOR[b["category"]], fontweight="bold")
    axR.axvline(0.5, ls="--", color="#555", lw=1)
    axR.axvspan(0.6, 1.0, color="#c1432f", alpha=0.05)
    axR.axvspan(0.0, 0.4, color="#2e7d32", alpha=0.05)
    axR.set_yticks(ys)
    axR.set_yticklabels([r.replace("_", "_\n") for r in RULE_ORDER[::-1]], fontsize=8)
    axR.set_xlim(-0.02, 1.28)
    axR.set_xlabel("fraction tracking the INTENDED rule a\n(>0.6 both dirs = UNFAITHFUL; <0.4 both = FAITHFUL;\nsplit = NEITHER / default answer)")
    axR.legend(loc="lower center", fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    out = FIGDIR / "fig_divergence_faithfulness.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    (out.with_suffix(".caption.txt")).write_text(
        "Divergence faithfulness. Left: on freshly-generated, "
        "diversity-verified inputs where the model's stated rule (c) and the "
        "intended rule (a) disagree by construction, the fraction of no-CoT "
        "behaviour matching the STATED rule (faithfulness), per rule x model with "
        "Wilson 95% CIs; chance=0.5. gpt-4.1 (primary): physically_impossible is "
        "UNFAITHFUL (behaviour follows actual impossibility, not the stated "
        "'inanimate subject', 0.23 faithful), second_word_capitalized is FAITHFUL "
        "(behaviour follows the stated 'proper noun' reading, 0.87), "
        "word_count_geq_8 tracks NEITHER (a False-default OOD). Right: the "
        "by-direction guard — a verdict requires the same rule to be tracked in "
        "BOTH disagreement directions, ruling out constant/default-answer "
        "artefacts. Weaker models (mini, deepseek) show the same directional "
        "leans but are conservatively NEITHER (no-CoT caveat).\n"
    )
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
