#!/usr/bin/env python
"""in-session figure (NO API, LOCAL): in-session per-item classify-then-articulate.

Reads results/figures/insession_articulation.json. Three panels: (A) in-session
classification accuracy Exp1 vs Exp2 per rule×model + the single-call Step-1
baseline (dashed); (B) articulation grade-2 ("names true rule") rate Exp1 vs Exp2;
(C) the per-turn-position accuracy drift curve. No-op if the json is absent.
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
# (model, experiment) -> (label, color, hatch)
STYLE = {
    ("gpt-4.1", "exp1-no-cot"): ("gpt-4.1 no-CoT", "#90caf9", ""),
    ("gpt-4.1", "exp2-cot"): ("gpt-4.1 CoT", "#1565c0", ""),
    ("claude-opus-4-8", "exp1-no-cot"): ("claude no-CoT", "#ef9a9a", ""),
    ("claude-opus-4-8", "exp2-cot"): ("claude CoT", "#c1432f", ""),
}


def main() -> None:
    src = FIGDIR / "insession_articulation.json"
    if not src.is_file():
        print("[fig] results/figures/insession_articulation.json absent — skipping (no-op).")
        return
    data = json.loads(src.read_text())
    blocks: dict[tuple[str, str, str], dict] = {}
    cells = []
    for pr in data["per_run"]:
        key = (pr["model"], pr["experiment"])
        if key not in cells:
            cells.append(key)
        for rule, b in pr["rules"].items():
            blocks[(pr["model"], pr["experiment"], rule)] = b
    cells = [c for c in STYLE if c in cells] + [c for c in cells if c not in STYLE]
    rules = [r for r in RULE_ORDER if any((m, e, r) in blocks for (m, e) in cells)]
    step1 = {(c["model"], c["rule_id"]): c.get("step1_single_call_acc") for c in data["exp1_vs_exp2"]}

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(17, 5.4))
    x = range(len(rules)); w = 0.8 / max(1, len(cells))

    def grouped(ax, getter, title, ylabel):
        for ci, (m, e) in enumerate(cells):
            label, color, hatch = STYLE.get((m, e), (f"{m}/{e}", "#888", ""))
            xs, ys = [], []
            for gi, rule in enumerate(rules):
                b = blocks.get((m, e, rule))
                if b is None:
                    continue
                v = getter(b)
                if v is None:
                    continue
                xs.append(gi + (ci - (len(cells) - 1) / 2) * w); ys.append(v)
            ax.bar(xs, ys, width=w * 0.9, color=color, edgecolor="black", linewidth=0.4, label=label)
        # single-call Step-1 baseline (dashed) per rule, gpt-4.1
        for gi, rule in enumerate(rules):
            s = step1.get(("gpt-4.1", rule))
            if s is not None and title.startswith("In-session"):
                ax.hlines(s, gi - 0.4, gi + 0.4, color="black", ls="--", lw=1.3,
                          label="Step-1 single-call (gpt-4.1)" if gi == 0 else None)
        ax.set_xticks(list(x)); ax.set_xticklabels([r.replace("_", "\n") for r in rules], fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=9); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower right")

    grouped(axA, lambda b: b["classification"]["rate"],
            "In-session classification accuracy", "accuracy (turn vs gold)")
    grouped(axB, lambda b: b["articulation"]["grade2_names_true_rate"],
            "Articulates the TRUE rule (grade-2 rate)", "fraction grade-2")

    # Panel C: per-turn drift, one line per cell, averaged over rules
    for (m, e) in cells:
        label, color, _ = STYLE.get((m, e), (f"{m}/{e}", "#888", ""))
        # average the turn curves across rules
        agg: dict[int, list[float]] = {}
        for rule in rules:
            b = blocks.get((m, e, rule))
            if not b:
                continue
            for pt in b["classification"]["turn_curve"]:
                agg.setdefault(pt["turn"], []).append(pt["accuracy"])
        if not agg:
            continue
        ts = sorted(agg)
        axC.plot([t + 1 for t in ts], [sum(agg[t]) / len(agg[t]) for t in ts],
                 marker="o", ms=3, color=color, label=label, lw=1.3)
    axC.set_title("In-session accuracy by item position (drift)", fontsize=10)
    axC.set_xlabel("item position in the conversation (1..N)", fontsize=9)
    axC.set_ylabel("accuracy (mean over rules)", fontsize=9); axC.set_ylim(0, 1.05)
    axC.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    out = FIGDIR / "fig_insession_articulation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    out.with_suffix(".caption.txt").write_text(
        "in-session in-session per-item classify-then-articulate. Two experiments differing ONLY by "
        "CoT (Exp 2): classify each held-out item one at a time in one preserved conversation, "
        "then ask the rule. Left: in-session classification accuracy (Exp1 no-CoT vs Exp2 CoT, "
        "per rule×model) with the base single-call Step-1 accuracy dashed for reference. Middle: "
        "fraction of articulations the judge grades as the TRUE rule (grade-2). Right: accuracy by "
        "item position in the conversation — surfaces any drift as the model accumulates its own "
        "answers. gpt-4.1 prompted-CoT and claude native-thinking are different interventions, not pooled.\n")
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
