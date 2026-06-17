#!/usr/bin/env python
"""Cross-rule confound-strength vs articulation-grade analysis (no API). Turns
representative compiled-predicate checks into an exploratory audit + one figure.

For every articulation-probed rule it computes, from the committed data and the
logged grades (NO API):
  * model Step-1 held-out accuracy (both models),
  * a naive-Bayes bag-of-words held-out baseline (how well a dumb LEXICAL shortcut
    does on the rule's own data) and the best single one-sided token,
  * the model's free-form judge grade (both models),
  * the in-distribution accuracy of a representative compiled stated-rule
    predicate, where such a predicate is available.

Honest headline (n = 11, exploratory): articulation grade broadly rises with
classification accuracy (Spearman ~ +0.8), so "apply >> state" is not a general
decoupling. The dissociation is local: a few well-classified rules
(word_count_geq_8, second_word_capitalized) get low intended-rule grades. These
are the strongest proxy cases, where a lexical shortcut already suffices (model
accuracy barely exceeds the bag-of-words baseline), and the compiled predicates
show that the low-grade stated rules are accurate in-distribution descriptions
of the data implementation.

Writes results/figures/confound_grade.json + fig_confound_grade.png (+ caption).
Run:  python scripts/analyze_confound_grade.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "results" / "figures"

# reuse the committed confound machinery (naive-Bayes BoW, best token, char, extensional)
_spec = importlib.util.spec_from_file_location("ac", str(REPO / "scripts" / "analyze_confounds.py"))
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)

PROBED = ["passive_voice", "food_topic", "positive_sentiment", "mentions_animal",
          "contains_first_name", "second_word_capitalized", "physically_impossible",
          "word_count_geq_8", "repeated_content_word", "contains_digit", "mentions_color"]
EXTENSIONAL_RULES = ["word_count_geq_8", "second_word_capitalized",
                     "physically_impossible", "food_topic"]
CATEGORY = {}


def _load_metrics():
    rows = list(csv.DictReader((FIG / "metrics_table.csv").open()))
    for r in rows:
        CATEGORY[r["rule_id"]] = r["category"]

    def cell(rule, model, col):
        for r in rows:
            if r["rule_id"] == rule and r["model"] == model and r[col]:
                return float(r[col])
        return None
    return cell


def _spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None, n
    xs2, ys2 = zip(*pairs)

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs2), rank(ys2)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return (num / den if den else None), n


def main() -> int:
    cell = _load_metrics()
    table = []
    for rule in PROBED:
        bow, _ = ac.naive_bayes_heldout(rule)
        tok = ac.best_single_token(rule)
        ext = None
        if rule in EXTENSIONAL_RULES:
            eg = ac.extensional_grade(rule)
            ext = eg.get("proper_noun_vocab_accuracy", eg["stated_rule_predicate_accuracy"])
        row = {
            "rule": rule,
            "category": CATEGORY.get(rule),
            "acc_gpt_4_1": cell(rule, "gpt-4.1", "step1_heldout_acc"),
            "acc_mini": cell(rule, "gpt-4.1-mini", "step1_heldout_acc"),
            "bow_baseline": bow,
            "best_single_token_acc": tok["accuracy"],
            "best_single_token": tok["token"],
            "grade_gpt_4_1": cell(rule, "gpt-4.1", "freeform_median_direct"),
            "grade_mini": cell(rule, "gpt-4.1-mini", "freeform_median_direct"),
            "stated_rule_extensional_acc": ext,
        }
        row["acc_minus_bow_gpt_4_1"] = (row["acc_gpt_4_1"] - bow) if (row["acc_gpt_4_1"] and bow) else None
        table.append(row)

    acc = [r["acc_gpt_4_1"] for r in table]
    bow = [r["bow_baseline"] for r in table]
    tok = [r["best_single_token_acc"] for r in table]
    grade = [r["grade_gpt_4_1"] for r in table]
    margin = [r["acc_minus_bow_gpt_4_1"] for r in table]
    corr = {
        "spearman_grade_vs_accuracy": _spearman(grade, acc),
        "spearman_grade_vs_bow": _spearman(grade, bow),
        "spearman_grade_vs_best_token": _spearman(grade, tok),
        "spearman_grade_vs_accuracy_minus_bow": _spearman(margin, grade),
    }

    out = {
        "_note": "Cross-rule confound-strength vs articulation-grade audit (gpt-4.1 unless noted). "
                 "BoW = naive-Bayes bag-of-words held-out baseline on the rule's own data. "
                 "Correlations are EXPLORATORY (n=11, coarse 0/1/2 grades).",
        "table": table,
        "spearman_gpt_4_1": {k: {"rho": v[0], "n": v[1]} for k, v in corr.items()},
    }
    (FIG / "confound_grade.json").write_text(json.dumps(out, indent=2) + "\n")

    # ---- figure: the compiled-predicate inversion (dumbbell) --------------------
    # For the rules where the model's stated rule compiles to a predicate, compare
    # the LLM-judge intended-rule grade (normalised to [0,1]) with the
    # in-distribution accuracy of that representative stated-rule predicate.
    ext_rows = [r for r in table if r["stated_rule_extensional_acc"] is not None]
    ext_rows.sort(key=lambda r: r["stated_rule_extensional_acc"] - (r["grade_gpt_4_1"] / 2))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ys = list(range(len(ext_rows)))
    for y, r in zip(ys, ext_rows):
        g = r["grade_gpt_4_1"] / 2.0
        e = r["stated_rule_extensional_acc"]
        ax.plot([g, e], [y, y], color="#bbb", lw=2, zorder=1)
        ax.scatter(g, y, s=110, color="#c33", edgecolor="k", zorder=3,
                   label="LLM-judge grade (÷2)" if y == 0 else None)
        ax.scatter(e, y, s=110, color="#27a", edgecolor="k", zorder=3,
                   label="compiled stated-rule accuracy" if y == 0 else None)
        ax.annotate(f"{e - g:+.2f}", ((g + e) / 2, y), fontsize=7, ha="center",
                    va="bottom", xytext=(0, 4), textcoords="offset points")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["rule"] + ("  (control)" if r["rule"] == "food_topic" else "")
                        for r in ext_rows], fontsize=9)
    ax.set_xlabel("score in [0,1]: judge grade / 2 (red) vs compiled stated-rule accuracy (blue)")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.6, len(ext_rows) - 0.3)
    ax.set_title("Compiled stated-rule checks expose in-distribution proxy articulations", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.01, 0.02))
    fig.tight_layout()
    fig.savefig(FIG / "fig_confound_grade.png", dpi=150, bbox_inches="tight")

    caption = (
        "LLM-judge intended-rule grading vs in-distribution accuracy of representative compiled "
        "stated-rule predicates, for the articulation-probed rules where a predicate is available. "
        "Red = the gpt-4.1 free-form articulation grade from the LLM judge, divided by 2 to share "
        "the [0,1] axis; blue = the compiled predicate scored on the rule's data. Numbers give the "
        "blue-red gap. For word_count_geq_8 a judge-grade-0 "
        "articulation ('an adverb/PP after the verb') is 0.98-accurate in-distribution — the dataset taught "
        "that confound and the model named it; second_word_capitalized's 'proper-noun' reading "
        "matches the true labels on this dataset under the dataset's proper-noun vocabulary (1.00). "
        "food_topic is the control "
        "(stated≈true rule, both high). The check separates intended-rule grading from "
        "in-distribution stated-rule accuracy. (Caveat: second_word_capitalized's 1.00 uses the "
        "dataset's own proper-noun vocabulary; the shallow name-list predicate scores lower — see "
        "confound_audit.json.) Cross-rule confound/grade audit for all 11 probed rules in confound_grade.json: "
        "articulation grade broadly tracks classification accuracy (Spearman +0.86), with no clean "
        "confound-strength->grade law; the low-grade proxy cases (word_count_geq_8, second_word_capitalized) "
        "are cases where a bag-of-words shortcut already approximates the in-distribution labels."
    )
    (FIG / "fig_confound_grade.caption.txt").write_text(caption + "\n")

    print("=== cross-rule audit (gpt-4.1) ===")
    print(f"{'rule':24s} {'acc':>5s} {'BoW':>5s} {'acc-BoW':>7s} {'grade':>5s} {'ext':>5s}")
    for r in table:
        f = lambda x, d=2: ("  --" if x is None else f"{x:.{d}f}")
        print(f"{r['rule']:24s} {f(r['acc_gpt_4_1'])} {f(r['bow_baseline'])} "
              f"{f(r['acc_minus_bow_gpt_4_1']):>7s} {f(r['grade_gpt_4_1'],1):>5s} {f(r['stated_rule_extensional_acc']):>5s}")
    print("\n=== Spearman (gpt-4.1, exploratory n=11) ===")
    for k, (rho, nn) in corr.items():
        print(f"  {k:42s} rho={rho:+.3f} n={nn}")
    print(f"\nwrote {FIG/'confound_grade.json'}, fig_confound_grade.png, caption")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
