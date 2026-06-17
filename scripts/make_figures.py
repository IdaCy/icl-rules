#!/usr/bin/env python
"""Consolidate all study metrics into one machine-readable table and render the
report figures from results/.

This script is the source for the consolidated metrics table and the five
standard report figures. It:
  (A) walks the run directories under results/, pulls each per-rule number out of
      each run's metrics.json (re-deriving from responses.jsonl only where a
      number is not present in metrics.json), and writes one row per
      (rule_id, model) to results/figures/metrics_table.{csv,json};
  (B) renders the five standard report figures (150 dpi PNG) with full axis
      labels, titles, legends, error bars where applicable, a colorblind-friendly
      palette, and a one-line caption saved alongside each as <fig>.caption.txt.

The extensional-grading Figure 3 is rendered by scripts/analyze_confound_grade.py
because it depends on the compiled-stated-rule audit rather than only the
metrics_table rows.

Everything here is local plotting / json parsing (CPU-trivial, no API calls), so
it is safe to run anywhere. Re-running is idempotent.

Run:  .venv/bin/python scripts/make_figures.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGDIR = RESULTS / "figures"

MODELS = ["gpt-4.1", "gpt-4.1-mini"]

# Rule -> category (drives colour). From the study spec.
RULE_CATEGORY: dict[str, str] = {}
for cat, rules in {
    "surface": [
        "all_lowercase", "contains_digit", "contains_exclamation", "title_case",
        "contains_letter_z", "repeated_content_word", "starts_with_vowel",
    ],
    "syntactic": [
        "past_tense", "passive_voice", "imperative", "question_word_order",
        "contains_comparative",
    ],
    "semantic": [
        "mentions_animal", "mentions_color", "positive_sentiment", "food_topic",
        "contains_first_name", "physically_impossible",
    ],
    "positional": [
        "first_word_longer_than_last", "last_word_ends_with_vowel",
        "the_appears_twice", "second_word_capitalized",
    ],
    "numeric": [
        "word_count_geq_8", "contains_number_gt_50", "even_word_count",
        "exactly_two_commas",
    ],
    "hard": [
        "first_last_same_letter", "double_letter_word",
        "first_two_words_alphabetical", "all_words_longer_than_3",
    ],
}.items():
    for r in rules:
        RULE_CATEGORY[r] = cat

CATEGORY_ORDER = ["surface", "syntactic", "semantic", "positional", "numeric", "hard"]

# Colourblind-friendly (Wong / Okabe-Ito) palette, one colour per category.
CATEGORY_COLOR = {
    "surface": "#0072B2",     # blue
    "syntactic": "#E69F00",   # orange
    "semantic": "#009E73",    # green
    "positional": "#CC79A7",  # reddish-purple
    "numeric": "#D55E00",     # vermillion
    "hard": "#56B4E9",        # sky blue
}

# Marker shape per model (so colour stays free for category).
MODEL_MARKER = {"gpt-4.1": "o", "gpt-4.1-mini": "^"}
MODEL_HATCH = {"gpt-4.1": "", "gpt-4.1-mini": "//"}

CHANCE = 0.5  # step-1 is balanced 2-way (true/false) -> chance 0.5

# Rules to call out in the headline figure.
HEADLINE_STANDOUTS = [
    "word_count_geq_8", "second_word_capitalized", "physically_impossible",
]


# --------------------------------------------------------------------------- #
# Run-directory discovery
# --------------------------------------------------------------------------- #

def _model_of_dir(name: str, prefix: str) -> str | None:
    """Extract the model id from a run-dir name of the form
    ``<prefix>-<model>-<timestamp>``. Returns None if it is not this run type.

    Note: 'gpt-4.1' is a prefix of 'gpt-4.1-mini', so we check the longer id
    first to disambiguate.
    """
    if not name.startswith(prefix + "-"):
        return None
    rest = name[len(prefix) + 1:]
    for model in sorted(MODELS, key=len, reverse=True):
        if rest.startswith(model + "-"):
            return model
    return None


def _n_rules(d: Path) -> int:
    """Number of rules in a run's metrics.json (0 if unreadable)."""
    try:
        return len(json.loads((d / "metrics.json").read_text()).get("rules", {}))
    except Exception:  # noqa: BLE001
        return 0


def find_runs(prefix: str) -> dict[str, Path]:
    """Map model -> run directory for a given run prefix. Prefer the run with the
    MOST rules, then the latest timestamp — so a small single-rule re-run (e.g. the
    deconfound experiment word_count_geq_8_v2, a separate step1-full dir) never
    clobbers the main 30-rule sweep."""
    cand: dict[str, list[Path]] = {}
    for d in sorted(RESULTS.glob(prefix + "-*")):
        if not d.is_dir():
            continue
        model = _model_of_dir(d.name, prefix)
        if model is None:
            continue
        cand.setdefault(model, []).append(d)
    out: dict[str, Path] = {}
    for model, dirs in cand.items():
        # sort key: (n_rules, name) ascending -> last is most-rules, latest ts
        out[model] = sorted(dirs, key=lambda d: (_n_rules(d), d.name))[-1]
    return out


def load_metrics(d: Path) -> dict[str, Any]:
    return json.loads((d / "metrics.json").read_text())


def load_corrected_step3() -> dict[str, dict]:
    """The corrected designed-divergence step-3 analysis written by
    scripts/analyze_step3.py: {model: {rule_id: corrected_block}}. Empty dict if
    the artifact has not been generated yet."""
    path = FIGDIR / "step3_corrected.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for model, blob in data.items():
        rules = blob.get("analysis", {}).get("rules", {})
        out[model] = {rid: rm["corrected_designed_divergence"] for rid, rm in rules.items()}
    return out


def load_corrected_step3_raw() -> dict[str, Any]:
    """Raw corrected Step-3 sidecar, for figure annotations that need family
    metadata not flattened into metrics_table.csv."""
    path = FIGDIR / "step3_corrected.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def load_survivor_set() -> set[str] | None:
    """Survivor rule_ids from the explicit-threshold artifact written by
    scripts/select_survivors.py, or None if it has not been generated.

    Decouples is_survivor from "the rule happens to appear in the confirmation
    run" (the old behaviour, which silently marked all 30 rules survivors if
    confirmation was run with --all). Falls back to the confirmation-run
    membership only when the artifact is absent."""
    path = FIGDIR / "survivors.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return set(data.get("survivors", []))


# --------------------------------------------------------------------------- #
# (A) Consolidated table
# --------------------------------------------------------------------------- #

TABLE_COLUMNS = [
    "rule_id", "model", "category",
    "step1_heldout_acc", "step1_heldout_ci_lo", "step1_heldout_ci_hi",
    "step1_confirmation_acc", "is_survivor",
    "rule_given_acc",
    "mc_frac_true", "mc_no_examples",
    "freeform_median_direct", "freeform_best",
    "faithfulness_all",
    "legacy_empirical_behavior_tracks_true", "legacy_empirical_behavior_tracks_self",
    "legacy_empirical_gap", "legacy_empirical_n_divergence",
    "legacy_empirical_true_ci_lo", "legacy_empirical_true_ci_hi",
    "legacy_empirical_self_ci_lo", "legacy_empirical_self_ci_hi",
    # CORRECTED step-3 (designed-divergence vs author's stated rule) — PRIMARY.
    "corrected_verdict", "corrected_n_designed", "corrected_n_discriminating",
    "corrected_tracks_true_rate", "corrected_tracks_true_ci_lo", "corrected_tracks_true_ci_hi",
    "corrected_binom_p", "corrected_behaviour_constant", "corrected_self_vs_stated_rate",
]


def _pooled_acc_and_ci(rule_block: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    """Held-out accuracy = pooled mean across the held-out contexts, with the
    cluster-bootstrap 95% CI (falls back to the lone context's Wilson CI when
    there is a single context, e.g. rule_given)."""
    pooled = rule_block.get("pooled", {})
    acc = pooled.get("mean_accuracy")
    ci = pooled.get("cluster_bootstrap_ci_95")
    if ci is None:
        # single context -> use that context's Wilson CI as the interval
        ctxs = rule_block.get("contexts", [])
        if len(ctxs) == 1 and ctxs[0].get("wilson_ci_95"):
            ci = ctxs[0]["wilson_ci_95"]
    lo, hi = (ci[0], ci[1]) if ci else (None, None)
    return acc, lo, hi


def build_table() -> list[dict[str, Any]]:
    step1 = find_runs("step1-full")
    conf = find_runs("step1-confirmation")
    rgiven = find_runs("step1-rule_given")
    mc = find_runs("step2mc")
    ff = find_runs("step2-freeform")
    faith = find_runs("step3-faithfulness")  # may be empty until step-3 runs

    # survivor flag comes from the explicit-threshold artifact (decoupled from
    # confirmation-run membership); None -> fall back to "appears in confirmation".
    survivor_set = load_survivor_set()
    corrected = load_corrected_step3()  # {model: {rule: corrected_block}}

    # cache loaded metrics
    M = {
        "step1": {m: load_metrics(p) for m, p in step1.items()},
        "conf": {m: load_metrics(p) for m, p in conf.items()},
        "rgiven": {m: load_metrics(p) for m, p in rgiven.items()},
        "mc": {m: load_metrics(p) for m, p in mc.items()},
        "ff": {m: load_metrics(p) for m, p in ff.items()},
        "faith": {m: load_metrics(p) for m, p in faith.items()},
    }

    rows: list[dict[str, Any]] = []
    for model in MODELS:
        s1 = M["step1"].get(model, {}).get("rules", {})
        cf = M["conf"].get(model, {}).get("rules", {})
        rg = M["rgiven"].get(model, {}).get("rules", {})
        mcm = M["mc"].get(model, {}).get("rules", {})
        ffm = M["ff"].get(model, {}).get("rules", {})
        # per-rule faithfulness lives at metrics.json["faithfulness"]["rules"]
        _faith_block = M["faith"].get(model, {}).get("faithfulness", {})
        fa = _faith_block.get("rules", {})
        _summary = _faith_block.get("summary", [])
        fa_summary = {r["rule_id"]: r for r in _summary} if isinstance(_summary, list) else {}

        for rule_id in sorted(s1.keys()):
            acc, lo, hi = _pooled_acc_and_ci(s1[rule_id])

            conf_acc = None
            if rule_id in cf:
                conf_acc = cf[rule_id].get("pooled", {}).get("mean_accuracy")
            is_survivor = (rule_id in survivor_set) if survivor_set is not None else (rule_id in cf)

            rgiven_acc = None
            if rule_id in rg:
                rgiven_acc = rg[rule_id].get("pooled", {}).get("mean_accuracy")

            mc_frac_true = mc_no_ex = None
            if rule_id in mcm:
                mc_frac_true = mcm[rule_id].get("examples", {}).get("accuracy")
                mc_no_ex = mcm[rule_id].get("no_examples", {}).get("accuracy")

            ff_median = ff_best = None
            if rule_id in ffm:
                ff_median = ffm[rule_id].get("primary_median_direct")
                ff_best = ffm[rule_id].get("secondary_best_variant")

            faith_all = b_true = b_self = gap = n_div = None
            true_ci_lo = true_ci_hi = self_ci_lo = self_ci_hi = None
            frow = fa_summary.get(rule_id)
            if frow is not None:
                faith_all = frow.get("faithfulness_overall")
                b_true = frow.get("legacy_empirical_behavior_tracks_true_rate",
                                  frow.get("behavior_tracks_true_rate"))
                b_self = frow.get("legacy_empirical_behavior_tracks_self_rate",
                                  frow.get("behavior_tracks_self_rate"))
                gap = frow.get("legacy_empirical_gap_true_minus_self",
                               frow.get("gap_true_minus_self"))
                n_div = frow.get("n_divergence")
            elif rule_id in fa:
                fr = fa[rule_id]
                faith_all = fr.get("overall", {}).get(
                    "faithfulness_behavior_vs_self", {}).get("rate")
                hd = fr.get("legacy_empirical_divergence") or fr.get("headline_divergence") or {}
                b_true = hd.get("behavior_tracks_true_rate")
                b_self = hd.get("behavior_tracks_self_rate")
                gap = hd.get("gap_true_minus_self")
                n_div = fr.get("n_divergence")
            # Wilson CIs for the two legacy empirical-divergence rates only ever
            # live in the per-rule legacy block, never in the flat summary, so
            # read them from there regardless of which branch supplied the rates.
            if rule_id in fa:
                hd = fa[rule_id].get("legacy_empirical_divergence") or fa[rule_id].get("headline_divergence") or {}
                tci = hd.get("behavior_tracks_true_ci")
                sci = hd.get("behavior_tracks_self_ci")
                if tci:
                    true_ci_lo, true_ci_hi = tci[0], tci[1]
                if sci:
                    self_ci_lo, self_ci_hi = sci[0], sci[1]

            # CORRECTED step-3 (designed-divergence vs author's stated rule)
            c_verdict = c_n_des = c_n_disc = c_true = c_true_lo = c_true_hi = None
            c_binom = c_const = c_self = None
            cblock = corrected.get(model, {}).get(rule_id)
            if cblock is not None:
                c_verdict = cblock["verdict"]
                c_n_des = cblock["n_designed"]
                disc = cblock["discriminating"]
                c_n_disc = disc["n"]
                c_true = disc["behaviour_tracks_true_rate"]
                tci = disc.get("behaviour_tracks_true_ci") or [None, None]
                c_true_lo, c_true_hi = tci[0], tci[1]
                c_binom = disc.get("binom_p_two_sided_vs_chance")
                c_const = cblock["behaviour_distribution"]["is_constant"]
                c_self = cblock["self_application_reliability"]["self_vs_stated"]["rate"]

            rows.append({
                "rule_id": rule_id,
                "model": model,
                "category": RULE_CATEGORY.get(rule_id, "unknown"),
                "step1_heldout_acc": acc,
                "step1_heldout_ci_lo": lo,
                "step1_heldout_ci_hi": hi,
                "step1_confirmation_acc": conf_acc,
                "is_survivor": is_survivor,
                "rule_given_acc": rgiven_acc,
                "mc_frac_true": mc_frac_true,
                "mc_no_examples": mc_no_ex,
                "freeform_median_direct": ff_median,
                "freeform_best": ff_best,
                "faithfulness_all": faith_all,
                "legacy_empirical_behavior_tracks_true": b_true,
                "legacy_empirical_behavior_tracks_self": b_self,
                "legacy_empirical_gap": gap,
                "legacy_empirical_n_divergence": n_div,
                "legacy_empirical_true_ci_lo": true_ci_lo,
                "legacy_empirical_true_ci_hi": true_ci_hi,
                "legacy_empirical_self_ci_lo": self_ci_lo,
                "legacy_empirical_self_ci_hi": self_ci_hi,
                "corrected_verdict": c_verdict,
                "corrected_n_designed": c_n_des,
                "corrected_n_discriminating": c_n_disc,
                "corrected_tracks_true_rate": c_true,
                "corrected_tracks_true_ci_lo": c_true_lo,
                "corrected_tracks_true_ci_hi": c_true_hi,
                "corrected_binom_p": c_binom,
                "corrected_behaviour_constant": c_const,
                "corrected_self_vs_stated_rate": c_self,
            })
    return rows


def write_table(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIGDIR / "metrics_table.csv"
    json_path = FIGDIR / "metrics_table.json"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


# --------------------------------------------------------------------------- #
# Figure helpers
# --------------------------------------------------------------------------- #

def save_caption(fig_path: Path, caption: str) -> None:
    fig_path.with_suffix(".caption.txt").write_text(caption.strip() + "\n")


def rows_by(rows: list[dict], model: str) -> dict[str, dict]:
    return {r["rule_id"]: r for r in rows if r["model"] == model}


def category_legend_handles(present: set[str] | None = None) -> list[Patch]:
    """Category swatches; if ``present`` is given, only categories that actually
    appear in the plot are shown (so the legend never lists, e.g., 'hard' when no
    hard rule is plotted)."""
    cats = [c for c in CATEGORY_ORDER if present is None or c in present]
    return [Patch(facecolor=CATEGORY_COLOR[c], edgecolor="none", label=c)
            for c in cats]


def model_legend_handles() -> list[Line2D]:
    return [Line2D([0], [0], marker=MODEL_MARKER[m], color="0.3", linestyle="none",
                   markersize=8, label=m) for m in MODELS]


# --------------------------------------------------------------------------- #
# Figure 1 — headline scatter
# --------------------------------------------------------------------------- #

def fig_headline_scatter(rows: list[dict]) -> Path:
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(9, 7))
    # jitter the discrete y (0/1/2 median, can be 0.5/1.5) for visibility
    rng = __import__("random").Random(0)

    # explicit region criterion (threshold-defined, NOT editorial):
    #   classified well (acc >= 0.90) AND articulated imperfectly (grade <= 1)
    ACC_THRESH, GRADE_THRESH = 0.90, 1.0
    present_cats: set[str] = set()
    region_points: list[tuple[str, str]] = []
    region_rules: set[str] = set()
    high_acc_grade2_points: list[tuple[str, str]] = []
    high_acc_grade2_rules: set[str] = set()
    for model in MODELS:
        for r in rows:
            if r["model"] != model:
                continue
            # x = confirmation acc for survivors, else held-out acc
            x = r["step1_confirmation_acc"] if r["is_survivor"] else r["step1_heldout_acc"]
            y = r["freeform_median_direct"]
            if x is None or y is None:
                continue
            present_cats.add(r["category"])
            if x >= ACC_THRESH and y <= GRADE_THRESH:
                region_points.append((r["rule_id"], model))
                region_rules.add(r["rule_id"])
            if x >= ACC_THRESH and y == 2:
                high_acc_grade2_points.append((r["rule_id"], model))
                high_acc_grade2_rules.add(r["rule_id"])
            jy = y + rng.uniform(-0.06, 0.06)
            jx = x + rng.uniform(-0.004, 0.004)
            ax.scatter(jx, jy, s=90, marker=MODEL_MARKER[model],
                       color=CATEGORY_COLOR[r["category"]],
                       edgecolor="black", linewidth=0.5, alpha=0.9, zorder=3)

    # explicit shaded region: acc >= 0.90 AND grade <= 1 (classified well but not
    # articulated at grade 2). Drawn in DATA coordinates so it matches the axes.
    ax.add_patch(Rectangle((ACC_THRESH, -0.25), 1.03 - ACC_THRESH, GRADE_THRESH + 0.25,
                           facecolor="0.85", alpha=0.4, zorder=0, edgecolor="0.6",
                           linestyle="--", linewidth=0.8))
    ax.text(ACC_THRESH + 0.005, GRADE_THRESH - 0.05,
            "classified well (acc ≥ 0.90),\narticulated ≤ partial (grade ≤ 1)",
            fontsize=9.5, color="0.25", style="italic", va="top")

    # Label standout rules (use the gpt-4.1 point for each). Custom text offsets
    # keep the labels from colliding.
    by41 = rows_by(rows, "gpt-4.1")
    label_offsets = {
        "word_count_geq_8": (-0.06, 0.55),
        "second_word_capitalized": (0.015, 0.42),
        "physically_impossible": (-0.09, 0.34),
    }
    for rule in HEADLINE_STANDOUTS:
        r = by41.get(rule)
        if not r:
            continue
        x = r["step1_confirmation_acc"] if r["is_survivor"] else r["step1_heldout_acc"]
        y = r["freeform_median_direct"]
        if x is None or y is None:
            continue
        dx, dy = label_offsets.get(rule, (-0.02, 0.28))
        ha = "left" if dx > 0 else "right"
        ax.annotate(rule, xy=(x, y), xytext=(x + dx, y + dy),
                    fontsize=9, ha=ha,
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))

    ax.set_xlabel("Step-1 classification accuracy\n(confirmation set for survivors, else held-out)")
    ax.set_ylabel("Step-2 free-form articulation grade\n(median of direct variant; 0 = none, 2 = correct)")
    ax.set_title("High classification accuracy does not guarantee strong articulation")
    ax.set_xlim(0.40, 1.03)
    ax.set_ylim(-0.25, 2.25)
    ax.set_yticks([0, 1, 2])
    ax.axvline(CHANCE, color="0.6", linestyle=":", lw=1)
    ax.text(CHANCE + 0.005, 2.15, "chance (0.5)", fontsize=8, color="0.5")
    ax.grid(True, axis="both", alpha=0.2)

    leg1 = ax.legend(handles=category_legend_handles(present_cats), title="rule category",
                     loc="upper left", fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=model_legend_handles(), title="model",
              loc="lower left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out = FIGDIR / "fig_headline_scatter.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    save_caption(out,
        "Each point is one rule x model; x is Step-1 classification accuracy "
        "(confirmation set for survivors, else held-out), y is the Step-2 "
        "free-form articulation grade (median of the direct variant, 0-2). The "
        "shaded region is threshold-defined: accuracy >= 0.90 AND grade <= 1 "
        f"(classified well, articulated at most partially). It contains {len(region_points)} "
        f"rule x model points across {len(region_rules)} unique rules. High "
        f"accuracy also includes {len(high_acc_grade2_points)} grade-2 points "
        f"across {len(high_acc_grade2_rules)} unique rules, so the plot should "
        "be read as a local apply-but-imperfectly-state region, not as an "
        "exclusive taxonomy of high-accuracy rules. "
        "word_count_geq_8, second_word_capitalized and physically_impossible "
        "(the Step-3 rules) are annotated; their grade-0/1 articulations are "
        "examined extensionally and counterfactually in the text (the grade is "
        "not the whole story — see Steps 2-3). Marker colour = category (only "
        "categories present are shown; no hard-category rule is a survivor); "
        "shape = model.")
    return out


# --------------------------------------------------------------------------- #
# Figure 2 — step-1 held-out accuracy bars
# --------------------------------------------------------------------------- #

def fig_step1_accuracy_bars(rows: list[dict]) -> Path:
    by = {m: rows_by(rows, m) for m in MODELS}
    rules = sorted(by["gpt-4.1"].keys(),
                   key=lambda r: by["gpt-4.1"][r]["step1_heldout_acc"] or 0.0)

    n = len(rules)
    y = list(range(n))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, max(8, 0.34 * n)))

    for i, model in enumerate(MODELS):
        offs = (0.5 - i) * h
        accs, los, his, colors = [], [], [], []
        for r in rules:
            row = by[model].get(r, {})
            a = row.get("step1_heldout_acc")
            lo = row.get("step1_heldout_ci_lo")
            hi = row.get("step1_heldout_ci_hi")
            accs.append(a if a is not None else 0)
            los.append((a - lo) if (a is not None and lo is not None) else 0)
            his.append((hi - a) if (a is not None and hi is not None) else 0)
            colors.append(CATEGORY_COLOR[RULE_CATEGORY.get(r, "surface")])
        ypos = [yy + offs for yy in y]
        ax.barh(ypos, accs, height=h, color=colors,
                edgecolor="black", linewidth=0.4,
                hatch=MODEL_HATCH[model], alpha=0.95, zorder=2)
        ax.errorbar(accs, ypos, xerr=[los, his], fmt="none",
                    ecolor="0.2", elinewidth=0.8, capsize=2, zorder=3)

    # mark survivors with a star just inside the left axis edge
    by41 = by["gpt-4.1"]
    for yy, r in zip(y, rules):
        if by41.get(r, {}).get("is_survivor"):
            ax.text(0.012, yy, "*", fontsize=15, va="center", ha="center",
                    color="black", zorder=5)

    ax.axvline(CHANCE, color="red", linestyle="--", lw=1.2, zorder=1)
    ax.text(CHANCE + 0.005, -0.6, "chance = 0.5", color="red", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(rules, fontsize=8)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Step-1 held-out classification accuracy (pooled over 3 contexts)")
    ax.set_title("Per-rule held-out accuracy with 95% CIs (* = Step-1 survivor)")

    cat_handles = category_legend_handles()
    model_handles = [Patch(facecolor="0.7", edgecolor="black",
                           hatch=MODEL_HATCH[m], label=m) for m in MODELS]
    leg1 = ax.legend(handles=cat_handles, title="rule category",
                     loc="lower right", fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="model (hatch)",
              loc="center right", fontsize=8, framealpha=0.9)
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    out = FIGDIR / "fig_step1_accuracy_bars.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    save_caption(out,
        "Step-1 held-out classification accuracy per rule for both models "
        "(pooled = mean of the 3 per-context accuracies), sorted by gpt-4.1 "
        "accuracy. Error bars are an ITEM-LEVEL bootstrap 95% CI (items resampled "
        "with replacement; the 3 contexts are held FIXED, so context-to-context "
        "variance is not captured — per-context spread can be large, e.g. "
        "contains_exclamation 0.99/0.93/0.75). Red dashed line is two-way chance "
        "(0.5); stars mark Step-1 survivors (held-out acc >= 0.85 for at least one "
        "model). Bar colour = rule category; hatch = model.")
    return out


# --------------------------------------------------------------------------- #
# Figure 3 — ICL vs rule_given dissociation
# --------------------------------------------------------------------------- #

def fig_rule_given_dissociation(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 8))

    for model in MODELS:
        for r in rows:
            if r["model"] != model:
                continue
            x = r["step1_heldout_acc"]
            y = r["rule_given_acc"]
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=85, marker=MODEL_MARKER[model],
                       color=CATEGORY_COLOR[r["category"]],
                       edgecolor="black", linewidth=0.5, alpha=0.9, zorder=3)

    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", lw=1, zorder=1,
            label="y = x (ICL == told the rule)")
    ax.axvline(CHANCE, color="0.7", linestyle=":", lw=1)
    ax.axhline(CHANCE, color="0.7", linestyle=":", lw=1)

    # quadrant guidance text
    ax.text(0.97, 0.52, "can apply when told,\ncan't learn from examples",
            fontsize=8.5, ha="right", color="0.3", style="italic")
    ax.text(0.52, 0.97, "learns from examples,\ncan't apply the stated rule",
            fontsize=8.5, ha="left", va="top", color="0.3", style="italic")

    ax.set_xlim(0.40, 1.03)
    ax.set_ylim(0.40, 1.03)
    ax.set_xlabel("Step-1 in-context-learning accuracy (examples only, no rule text)")
    ax.set_ylabel("Rule-given accuracy (rule stated explicitly, no examples)")
    ax.set_title("Learning a rule from examples vs. applying it when told")
    ax.grid(True, alpha=0.2)

    handles = category_legend_handles() + [
        Line2D([0], [0], color="0.5", linestyle="--", lw=1, label="y = x")]
    leg1 = ax.legend(handles=handles, title="rule category", loc="lower left",
                     fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=model_legend_handles(), title="model", loc="upper left",
              fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out = FIGDIR / "fig_rule_given_dissociation.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    save_caption(out,
        "Dissociation between in-context learning and instruction following: "
        "x is Step-1 accuracy from examples alone, y is accuracy when the rule "
        "is stated explicitly (no examples). Points above the y=x diagonal are "
        "rules the model applies better when told than it learns from examples "
        "(e.g. all_lowercase 0.50->0.98); points below it are learned from "
        "examples but applied worse when told - word_count_geq_8 (0.92, 0.58) "
        "learned a proxy, and even_word_count (0.81, 0.50) cannot be applied "
        "even when told. Colour = category, marker = model.")
    return out


# --------------------------------------------------------------------------- #
# Figure 4 — multiple-choice vs free-form articulation (survivors)
# --------------------------------------------------------------------------- #

def fig_articulation_mc_vs_freeform(rows: list[dict]) -> Path:
    # Use rules that have free-form / multiple-choice data (the 11 articulation-probe rules).
    by41 = rows_by(rows, "gpt-4.1")
    probe_rules = sorted(
        [r for r, row in by41.items() if row["mc_frac_true"] is not None],
        key=lambda r: (by41[r]["freeform_median_direct"] or 0,
                       by41[r]["mc_frac_true"] or 0),
    )
    n = len(probe_rules)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(6.5, 0.42 * n)),
                             sharey=True)

    # The multiple-choice no-examples control is uniformly 0 (the model parse-fails without
    # examples), so we draw the two informative series as bars and mark the
    # control with a small "x at 0" + a note, instead of an invisible bar.
    metric_specs = [
        ("mc_frac_true", "multiple-choice frac-true (recognition)", "#0072B2", lambda v: v),
        ("freeform_median_direct", "Free-form median grade /2 (production)",
         "#D55E00", lambda v: (v / 2.0) if v is not None else None),
    ]

    y = list(range(n))
    bw = 0.36
    for ax, model in zip(axes, MODELS):
        by = rows_by(rows, model)
        for j, (key, label, color, scale) in enumerate(metric_specs):
            offs = (0.5 - j) * bw
            vals = []
            for r in probe_rules:
                v = by.get(r, {}).get(key)
                vals.append(scale(v) if v is not None else 0)
            ax.barh([yy + offs for yy in y], vals, height=bw, color=color,
                    edgecolor="black", linewidth=0.4, label=label, zorder=2)
        # control markers at x=0
        ctrl_y = [yy for yy, r in zip(y, probe_rules)
                  if by.get(r, {}).get("mc_no_examples") is not None]
        ax.scatter([0.0] * len(ctrl_y), ctrl_y, marker="x", color="0.45",
                   s=22, zorder=4,
                   label="multiple-choice no-examples control = 0 (all rules)")
        ax.set_title(model)
        ax.set_xlim(-0.02, 1.05)
        ax.axvline(0.125, color="0.5", linestyle=":", lw=1)
        # label the chance line in-axes, anchored to the bottom so it never clips
        ax.text(0.13, -0.45, "multiple-choice chance 0.125", fontsize=7.5, color="0.4",
                va="bottom", rotation=90)
        ax.set_xlabel("rate (multiple-choice: frac-true; free-form: grade/2)")
        ax.grid(True, axis="x", alpha=0.25)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(probe_rules, fontsize=8.5)
    axes[0].set_ylim(-0.6, n - 0.2)
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.suptitle("Articulation by elicitation mode: recognition (multiple-choice) vs production (free-form)",
                 fontsize=13)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGDIR / "fig_articulation_mc_vs_freeform.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    save_caption(out,
        "Articulation by elicitation mode for the probe rules: multiple-choice "
        "frac-true (the model picks the correct rule description from 8 options), "
        "the multiple-choice no-examples control (x at 0), and free-form median grade rescaled "
        "to [0,1] (grade/2). Dotted line is 8-way multiple-choice chance (0.125). Recognition "
        "generally exceeds production. physically_impossible multiple-choice is shown AFTER an "
        "instrument fix: the original 0.08 (below chance) was an artefact of a "
        "vacuous distractor gate that admitted a near-synonym distractor ('an "
        "object doing a human action'); with a real predicate + that near-synonym "
        "banned and the option set rebuilt, recognition recovers to 1.0 (gpt-4.1) "
        "and 0.5 (gpt-4.1-mini). second_word_capitalized is a GENUINE "
        "low-recognition case (multiple-choice 0.25 for gpt-4.1, 0.167 for mini; its "
        "distractor gate is exact). The gpt-4.1-mini "
        "panel is the de-truncated re-run (max_tokens=256, CoT-tolerant parse): at "
        "the original max_tokens=2 mini truncated mid-reasoning on 27% of calls "
        "(examples-arm parse failures 35 -> 1). The no-examples control is a "
        "refusal artefact (the model declines without examples), not a "
        "forced-choice chance floor.")
    return out


# --------------------------------------------------------------------------- #
# Figure 5 — faithfulness gap (step-3)
# --------------------------------------------------------------------------- #

# Verdict -> (colour, short label) for the corrected step-3 figure.
_VERDICT_STYLE = {
    "true": ("#D55E00", "unfaithful\n(tracks true)"),      # vermillion
    "stated": ("#009E73", "~faithful\n(tracks stated)"),   # green
    "neither": ("0.6", "ambiguous /\ndefault"),            # grey
    "control": ("#0072B2", "control\n(no divergence)"),    # blue
}


def _verdict_key(verdict: str | None) -> str:
    if not verdict:
        return "neither"
    if verdict.startswith("control"):
        return "control"
    if "TRUE" in verdict:
        return "true"
    if "STATED" in verdict:
        return "stated"
    return "neither"


def fig_faithfulness_gap(rows: list[dict]) -> Path:
    """CORRECTED step-3 figure: on the FIXED designed-divergence probe set, the
    rate at which in-context behaviour tracks the TRUE rule (scored against the
    author's stated-rule label, not arm-2 behaviour). One bar per rule x model
    (no complementary double-plotting); chance = 0.5; high => unfaithful (tracks
    true), low => faithful (tracks stated), grey => neither/ambiguous/default."""
    out = FIGDIR / "fig_faithfulness_gap.png"
    by41 = rows_by(rows, "gpt-4.1")
    target = [r for r, row in by41.items() if row.get("corrected_n_designed") is not None]
    corrected_raw = load_corrected_step3_raw()

    if not target:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axis("off")
        ax.text(0.5, 0.6, "Corrected step-3 analysis not yet generated",
                ha="center", va="center", fontsize=14, weight="bold")
        ax.text(0.5, 0.38,
                "Run scripts/analyze_step3.py to write results/figures/"
                "step3_corrected.json, then re-run scripts/make_figures.py.",
                ha="center", va="center", fontsize=10, color="0.3")
        fig.savefig(out, dpi=170)
        plt.close(fig)
        save_caption(out, "PLACEHOLDER: run scripts/analyze_step3.py first.")
        return out

    # fixed, interpretable order: unfaithful first, then faithful, then control
    order = {"physically_impossible": 0, "second_word_capitalized": 1,
             "word_count_geq_8": 2, "food_topic": 3}
    rules = sorted(target, key=lambda r: order.get(r, 9))
    n = len(rules)
    x = list(range(n))

    fig, axes = plt.subplots(1, len(MODELS), figsize=(6.6 * len(MODELS), 6.2),
                             sharey=True)
    if len(MODELS) == 1:
        axes = [axes]
    def family_disc_stats(model: str, rule: str, family: str) -> dict[str, Any] | None:
        try:
            families = (
                corrected_raw[model]["analysis"]["rules"][rule]
                ["corrected_designed_divergence"]["families"]
            )
        except KeyError:
            return None
        for item in families:
            if item.get("family") == family:
                return item.get("discriminating")
        return None

    pi_handles_added = False
    for ax, model in zip(axes, MODELS):
        by = rows_by(rows, model)
        for xx, r in zip(x, rules):
            row = by.get(r, {})
            vkey = _verdict_key(row.get("corrected_verdict"))
            color, _ = _VERDICT_STYLE[vkey]
            rate = row.get("corrected_tracks_true_rate")
            n_disc = row.get("corrected_n_discriminating")
            constant = row.get("corrected_behaviour_constant")
            if rate is None:  # control: no discriminating probes
                ax.scatter([xx], [0.5], marker="D", s=55, color=color, zorder=4)
                ax.annotate("control\n(stated=true,\nno divergence)", (xx, 0.5),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", va="bottom", fontsize=8, color="0.3")
                continue
            lo = row.get("corrected_tracks_true_ci_lo")
            hi = row.get("corrected_tracks_true_ci_hi")
            if r == "physically_impossible":
                fam = family_disc_stats(model, r, "A_animate_impossible")
                if fam and fam["behaviour_vs_true"]["rate"] is not None:
                    clean_rate = fam["behaviour_vs_true"]["rate"]
                    clean_n = fam["behaviour_vs_true"]["n"]
                    clean_agree = fam["behaviour_vs_true"]["agree"]
                    clean_lo = fam["behaviour_vs_true"]["ci_low"]
                    clean_hi = fam["behaviour_vs_true"]["ci_high"]
                    clean_yerr = [[max(0.0, clean_rate - clean_lo)],
                                  [max(0.0, clean_hi - clean_rate)]]
                    ax.bar([xx - 0.17], [clean_rate], width=0.32, color=color,
                           edgecolor="black", linewidth=0.6,
                           yerr=clean_yerr, ecolor="0.15",
                           error_kw=dict(elinewidth=1.0, capsize=3, capthick=1.0, zorder=5),
                           label="PI clean Family A" if not pi_handles_added else None)
                    ax.text(xx - 0.17, max(clean_rate, 0.5) + 0.07 + (clean_hi - clean_rate),
                            f"clean A\n{clean_agree}/{clean_n}",
                            ha="center", va="bottom", fontsize=8.5)
                full_yerr = [[max(0.0, rate - lo)], [max(0.0, hi - rate)]] if (lo is not None) else None
                ax.bar([xx + 0.17], [rate], width=0.32, color=color, alpha=0.35,
                       edgecolor="black", linewidth=0.6, hatch="///",
                       yerr=full_yerr, ecolor="0.15",
                       error_kw=dict(elinewidth=1.0, capsize=3, capthick=1.0, zorder=5),
                       label="PI extended audit" if not pi_handles_added else None)
                ax.text(xx + 0.17, max(rate, 0.5) + 0.07 + (hi - rate if hi else 0),
                        f"audit set\n{int(round(rate * n_disc))}/{n_disc}",
                        ha="center", va="bottom", fontsize=8.5)
                pi_handles_added = True
            else:
                yerr = [[max(0.0, rate - lo)], [max(0.0, hi - rate)]] if (lo is not None) else None
                ax.bar([xx], [rate], width=0.6, color=color, edgecolor="black",
                       linewidth=0.5, hatch=("xx" if constant else ""),
                       yerr=yerr, ecolor="0.15",
                       error_kw=dict(elinewidth=1.0, capsize=3, capthick=1.0, zorder=5))
                p = row.get("corrected_binom_p")
                ptxt = f"\np={p:.0e}" if (p is not None and p < 0.01) else (
                       f"\np={p:.2f}" if p is not None else "")
                note = " constant" if constant else ""
                ax.text(xx, max(rate, 0.5) + 0.04 + (hi - rate if hi else 0),
                        f"track-true\n{rate:.2f}\nn={n_disc}{note}{ptxt}",
                        ha="center", va="bottom", fontsize=8.5)
        ax.axhline(0.5, color="0.5", linestyle="--", lw=1.1)
        ax.text(n - 0.5, 0.515, "chance 0.5", fontsize=8.5, color="0.5", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels(rules, rotation=22, ha="right", fontsize=9)
        ax.set_ylim(0, 1.32)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(model)
        ax.set_ylabel("rate in-context behaviour tracks the TRUE rule\n"
                      "(discriminating designed-divergence probes)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.text(0.5, -0.30, "↑ unfaithful  ·  ↓ faithful", transform=ax.transAxes,
                ha="center", fontsize=9, color="0.35")

    legend_handles = [Patch(facecolor=c, edgecolor="black", label=lab.replace("\n", " "))
                      for c, lab in _VERDICT_STYLE.values()]
    legend_handles.extend([
        Patch(facecolor="#D55E00", edgecolor="black", label="PI clean Family A"),
        Patch(facecolor="#D55E00", alpha=0.35, hatch="///", edgecolor="black",
              label="PI extended audit"),
    ])
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Step-3 faithfulness (corrected): does behaviour track the true rule, "
                 "the stated rule, or neither?")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    save_caption(out,
        "Corrected step-3 faithfulness on the FIXED designed-divergence probe set "
        "(scored against the author's stated-rule label, NOT the model's own arm-2 "
        "behaviour, which the original empirical conditioning used). Bars show the "
        "rate at which in-context behaviour tracks the TRUE rule on the "
        "discriminating probes (true != stated), with Wilson "
        "95% CIs and an exact two-sided binomial p vs chance (0.5, dashed). High => "
        "unfaithful (behaviour follows the learned rule, not the stated one); low "
        "=> approximately faithful; grey marks behaviour that is ambiguous, "
        "default-confounded, or tracks neither, with hatching for a constant OOD "
        "default. "
        "For physically_impossible, the solid bar is the clean Family-A "
        "animate-impossibility subset and the hatched translucent bar is the "
        "extended historical audit set. The conservative gpt-4.1 claim is the clean "
        "historical predicate-discriminating Family-A result (11/11, p~1e-3); "
        "the semantic family includes a twelfth true-tracking runner item "
        "excluded from that historical denominator by a shallow predicate bug. "
        "The 23-probe gpt-4.1 extended audit is 23/23, but its other 12 probes rest "
        "on a contestable stated-rule label that gpt-4.1's own self-application "
        "endorses only 1/12. gpt-4.1-mini's articulation lacks the impossibility "
        "qualifier and endorses 11/12 of that family, so its 20/23 full-set "
        "audit is less label-contingent but still small-n and exact-articulation "
        "dependent. "
        "second_word_capitalized does not track the true position rule, but its "
        "low track-true rate is stated/default ambiguous because the probes are "
        "direction-imbalanced and behaviour mostly defaults False; word_count_geq_8 "
        "is constant-False OOD; food_topic is the "
        "no-divergence control.")
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="write metrics_table.csv/json and exit without rendering figures",
    )
    args = parser.parse_args(argv)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = build_table()
    csv_path, json_path = write_table(rows)
    print(f"[table] wrote {csv_path}  ({len(rows)} rows)")
    print(f"[table] wrote {json_path}")

    if args.tables_only:
        print("[table] tables-only mode; skipped figure rendering")
        return

    for fn in (
        fig_headline_scatter,
        fig_step1_accuracy_bars,
        fig_rule_given_dissociation,
        fig_articulation_mc_vs_freeform,
        fig_faithfulness_gap,
    ):
        p = fn(rows)
        print(f"[figure] wrote {p}")


if __name__ == "__main__":
    main()
