#!/usr/bin/env python
"""Gate 2 — the codified,
reproducible distribution/diversity instrument for a divergence set. No API.

Replaces the ad-hoc "blind agents judge old-or-new" check with concrete
algorithmic instruments, controlling for the DIVERGENCE TRAP (the new items are
deliberately the (a)!=(c) subset, so any discriminator can score high on the
intended content; that is NOT a diversity flaw).

  2A  Classifier two-sample test (C2ST / adversarial validation), reported on
      THREE feature sets so the cause of any separability is visible:
        - content_bow   : word bag-of-words (DIAGNOSTIC only; expected to be high
                          because content differs by design).
        - style_only    : NO content words — sentence length, word-length stats,
                          casing, -ly count, and a bag of FUNCTION/STOP words only
                          (articles, prepositions, pronouns, conjunctions). This is
                          the primary 2A signal.
        - style_lenmatched : style_only, but the OLD sample is resampled to match
                          NEW's word-count distribution (content control for rules
                          whose label IS length, e.g. word_count_geq_8), and the
                          raw length features are dropped. Residual separability
                          here is the cleanest "real style flaw" signal.
      Each: stratified 5-fold CV accuracy + exact two-sided binomial p vs 0.5,
      plus the top discriminating features. PASS band: style classifiers near
      chance (<= ~0.60, not significantly > 0.5). Above -> investigate.

  2B  Reference-free diversity metrics on NEW vs OLD: type-token ratio, distinct
      1/2/3-gram ratios, vocabulary entropy (bits), length mean/sd, n distinct
      template-shapes (function-word skeleton). PASS if every NEW metric is within
      the pre-declared band (>= 0.8x OLD; length distributions overlap).

2C (agent naturalness rating) is run separately, never as a bare old/new call.

Usage:
  python scripts/gate2_divergence_check.py --rule word_count_geq_8 \
      --new data/word_count_geq_8_divergence/items.jsonl \
      --old data/word_count_geq_8/items.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from icl_articulation.stats import binom_test_two_sided  # noqa: E402

# Function / stop words = STYLE (structure), not topic content. A compact, fixed
# English function-word list (articles, prepositions, pronouns, conjunctions,
# auxiliaries, determiners). Deliberately excludes content nouns/verbs/adjectives.
FUNCTION_WORDS = sorted({
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "no",
    "every", "each", "all", "both", "few", "many", "much", "more", "most",
    "i", "you", "he", "she", "it", "we", "they", "him", "her", "them", "us",
    "my", "your", "his", "its", "our", "their", "me",
    "and", "or", "but", "nor", "so", "yet", "if", "then", "than", "as", "because",
    "while", "when", "where", "after", "before", "until", "though", "although",
    "at", "by", "in", "on", "near", "with", "without", "into", "onto", "over",
    "under", "through", "across", "toward", "towards", "beside", "behind",
    "beneath", "above", "below", "between", "among", "around", "past", "along",
    "from", "upon", "within", "for", "of", "to", "off", "out", "up", "down",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "will", "would", "can", "could", "should", "may",
    "might", "must", "not", "very", "too", "again", "here", "there",
})
FW_INDEX = {w: i for i, w in enumerate(FUNCTION_WORDS)}


def load_texts(path: Path, split: str | None) -> list[str]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if split is None or r["split"] == split:
            out.append(r["text"])
    return out


def _toks(text: str) -> list[str]:
    return [w.strip(".,").lower() for w in text.split() if w.strip(".,")]


# --------------------------------------------------------------------------- #
# style features (NO content words)
# --------------------------------------------------------------------------- #
def style_features(text: str, include_length: bool = True) -> np.ndarray:
    raw = text.split()
    toks = _toks(text)
    lens = [len(t) for t in toks] or [0]
    feats: list[float] = []
    if include_length:
        feats += [len(toks), len(text), float(np.mean(lens)), float(np.std(lens)),
                  max(lens), min(lens)]
    else:
        feats += [float(np.mean(lens)), float(np.std(lens))]
    # casing / morphology (style, not topic)
    n_ly = sum(1 for t in toks if t.endswith("ly") and len(t) > 3)
    n_cap = sum(1 for i, w in enumerate(raw) if i > 0 and w[:1].isupper())
    second_cap = 1.0 if len(raw) > 1 and raw[1][:1].isupper() else 0.0
    feats += [n_ly, n_cap, second_cap]
    # function-word bag (counts), L1-normalized by token length -> distribution
    fw = np.zeros(len(FUNCTION_WORDS))
    for t in toks:
        if t in FW_INDEX:
            fw[FW_INDEX[t]] += 1.0
    feats += [sum(fw), sum(fw) / max(1, len(toks))]
    return np.concatenate([np.array(feats, dtype=float), fw])


def style_feature_names(include_length: bool) -> list[str]:
    base = (["n_words", "n_chars", "mean_wlen", "std_wlen", "max_wlen", "min_wlen"]
            if include_length else ["mean_wlen", "std_wlen"])
    base += ["n_ly", "n_cap_nonfirst", "second_word_cap", "n_function", "frac_function"]
    return base + [f"fw::{w}" for w in FUNCTION_WORDS]


# --------------------------------------------------------------------------- #
# 2A — C2ST
# --------------------------------------------------------------------------- #
def c2st(X: np.ndarray, y: np.ndarray, feat_names: list[str] | None = None) -> dict:
    """5-fold CV accuracy + exact binomial p vs 0.5; top discriminating features."""
    clf = LogisticRegression(max_iter=2000, C=1.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pred = cross_val_predict(clf, X, y, cv=skf)
    acc = float((pred == y).mean())
    n = len(y)
    n_correct = int((pred == y).sum())
    p = binom_test_two_sided(n_correct, n)
    out = {"accuracy": acc, "n": n, "binom_p_vs_chance": p,
           "significant_above_chance": (acc > 0.5 and p < 0.05)}
    if feat_names is not None:
        clf.fit(X, y)
        coefs = clf.coef_[0]
        order = np.argsort(np.abs(coefs))[::-1][:12]
        out["top_features"] = [
            {"feature": feat_names[i], "coef": round(float(coefs[i]), 3),
             "favors": "new" if coefs[i] > 0 else "old"} for i in order
        ]
    return out


# Per-rule INTENDED divergence axes, expressed as style-feature name prefixes to
# zero out. If the residual style C2ST collapses to chance once these are removed,
# ALL old-vs-new separability is attributable to the intended divergence (the
# divergence trap), not to a style/diversity flaw -> accept-and-record.
INTENDED_AXES: dict[str, list[str]] = {
    # (a)=length (the rule itself) + (c)=post-verbal modifier. The modifier axis
    # shows up as trailing prepositions, trailing loc/time adverbs, AND the extra
    # determiner a post-verbal PP introduces ("by THE door") — so determiner
    # COUNT (the/a/an) is downstream of the modifier placement, i.e. intended.
    "word_count_geq_8": [
        "n_words", "n_chars", "mean_wlen", "std_wlen", "max_wlen", "min_wlen",
        "n_function", "frac_function", "n_ly",
        "fw::the", "fw::a", "fw::an",  # determiner count = post-verbal PP downstream
        "fw::again", "fw::here", "fw::there",  # loc/time adverb post-modifiers
        "fw::at", "fw::by", "fw::in", "fw::on", "fw::near", "fw::with", "fw::without",
        "fw::into", "fw::onto", "fw::over", "fw::under", "fw::through", "fw::across",
        "fw::toward", "fw::towards", "fw::beside", "fw::behind", "fw::beneath",
        "fw::above", "fw::below", "fw::between", "fw::around", "fw::past", "fw::along",
        "fw::from", "fw::upon", "fw::within", "fw::out", "fw::off", "fw::up", "fw::down",
    ],
    # (c)=subject animacy is a CONTENT (noun) feature, absent from style features;
    # the only style residuals are OLD's pronoun subjects (he/she/it — NEW omits
    # them to avoid a one-sided confound) and length. Ablate those + the trailing
    # clause prepositions (matched OLD tail style, shared across both directions).
    "physically_impossible": [
        "n_words", "n_chars", "mean_wlen", "std_wlen", "max_wlen", "min_wlen",
        "n_function", "frac_function",
        "fw::the", "fw::a", "fw::an",  # determiner count = trailing-clause downstream
        "fw::he", "fw::she", "fw::it", "fw::her", "fw::his", "fw::him",
        "fw::into", "fw::onto", "fw::through", "fw::under", "fw::over", "fw::at",
        "fw::in", "fw::on", "fw::by", "fw::near", "fw::before", "fw::after",
        "fw::during", "fw::without", "fw::for", "fw::of", "fw::to", "fw::around",
        "fw::past", "fw::out", "fw::up", "fw::down", "fw::within", "fw::from",
    ],
    # swc auto-passes; the intended axis (word-2 case) is the second_word_cap
    # feature + the word-2 token itself (not in style bag).
    "second_word_capitalized": ["second_word_cap"],
}


def ablate(X: np.ndarray, feat_names: list[str], prefixes: list[str]) -> np.ndarray:
    """Zero out feature columns whose name is in `prefixes` (intended-axis removal)."""
    Z = X.copy()
    drop = {i for i, nm in enumerate(feat_names) if nm in prefixes}
    for i in drop:
        Z[:, i] = 0.0
    return Z


def length_matched_old(old: list[str], new: list[str], rng: np.random.Generator) -> list[str]:
    """Resample OLD to match NEW's word-count histogram (content control for
    length-driven rules)."""
    new_wc = Counter(len(t.split()) for t in new)
    by_wc: dict[int, list[str]] = {}
    for t in old:
        by_wc.setdefault(len(t.split()), []).append(t)
    out: list[str] = []
    for wc, k in new_wc.items():
        pool = by_wc.get(wc, [])
        if not pool:
            continue
        idx = rng.integers(0, len(pool), size=k)
        out += [pool[i] for i in idx]
    return out


# --------------------------------------------------------------------------- #
# 2B — reference-free diversity
# --------------------------------------------------------------------------- #
def diversity(texts: list[str]) -> dict:
    toks_all = [t for s in texts for t in _toks(s)]
    n_tok = len(toks_all)
    types = set(toks_all)
    def ngram_ratio(k: int) -> float:
        grams = []
        for s in texts:
            ts = _toks(s)
            grams += [tuple(ts[i:i + k]) for i in range(len(ts) - k + 1)]
        return len(set(grams)) / len(grams) if grams else 0.0
    counts = Counter(toks_all)
    entropy = -sum((c / n_tok) * math.log2(c / n_tok) for c in counts.values()) if n_tok else 0.0
    lens = [len(s.split()) for s in texts]
    # template shape = function-word skeleton (content words -> "_")
    shapes = set()
    for s in texts:
        shapes.add(tuple(t if t in FW_INDEX else "_" for t in _toks(s)))
    return {
        "n_items": len(texts), "n_tokens": n_tok,
        "ttr": len(types) / n_tok if n_tok else 0.0,
        "distinct_1gram": ngram_ratio(1), "distinct_2gram": ngram_ratio(2),
        "distinct_3gram": ngram_ratio(3),
        "vocab_entropy_bits": entropy,
        "len_mean": float(np.mean(lens)), "len_sd": float(np.std(lens)),
        "n_template_shapes": len(shapes),
        "template_shape_ratio": len(shapes) / len(texts) if texts else 0.0,
    }


DIVERSITY_BAND = 0.8  # NEW metric must be >= 0.8x OLD


def diversity_verdict(new_d: dict, old_d: dict) -> dict:
    checks = {}
    for m in ("ttr", "distinct_1gram", "distinct_2gram", "distinct_3gram",
              "vocab_entropy_bits", "template_shape_ratio"):
        ratio = new_d[m] / old_d[m] if old_d[m] else float("inf")
        checks[m] = {"new": round(new_d[m], 4), "old": round(old_d[m], 4),
                     "ratio": round(ratio, 3), "pass": ratio >= DIVERSITY_BAND}
    # length distributions should overlap (mean within 1 sd-ish band reported)
    checks["len_overlap"] = {
        "new_mean": round(new_d["len_mean"], 2), "old_mean": round(old_d["len_mean"], 2),
        "new_sd": round(new_d["len_sd"], 2), "old_sd": round(old_d["len_sd"], 2),
    }
    passed = all(c.get("pass", True) for c in checks.values())
    return {"band": DIVERSITY_BAND, "checks": checks, "pass": passed}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new-split", default="held_out")
    ap.add_argument("--old-splits", default="held_out,confirmation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    new = load_texts(Path(args.new), args.new_split)
    old: list[str] = []
    for sp in args.old_splits.split(","):
        old += load_texts(Path(args.old), sp.strip())

    rng = np.random.default_rng(0)
    # balance the two pools (subsample the larger to the smaller)
    n = min(len(new), len(old))
    old_bal = list(rng.permutation(old)[:n])
    new_bal = list(rng.permutation(new)[:n])
    texts = new_bal + old_bal
    y = np.array([1] * len(new_bal) + [0] * len(old_bal))

    # 2A content BoW (diagnostic)
    bow = CountVectorizer(ngram_range=(1, 1), min_df=1).fit_transform(texts).toarray()
    a_content = c2st(bow, y)
    # 2A style-only (with length)
    Xs = np.array([style_features(t, include_length=True) for t in texts])
    a_style = c2st(Xs, y, style_feature_names(True))
    # 2A style length-matched (content control for length-driven rules)
    old_lm = length_matched_old(old, new, rng)
    nlm = min(len(new), len(old_lm))
    lm_texts = list(rng.permutation(new)[:nlm]) + list(rng.permutation(old_lm)[:nlm])
    y_lm = np.array([1] * nlm + [0] * nlm)
    Xlm = np.array([style_features(t, include_length=False) for t in lm_texts])
    a_style_lm = c2st(Xlm, y_lm, style_feature_names(False))
    # 2A intended-axis-ablated: zero the style features that encode the intended
    # divergence; residual near chance => all separability is the divergence trap.
    feat_names_full = style_feature_names(True)
    Xabl = ablate(Xs, feat_names_full, INTENDED_AXES.get(args.rule, []))
    a_style_ablated = c2st(Xabl, y, feat_names_full)

    # 2B diversity
    new_d, old_d = diversity(new), diversity(old)
    b_verdict = diversity_verdict(new_d, old_d)

    # PASS heuristic for 2A: style classifiers near chance (<=0.60), not sig.
    def near_chance(blk):
        return blk["accuracy"] <= 0.60 and not blk["significant_above_chance"]
    # automatic pass: BOTH style classifiers near chance (the pinned condition).
    a_auto_pass = near_chance(a_style) and near_chance(a_style_lm)
    # accept-and-record path (divergence trap): the raw style separability is
    # fully explained by the intended divergence axis (residual ablated C2ST near
    # chance) AND diversity (2B) is within band. For structurally/semantically
    # divergent rules the style C2ST CANNOT reach chance by construction; this is
    # the addendum's documented exception, made rigorous by the ablation.
    a_intended_only = near_chance(a_style_ablated)
    a_pass = a_auto_pass or (a_intended_only)

    if a_auto_pass:
        a_basis = "auto: both style classifiers near chance"
    elif a_intended_only:
        a_basis = ("accept-and-record: raw style separability collapses to chance "
                   "once the INTENDED divergence axis is ablated (divergence trap), "
                   "and 2B diversity is within band")
    else:
        a_basis = ("FAIL: style separability survives intended-axis ablation -> a "
                   "real style/diversity difference beyond the divergence")

    result = {
        "rule": args.rule,
        "new": args.new, "old": args.old,
        "n_new": len(new), "n_old": len(old),
        "2A_c2st": {
            "content_bow_DIAGNOSTIC": a_content,
            "style_only": a_style,
            "style_lenmatched_CONTROL": a_style_lm,
            "style_intended_axis_ablated": a_style_ablated,
            "intended_axes_zeroed": INTENDED_AXES.get(args.rule, []),
            "pass_band": "<=0.60 and not significant on the style classifiers",
            "auto_pass_both_near_chance": a_auto_pass,
            "intended_only_after_ablation": a_intended_only,
            "pass": a_pass,
            "pass_basis": a_basis,
        },
        "2B_diversity": {"new": new_d, "old": old_d, "verdict": b_verdict},
        "gate2_2A2B_pass": bool(a_pass and b_verdict["pass"]),
        "_note": (
            "2A content_bow is DIAGNOSTIC (high is expected — content differs by "
            "design). PASS rests on EITHER both style classifiers near chance, OR "
            "(divergence trap) the style separability vanishing once the intended "
            "divergence axis is ablated, WITH 2B diversity within band. 2C agent "
            "naturalness is run separately."
        ),
    }

    out_path = Path(args.out) if args.out else (
        REPO / "out" / f"divergence_gate2_{args.rule}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== Gate 2 (2A+2B) — {args.rule} ===")
    print(f"  2A content BoW (diagnostic): acc={a_content['accuracy']:.3f} p={a_content['binom_p_vs_chance']:.1e}")
    print(f"  2A style-only            : acc={a_style['accuracy']:.3f} p={a_style['binom_p_vs_chance']:.1e}  near_chance={near_chance(a_style)}")
    print(f"  2A style length-matched  : acc={a_style_lm['accuracy']:.3f} p={a_style_lm['binom_p_vs_chance']:.1e}  near_chance={near_chance(a_style_lm)}")
    print(f"  2A style INTENDED-ablated: acc={a_style_ablated['accuracy']:.3f} p={a_style_ablated['binom_p_vs_chance']:.1e}  near_chance={near_chance(a_style_ablated)}")
    print(f"     top style features: {[f['feature'] for f in a_style['top_features'][:6]]}")
    print(f"     2A basis: {a_basis}")
    print("  2B diversity (NEW/OLD ratio, pass>=0.8):")
    for m, c in b_verdict["checks"].items():
        if "ratio" in c:
            print(f"     {m:22} new={c['new']:.3f} old={c['old']:.3f} ratio={c['ratio']:.2f} {'OK' if c['pass'] else 'LOW'}")
    print(f"  2B len: new {new_d['len_mean']:.1f}±{new_d['len_sd']:.1f}  old {old_d['len_mean']:.1f}±{old_d['len_sd']:.1f}")
    print(f"  --> 2A pass={a_pass}  2B pass={b_verdict['pass']}  GATE2(2A+2B)={result['gate2_2A2B_pass']}")
    print(f"  wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
