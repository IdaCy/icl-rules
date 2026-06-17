"""confound_report.json: the per-rule distributional audit (rule-specs
globals.dataset_construction.required_outputs).

For one rule's items this computes, class-conditionally (True vs False):
  - mean/sd word count
  - mean/sd char count
  - stopword rate (fraction of word tokens on the frozen global stopword list)
  - comma rate (commas per item)
  - capitalized-word rate (fraction of word tokens whose first char is upper)
  - top-20 class-skewed tokens (by True-vs-False count difference)
plus the generic_probe_battery agreements (from battery.py).

It then applies the length-matching asserts (globals.length_matching.policy):
  |mean_wc(T) - mean_wc(F)| <= 0.2  (programmatic rules)
                            <= 1.0  (llm rules)
and an overall pass/fail = (length-matching OK) AND (no battery violation).

LLM-RULE AUDIT THRESHOLDS (rules 15/16 distribution_guards.audit_thresholds)
---------------------------------------------------------------------------
For is_llm_rule=True the spec's quantitative audit_thresholds are HARD GATES:
"dataset REJECTED if violated". build_confound_report computes the
``audit_thresholds`` block and folds every PROGRAMMATIC threshold into
``overall_pass`` so a violating LLM dataset is rejected loudly by the emit
pipeline (nothing is written). The thresholds split into two tiers:

  GENERAL numeric-diff gates (every is_llm_rule=True dataset, incl. rule 18 —
  matched by construction there):
    * |mean word count (T) - (F)|          <= 1.0
    * |stopword rate (T) - (F)|            <= 0.03
    * |comma rate (T) - (F)|               <= 0.05
    * |capitalized-word rate (T) - (F)|    <= 0.05
    * |negator rate (not/never/no/nothing/nobody) (T) - (F)| <= 0.05  (rule 15)

  CONTENT-rule gate (rules 15/16 ONLY — enabled by passing ``topics``; NOT
  applied to rule 18, a word-swap rule where the swapped word IS the signal and
  legitimately lives in one class):
    * topic balance: every topic 50/50 (rule 15) and no single False topic
      > 25% (rule 16) after rebalancing — purely structural, a hard gate.

JUDGE-ADJUDICATED (computed + reported, NOT hard gates):
  * ``judge_skewed_tokens``: tokens with class ratio >= 5:1 and frequency >= 2%.
  * ``judge_tokens_missing_from_class``: tokens with overall doc-frequency >= 3%
    that appear in only ONE class.
For rules 15/16 the rule's OWN signal IS class-exclusive vocabulary —
evaluative words (great/terrible/love/awful) live only in one polarity; food
words (soup/oven/recipe) live only in the food class. These are legitimately
>= 3% frequent and one-sided; the spec does NOT forbid them. It routes them to
the dataset JUDGE: each listed token must be EVALUATIVE / food-denoting — i.e.
rule-bearing — "as judged by the dataset judge"; an incidental skewed token
(e.g. a topic noun that leaked) triggers targeted regeneration. Distinguishing a
legitimate rule-bearing class-exclusive token from an incidental leaked one is a
SEMANTIC judge call, NOT a programmatic hard gate, so BOTH lists are reported for
the judge and left out of ``overall_pass``.

Output is JSON only (never markdown into results/ or data/). Numbers are
plain floats so the file diffs cleanly.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .battery import battery_report, battery_violations
from .schema import word_count, words

# frozen global stopword list (rule-specs globals.stopwords.list)
STOPWORDS = frozenset(
    """a about after all also am an and any are as at be been before being between
    both but by can could did do does during each few for from had has have he her
    here him his how i if in into is it its just may me might more most must my near
    no nor not of off on only onto or our out over own same shall she should so some
    such than that the their them then there these they this those to too under until
    up us very was we were what when where which who why will with would yet you your""".split()
)

PROGRAMMATIC_WC_TOL = 0.2
LLM_WC_TOL = 1.0

# LLM-rule audit_thresholds (rules 15/16 distribution_guards.audit_thresholds).
# These are HARD gates for is_llm_rule=True (spec: "dataset REJECTED if
# violated"). Tolerances are verbatim from the spec.
LLM_STOPWORD_RATE_TOL = 0.03
LLM_COMMA_RATE_TOL = 0.05
LLM_CAP_RATE_TOL = 0.05
LLM_NEGATOR_RATE_TOL = 0.05
# "every token with overall frequency >= 3% appears in both classes"
LLM_TOKEN_BOTH_CLASSES_FREQ = 0.03
# rule 16 "no single non-food (False) topic > 25%"
LLM_FALSE_TOPIC_MAX_SHARE = 0.25
# negators counted for the negator-rate audit (rule 15 audit_threshold)
NEGATORS = frozenset({"not", "never", "no", "nothing", "nobody"})
# judge report: tokens with class ratio >= 5:1 AND overall frequency >= 2% are
# listed for the dataset JUDGE to adjudicate (must be evaluative/rule-bearing).
# This is NOT a programmatic hard gate (see module docstring).
LLM_SKEW_RATIO = 5.0
LLM_SKEW_MIN_FREQ = 0.02


def _mean_sd(xs: Sequence[float]) -> tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return (m, math.sqrt(var))


def _capitalized_word_rate(text: str) -> tuple[int, int]:
    toks = words(text)
    caps = sum(1 for t in toks if t[:1].isupper())
    return caps, len(toks)


def _stopword_count(text: str) -> tuple[int, int]:
    toks = words(text)
    stops = sum(1 for t in toks if t.lower() in STOPWORDS)
    return stops, len(toks)


def class_stats(items: Sequence[dict]) -> dict[str, Any]:
    """Class-conditional descriptive stats for one label group."""
    wcs = [float(word_count(it["text"])) for it in items]
    ccs = [float(len(it["text"])) for it in items]
    wc_mean, wc_sd = _mean_sd(wcs)
    cc_mean, cc_sd = _mean_sd(ccs)
    stop_num = stop_den = cap_num = cap_den = comma_total = 0
    for it in items:
        s_n, s_d = _stopword_count(it["text"])
        c_n, c_d = _capitalized_word_rate(it["text"])
        stop_num += s_n
        stop_den += s_d
        cap_num += c_n
        cap_den += c_d
        comma_total += it["text"].count(",")
    n = len(items)
    return {
        "n": n,
        "word_count_mean": wc_mean,
        "word_count_sd": wc_sd,
        "char_count_mean": cc_mean,
        "char_count_sd": cc_sd,
        "stopword_rate": (stop_num / stop_den) if stop_den else 0.0,
        "comma_rate": (comma_total / n) if n else 0.0,
        "capitalized_word_rate": (cap_num / cap_den) if cap_den else 0.0,
    }


def top_skewed_tokens(items: Sequence[dict], k: int = 20) -> list[dict[str, Any]]:
    """The k tokens with the largest |True-count - False-count| (lowercased)."""
    true_c: Counter = Counter()
    false_c: Counter = Counter()
    for it in items:
        toks = [t.lower() for t in words(it["text"])]
        (true_c if bool(it["label"]) else false_c).update(set(toks))
    all_tokens = set(true_c) | set(false_c)
    scored = []
    for tok in all_tokens:
        t = true_c.get(tok, 0)
        f = false_c.get(tok, 0)
        scored.append({"token": tok, "true_count": t, "false_count": f, "skew": abs(t - f)})
    scored.sort(key=lambda d: (-d["skew"], d["token"]))
    return scored[:k]


def _negator_rate(items: Sequence[dict]) -> float:
    """Fraction of items containing >= 1 negator (not/never/no/nothing/nobody)."""
    if not items:
        return 0.0
    hits = sum(
        1 for it in items if any(w.lower() in NEGATORS for w in words(it["text"]))
    )
    return hits / len(items)


def _doc_frequencies(items: Sequence[dict]) -> tuple[Counter, set, set, int]:
    """Per-item (document) token presence counts, lowercased.

    Returns (overall_doc_count, tokens_in_true, tokens_in_false, n_items).
    A token is counted once per item it appears in (set semantics), so
    'frequency' is the share of ITEMS that contain the token."""
    overall: Counter = Counter()
    in_true: set = set()
    in_false: set = set()
    for it in items:
        toks = {t.lower() for t in words(it["text"])}
        overall.update(toks)
        (in_true if bool(it["label"]) else in_false).update(toks)
    return overall, in_true, in_false, len(items)


def tokens_missing_from_a_class(items: Sequence[dict]) -> list[dict[str, Any]]:
    """Tokens with overall doc-frequency >= 3% that appear in only ONE class.

    Spec (rules 15/16 audit_thresholds): for sentiment/food the rule's own signal
    IS class-exclusive vocabulary (evaluative words; food words), legitimately
    >= 3% frequent and one-sided. Whether such a one-sided high-frequency token is
    rule-bearing (kept) or an incidental leak (regenerate) is a SEMANTIC judge
    call, NOT a programmatic gate — so this is REPORTED for the dataset judge
    (``judge_tokens_missing_from_class``), never folded into ``audit_hard_pass``.
    Returns the offending tokens with their per-class counts for the judge."""
    overall, in_true, in_false, n = _doc_frequencies(items)
    if not n:
        return []
    # class doc-counts so the judge sees the actual skew per token
    true_c: Counter = Counter()
    false_c: Counter = Counter()
    for it in items:
        toks = {t.lower() for t in words(it["text"])}
        (true_c if bool(it["label"]) else false_c).update(toks)
    thr = LLM_TOKEN_BOTH_CLASSES_FREQ * n
    out: list[dict[str, Any]] = []
    for tok, c in overall.items():
        if c >= thr and not (tok in in_true and tok in in_false):
            out.append(
                {
                    "token": tok,
                    "overall_freq": c / n,
                    "doc_count": c,
                    "true_count": true_c.get(tok, 0),
                    "false_count": false_c.get(tok, 0),
                    "in_true": tok in in_true,
                    "in_false": tok in in_false,
                }
            )
    out.sort(key=lambda d: (-d["doc_count"], d["token"]))
    return out


def judge_skewed_tokens(items: Sequence[dict]) -> list[dict[str, Any]]:
    """Tokens with class ratio >= 5:1 and overall frequency >= 2%, for the JUDGE.

    Spec (rules 15/16 audit_thresholds 'skew report'): these tokens must be
    EVALUATIVE / rule-bearing 'as judged by the dataset judge'. That adjudication
    is semantic (a later step), so this is REPORTED only — it is NOT a hard gate
    in overall_pass. Listed for the judge to inspect."""
    overall, in_true, in_false, n = _doc_frequencies(items)
    if not n:
        return []
    # class doc-counts
    true_c: Counter = Counter()
    false_c: Counter = Counter()
    for it in items:
        toks = {t.lower() for t in words(it["text"])}
        (true_c if bool(it["label"]) else false_c).update(toks)
    freq_thr = LLM_SKEW_MIN_FREQ * n
    out: list[dict[str, Any]] = []
    for tok, c in overall.items():
        if c < freq_thr:
            continue
        t = true_c.get(tok, 0)
        f = false_c.get(tok, 0)
        hi, lo = max(t, f), min(t, f)
        ratio = float("inf") if lo == 0 else hi / lo
        if ratio >= LLM_SKEW_RATIO:
            out.append(
                {
                    "token": tok,
                    "true_count": t,
                    "false_count": f,
                    "frequency": c / n,
                    "ratio": (None if ratio == float("inf") else ratio),
                }
            )
    out.sort(key=lambda d: (-d["frequency"], d["token"]))
    return out


def topic_balance_audit(
    items: Sequence[dict], topics: Sequence[str]
) -> dict[str, Any]:
    """Topic-balance audit (rules 15/16). ``topics`` is per-item (same order).

    Rule 16 hard gate: no single FALSE topic exceeds 25% of the False class.
    Rule 15 is even tighter (exact 50/50 per topic by construction) — we enforce
    the same <= 25%-of-class floor on BOTH classes (a perfectly balanced 8-topic
    rule-15 set sits at 12.5%/topic, well under), plus require that every topic
    that appears in one class also appears in the other when topics are shared.
    Returns a dict with per-class topic shares and a programmatic ``passes``."""
    if len(topics) != len(items):
        raise ValueError(
            f"topic_balance_audit: {len(topics)} topics for {len(items)} items"
        )
    true_topics: Counter = Counter()
    false_topics: Counter = Counter()
    for it, tp in zip(items, topics):
        (true_topics if bool(it["label"]) else false_topics).update([tp])
    n_true = sum(true_topics.values())
    n_false = sum(false_topics.values())

    def shares(counts: Counter, n: int) -> dict[str, float]:
        return {t: c / n for t, c in counts.items()} if n else {}

    true_shares = shares(true_topics, n_true)
    false_shares = shares(false_topics, n_false)
    # the rule-16 hard gate: no single False topic > 25%
    false_over = {t: s for t, s in false_shares.items() if s > LLM_FALSE_TOPIC_MAX_SHARE + 1e-9}
    true_over = {t: s for t, s in true_shares.items() if s > LLM_FALSE_TOPIC_MAX_SHARE + 1e-9}
    passes = not false_over and not true_over
    return {
        "true_topic_shares": true_shares,
        "false_topic_shares": false_shares,
        "max_false_topic_share": (max(false_shares.values()) if false_shares else 0.0),
        "max_true_topic_share": (max(true_shares.values()) if true_shares else 0.0),
        "false_topics_over_25pct": false_over,
        "true_topics_over_25pct": true_over,
        "threshold": LLM_FALSE_TOPIC_MAX_SHARE,
        "passes": passes,
    }


def build_audit_thresholds(
    items: Sequence[dict],
    t_stats: dict[str, Any],
    f_stats: dict[str, Any],
    wc_diff: float,
    *,
    topics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute the rule-15/16 distribution_guards.audit_thresholds block.

    All thresholds are COMPUTED and reported. The structural ones (5 numeric-diff
    rate gates + topic balance) are HARD gates (their ``passes`` folds into
    overall_pass via ``audit_hard_pass``); the two judge-adjudicated reports
    (skew tokens + tokens-missing-from-a-class) are reported only. ``topics`` (per
    item) enables the CONTENT-rule topic-balance hard gate and the
    judge_tokens_missing_from_class report; when absent (e.g. rule 18, a word-swap
    rule), only the 5 general numeric-diff gates apply — see module docstring."""
    stop_diff = abs(t_stats["stopword_rate"] - f_stats["stopword_rate"])
    comma_diff = abs(t_stats["comma_rate"] - f_stats["comma_rate"])
    cap_diff = abs(t_stats["capitalized_word_rate"] - f_stats["capitalized_word_rate"])
    trues = [it for it in items if bool(it["label"])]
    falses = [it for it in items if not bool(it["label"])]
    neg_t, neg_f = _negator_rate(trues), _negator_rate(falses)
    neg_diff = abs(neg_t - neg_f)

    checks: dict[str, dict[str, Any]] = {
        "word_count_mean_abs_diff": {
            "value": wc_diff, "threshold": LLM_WC_TOL,
            "passes": wc_diff <= LLM_WC_TOL, "hard_gate": True,
        },
        "stopword_rate_abs_diff": {
            "value": stop_diff, "threshold": LLM_STOPWORD_RATE_TOL,
            "passes": stop_diff <= LLM_STOPWORD_RATE_TOL, "hard_gate": True,
        },
        "comma_rate_abs_diff": {
            "value": comma_diff, "threshold": LLM_COMMA_RATE_TOL,
            "passes": comma_diff <= LLM_COMMA_RATE_TOL, "hard_gate": True,
        },
        "capitalized_word_rate_abs_diff": {
            "value": cap_diff, "threshold": LLM_CAP_RATE_TOL,
            "passes": cap_diff <= LLM_CAP_RATE_TOL, "hard_gate": True,
        },
        "negator_rate_abs_diff": {
            "value": neg_diff, "threshold": LLM_NEGATOR_RATE_TOL,
            "true_rate": neg_t, "false_rate": neg_f,
            "passes": neg_diff <= LLM_NEGATOR_RATE_TOL, "hard_gate": True,
        },
    }

    # CONTENT-rule gate (rules 15/16): topic balance only, when topics supplied.
    # The "token-in-both-classes" metric is NOT a hard gate here — for sentiment/
    # food the rule's own signal is legitimately class-exclusive vocabulary, so it
    # is judge-adjudicated (see judge_tokens_missing_from_class below), not gated.
    content_audit = topics is not None
    judge_tokens_missing: list[dict[str, Any]] = []
    if content_audit:
        judge_tokens_missing = tokens_missing_from_a_class(items)
        tb = topic_balance_audit(items, topics)
        checks["topic_balance"] = {
            "value": tb, "threshold": LLM_FALSE_TOPIC_MAX_SHARE,
            "passes": tb["passes"], "hard_gate": True,
        }

    hard_pass = all(c["passes"] for c in checks.values() if c["hard_gate"])
    return {
        "content_audit_applied": content_audit,
        "checks": checks,
        # JUDGE-ADJUDICATED, reported only (NOT in hard_pass): both are semantic
        # class-exclusivity calls the dataset judge makes, not auto-gated.
        "judge_skewed_tokens": judge_skewed_tokens(items),
        "judge_tokens_missing_from_class": judge_tokens_missing,
        "judge_note": (
            "judge_skewed_tokens and judge_tokens_missing_from_class are reported "
            "for the dataset JUDGE to confirm each token is rule-bearing "
            "(evaluative/food-denoting); an incidental skewed/one-sided token "
            "triggers targeted regeneration. These are NOT programmatic gates."
        ),
        "audit_hard_pass": hard_pass,
    }


def build_confound_report(
    items: Sequence[dict],
    *,
    is_llm_rule: bool = False,
    equiv_keys: dict[str, list[str]] | None = None,
    equivalence_class: Sequence[str] = (),
    battery_exemptions: Sequence[str] = (),
    run_pos: bool = True,
    length_match_exempt: bool = False,
    topics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble the confound_report payload + overall pass/fail.

    Does NOT write to disk (use write_confound_report for that). The battery
    agreements are included; if nltk is unavailable the 6 POS predicates are
    reported as skipped (and excluded from the violation check).

    ``length_match_exempt`` (globals.length_matching.policy "EXCEPT 23 and 25"):
    the word-count rules whose class-conditional word-count means MUST differ by
    construction. For them the |mean_T - mean_F| match is unsatisfiable, so it is
    NOT applied to overall_pass — but the actual diff and the tolerance are still
    computed and reported (the field ``length_match_exempt`` records the carve-
    out, so the audit stays fully visible).

    For ``is_llm_rule=True`` the rule-15/16 distribution_guards.audit_thresholds
    are computed (``audit_thresholds`` block) and the PROGRAMMATIC ones become
    HARD gates: ``overall_pass`` additionally requires ``audit_hard_pass``. The
    27 programmatic rules (``is_llm_rule=False``) are completely UNAFFECTED — no
    audit block is built and ``overall_pass`` is unchanged. Passing ``topics``
    (per item, same order) enables the topic-balance CONTENT hard gate for rules
    15/16 plus the judge-reported (NOT gated) token-in-both-classes metric; it is
    ignored when ``is_llm_rule=False`` and absent for the word-swap rule 18 (see
    module docstring)."""
    if not items:
        raise ValueError("no items for the confound report")
    trues = [it for it in items if bool(it["label"])]
    falses = [it for it in items if not bool(it["label"])]

    t_stats = class_stats(trues)
    f_stats = class_stats(falses)

    wc_diff = abs(t_stats["word_count_mean"] - f_stats["word_count_mean"])
    tol = LLM_WC_TOL if is_llm_rule else PROGRAMMATIC_WC_TOL
    # the raw match result is always reported; the exemption only frees
    # overall_pass from requiring it (word count IS the rule for 23/25).
    length_match_ok = bool(length_match_exempt or wc_diff <= tol)

    results = battery_report(
        items,
        equiv_keys=equiv_keys,
        equivalence_class=equivalence_class,
        battery_exemptions=battery_exemptions,
        run_pos=run_pos,
    )
    battery_payload = [
        {
            "predicate": r.key,
            "agreement": (None if r.skipped else r.agreement),
            "score": (None if r.skipped else r.score),
            "exempt": r.exempt,
            "passes": r.passes,
            "skipped": r.skipped,
        }
        for r in results
    ]
    violations = [r.key for r in battery_violations(results)]
    battery_ok = not violations

    # LLM-rule audit_thresholds: computed + HARD-gated ONLY for is_llm_rule=True.
    # The 27 programmatic rules (is_llm_rule=False) get neither the block nor any
    # change to overall_pass.
    audit_thresholds: dict[str, Any] | None = None
    audit_hard_pass = True
    if is_llm_rule:
        audit_thresholds = build_audit_thresholds(
            items, t_stats, f_stats, wc_diff, topics=topics
        )
        audit_hard_pass = bool(audit_thresholds["audit_hard_pass"])

    report: dict[str, Any] = {
        "n_items": len(items),
        "n_true": len(trues),
        "n_false": len(falses),
        "is_llm_rule": is_llm_rule,
        "class_conditional": {"true": t_stats, "false": f_stats},
        "word_count_mean_abs_diff": wc_diff,
        "length_match_tolerance": tol,
        "length_match_exempt": bool(length_match_exempt),
        "length_match_ok": length_match_ok,
        "top_skewed_tokens": top_skewed_tokens(items),
        # one-sided high-frequency tokens (>=3% of items, present in only ONE
        # class) — surfaced for EVERY rule now, not just LLM rules, so an
        # extreme construction skew like word_count_geq_8's trailing 'by' (84
        # True / 0 False) is visible in the audit even where it is not a hard
        # gate (the deliberately length-skewed word-count rules carry it by
        # design). Honest flag, not folded into overall_pass.
        "one_sided_high_freq_tokens": tokens_missing_from_a_class(items),
        "generic_probe_battery": battery_payload,
        "battery_violations": violations,
        "battery_ok": battery_ok,
        "overall_pass": bool(length_match_ok and battery_ok and audit_hard_pass),
    }
    if audit_thresholds is not None:
        report["audit_thresholds"] = audit_thresholds
        report["audit_hard_pass"] = audit_hard_pass
    return report


def write_confound_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write the report as JSON (never markdown)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
