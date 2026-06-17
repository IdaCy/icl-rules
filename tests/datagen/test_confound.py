"""confound.py tests: class-conditional stats math, skew tokens, length-match
asserts, and the overall pass/fail."""

from __future__ import annotations

import json

from icl_articulation.datagen.confound import (
    LLM_WC_TOL,
    PROGRAMMATIC_WC_TOL,
    build_confound_report,
    class_stats,
    top_skewed_tokens,
    write_confound_report,
)


def _items(pairs):
    return [{"item_id": f"i{i}", "text": t, "label": l} for i, (t, l) in enumerate(pairs)]


# --- shared builders for the is_llm_rule audit_thresholds tests ----------------
#
# A clean, AUDIT-PASSING llm corpus: a SHARED neutral scaffold (every token lands
# in both classes) with one rotating class marker drawn from a large pool (so no
# single marker reaches the 3% token-in-both-classes floor, keeping the judge
# token-missing report empty here too). Mirrors how the real pipeline's mock
# corpus passes the structural gates. Word count is identical (scaffold + one
# marker + tail), so the 5 numeric-diff gates and topic balance pass too.
_POS_MARKERS = [
    "wonderful", "delightful", "brilliant", "lovely", "pleasant", "superb",
    "excellent", "charming", "splendid", "gorgeous", "radiant", "cheerful",
    "elegant", "soothing", "uplifting", "glorious", "vibrant", "refreshing",
    "amazing", "satisfying", "joyful", "serene", "stellar", "magical",
]
_NEG_MARKERS = [
    "bland", "tiring", "dull", "clumsy", "miserable", "rude", "stressful",
    "frustrating", "careless", "dreary", "tedious", "awful", "dismal", "gloomy",
    "irritating", "shabby", "tiresome", "dreadful", "lousy", "grim", "annoying",
    "horrible", "draining", "bleak",
]


def _clean_llm_corpus(n_per_class=60, topics=("food", "movies", "work", "music")):
    """A clean is_llm_rule corpus that passes EVERY audit_threshold gate.

    Returns (items, item_topics). Topics are spread evenly across both classes
    (each <= 25%). The only class-skewed token is the rotating marker, kept under
    the 3% floor by the large marker pool."""
    items = []
    topic_list = []
    for cls, markers in ((True, _POS_MARKERS), (False, _NEG_MARKERS)):
        for j in range(n_per_class):
            m = markers[j % len(markers)]
            tp = topics[j % len(topics)]
            text = f"the whole visit there felt {m} to everyone today number {j}"
            items.append({"item_id": f"{int(cls)}-{j}", "text": text, "label": cls})
            topic_list.append(tp)
    return items, topic_list


# --- is_llm_rule audit_thresholds: clean corpus PASSES -------------------------


def test_llm_audit_clean_corpus_passes() -> None:
    items, topics = _clean_llm_corpus()
    rep = build_confound_report(items, is_llm_rule=True, run_pos=False, topics=topics)
    at = rep["audit_thresholds"]
    assert at["content_audit_applied"] is True
    # every hard gate passes
    for name, c in at["checks"].items():
        if c["hard_gate"]:
            assert c["passes"], (name, c)
    # the token-in-both-classes check is NOT a hard gate any more
    assert "tokens_missing_from_a_class" not in at["checks"]
    # clean corpus -> judge token-missing report empty (markers under 3% floor)
    assert at["judge_tokens_missing_from_class"] == []
    assert rep["audit_hard_pass"] is True
    assert rep["overall_pass"] is True


# --- each HARD gate fires on a deliberate violation ----------------------------


def test_llm_audit_token_in_one_class_only_is_judge_reported_not_gated() -> None:
    """A high-frequency token present in only ONE class is REPORTED for the
    dataset judge (judge_tokens_missing_from_class) but is NOT a programmatic
    hard gate: for sentiment/food the rule's own signal IS class-exclusive
    vocabulary, so audit_hard_pass / overall_pass stay True and nothing raises."""
    items, topics = _clean_llm_corpus()
    # inject the SAME word into 20% of True items only (well over the 3% floor):
    n_true = sum(1 for it in items if it["label"])
    injected = 0
    for it in items:
        if it["label"] and injected < n_true // 5:
            it["text"] = it["text"] + " spaghetti"
            injected += 1
    rep = build_confound_report(items, is_llm_rule=True, run_pos=False, topics=topics)
    at = rep["audit_thresholds"]
    # NOT a hard gate any more — must not be in checks at all
    assert "tokens_missing_from_a_class" not in at["checks"]
    # reported for the judge, with per-class counts the judge can inspect
    missing = at["judge_tokens_missing_from_class"]
    bad = {d["token"]: d for d in missing}
    assert "spaghetti" in bad
    assert bad["spaghetti"]["true_count"] > 0
    assert bad["spaghetti"]["false_count"] == 0
    assert bad["spaghetti"]["overall_freq"] >= 0.03
    # a documented judge_note routes this to the dataset judge, not a gate
    assert "judge" in at["judge_note"].lower()
    # the class-exclusive token does NOT fail the audit
    assert rep["audit_hard_pass"] is True
    assert rep["overall_pass"] is True


def test_llm_audit_false_topic_over_25pct_fails() -> None:
    """A single False topic exceeding 25% of the False class trips the
    topic-balance hard gate (rule 16)."""
    items, topics = _clean_llm_corpus()
    # re-label every False item's topic to one dominant topic -> 100% > 25%
    for k, it in enumerate(items):
        if not it["label"]:
            topics[k] = "sports"
    rep = build_confound_report(items, is_llm_rule=True, run_pos=False, topics=topics)
    tb = rep["audit_thresholds"]["checks"]["topic_balance"]
    assert tb["passes"] is False
    assert "sports" in tb["value"]["false_topics_over_25pct"]
    assert rep["overall_pass"] is False


def test_llm_audit_negator_rate_gap_fails() -> None:
    """A negator-rate gap > 0.05 between classes trips the negator hard gate
    (rule 15 audit_threshold)."""
    items, topics = _clean_llm_corpus()
    # add 'never' to HALF the True items only -> negator-rate gap ~0.5 > 0.05
    n_true = sum(1 for it in items if it["label"])
    added = 0
    for it in items:
        if it["label"] and added < n_true // 2:
            it["text"] = it["text"] + " never"
            added += 1
    rep = build_confound_report(items, is_llm_rule=True, run_pos=False, topics=topics)
    neg = rep["audit_thresholds"]["checks"]["negator_rate_abs_diff"]
    assert neg["passes"] is False
    assert neg["value"] > 0.05
    assert rep["overall_pass"] is False


def test_llm_audit_stopword_rate_gap_fails() -> None:
    """A stopword-rate gap > 0.03 trips the stopword hard gate."""
    items, topics = _clean_llm_corpus()
    # append a run of stopwords to every True item only -> stopword rate diverges
    for it in items:
        if it["label"]:
            it["text"] = it["text"] + " of the and to in for"
    rep = build_confound_report(items, is_llm_rule=True, run_pos=False, topics=topics)
    sw = rep["audit_thresholds"]["checks"]["stopword_rate_abs_diff"]
    assert sw["passes"] is False
    assert sw["value"] > 0.03
    assert rep["overall_pass"] is False


def test_llm_audit_word_count_gap_fails() -> None:
    """A mean-word-count gap > 1.0 trips the word-count hard gate."""
    items, topics = _clean_llm_corpus()
    for it in items:
        if it["label"]:
            it["text"] = it["text"] + " alpha beta gamma delta epsilon"
    rep = build_confound_report(items, is_llm_rule=True, run_pos=False, topics=topics)
    wc = rep["audit_thresholds"]["checks"]["word_count_mean_abs_diff"]
    assert wc["passes"] is False
    assert wc["value"] > 1.0
    assert rep["overall_pass"] is False


# --- the content gates are SKIPPED when topics are absent (rule-18 word-swap) --


def test_llm_audit_without_topics_skips_content_gates() -> None:
    """Rule 18 (a word-swap llm rule) passes topics=None: the 5 numeric-diff
    gates still apply, but the two CONTENT gates (token-in-both-classes,
    topic-balance) are NOT applied — a class-skewed slot word is the rule there,
    not a confound."""
    # a single high-frequency token ('barn') confined to the True class would
    # FAIL the content gate IF it were applied; without topics it must NOT be.
    items = []
    for j in range(40):
        items.append({"item_id": f"t{j}", "text": f"the barn floated above the field number {j}", "label": True})
        items.append({"item_id": f"f{j}", "text": f"the kite floated above the field number {j}", "label": False})
    rep = build_confound_report(items, is_llm_rule=True, run_pos=False)  # topics=None
    at = rep["audit_thresholds"]
    assert at["content_audit_applied"] is False
    assert "tokens_missing_from_a_class" not in at["checks"]
    assert "topic_balance" not in at["checks"]
    # only the 5 numeric-diff gates, and they pass (matched by construction)
    assert set(at["checks"]) == {
        "word_count_mean_abs_diff", "stopword_rate_abs_diff", "comma_rate_abs_diff",
        "capitalized_word_rate_abs_diff", "negator_rate_abs_diff",
    }
    # the judge token-missing report is empty without topics (content audit off)
    assert at["judge_tokens_missing_from_class"] == []
    assert rep["audit_hard_pass"] is True
    assert rep["overall_pass"] is True


# --- the skew report is JUDGE-adjudicated, NOT a hard gate ----------------------


def test_judge_skewed_tokens_reported_not_gated() -> None:
    """A class-skewed token at >= 5:1 ratio and >= 2% frequency is REPORTED for
    the judge but does NOT by itself fail overall_pass (it is below the 3%
    token-in-both floor, so the hard gate does not fire)."""
    from icl_articulation.datagen.confound import judge_skewed_tokens

    items, topics = _clean_llm_corpus(n_per_class=60)
    # 'cosy' in exactly 2.5% of items (3 of 120), True-only: above the 2% judge
    # floor and 5:1 ratio, but below the 3% (= 3.6 of 120) hard-gate floor ->
    # reported for the judge, not programmatically gated.
    n = len(items)
    assert n == 120
    target = 3  # 3/120 = 2.5%  (2% <= 2.5% < 3%)
    added = 0
    for it in items:
        if it["label"] and added < target:
            it["text"] = it["text"] + " cosy"
            added += 1
    skew = judge_skewed_tokens(items)
    assert any(d["token"] == "cosy" for d in skew)
    rep = build_confound_report(items, is_llm_rule=True, run_pos=False, topics=topics)
    at = rep["audit_thresholds"]
    # 'cosy' is in the judge skew list...
    assert any(d["token"] == "cosy" for d in at["judge_skewed_tokens"])
    # ...but it is below the 3% floor, so the judge token-missing report does not
    # list it; either way these are judge-reported, NOT hard gates -> still passes.
    assert all(d["token"] != "cosy" for d in at["judge_tokens_missing_from_class"])
    assert rep["overall_pass"] is True


# --- is_llm_rule=False (the 27 programmatic rules) is UNAFFECTED ----------------


def test_programmatic_rule_has_no_audit_block_and_unchanged_overall_pass() -> None:
    """build_confound_report for is_llm_rule=False must NOT add the audit block
    and must keep overall_pass == (length_match_ok and battery_ok)."""
    items = _items([("the cat sat on a mat", True), ("the dog ran by a log", False)])
    rep = build_confound_report(items, run_pos=False)  # is_llm_rule=False default
    assert rep["is_llm_rule"] is False
    assert "audit_thresholds" not in rep
    assert "audit_hard_pass" not in rep
    assert rep["overall_pass"] == bool(rep["length_match_ok"] and rep["battery_ok"])


def test_programmatic_rule_ignores_a_skewed_token() -> None:
    """A class-skewed high-frequency token does NOT fail a programmatic
    (is_llm_rule=False) report — the content gate is llm-only."""
    items = []
    for j in range(20):
        items.append({"item_id": f"t{j}", "text": f"the zebra walked home number {j}", "label": True})
        items.append({"item_id": f"f{j}", "text": f"the rabbit walked home number {j}", "label": False})
    rep = build_confound_report(items, run_pos=False)  # is_llm_rule=False
    assert "audit_thresholds" not in rep
    # 'zebra' (True-only, high freq) does not gate the programmatic report
    assert rep["overall_pass"] == bool(rep["length_match_ok"] and rep["battery_ok"])


def test_class_stats_math() -> None:
    # 'cat'/'sat'/'dog'/'ran' are content words; only 'the' is a stopword here
    items = _items([("the cat sat down", True), ("the dog ran fast", True)])
    s = class_stats(items)
    assert s["n"] == 2
    assert s["word_count_mean"] == 4.0
    assert s["word_count_sd"] == 0.0
    # stopword rate: "the" (1 of 4) in each sentence -> 2/8 = 0.25
    assert abs(s["stopword_rate"] - 0.25) < 1e-9
    assert s["comma_rate"] == 0.0
    # texts are lowercase, so capitalized-word rate 0
    assert s["capitalized_word_rate"] == 0.0


def test_comma_and_cap_rates() -> None:
    items = _items([("Later, the Dog ran", True)])
    s = class_stats(items)
    assert s["comma_rate"] == 1.0
    # tokens: Later the Dog ran -> 'Later' and 'Dog' capitalized = 2/4
    assert abs(s["capitalized_word_rate"] - 0.5) < 1e-9


def test_top_skewed_tokens() -> None:
    items = _items(
        [
            ("alpha here now today", True),
            ("alpha there now today", True),
            ("beta here now today", False),
            ("beta there now today", False),
        ]
    )
    skew = top_skewed_tokens(items, k=5)
    by_tok = {d["token"]: d for d in skew}
    # 'alpha' appears in 2 True / 0 False -> skew 2; 'beta' 0/2 -> skew 2
    assert by_tok["alpha"]["skew"] == 2
    assert by_tok["beta"]["skew"] == 2
    # 'now'/'today' appear in all -> skew 0 (low)
    assert by_tok["now"]["skew"] == 0


def test_length_match_programmatic_pass_and_fail() -> None:
    # matched lengths -> pass
    matched = _items([("a b c d e", True), ("f g h i j", False)])
    rep = build_confound_report(matched, run_pos=False)
    assert rep["length_match_ok"] is True
    assert rep["length_match_tolerance"] == PROGRAMMATIC_WC_TOL

    # True items much longer than False -> programmatic 0.2 tol fails
    skewed = _items(
        [("a b c d e f g h", True), ("a b c d e f g i", True), ("x y z", False), ("p q r", False)]
    )
    rep2 = build_confound_report(skewed, run_pos=False)
    assert rep2["length_match_ok"] is False
    assert rep2["overall_pass"] is False


def test_length_match_exempt_frees_overall_pass() -> None:
    # True items much longer than False (word-count diff > 0.2): without the
    # exemption length-matching fails; WITH it the word-count rules
    # (length_match_exempt) are carved out of that assert. The match result is
    # the ONLY thing the exemption changes -- overall_pass then equals the
    # battery result -- and the real diff stays reported in the audit.
    skewed = _items(
        [("a b c d e f g h", True), ("a b c d e f g i", True), ("x y z", False), ("p q r", False)]
    )
    base = build_confound_report(skewed, run_pos=False)
    assert base["length_match_exempt"] is False
    assert base["length_match_ok"] is False          # diff > 0.2 -> fails
    assert base["overall_pass"] is False

    ex = build_confound_report(skewed, run_pos=False, length_match_exempt=True)
    assert ex["length_match_exempt"] is True
    assert ex["length_match_ok"] is True             # exemption frees the match
    # the exemption ONLY frees length-matching: overall_pass now == battery_ok
    assert ex["overall_pass"] == ex["battery_ok"]
    assert base["battery_ok"] == ex["battery_ok"]    # battery itself unchanged
    # the real diff is still computed and reported (audit stays visible)
    assert ex["word_count_mean_abs_diff"] == base["word_count_mean_abs_diff"]
    assert ex["word_count_mean_abs_diff"] > 0.2


def test_length_match_llm_wider_tolerance() -> None:
    items = _items([("a b c d e f", True), ("a b c d", False)])  # diff = 2 words
    prog = build_confound_report(items, run_pos=False)
    assert prog["length_match_ok"] is False  # 2 > 0.2
    llm = build_confound_report(items, is_llm_rule=True, run_pos=False)
    assert llm["length_match_tolerance"] == LLM_WC_TOL
    # diff 2 still > 1.0 -> fails; verify a 1-word diff passes under llm
    items2 = _items([("a b c d e", True), ("a b c d", False)])
    llm2 = build_confound_report(items2, is_llm_rule=True, run_pos=False)
    assert llm2["length_match_ok"] is True


def test_build_report_has_required_outputs() -> None:
    items = _items([("the cat sat on a mat", True), ("the dog ran by a log", False)])
    rep = build_confound_report(items, run_pos=False)
    cc = rep["class_conditional"]
    for cls in ("true", "false"):
        for field in (
            "word_count_mean", "word_count_sd", "char_count_mean", "char_count_sd",
            "stopword_rate", "comma_rate", "capitalized_word_rate",
        ):
            assert field in cc[cls]
    assert "top_skewed_tokens" in rep
    assert "generic_probe_battery" in rep
    assert len(rep["generic_probe_battery"]) == 40


def test_one_sided_high_freq_tokens_reported_for_programmatic_rules() -> None:
    # A programmatic rule (is_llm_rule=False) where a token sits in ONLY the True
    # class at high frequency (the word_count_geq_8 "by 84/0" shape). It must be
    # surfaced in one_sided_high_freq_tokens for EVERY rule now, even though it is
    # not folded into overall_pass.
    items = _items(
        [(f"the cat ran fast today by the wall {i}", True) for i in range(20)]
        + [(f"the cat ran fast today {i}", False) for i in range(20)]
    )
    rep = build_confound_report(items, is_llm_rule=False, run_pos=False)
    one_sided = {t["token"]: t for t in rep["one_sided_high_freq_tokens"]}
    assert "by" in one_sided  # 20 True / 0 False, doc-freq 50% >= 3%
    assert one_sided["by"]["true_count"] == 20 and one_sided["by"]["false_count"] == 0
    # "the" is in both classes -> not flagged
    assert "the" not in one_sided


def test_write_confound_report_json(tmp_path) -> None:
    items = _items([("a b c d e", True), ("f g h i j", False)])
    rep = build_confound_report(items, run_pos=False)
    path = write_confound_report(rep, tmp_path / "confound_report.json")
    assert path.is_file()
    loaded = json.loads(path.read_text())
    assert loaded["n_items"] == 2
    assert loaded["overall_pass"] in (True, False)
