"""battery.py tests: the 40 frozen predicates, agreement math, the EXACTLY-75%
boundary (must PASS), exemption logic, and the POS predicates (importorskip)."""

from __future__ import annotations

import pytest

from icl_articulation.datagen import battery
from icl_articulation.datagen.battery import (
    PASS_THRESHOLD,
    PREDICATES,
    PREDICATES_BY_KEY,
    Predicate,
    battery_report,
    battery_violations,
    is_exempt,
    predicate_agreement,
    predicate_score,
)


def _items(pairs):
    """pairs: list of (text, label) -> item dicts."""
    return [{"item_id": f"i{i}", "text": t, "label": l} for i, (t, l) in enumerate(pairs)]


# --- the frozen 40 ------------------------------------------------------------


def test_exactly_40_predicates() -> None:
    assert len(PREDICATES) == 40
    assert len(PREDICATES_BY_KEY) == 40  # unique keys


def test_predicate_breakdown_counts() -> None:
    keys = [p.key for p in PREDICATES]
    assert sum(k.startswith("word_count>=") for k in keys) == 6
    assert sum(k.startswith("char_count>=") for k in keys) == 3
    assert sum(k in ("contains_the", "contains_a", "contains_and") for k in keys) == 3
    assert "count_the>=2" in keys
    assert sum(k.startswith("first_word_len>=") for k in keys) == 5
    assert sum(k.startswith("last_word_len<=") for k in keys) == 4
    assert sum(k.startswith("first_letter_bucket_") for k in keys) == 4
    assert sum(k.startswith("first_word_pos=") for k in keys) == 6
    assert sum(p.needs_pos for p in PREDICATES) == 6


# --- predicate behavior -------------------------------------------------------


def test_contains_digit_and_comma() -> None:
    cd = PREDICATES_BY_KEY["contains_digit"]
    assert cd.fn("They counted 10 boxes")
    assert not cd.fn("They counted ten boxes")
    cc = PREDICATES_BY_KEY["contains_comma"]
    assert cc.fn("Later, the dog ran")
    assert not cc.fn("The dog ran later")


def test_all_lowercase_predicate() -> None:
    p = PREDICATES_BY_KEY["all_lowercase"]
    assert p.fn("the dog ran home")
    assert not p.fn("The dog ran home")


def test_first_last_letter_predicates() -> None:
    assert PREDICATES_BY_KEY["first_starts_vowel"].fn("Apples fell down")
    assert not PREDICATES_BY_KEY["first_starts_vowel"].fn("Bananas fell down")
    assert PREDICATES_BY_KEY["last_ends_vowel"].fn("They saw the sofa")
    assert not PREDICATES_BY_KEY["last_ends_vowel"].fn("They saw the cat")


def test_word_and_char_length_predicates() -> None:
    assert PREDICATES_BY_KEY["word_count>=5"].fn("a b c d e")
    assert not PREDICATES_BY_KEY["word_count>=5"].fn("a b c d")
    assert PREDICATES_BY_KEY["first_word_len>=4"].fn("Apples here")
    assert PREDICATES_BY_KEY["last_word_len<=3"].fn("Apples cat")


# --- agreement math + the EXACTLY-75% boundary --------------------------------


def test_agreement_and_score() -> None:
    p = PREDICATES_BY_KEY["contains_digit"]
    items = _items([("has 1", True), ("has 2", True), ("none here", False), ("none too", True)])
    # pred True on items 0,1,2? "has 1"->T, "has 2"->T, "none here"->F, "none too"->F
    # labels: T,T,F,T ; agreement on 0(T==T),1(T==T),2(F==F),3(F!=T) = 3/4
    assert predicate_agreement(p, items) == 0.75
    assert predicate_score(p, items) == 0.75


def test_boundary_exactly_75_passes() -> None:
    """A predicate landing at EXACTLY 75% agreement must PASS (inclusive)."""
    # 8 items: predicate agrees on 6, disagrees on 2 -> 75%
    p = PREDICATES_BY_KEY["contains_digit"]
    items = _items(
        [
            ("a 1", True), ("a 2", True), ("a 3", True),  # T, pred True -> agree x3
            ("none p", False), ("none q", False), ("none r", False),  # F, pred False -> agree x3
            ("none s", True),  # T, pred False -> disagree
            ("none t", True),  # T, pred False -> disagree
        ]
    )
    assert predicate_agreement(p, items) == 0.75
    results = battery_report(items, run_pos=False)
    digit_res = next(r for r in results if r.key == "contains_digit")
    assert digit_res.score == 0.75
    assert digit_res.score <= PASS_THRESHOLD
    assert digit_res.passes is True  # exactly 75% PASSES
    # and it is NOT counted as a violation
    assert "contains_digit" not in [v.key for v in battery_violations(results)]


def test_boundary_just_over_75_fails() -> None:
    # 8 items, agree on 7 -> 87.5% > 75% -> fail (no exemption)
    p = PREDICATES_BY_KEY["contains_digit"]
    items = _items(
        [
            ("a 1", True), ("a 2", True), ("a 3", True), ("a 4", True),
            ("none p", False), ("none q", False), ("none r", False),
            ("none s", True),  # disagree
        ]
    )
    assert predicate_agreement(p, items) == 0.875
    results = battery_report(items, run_pos=False)
    assert "contains_digit" in [v.key for v in battery_violations(results)]


# --- exemption logic ----------------------------------------------------------


def test_exempt_via_equivalence_class() -> None:
    p = PREDICATES_BY_KEY["contains_digit"]
    assert is_exempt(
        p,
        equiv_keys={"contains_digit": ["contains a number written in digits"]},
        equivalence_class=["contains a number written in digits"],
        battery_exemptions=[],
    )
    # not exempt if the equiv string isn't actually in this rule's class
    assert not is_exempt(
        p,
        equiv_keys={"contains_digit": ["contains a number written in digits"]},
        equivalence_class=["something else"],
        battery_exemptions=[],
    )


def test_exempt_via_battery_exemptions() -> None:
    p = PREDICATES_BY_KEY["first_word_pos=verb"]
    assert is_exempt(
        p, equiv_keys={}, equivalence_class=[], battery_exemptions=["first_word_pos=verb"]
    )


def test_exempt_predicate_over_75_still_passes() -> None:
    # a digit-rule-like dataset where contains_digit is ~100% but exempt
    p = PREDICATES_BY_KEY["contains_digit"]
    items = _items([("a 1", True), ("a 2", True), ("none p", False), ("none q", False)])
    results = battery_report(
        items,
        equiv_keys={"contains_digit": ["contains a number written in digits"]},
        equivalence_class=["contains a number written in digits"],
        run_pos=False,
    )
    digit_res = next(r for r in results if r.key == "contains_digit")
    assert digit_res.agreement == 1.0
    assert digit_res.exempt is True
    assert digit_res.passes is True
    assert "contains_digit" not in [v.key for v in battery_violations(results)]


# --- POS predicates (optional nltk) -------------------------------------------


def test_pos_predicates_skip_cleanly_without_pos() -> None:
    items = _items([("Close the window", True), ("She closes it", False)])
    results = battery_report(items, run_pos=False)
    pos_results = [r for r in results if r.key.startswith("first_word_pos=")]
    assert len(pos_results) == 6
    assert all(r.skipped for r in pos_results)
    # skipped predicates never count as violations
    assert not any(v.key.startswith("first_word_pos=") for v in battery_violations(results))


def test_battery_report_skips_pos_when_tagger_data_missing(monkeypatch) -> None:
    """nltk installed but tagger DATA missing (fresh-instance state): the eager
    probe in battery_report must catch LookupError and mark the 6 POS predicates
    skipped instead of crashing the battery (docstring: never raises for a
    missing optional dep)."""
    nltk = pytest.importorskip("nltk")

    def _raise(*_a, **_k):
        raise LookupError("Resource averaged_perceptron_tagger not found.")

    monkeypatch.setattr(nltk, "pos_tag", _raise)
    items = _items([("Close the window", True), ("She closes it", False)])
    results = battery_report(items, run_pos=True)  # must NOT raise
    pos_results = [r for r in results if r.key.startswith("first_word_pos=")]
    assert len(pos_results) == 6
    assert all(r.skipped for r in pos_results)
    assert not any(v.key.startswith("first_word_pos=") for v in battery_violations(results))


def test_pos_tagger_when_available() -> None:
    pytest.importorskip("nltk")
    try:
        from nltk import pos_tag

        pos_tag(["close", "the", "window"])  # triggers data load; skip if absent
    except LookupError:
        pytest.skip("nltk tagger data not downloaded (run scripts/setup_nltk.py)")
    # the tagger maps PTB tags into the coarse set, and the determiner /
    # pronoun / noun openers it handles reliably are what the battery uses.
    # (Imperatives mis-tag in isolation — a known tagger limitation; the
    # battery's POS predicates are still used because rule data is scored
    # in-context, and exemptions cover the rule-inherent cases.)
    assert battery.coarse_pos_first_word("The window is open") == "determiner"
    assert battery.coarse_pos_first_word("They walked home after dinner") == "pronoun"
    assert battery.coarse_pos_first_word("Apples fell from the tree") == "noun"
    # every output is in the coarse POS set
    assert battery.coarse_pos_first_word("Close the window today") in {
        "noun", "verb", "adjective", "adverb", "pronoun", "determiner", "other"
    }


def test_aux_copular_first_word_pos_is_verb() -> None:
    """Rule 11's True-class signal: every spec-named aux/copular opener
    {is,are,was,were,can,will,should,must,could} coarse-maps to 'verb'.

    KNOWN TAGGER MIS-TAGS handled by the closed-set override in
    coarse_pos_first_word (NOT the PTB->coarse map): the pinned
    averaged_perceptron_tagger tags several of these wrongly in isolation —
    'Were'->WRB (would fall to 'adverb'), 'Must'->NN (->'noun'), and
    'Can'/'Will'/'Should'/'Could' to non-MD tags (->'other'). Only Is/Are/Was
    reach 'verb' via the tagger (VBZ/VBP/VBD) and the new MD->verb entry covers
    genuine modal tags; the override pins the rest. Content words are untouched
    because the override set is exactly the closed aux/copular family."""
    pytest.importorskip("nltk")
    try:
        from nltk import pos_tag

        pos_tag(["ok"])  # triggers data load; skip if absent
    except LookupError:
        pytest.skip("nltk tagger data not downloaded (run scripts/setup_nltk.py)")

    pos = PREDICATES_BY_KEY["first_word_pos=verb"]
    for opener in ("Can", "Will", "Should", "Must", "Could", "Were", "Is", "Are", "Was"):
        text = f"{opener} the dog run home"
        assert battery.coarse_pos_first_word(text) == "verb", opener
        assert pos.fn(text) is True, opener  # the battery predicate fires
    # the override is tight: a content word with the same shape is NOT forced
    assert battery.coarse_pos_first_word("Apples fell from the tree") == "noun"
