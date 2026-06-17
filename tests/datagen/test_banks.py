"""Bank infrastructure tests.

Three groups:
- code-logic (GREEN): computed tags, entry-contract enforcement, quota-check
  mechanics, the matched-pair logic, and the reference bank NUMBER_WORDS passing
  end-to-end.
- group quotas (GREEN, all banks now authored): every spec'd bank is populated
  and passes its BANK_QUOTAS contract via check_bank. (This was an xfail(strict)
  RED marker during B1-B4 authoring; it was un-xfailed once the banks landed, as
  the marker's docstring promised.)
- optional tool verification (SKIP if wordfreq/nltk absent): frequency tiers are
  checked against the pinned wordfreq top-n list for the reference bank AND for
  every authored word bank, with the spec-mandated COLORS exemption recorded.
"""

from __future__ import annotations

import pytest

from icl_articulation.datagen import banks
from icl_articulation.datagen.banks import (
    BankError,
    BankQuotaError,
    alphabetic_length,
    build_entry,
    check_bank,
    has_adjacent_double,
    has_nonadjacent_repeat,
)

# bank -> stub file map (documented so authors know their target file)
GROUP_BANKS = {
    "g1_general": [
        "NOUN_CONCRETE", "VERB_REGULAR", "ADJ_PLAIN", "ADVERB_PLACE",
        "ADVERB_SENT_INITIAL", "ER_NONCOMPARATIVE", "ADJ_COMPARABLE",
        "FRAME_NEUTRAL", "FRAME_PROPER",
    ],
    "g2_zdouble": ["Z_WORDS", "Z_FREE_MATCHED", "DOUBLE_WORDS", "NONADJ_REPEAT_WORDS"],
    "g3_semantic": [
        "ANIMALS", "OBJECTS_PLANTS_VEHICLES", "COLORS", "ADJ_NONCOLOR_MATCHED",
        "FIRST_NAMES", "NONNAME_PROPER",
    ],
    "g4a_byletter": ["NOUN_PLURAL_BY_LETTER", "ADJ_BY_LETTER", "NOUN_FINAL_BY_LETTER"],
    "g4b_bylength": [
        "INITIAL_BY_LENGTH", "FINAL_BY_LENGTH", "VOWEL_INITIAL", "CONSONANT_INITIAL",
        "TERMINAL_VOWEL", "TERMINAL_CONSONANT",
    ],
    "g5_numhard": ["SHORT_WORDS_BY_POS", "LONG_ONLY_VOCAB"],
}
GROUP_BANK_NAMES = [b for names in GROUP_BANKS.values() for b in names]


# --- computed tags ------------------------------------------------------------


def test_alphabetic_length_ignores_nonalpha() -> None:
    assert alphabetic_length("don't") == 4
    assert alphabetic_length("well-known") == 9
    assert alphabetic_length("10") == 0
    assert alphabetic_length("three") == 5


def test_adjacent_double() -> None:
    assert has_adjacent_double("coffee")
    assert has_adjacent_double("wall")
    assert not has_adjacent_double("window")  # two w's, not adjacent
    assert not has_adjacent_double("cat")


def test_nonadjacent_repeat() -> None:
    assert has_nonadjacent_repeat("window")  # w...w
    assert has_nonadjacent_repeat("banana")  # a..a..a
    assert has_nonadjacent_repeat("level")   # l...l, e..e
    assert not has_nonadjacent_repeat("cat")
    # 'coffee': ff adjacent, but ee also adjacent and only one e-pair adjacent;
    # 'o' appears once, 'f' twice adjacent, 'e' twice adjacent -> no NONadjacent
    assert not has_nonadjacent_repeat("coffee")


# --- entry contract -----------------------------------------------------------


def test_build_entry_computes_tags() -> None:
    e = build_entry({"word": "coffee", "pos": "noun", "frequency_tier": 2})
    assert e.length == 6
    assert e.initial == "c"
    assert e.final == "e"
    assert e.has_adjacent_double is True
    assert e.has_nonadjacent_repeat is False


def test_build_entry_rejects_missing_required() -> None:
    with pytest.raises(BankError, match="missing required"):
        build_entry({"word": "x", "pos": "noun"})


def test_build_entry_rejects_bad_pos() -> None:
    with pytest.raises(BankError, match="pos"):
        build_entry({"word": "x", "pos": "gerund", "frequency_tier": 1})


def test_build_entry_rejects_bad_tier() -> None:
    with pytest.raises(BankError, match="frequency_tier"):
        build_entry({"word": "x", "pos": "noun", "frequency_tier": 3})


def test_build_entry_rejects_unknown_key() -> None:
    with pytest.raises(BankError, match="unknown keys"):
        build_entry({"word": "x", "pos": "noun", "frequency_tier": 1, "color": "red"})


def test_build_entry_rejects_disagreeing_computed_tag() -> None:
    with pytest.raises(BankError, match="disagrees"):
        build_entry({"word": "cat", "pos": "noun", "frequency_tier": 1, "length": 9})


# --- reference bank passes end-to-end -----------------------------------------


def test_reference_bank_number_words_passes() -> None:
    bank = check_bank("NUMBER_WORDS")
    assert bank is not None
    assert len(bank) == 20
    assert all(e.pos == "numeral" for e in bank.entries)
    assert all(e.frequency_tier in (1, 2) for e in bank.entries)


def test_all_quota_banks_are_populated() -> None:
    # B0 is complete: every bank that has a quota is now authored (no empties).
    assert set(banks.populated_banks()) == set(banks.bank_names())


def test_check_all_banks_passes() -> None:
    # all banks authored: check_all_banks must return 'ok' for every quota bank
    # and raise on the first violation otherwise.
    result = banks.check_all_banks(only_populated=True)
    assert set(result) == set(banks.bank_names())
    assert set(result.values()) == {"ok"}


# --- quota-check mechanics (synthetic banks) ----------------------------------


def test_quota_size_violation_raises(monkeypatch) -> None:
    monkeypatch.setitem(banks.BANKS, "NUMBER_WORDS", banks.BANKS["NUMBER_WORDS"][:5])
    with pytest.raises(BankQuotaError, match="< required size"):
        check_bank("NUMBER_WORDS")


def test_by_letter_min_logic(monkeypatch) -> None:
    # synthetic bank with quota requiring >=2 per letter b,c
    monkeypatch.setitem(
        banks.BANK_QUOTAS, "_synthetic", {"size": 3, "by_letter_min": (("b", "c"), 2)}
    )
    monkeypatch.setitem(
        banks.BANKS,
        "_synthetic",
        [
            {"word": "bats", "pos": "noun", "frequency_tier": 1},
            {"word": "bins", "pos": "noun", "frequency_tier": 1},
            {"word": "cats", "pos": "noun", "frequency_tier": 1},
        ],
    )
    with pytest.raises(BankQuotaError, match="initial 'c'"):
        check_bank("_synthetic")


def test_matched_pair_per_entry(monkeypatch) -> None:
    # a Z-style matched pair where the counterpart has a length mismatch
    monkeypatch.setitem(
        banks.BANK_QUOTAS,
        "_zsrc",
        {"size": 1, "matched_pair": ("_zdst", ["pos", "length_pm2", "no_z"])},
    )
    monkeypatch.setitem(banks.BANK_QUOTAS, "_zdst", {"size": 1})
    monkeypatch.setitem(
        banks.BANKS,
        "_zsrc",
        [{"word": "zoo", "pos": "noun", "frequency_tier": 1, "pair": "p1"}],
    )
    monkeypatch.setitem(
        banks.BANKS,
        "_zdst",
        [{"word": "encyclopedia", "pos": "noun", "frequency_tier": 1, "pair": "p1"}],
    )
    with pytest.raises(BankQuotaError, match="len"):
        check_bank("_zsrc")


def test_max_phrase_words_rejects_four_word_phrase(monkeypatch) -> None:
    # MAJOR: ADVERB_PLACE allows only 1/2/3-word phrases (spec line 326 / rule-9
    # line 799). A 4-word adjunct must raise via max_phrase_words=3.
    good = list(banks.BANKS["ADVERB_PLACE"])
    mutated = good + [{"word": "near the old fence", "pos": "adverb", "frequency_tier": 1}]
    monkeypatch.setitem(banks.BANKS, "ADVERB_PLACE", mutated)
    with pytest.raises(BankQuotaError, match="longer than 3 words"):
        check_bank("ADVERB_PLACE")


def test_verb_regular_form_derivation() -> None:
    # the four-form deriver underpinning the VERB_REGULAR custom check
    assert banks._regular_verb_forms("walk") == ("walk", "walks", "walked", "walking")
    assert banks._regular_verb_forms("carry") == ("carry", "carries", "carried", "carrying")
    assert banks._regular_verb_forms("wash") == ("wash", "washes", "washed", "washing")
    assert banks._regular_verb_forms("close") == ("close", "closes", "closed", "closing")
    assert banks._regular_verb_forms("enjoy") == ("enjoy", "enjoys", "enjoyed", "enjoying")
    assert banks._regular_verb_forms("stop") == ("stop", "stops", "stopped", "stopping")
    # multi-syllable -er/-it verbs take PLAIN -ed (no doubling)
    assert banks._regular_verb_forms("open") == ("open", "opens", "opened", "opening")
    assert banks._regular_verb_forms("visit") == ("visit", "visits", "visited", "visiting")


def test_verb_regular_rejects_ambiguous_past(monkeypatch) -> None:
    # MINOR: a zero-past verb (put/cut/set/hit/read/...) must raise
    mutated = list(banks.BANKS["VERB_REGULAR"]) + [
        {"word": "put", "pos": "verb", "frequency_tier": 1}
    ]
    monkeypatch.setitem(banks.BANKS, "VERB_REGULAR", mutated)
    with pytest.raises(BankQuotaError, match="ambiguous/zero-past"):
        check_bank("VERB_REGULAR")


def test_verb_regular_rejects_nondistinct_forms(monkeypatch) -> None:
    # MINOR: an entry whose four derived forms are not all distinct must raise.
    # Force the deriver to collapse 3sg into the base so set(forms) != 4 while
    # past != base (isolating the distinctness branch from the past==base one).
    monkeypatch.setattr(
        banks, "_regular_verb_forms", lambda b: (b, b, b + "ed", b + "ing")
    )
    with pytest.raises(BankQuotaError, match="4 distinct forms"):
        check_bank("VERB_REGULAR")


def test_adj_comparable_subtype_split_locked(monkeypatch) -> None:
    # MINOR: the 40 '-er' / 20 'more' subtype split is locked (subtype_min).
    # Flipping ten '-er' entries to 'more' (30/30) must raise on the 'er' floor.
    good = banks.BANKS["ADJ_COMPARABLE"]
    mutated = []
    flipped = 0
    for e in good:
        if e.get("subtype") == "er" and flipped < 11:
            e2 = dict(e)
            e2["subtype"] = "more"
            mutated.append(e2)
            flipped += 1
        else:
            mutated.append(dict(e))
    monkeypatch.setitem(banks.BANKS, "ADJ_COMPARABLE", mutated)
    with pytest.raises(BankQuotaError, match="subtype 'er'"):
        check_bank("ADJ_COMPARABLE")


def test_matched_pair_symmetric_from_dependent(monkeypatch) -> None:
    # MINOR: check_bank on the DEPENDENT side of a matched pair must re-run the
    # owner's cross-validation. Flip a Z_FREE_MATCHED counterpart's POS; checking
    # the DEPENDENT bank alone (not check_all_banks) must catch it.
    zfree = [dict(e) for e in banks.BANKS["Z_FREE_MATCHED"]]
    # find an entry that pairs with a Z_WORDS noun and break its POS
    zwords_by_pair = {e["pair"]: e for e in banks.BANKS["Z_WORDS"]}
    broke = False
    for e in zfree:
        mate = zwords_by_pair.get(e.get("pair"))
        if mate is not None and e["pos"] == mate["pos"]:
            e["pos"] = "adverb" if mate["pos"] != "adverb" else "noun"
            broke = True
            break
    assert broke, "could not find a pair to mutate"
    monkeypatch.setitem(banks.BANKS, "Z_FREE_MATCHED", zfree)
    with pytest.raises(BankQuotaError, match="POS"):
        check_bank("Z_FREE_MATCHED")


def test_frame_bank_requires_single_slot(monkeypatch) -> None:
    monkeypatch.setitem(banks.BANKS, "FRAME_NEUTRAL", ["The {X} ran", "no slot here"])
    with pytest.raises(BankQuotaError, match="exactly one"):
        # size 30 not met either, but the slot check fires after size; lower size
        monkeypatch.setitem(banks.BANK_QUOTAS["FRAME_NEUTRAL"], "size", 2)
        check_bank("FRAME_NEUTRAL")


# --- group banks: GREEN now that every group is authored ----------------------


@pytest.mark.parametrize("name", GROUP_BANK_NAMES)
def test_group_bank_quota_met(name) -> None:
    # Every group bank is authored at B0; its BANK_QUOTAS contract must pass.
    # (Was xfail(strict) RED during B1-B4 authoring; un-xfailed on landing, per
    # that marker's docstring.) check_bank raises BankQuotaError on any
    # violation; it returns the built Bank for word banks and None for frames.
    if name in banks.FRAME_BANKS:
        assert check_bank(name) is None
    else:
        assert check_bank(name) is not None


# --- optional tool verification (frequency tiers) -----------------------------


def test_number_words_tiers_match_wordfreq() -> None:
    wordfreq = pytest.importorskip("wordfreq")
    top2000 = set(wordfreq.top_n_list("en", 2000))
    top10000 = set(wordfreq.top_n_list("en", 10000))
    bank = check_bank("NUMBER_WORDS")
    for e in bank.entries:
        if e.frequency_tier == 1:
            assert e.word in top2000, f"{e.word} tagged tier 1 but not in top 2000"
        else:
            assert e.word in top10000, f"{e.word} tagged tier 2 but not in top 10000"
            assert e.word not in top2000, f"{e.word} tagged tier 2 but is in top 2000"


# Spec-mandated tier exemption: COLORS enumerates beige/maroon/crimson/turquoise
# verbatim (rule-specs banks.COLORS note). wordfreq ranks them past the
# top-10000 boundary, but the spec binds the exact list and COLORS' quota checks
# only size + POS (not tier), so they are tagged tier 2 (nearest allowed value).
# See g3_semantic.py TENSIONS. Any tier mismatch OUTSIDE this set is a tag bug.
_TIER_EXEMPT = {("COLORS", "beige"), ("COLORS", "maroon"),
                ("COLORS", "crimson"), ("COLORS", "turquoise")}


def test_all_bank_tiers_match_wordfreq() -> None:
    # Verify every authored word-bank entry's frequency_tier against the pinned
    # wordfreq top-n list (the codebase convention: surface-form membership,
    # tier 1 = top 2000, tier 2 = top 10000). Proper nouns and multi-word phrase
    # entries (ADVERB_PLACE) are not reliably in the top-n list and are skipped.
    wordfreq = pytest.importorskip("wordfreq")
    top2000 = set(wordfreq.top_n_list("en", 2000))
    top10000 = set(wordfreq.top_n_list("en", 10000))
    bad: list[str] = []
    for name in banks.bank_names():
        if name in banks.FRAME_BANKS or not banks.BANKS.get(name):
            continue
        for e in banks.get_bank(name).entries:
            w = e.word.lower()
            if e.proper or " " in w:
                continue
            if (name, w) in _TIER_EXEMPT:
                continue
            if e.frequency_tier == 1:
                if w not in top2000:
                    bad.append(f"{name}:{e.word} tier 1 but not in top 2000")
            else:
                if w not in top10000:
                    bad.append(f"{name}:{e.word} tier 2 but not in top 10000")
                elif w in top2000:
                    bad.append(f"{name}:{e.word} tier 2 but is in top 2000")
    assert not bad, "tier mismatches: " + "; ".join(bad)
