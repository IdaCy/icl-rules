"""genutils.py tests: seeded RNG, frame fill, casing, and the EXACT word-count
equalizer (the load-bearing piece for length matching)."""

from __future__ import annotations

import pytest

from icl_articulation.datagen.genutils import (
    Gen,
    GenError,
    adjunct_word_lengths,
    base_id,
    equalize_word_count,
    fill_frame,
    fix_indefinite_articles,
    frame_slots,
    item_id,
    solve_adjuncts,
    to_lower,
    to_sentence_case,
    to_title_case,
)
from icl_articulation.datagen.schema import word_count


# --- seeded RNG ---------------------------------------------------------------


def test_gen_deterministic() -> None:
    a = Gen(42)
    b = Gen(42)
    seq = list(range(20))
    assert [a.choice(seq) for _ in range(10)] == [b.choice(seq) for _ in range(10)]


def test_gen_derive_independent_but_reproducible() -> None:
    parent = Gen(7)
    c1 = parent.derive("phaseA")
    c2 = Gen(7).derive("phaseA")
    assert c1.seed == c2.seed
    assert parent.derive("phaseA").seed != parent.derive("phaseB").seed


# --- frame fill ---------------------------------------------------------------


def test_frame_slots_and_fill() -> None:
    fr = "The {X} watched the {Y}"
    assert frame_slots(fr) == ["X", "Y"]
    assert fill_frame(fr, {"X": "dog", "Y": "cat"}) == "The dog watched the cat"


def test_fill_frame_missing_filler_raises() -> None:
    with pytest.raises(GenError, match="missing fillers"):
        fill_frame("The {X} ran", {})


def test_fill_frame_empty_filler_raises() -> None:
    with pytest.raises(GenError, match="empty filler"):
        fill_frame("The {X} ran", {"X": ""})


# --- casing -------------------------------------------------------------------


def test_sentence_case() -> None:
    assert to_sentence_case("the DOG ran HOME") == "The dog ran home"
    assert to_lower("The Dog Ran") == "the dog ran"


def test_title_case_capitalizes_every_word() -> None:
    # rule 4 definition: EVERY word, including stopwords
    assert to_title_case("the last train to madrid") == "The Last Train To Madrid"


# --- exact word-count equalizer ----------------------------------------------


def test_solve_adjuncts_exact() -> None:
    assert solve_adjuncts(0, [1, 2, 3]) == []
    assert sum(solve_adjuncts(5, [1, 2, 3])) == 5
    assert sum(solve_adjuncts(7, [2, 3])) == 7  # 2+2+3
    # minimal count preferred: 6 from {1,2,3} -> two 3s, not 1+2+3
    assert solve_adjuncts(6, [1, 2, 3]) == [3, 3]


def test_solve_adjuncts_unreachable_raises() -> None:
    with pytest.raises(GenError, match="cannot reach"):
        solve_adjuncts(1, [2, 3])  # no way to make 1 from {2,3}


def test_solve_adjuncts_negative_raises() -> None:
    with pytest.raises(GenError, match="negative"):
        solve_adjuncts(-1, [1])


def test_equalize_word_count_hits_exact_target() -> None:
    adjuncts = ["downtown", "in the kitchen", "near the old bridge"]  # 1, 3, 3 words
    by_len = adjunct_word_lengths(adjuncts)
    gen = Gen(0)
    text = "The dog ran"  # 3 words
    out = equalize_word_count(text, target=7, adjuncts_by_len=by_len, gen=gen)
    assert word_count(out) == 7
    assert out.startswith("The dog ran")


def test_equalize_word_count_many_targets_exact() -> None:
    adjuncts = ["downtown", "in the kitchen", "near the old bridge", "outside today"]
    by_len = adjunct_word_lengths(adjuncts)  # lengths {1,2,3}
    for target in range(4, 13):
        gen = Gen(target)
        out = equalize_word_count("The cat sat", target=target, adjuncts_by_len=by_len, gen=gen)
        assert word_count(out) == target, f"target {target} not hit exactly"


def test_equalize_word_count_over_target_raises() -> None:
    by_len = adjunct_word_lengths(["downtown"])
    with pytest.raises(GenError, match="over target"):
        equalize_word_count("a b c d e", target=3, adjuncts_by_len=by_len, gen=Gen(0))


# --- indefinite-article normalizer -------------------------------------------


def test_fix_articles_basic_vowel_words() -> None:
    assert fix_indefinite_articles("a apple") == "an apple"
    assert fix_indefinite_articles("a engine") == "an engine"
    assert fix_indefinite_articles("a office") == "an office"
    assert fix_indefinite_articles("a arm") == "an arm"


def test_fix_articles_sentence_initial_case_preserved() -> None:
    assert fix_indefinite_articles("A engine") == "An engine"
    # capital A before a consonant is left alone
    assert fix_indefinite_articles("A dog ran") == "A dog ran"


def test_fix_articles_consonant_sound_vowel_letter_keeps_a() -> None:
    assert fix_indefinite_articles("a university") == "a university"
    assert fix_indefinite_articles("a unit") == "a unit"
    assert fix_indefinite_articles("a one") == "a one"
    assert fix_indefinite_articles("a European trip") == "a European trip"
    assert fix_indefinite_articles("a useful tool") == "a useful tool"


def test_fix_articles_vowel_sound_consonant_letter_forces_an() -> None:
    assert fix_indefinite_articles("a hour") == "an hour"
    assert fix_indefinite_articles("a honest man") == "an honest man"
    assert fix_indefinite_articles("a honor") == "an honor"


def test_fix_articles_an_unchanged() -> None:
    assert fix_indefinite_articles("an") == "an"
    assert fix_indefinite_articles("an apple") == "an apple"
    assert fix_indefinite_articles("an engine fell") == "an engine fell"


def test_fix_articles_only_touches_article_token() -> None:
    # the lone 'a' is the only thing that can change; other tokens are untouched
    assert fix_indefinite_articles("The cat ate a apple today") == "The cat ate an apple today"
    # a word that merely contains 'a' is not an article
    assert fix_indefinite_articles("Anna ate apples") == "Anna ate apples"
    # consonant follower -> no change
    assert fix_indefinite_articles("a dog") == "a dog"


def test_fix_articles_word_count_unchanged() -> None:
    for src in ("a apple", "A engine", "a hour", "a university", "an apple", "a dog"):
        assert word_count(fix_indefinite_articles(src)) == word_count(src)


def test_fix_articles_preserves_punctuation_on_following_word() -> None:
    # follower classified on its bare form
    assert fix_indefinite_articles("It was a (apple)") == "It was an (apple)"
    assert fix_indefinite_articles("a apple,") == "an apple,"


def test_fix_articles_idempotent() -> None:
    once = fix_indefinite_articles("a apple and a hour and a university")
    assert fix_indefinite_articles(once) == once


# --- base_id / item_id --------------------------------------------------------


def test_base_id_stable_and_variant_sharing() -> None:
    b1 = base_id("frame7", "dog", "cat")
    b2 = base_id("frame7", "dog", "cat")
    assert b1 == b2  # two variants of a base share it
    assert base_id("frame7", "dog", "fox") != b1
    assert item_id(b1, True) == f"{b1}-T"
    assert item_id(b1, False) == f"{b1}-F"
    assert item_id(b1, True, "v2") == f"{b1}-T-v2"
