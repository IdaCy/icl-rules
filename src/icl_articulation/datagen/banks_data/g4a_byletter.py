"""Bank group G4a (by-initial-letter banks). Authored content for:
  NOUN_PLURAL_BY_LETTER, ADJ_BY_LETTER, NOUN_FINAL_BY_LETTER

All three index entries by INITIAL LETTER over the shared letter set
{b, c, d, f, g, h, l, m, n, p, r, s, t, w} with >= 5 entries per letter
(banks.BANK_QUOTAS by_letter_min = (BY_LETTER_SET, 5)). The size quota for
each bank is 70, and 14 letters x 5 = 70, so each bank is authored at exactly
5 entries per letter: the per-letter minimum and the size quota are met
simultaneously with no padding letter. ``banks.py`` computes ``initial`` from
the surface word, so the per-letter count is checked mechanically; every word
below was selected so that its first ALPHABETIC character is its intended
index letter (no silent-letter surprises, e.g. 'knights' -> initial 'k' is
excluded from the 'n' group).

Consumers (rule recipes that pin these banks; see rule-specs.yaml):
  - rule 27 first_last_same_letter: W1 (sentence-INITIAL, capitalized) from
    NOUN_PLURAL_BY_LETTER, W2 (sentence-FINAL) from NOUN_FINAL_BY_LETTER, both
    drawn with the same initial letter L. The two banks are independent word
    sets; the generator additionally rejects first==last word, so per-letter
    diversity matters more than total size.
  - rule 29 first_two_words_alphabetical: first word = adjective from
    ADJ_BY_LETTER, second word = plural noun from NOUN_PLURAL_BY_LETTER, with
    the True/False label decided by initial-letter order (same-initial pairs,
    decided at letter 2, are kept to <= 10% of items).

Authoring honesty (globals.banks header: nothing rarer than tier 2 anywhere):
  - frequency_tier is 1 (wordfreq top-2000) or 2 (wordfreq top-10000),
    verified against ``wordfreq.top_n_list('en', ...)`` for the PLURAL surface
    form of each noun (plurals are often rarer than their singulars, so this
    was checked on the inflected word actually emitted). No tier-3 word is
    used.
  - NOUN_PLURAL_BY_LETTER: sentence-initial-capable plural common nouns, all
    pos='noun'.
  - ADJ_BY_LETTER: ordinary (non-comparative) adjectives, all pos='adjective'.
  - NOUN_FINAL_BY_LETTER: sentence-final-capable words. Mostly singular common
    nouns (pos='noun'); two 1-word place adverbs ('downtown', 'nearby') are
    tagged pos='adverb' honestly. Every entry is grammatical as the last word
    of a sentence ("They walked downtown", "The shop was nearby").

Each entry is a word-bank dict per the entry contract in banks.py: the author
supplies only ``word``, ``pos``, ``frequency_tier`` (the computed tags
length / initial / final / doubles are derived by banks.py).
"""

from __future__ import annotations

from typing import Any


def _n(word: str, tier: int) -> dict[str, Any]:
    """A plural / common noun entry."""
    return {"word": word, "pos": "noun", "frequency_tier": tier}


def _a(word: str, tier: int) -> dict[str, Any]:
    """An adjective entry."""
    return {"word": word, "pos": "adjective", "frequency_tier": tier}


def _adv(word: str, tier: int) -> dict[str, Any]:
    """A 1-word place adverb (sentence-final-capable)."""
    return {"word": word, "pos": "adverb", "frequency_tier": tier}


# --- NOUN_PLURAL_BY_LETTER ----------------------------------------------------
# Sentence-initial-capable plural common nouns; 5 per letter (70 total).
# Tiers verified on the plural surface form via wordfreq.

NOUN_PLURAL_BY_LETTER: list[dict[str, Any]] = [
    # b
    _n("birds", 2), _n("books", 1), _n("bottles", 2), _n("bridges", 2), _n("buildings", 2),
    # c
    _n("cars", 1), _n("cities", 1), _n("children", 1), _n("chairs", 2), _n("clouds", 2),
    # d
    _n("dogs", 1), _n("doors", 2), _n("doctors", 2), _n("drivers", 2), _n("dreams", 2),
    # f
    _n("friends", 1), _n("families", 1), _n("farmers", 2), _n("fields", 2), _n("flowers", 2),
    # g
    _n("girls", 1), _n("games", 1), _n("gardens", 2), _n("groups", 1), _n("guards", 2),
    # h
    _n("houses", 1), _n("horses", 2), _n("hills", 2), _n("hotels", 2), _n("hunters", 2),
    # l
    _n("lakes", 2), _n("letters", 2), _n("lions", 2), _n("leaders", 1), _n("lawyers", 2),
    # m
    _n("markets", 2), _n("mountains", 2), _n("mothers", 2), _n("machines", 2), _n("members", 1),
    # n
    _n("names", 1), _n("numbers", 1), _n("nurses", 2), _n("neighbors", 2), _n("novels", 2),
    # p
    _n("parents", 1), _n("plants", 2), _n("players", 1), _n("pictures", 1), _n("pilots", 2),
    # r
    _n("rivers", 2), _n("roads", 2), _n("readers", 2), _n("rockets", 2), _n("robots", 2),
    # s
    _n("streets", 2), _n("students", 1), _n("singers", 2), _n("sisters", 2), _n("soldiers", 2),
    # t
    _n("trees", 2), _n("teachers", 1), _n("tigers", 2), _n("towers", 2), _n("trains", 2),
    # w
    _n("windows", 1), _n("workers", 1), _n("writers", 2), _n("wolves", 2), _n("warriors", 2),
]


# --- ADJ_BY_LETTER ------------------------------------------------------------
# Ordinary (non-comparative) adjectives; 5 per letter (70 total).

ADJ_BY_LETTER: list[dict[str, Any]] = [
    # b
    _a("big", 1), _a("black", 1), _a("brown", 1), _a("bright", 2), _a("bitter", 2),
    # c
    _a("clean", 1), _a("clear", 1), _a("cold", 1), _a("cheap", 1), _a("careful", 2),
    # d
    _a("dark", 1), _a("deep", 1), _a("dirty", 2), _a("difficult", 1), _a("dangerous", 1),
    # f
    _a("fast", 1), _a("flat", 1), _a("fresh", 1), _a("famous", 1), _a("friendly", 2),
    # g
    _a("good", 1), _a("green", 1), _a("great", 1), _a("gentle", 2), _a("grand", 1),
    # h
    _a("happy", 1), _a("heavy", 1), _a("hard", 1), _a("huge", 1), _a("honest", 1),
    # l
    _a("large", 1), _a("long", 1), _a("loud", 2), _a("light", 1), _a("lazy", 2),
    # m
    _a("modern", 1), _a("mild", 2), _a("massive", 1), _a("merry", 2), _a("mature", 2),
    # n
    _a("narrow", 2), _a("nice", 1), _a("normal", 1), _a("nervous", 2), _a("natural", 1),
    # p
    _a("plain", 2), _a("polite", 2), _a("proud", 1), _a("pale", 2), _a("perfect", 1),
    # r
    _a("red", 1), _a("round", 1), _a("rich", 1), _a("rough", 2), _a("rapid", 2),
    # s
    _a("small", 1), _a("soft", 2), _a("strong", 1), _a("sweet", 1), _a("smooth", 2),
    # t
    _a("tall", 2), _a("thin", 2), _a("thick", 2), _a("tiny", 2), _a("tired", 2),
    # w
    _a("warm", 2), _a("wide", 1), _a("wild", 1), _a("weak", 2), _a("wooden", 2),
]


# --- NOUN_FINAL_BY_LETTER -----------------------------------------------------
# Sentence-final-capable words (singular common nouns + 1-word place adverbs);
# 5 per letter (70 total). 'downtown' and 'nearby' are place adverbs.

NOUN_FINAL_BY_LETTER: list[dict[str, Any]] = [
    # b
    _n("bridge", 1), _n("barn", 2), _n("beach", 1), _n("basement", 2), _n("bank", 1),
    # c
    _n("corner", 1), _n("cottage", 2), _n("city", 1), _n("castle", 2), _n("cave", 2),
    # d
    _adv("downtown", 2), _n("desk", 2), _n("door", 1), _n("dock", 2), _n("district", 1),
    # f
    _n("field", 1), _n("farm", 1), _n("forest", 2), _n("factory", 2), _n("fountain", 2),
    # g
    _n("garden", 1), _n("garage", 2), _n("gate", 2), _n("gallery", 2), _n("ground", 1),
    # h
    _n("house", 1), _n("hill", 1), _n("harbor", 2), _n("hotel", 1), _n("hospital", 1),
    # l
    _n("lake", 1), _n("library", 1), _n("lobby", 2), _n("lawn", 2), _n("lane", 2),
    # m
    _n("market", 1), _n("mountain", 1), _n("mall", 2), _n("museum", 2), _n("mill", 2),
    # n
    _adv("nearby", 2), _n("nursery", 2), _n("nest", 2), _n("nation", 1), _n("neighborhood", 2),
    # p
    _n("park", 1), _n("porch", 2), _n("pond", 2), _n("palace", 2), _n("pier", 2),
    # r
    _n("river", 1), _n("road", 1), _n("room", 1), _n("ranch", 2), _n("restaurant", 2),
    # s
    _n("street", 1), _n("school", 1), _n("station", 1), _n("store", 1), _n("stadium", 2),
    # t
    _n("town", 1), _n("theater", 2), _n("tower", 2), _n("tunnel", 2), _n("temple", 2),
    # w
    _n("warehouse", 2), _n("workshop", 2), _n("well", 1), _n("wing", 2), _n("wood", 1),
]


BANKS: dict[str, list[Any]] = {
    "NOUN_PLURAL_BY_LETTER": NOUN_PLURAL_BY_LETTER,
    "ADJ_BY_LETTER": ADJ_BY_LETTER,
    "NOUN_FINAL_BY_LETTER": NOUN_FINAL_BY_LETTER,
}
