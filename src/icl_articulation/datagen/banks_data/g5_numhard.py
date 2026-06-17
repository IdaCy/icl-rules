"""Bank group G5 (comparatives + length-threshold vocab for rules 12/23/30).

Owns these banks (see banks.BANK_QUOTAS for the per-bank contract):
  ADJ_COMPARABLE, ER_NONCOMPARATIVE, SHORT_WORDS_BY_POS, LONG_ONLY_VOCAB

(The g1_general stub docstring lists ADJ_COMPARABLE / ER_NONCOMPARATIVE among
its banks, but this group authors them per the B0 stage-B assignment. Each bank
has exactly ONE owner; banks._raw_registry() raises if two groups define the
same name, so g1_general must NOT also define these two. See the open concern
recorded with this stage's summary.)

Author/computed tag split: the author supplies word + pos + frequency_tier (and,
where a quota uses them, the optional keys); banks.py computes length / initial /
final / has_adjacent_double / has_nonadjacent_repeat. frequency_tier is honest
per the globals.implementation_pins definition: tier 1 = wordfreq
top_n_list('en', 2000), tier 2 = top_n_list('en', 10000); nothing rarer than
tier 2 appears anywhere (every surface word here was verified against the pinned
wordfreq list, including the inflected plural/past surfaces in LONG_ONLY_VOCAB —
a few near-cutoff lemmas like 'baskets'/'meadows'/'chased' fall just below the
top-10000 threshold and were replaced by in-list equivalents).

Per-bank notes the author honors (all enforced by banks.check_bank):

  ADJ_COMPARABLE (size 60, pos_required adjective). Rule 12's True-class
  adjective bank. The bank stores BASE adjective forms; the generator derives
  the comparative surface ('-er' or 'more X'). Spec split: 40 adjectives with a
  productive '-er' comparative (tall->taller, cheap->cheaper, old->older) and 20
  that take the analytic 'more X' form (more careful, more famous). The split is
  recorded in subtype ('er' / 'more') so the generator can honor the recipe's
  70% '-er' / 30% 'more {adj}' COMP mix and the count-solver note ('more {adj}'
  is 2 words, '-er' is 1). All are plain (non-comparative) base adjectives, so
  they double as plausible ADJ_PLAIN-style fillers.

  ER_NONCOMPARATIVE (size 25, pos_required noun, custom_final_er). Nouns ending
  in the letters '-er' that are NOT comparative adjectives (teacher, river,
  corner). Rule 12 salts 30% of its False items with one of these so a naive
  'contains a word ending in -er' distractor disagrees. Every entry is a common
  noun ending in 'er'; '-or'/'-ar' nouns (doctor, dollar) are deliberately
  excluded because the rule-12 salt and the '-er' distractor both key on the
  literal '-er' suffix.

  SHORT_WORDS_BY_POS (size 45, max_alpha_len 3, pos_required noun/verb/adjective/
  numeral). <= 3 alphabetic letters, drawn from the exact members the spec lists,
  spread across the four POS so rule 30's one-slot short-word substitution has a
  same-POS replacement for every long word it swaps out (and so the rule-30
  short-word POS-spread quota >= 15%/POS is feasible). Verbs are PAST-tense
  forms (ran, sat, ate) to match LONG_ONLY_VOCAB's past verbs grammatically.

  LONG_ONLY_VOCAB (size 150, min_alpha_len 4, exactly_four_min 0.40, no_articles).
  Rule 30's True-class vocabulary: every word >= 4 letters, NO articles, and
  >= 40% of entries EXACTLY 4 letters (this group is 72/169 = 42.6%, supporting
  the rule-30 item-level quota that >= 60% of True items carry a 4-letter word).
  Coverage spans plural bare nouns, past verbs, adjectives, the >= 4-letter
  numerals (three, seven, eight, nine) and the >= 4-letter prepositions the spec
  names (across, beside, under, toward, during, near + above/behind and other
  in-list directional/temporal preps). subtype records the coarse class
  (plural_noun / past_verb / adjective / numeral / preposition) for the
  generator's POS-matched substitution against SHORT_WORDS_BY_POS.

Each WORD-bank entry is a dict per the entry contract documented in banks.py.
"""

from __future__ import annotations

from typing import Any


def _w(word: str, pos: str, tier: int, **opt: Any) -> dict[str, Any]:
    """Build one authored entry dict (word + pos + frequency_tier + optionals)."""
    entry: dict[str, Any] = {"word": word, "pos": pos, "frequency_tier": tier}
    entry.update(opt)
    return entry


# =============================================================================
# ADJ_COMPARABLE — 40 '-er' base adjectives + 20 'more X' base adjectives.
# subtype: 'er' (productive -er comparative) | 'more' (analytic comparative).
# Tiers verified against wordfreq top-2000 / top-10000.
# =============================================================================

_ADJ_ER: list[tuple[str, int]] = [
    ("tall", 2), ("short", 1), ("old", 1), ("young", 1), ("cheap", 1),
    ("rich", 1), ("poor", 1), ("fast", 1), ("slow", 1), ("high", 1),
    ("low", 1), ("big", 1), ("small", 1), ("long", 1), ("wide", 1),
    ("thin", 2), ("thick", 2), ("weak", 2), ("strong", 1), ("hard", 1),
    ("soft", 2), ("warm", 2), ("cold", 1), ("clean", 1), ("dark", 1),
    ("bright", 2), ("deep", 1), ("fresh", 1), ("kind", 1), ("loud", 2),
    ("quiet", 2), ("sharp", 2), ("smooth", 2), ("sweet", 1), ("wet", 2),
    ("fair", 1), ("late", 1), ("nice", 1), ("safe", 1), ("brave", 2),
]

_ADJ_MORE: list[tuple[str, int]] = [
    ("careful", 2), ("useful", 1), ("helpful", 2), ("famous", 1),
    ("modern", 1), ("honest", 1), ("nervous", 2), ("serious", 1),
    ("popular", 1), ("perfect", 1), ("obvious", 2), ("curious", 2),
    ("anxious", 2), ("precious", 2), ("generous", 2), ("expensive", 2),
    ("difficult", 1), ("important", 1), ("beautiful", 1), ("dangerous", 1),
]

ADJ_COMPARABLE: list[dict[str, Any]] = (
    [_w(w, "adjective", t, subtype="er") for w, t in _ADJ_ER]
    + [_w(w, "adjective", t, subtype="more") for w, t in _ADJ_MORE]
)


# =============================================================================
# ER_NONCOMPARATIVE — 25 common nouns ending in the letters '-er', not
# comparative adjectives (rule-12 False-class salt).
# =============================================================================

_ER_NONCOMP: list[tuple[str, int]] = [
    ("teacher", 1), ("river", 1), ("corner", 1), ("paper", 1), ("ladder", 2),
    ("summer", 1), ("dinner", 1), ("letter", 1), ("number", 1), ("water", 1),
    ("mother", 1), ("father", 1), ("brother", 1), ("sister", 1), ("winter", 1),
    ("finger", 2), ("member", 1), ("officer", 1), ("answer", 1), ("matter", 1),
    ("center", 1), ("driver", 1), ("leader", 1), ("owner", 1), ("power", 1),
]

ER_NONCOMPARATIVE: list[dict[str, Any]] = [
    _w(w, "noun", t) for w, t in _ER_NONCOMP
]


# =============================================================================
# SHORT_WORDS_BY_POS — <= 3 alphabetic letters, POS spread across
# noun / verb (past) / adjective / numeral. From the spec's exact members.
# =============================================================================

_SHORT_NOUN: list[tuple[str, int]] = [
    ("dog", 1), ("cat", 1), ("fox", 2), ("bus", 1), ("car", 1), ("cup", 1),
    ("hat", 2), ("map", 1), ("pen", 2), ("box", 1), ("key", 1), ("egg", 2),
    ("jar", 2), ("arm", 1), ("sky", 1), ("sun", 1), ("ice", 1), ("oil", 1),
    ("tea", 1),
]

_SHORT_PAST_VERB: list[tuple[str, int]] = [
    ("ran", 1), ("sat", 2), ("ate", 2), ("saw", 1), ("hid", 2), ("won", 1),
    ("got", 1), ("met", 1), ("led", 1), ("fed", 2),
]

_SHORT_ADJ: list[tuple[str, int]] = [
    ("big", 1), ("old", 1), ("new", 1), ("red", 1), ("hot", 1), ("wet", 2),
    ("dry", 1), ("sad", 1), ("shy", 2), ("raw", 2), ("odd", 2), ("fit", 1),
    ("low", 1),
]

_SHORT_NUM: list[tuple[str, int]] = [
    ("two", 1), ("six", 1), ("ten", 1), ("one", 1),
]

SHORT_WORDS_BY_POS: list[dict[str, Any]] = (
    [_w(w, "noun", t) for w, t in _SHORT_NOUN]
    + [_w(w, "verb", t) for w, t in _SHORT_PAST_VERB]
    + [_w(w, "adjective", t) for w, t in _SHORT_ADJ]
    + [_w(w, "numeral", t) for w, t in _SHORT_NUM]
)


# =============================================================================
# LONG_ONLY_VOCAB — every word >= 4 letters, no articles, >= 40% exactly 4
# letters. subtype = coarse class for the rule-30 POS-matched substitution.
# 72 of 169 entries are exactly 4 letters (42.6%).
# =============================================================================

# ---- 4-letter words (72) -----------------------------------------------------
_L4_PLURAL: list[tuple[str, int]] = [
    ("dogs", 1), ("cats", 2), ("cars", 1), ("cups", 2), ("hats", 2),
    ("maps", 2), ("keys", 2), ("eggs", 2), ("arms", 1), ("guns", 2),
    ("jobs", 1), ("legs", 2), ("beds", 2), ("toys", 2), ("bags", 2),
    ("cows", 2), ("pigs", 2), ("bees", 2), ("jets", 2), ("kids", 1),
    ("fans", 1), ("lips", 2), ("ribs", 2),
]
_L4_PAST: list[tuple[str, int]] = [
    ("used", 1), ("made", 1), ("told", 1), ("took", 1), ("gave", 1),
    ("came", 1), ("knew", 1), ("felt", 1), ("kept", 1), ("held", 1),
    ("sold", 1), ("fell", 1), ("grew", 2), ("drew", 2), ("blew", 2),
    ("flew", 2), ("wore", 2), ("rode", 2), ("woke", 2), ("shut", 1),
    ("sang", 2), ("hung", 2),
]
_L4_ADJ: list[tuple[str, int]] = [
    ("wide", 1), ("cold", 1), ("warm", 2), ("cool", 1), ("dark", 1),
    ("deep", 1), ("fast", 1), ("slow", 1), ("tall", 2), ("thin", 2),
    ("weak", 2), ("soft", 2), ("hard", 1), ("good", 1), ("poor", 1),
    ("rich", 1), ("wild", 1), ("calm", 2), ("fair", 1), ("fine", 1),
]
_L4_PREP: list[tuple[str, int]] = [
    ("over", 1), ("near", 1), ("into", 1), ("onto", 1), ("upon", 1),
]
_L4_NUM: list[tuple[str, int]] = [
    ("nine", 1),
]

# ---- words >= 5 letters (97) -------------------------------------------------
_L5_PLURAL: list[tuple[str, int]] = [
    ("tables", 2), ("gardens", 2), ("forests", 2), ("roads", 2), ("houses", 1),
    ("horses", 2), ("apples", 2), ("villages", 2), ("rivers", 2), ("fields", 2),
    ("clouds", 2), ("streets", 2), ("bridges", 2), ("flowers", 2),
    ("mountains", 2), ("insects", 2), ("wolves", 2), ("tigers", 2),
    ("eagles", 2), ("stones", 2), ("farmers", 2), ("children", 1),
    ("engines", 2), ("windows", 1), ("branches", 2), ("letters", 2),
    ("singers", 2), ("drivers", 2), ("leaders", 1), ("feathers", 2),
]
_L5_PAST: list[tuple[str, int]] = [
    ("jumped", 2), ("walked", 2), ("crossed", 2), ("carried", 1),
    ("painted", 2), ("opened", 1), ("closed", 1), ("wanted", 1),
    ("played", 1), ("wished", 2), ("cooked", 2), ("cleaned", 2),
    ("watched", 1), ("escaped", 2), ("entered", 1), ("reached", 1),
    ("followed", 1), ("learned", 1), ("brought", 1), ("started", 1),
    ("stopped", 1), ("talked", 2), ("turned", 1), ("visited", 2),
    ("waited", 2),
]
_L5_ADJ: list[tuple[str, int]] = [
    ("large", 1), ("small", 1), ("heavy", 1), ("light", 1), ("clean", 1),
    ("fresh", 1), ("quiet", 2), ("sharp", 2), ("smooth", 2), ("loud", 2),
    ("hungry", 2), ("frozen", 2), ("ancient", 2), ("gentle", 2), ("strong", 1),
    ("bright", 2), ("quick", 1), ("green", 1), ("brown", 1), ("silver", 1),
    ("golden", 2), ("silent", 2), ("narrow", 2), ("steady", 2), ("hidden", 2),
]
_L5_NUM: list[tuple[str, int]] = [
    ("three", 1), ("seven", 1), ("eight", 1),
]
_L5_PREP: list[tuple[str, int]] = [
    ("across", 1), ("beside", 2), ("under", 1), ("toward", 1), ("during", 1),
    ("behind", 1), ("above", 1), ("below", 1), ("along", 1), ("among", 1),
    ("around", 1), ("before", 1), ("beyond", 1), ("inside", 1), ("within", 1),
]


def _long(entries: list[tuple[str, int]], subtype: str, pos: str) -> list[dict[str, Any]]:
    return [_w(w, pos, t, subtype=subtype) for w, t in entries]


LONG_ONLY_VOCAB: list[dict[str, Any]] = (
    _long(_L4_PLURAL, "plural_noun", "noun")
    + _long(_L4_PAST, "past_verb", "verb")
    + _long(_L4_ADJ, "adjective", "adjective")
    + _long(_L4_PREP, "preposition", "preposition")
    + _long(_L4_NUM, "numeral", "numeral")
    + _long(_L5_PLURAL, "plural_noun", "noun")
    + _long(_L5_PAST, "past_verb", "verb")
    + _long(_L5_ADJ, "adjective", "adjective")
    + _long(_L5_NUM, "numeral", "numeral")
    + _long(_L5_PREP, "preposition", "preposition")
)


BANKS: dict[str, list[Any]] = {
    "ADJ_COMPARABLE": ADJ_COMPARABLE,
    "ER_NONCOMPARATIVE": ER_NONCOMPARATIVE,
    "SHORT_WORDS_BY_POS": SHORT_WORDS_BY_POS,
    "LONG_ONLY_VOCAB": LONG_ONLY_VOCAB,
}
