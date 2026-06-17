"""Step-2 multiple-choice articulation: distractor builder + multiple-choice option assembly.

PLAN (locked) + rule-specs globals.step2_mc_policy:

  Every multiple-choice query presents 8 options = the rule's canonical_articulation (the
  TRUE option) + 7 distractors drawn from the rule's ``mc_distractor_seeds``
  (data/spec_extract.json). banned_distractors and any seed extensionally equal
  to the true rule are excluded up front.

  THE LOAD-BEARING CHECK: every distractor must DISAGREE with the true rule on
  >= 25% of the 32 SHOWN examples of the ACTUAL context, computed
  programmatically. To label an example under a distractor we need a CHECKING
  PREDICATE per seed: cross-rule canonical seeds ("... cross-rule canonical
  (rN)") reuse ``groundtruth.label_of`` for rule rN; the rest carry a bespoke
  predicate implementing the seed's stated checking hint. The true rule's label
  on the 32 examples is the SHOWN label (verified against groundtruth.label_of
  where the rule is recomputable).

  disagreement(distractor, context) = fraction of the 32 shown examples where
  predicate(text) != shown_label. Require >= 0.25 in EVERY one of the rule's 3
  contexts. The option set is IDENTICAL across the 3 contexts: a seed that fails
  the check in ANY context is replaced GLOBALLY from the remaining pool. Two
  options may never share a checking predicate (``predicate_key`` dedupe).

  Per rule: 4 option ORDERS x 3 contexts = 12 multiple-choice queries; order shuffled with a
  logged seed; the letter of the true option is recorded per query.

This module builds the option SETS and the per-query ORDERS. The runner
(scripts/run_step2_mc.py) renders prompts, calls the model, and scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Callable

from .contexts import load_items, sample_context
from .datagen import banks, groundtruth
from .datagen.schema import words as _words
from .rule_ids import canonical_rule_id
from .step3_probes import _art_physically_impossible

# locked structure (PLAN "Step-2 multiple-choice")
N_OPTIONS = 8
N_DISTRACTORS = N_OPTIONS - 1  # 7
K_FEW_SHOT = 32
N_CONTEXTS = 3
N_ORDERS = 4
DISAGREEMENT_FLOOR = 0.25  # the PLAN-locked hard per-context check
LETTERS = "ABCDEFGH"  # A..H for 8 options

# BEHAVIOURAL near-synonym bans (external-review fix). The >=25% extensional
# disagreement gate cannot catch a distractor that is SEMANTICALLY a near-synonym
# of the true rule yet extensionally overlaps it only ~70% (just under the spec's
# 75% agreement ban). physically_impossible's "an object doing a human action" is
# exactly this: with the real predicate it agrees with the true rule up to 0.72,
# and BOTH models picked it 11-12/12 in the original multiple-choice run — a recognition
# artifact, not a failure. We ban it here and the builder replaces it from the
# pool (re-run produces a clean recognition number).
BEHAVIORAL_NEAR_SYNONYM_BANS: dict[str, set[str]] = {
    "physically_impossible": {"The input mentions an object doing a human action"},
}


class MCBuildError(ValueError):
    """The multiple-choice option set for a rule could not be assembled under the policy
    (too few surviving seeds, a duplicate predicate, or a missing predicate)."""


# --- construct-validity: stylistically uniform option text ---------------------
#
# The TRUE option must NOT be identifiable by surface form. The raw
# canonical_articulation is unusable as displayed text: it uniquely carries the
# "is labeled True iff" framing, parenthetical clarifications, and "— ..." tails,
# and is far longer than the terse distractors — a model could ace the multiple-choice by
# format-matching alone. We therefore render the true option as a terse "The
# input ..." predicate in the SAME register/length band as the distractors, and
# strip the equivalent author-note suffixes from distractor display text. The
# CHECKING PREDICATES and the >= 25%-disagreement logic are UNCHANGED — only the
# DISPLAYED text is normalized.
#
# Faithful one-line true-option paraphrases for the 11 target rules, authored in
# the distractor register (auditable; each denotes the SAME predicate as the
# rule's canonical_articulation, never changing which items it labels True). For
# semantic rules the essential qualifier is retained (e.g. "a real living animal"
# vs the distractor "a living thing"/"a pet"; "a specific color" vs "the color
# red") so the true option stays distinguishable from near-miss distractors by
# CONTENT — not by surface form.

_TRUE_OPTION_TEXT: dict[str, str] = {
    # canonical: "... its main clause is in the passive voice ..."
    "passive_voice": "The input is in the passive voice",
    # canonical: "... its topic is food or cooking ..."
    "food_topic": "The input is about food or cooking",
    # canonical: "... it expresses a positive sentiment or opinion ..."
    "positive_sentiment": "The input expresses a positive sentiment",
    # canonical: "... a real, living, non-human animal kind ..."; the qualifier
    # "real living" matters against the distractors "a pet" / "a living thing".
    "mentions_animal": "The input mentions a real living animal",
    # canonical: "... a person's first name ... NOT city/month/brand names";
    # "first name" matters against "a city"/"a proper noun"/"a woman's name".
    "contains_first_name": "The input contains a person's first name",
    # canonical: "... its second word begins with a capital letter."
    "second_word_capitalized": "The input has a capitalized second word",
    # canonical: "... describes something physically impossible ..."; "physically
    # impossible" matters against "magic or fantasy"/"an object doing a human
    # action".
    # shortened to the distractor length band after the near-synonym "an object
    # doing a human action" was banned (it had padded the band); still a faithful
    # terse paraphrase of the canonical rule.
    "physically_impossible": "The input is physically impossible",
    # canonical: "... contains at least 8 words."
    "word_count_geq_8": "The input has at least 8 words",
    # canonical: "... some content word ... occurs at least twice ..."; "content
    # word" matters against "Some word appears at least twice, counting words
    # like 'the' and 'a'".
    "repeated_content_word": "The input repeats a content word",
    # canonical: "... contains at least one digit character (0-9)."; "character"
    # keeps it distinct from the (display-normalized) distractor "contains a
    # digit".
    "contains_digit": "The input contains a digit character",
    # canonical: "... a specific color term (such as red, blue, ...)"; "specific
    # color" matters against the distractor "the color red".
    "mentions_color": "The input mentions a specific color",
}

# Giveaway tokens that must never appear in any displayed option (they are the
# format tell of the raw canonical_articulation).
_GIVEAWAY_TOKENS = ("is labeled true iff", "labeled true", " iff ", " iff.", "iff it")

_PARENTHETICAL_RE = re.compile(r"\s*\([^()]*\)")


def normalize_display(text: str) -> str:
    """The terse DISPLAY text for a distractor: drop any ' — ' rationale tail
    (already done by _seed_head for heads) and strip parenthetical clauses —
    both the author-note kind ("(predicate: ...)") and short inline clarifiers
    ("(0-9)", "(breakfast/lunch/dinner)") — so distractors are uniformly terse.
    The CHECKING PREDICATE is unchanged; only the shown string is normalized."""
    text = text.split(" — ")[0]
    text = _PARENTHETICAL_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def true_option_text(rule_id: str, rule: dict[str, Any]) -> str:
    """The terse, distractor-register display text for the TRUE option. Uses the
    authored paraphrase for the 11 target rules; for any other rule falls back to
    a normalized projection of the canonical_articulation (strip the "is labeled
    True iff" frame, parentheticals, and "— ..." tails) so no rule can leak the
    raw-canonical format tell."""
    if rule_id in _TRUE_OPTION_TEXT:
        return _TRUE_OPTION_TEXT[rule_id]
    return _project_canonical(rule["canonical_articulation"])


_CANON_FRAME_RE = re.compile(r"^the input is labeled true iff\s+", re.IGNORECASE)


def _project_canonical(canonical: str) -> str:
    """Best-effort terse projection for a non-target rule's canonical text:
    take the clause before the first ' — '/'(' /', and False', strip the
    'is labeled True iff' frame, and re-cap to 'The input ...'."""
    core = canonical.strip()
    core = core.split(" — ")[0]
    core = re.split(r", and False| and False|\(", core, maxsplit=1)[0]
    core = _PARENTHETICAL_RE.sub("", core).strip().rstrip(".")
    m = _CANON_FRAME_RE.match(core)
    if m:
        core = "The input " + core[m.end():]
    return re.sub(r"\s+", " ", core).strip()


# --- checking predicates -------------------------------------------------------
#
# A checking predicate maps a sentence -> True/False (the distractor's label on
# that sentence). ``key`` makes two options with the SAME checker comparable so
# duplicates are forbidden in one option set; ``label_of`` is the pure function.


@dataclass(frozen=True)
class CheckingPredicate:
    key: str  # identity for duplicate detection (cross-rule rN, or seed slug)
    label_of: Callable[[str], bool]


def _toks(text: str) -> list[str]:
    return [w.lower() for w in _words(text)]


def _bank_set(name: str) -> frozenset[str]:
    return frozenset(e.word.lower() for e in banks.get_bank(name).entries)


# Lexicons for bespoke predicates. Bank-backed where a bank exists; otherwise a
# small frozen lexicon transcribed to match this data's vocabulary. These are
# CHECKING predicates for DISTRACTORS — they need only be deterministic pure
# functions of text; the empirical >= 25% per-context check is what admits or
# globally drops each seed, so an over/under-inclusive lexicon never corrupts a
# run (it just changes which seeds survive).

_FEMALE_NAMES = frozenset(
    {
        "anna", "maria", "emma", "sophie", "laura", "julia", "sarah", "hannah",
        "clara", "lena", "nina", "lisa", "eva", "mia", "lucy", "grace", "alice",
        "rosa", "ella", "ruby", "olivia", "chloe", "amelia", "isabella", "lily",
        "zoe", "amy", "kate", "jane", "rose",
    }
)
_MEAL_TIMES = frozenset({"breakfast", "lunch", "dinner", "brunch", "supper"})
_EAT_DRINK_VERBS = frozenset(
    {
        "eat", "eats", "ate", "eating", "drink", "drinks", "drank", "drinking",
        "taste", "tastes", "tasted", "tasting", "chew", "chews", "chewed",
        "swallow", "sip", "sips", "sipped", "bite", "bites", "cook", "cooks",
        "cooked", "cooking", "bake", "bakes", "baked", "baking", "serve",
        "serves", "served", "dine", "dines", "dined",
    }
)
_NOT_WORDS = frozenset({"not", "n't", "never", "no", "nothing", "nobody", "none"})
_COORD_CLAUSE = frozenset({"and", "while", "because", "but", "although", "since", "when"})
_VISUAL_ADJ = frozenset(
    {
        "bright", "dark", "shiny", "dull", "pale", "vivid", "glossy", "matte",
        "clear", "cloudy", "transparent", "smooth", "rough", "round", "square",
        "tall", "short", "big", "small", "tiny", "huge", "wide", "narrow",
        "thin", "thick", "long", "flat", "curved",
    }
)


def _membership(name: str, bank: str) -> CheckingPredicate:
    wordset = _bank_set(bank)
    return CheckingPredicate(name, lambda t: any(tok in wordset for tok in _toks(t)))


def _lexicon(name: str, lex: frozenset[str]) -> CheckingPredicate:
    return CheckingPredicate(name, lambda t: any(tok in lex for tok in _toks(t)))


def _cross_rule(rule_id: str) -> CheckingPredicate:
    """A cross-rule canonical distractor: reuse the cited rule's label_of."""
    return CheckingPredicate(f"cross:{rule_id}", lambda t: groundtruth.label_of(rule_id, t))


# --- bespoke predicate builders (keyed on the seed head text) ------------------


def _pred_contains_word(word: str) -> Callable[[str], bool]:
    w = word.lower()
    return lambda t: w in _toks(t)


def _pred_contains_word_twice(word: str) -> Callable[[str], bool]:
    w = word.lower()
    return lambda t: sum(1 for tok in _toks(t) if tok == w) >= 2


def _pred_some_word_repeats(t: str) -> bool:
    """ANY token repeats, INCLUDING stopwords (the 'counting words like the/a'
    variant of repeated_content_word — fires on stopword repeats too)."""
    seen: set[str] = set()
    for tok in _toks(t):
        if tok in seen:
            return True
        seen.add(tok)
    return False


def _pred_word_count_geq(n: int) -> Callable[[str], bool]:
    return lambda t: len(_words(t)) >= n


def _pred_word_count_gt(n: int) -> Callable[[str], bool]:
    return lambda t: len(_words(t)) > n


def _pred_char_len_gt(n: int) -> Callable[[str], bool]:
    return lambda t: len(t) > n


def _pred_n_digits_eq(n: int) -> Callable[[str], bool]:
    return lambda t: sum(ch.isdigit() for ch in t) == n


_DIGIT_RUN = re.compile(r"\d+")


def _pred_number_gt(n: int) -> Callable[[str], bool]:
    return lambda t: any(int(m.group()) > n for m in _DIGIT_RUN.finditer(t))


def _pred_digit_or_number_word(t: str) -> bool:
    if any(ch.isdigit() for ch in t):
        return True
    nums = _bank_set("NUMBER_WORDS")
    return any(tok in nums for tok in _toks(t))


def _pred_has_comma(t: str) -> bool:
    return "," in t


def _pred_distinct_nouns_geq(n: int) -> Callable[[str], bool]:
    """Approximate '>= n distinct things': count distinct CONTENT nouns via the
    concrete-noun + animal + color + food banks. Deterministic proxy; the
    per-context check decides admissibility."""
    nounset = _bank_set("NOUN_CONCRETE") | _bank_set("ANIMALS") | _bank_set("COLORS")

    def pred(t: str) -> bool:
        return len({tok for tok in _toks(t) if tok in nounset}) >= n

    return pred


def _pred_capitalized_after_first(t: str) -> bool:
    """A capitalized word appears anywhere AFTER the first word."""
    toks = _words(t)
    return any(tok[:1].isupper() for tok in toks[1:])


def _pred_second_word_is_name(t: str) -> bool:
    toks = _words(t)
    if len(toks) < 2:
        return False
    return toks[1].lower() in _bank_set("FIRST_NAMES")


def _pred_first_noun_is_person(t: str) -> bool:
    """The first name/person token in the sentence is a first name."""
    names = _bank_set("FIRST_NAMES")
    return any(tok in names for tok in _toks(t))


def _pred_contains_proper_noun(t: str) -> bool:
    """A capitalized word other than a sentence-initial 'I' / first word."""
    toks = _words(t)
    return any(tok[:1].isupper() for tok in toks[1:]) or (
        len(toks) >= 1 and toks[0][:1].isupper() and toks[0].lower() not in _SENT_INITIAL_OK
    )


_SENT_INITIAL_OK = frozenset({"the", "a", "an"})


def _pred_starts_with_adverb(t: str) -> bool:
    toks = _toks(t)
    if not toks:
        return False
    return toks[0] in _bank_set("ADVERB_SENT_INITIAL")


def _pred_adj_before_noun(t: str) -> bool:
    adjs = _bank_set("ADJ_PLAIN") | _VISUAL_ADJ | _bank_set("COLORS")
    nouns = _bank_set("NOUN_CONCRETE") | _bank_set("ANIMALS")
    toks = _toks(t)
    return any(toks[i] in adjs and toks[i + 1] in nouns for i in range(len(toks) - 1))


_VOWELS = frozenset("aeiou")
_ARTICLES = frozenset({"a", "an", "the"})


def _alpha_len(tok: str) -> int:
    return sum(ch.isalpha() for ch in tok)


def _alpha_word(tok: str) -> str:
    return re.sub(r"[^A-Za-z]", "", tok).lower()


def _first_word(t: str) -> str:
    toks = _words(t)
    return _alpha_word(toks[0]) if toks else ""


def _last_word(t: str) -> str:
    toks = _words(t)
    return _alpha_word(toks[-1]) if toks else ""


def _noun_words() -> frozenset[str]:
    return (
        _bank_set("NOUN_CONCRETE")
        | _bank_set("ANIMALS")
        | _FOOD_WORDS
        | _PLANTS
        | _VEHICLES
    )


def _is_plural(tok: str) -> bool:
    return len(tok) > 2 and tok.endswith("s") and not tok.endswith("ss")


def _last_alpha(tok: str) -> str:
    for ch in reversed(tok):
        if ch.isalpha():
            return ch.lower()
    return ""


def _last_word_ends_vowel(t: str) -> bool:
    return _last_alpha(_last_word(t)) in _VOWELS


def _count_syllables(word: str) -> int:
    word = _alpha_word(word)
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _avg_alpha_len(t: str) -> float:
    toks = [_alpha_word(tok) for tok in _words(t)]
    lens = [_alpha_len(tok) for tok in toks if tok]
    return sum(lens) / len(lens) if lens else 0.0


def _first_initial(t: str) -> str:
    first = _first_word(t)
    return first[:1]


def _is_verb_proxy(tok: str) -> bool:
    return tok in _bank_set("VERB_REGULAR") or tok.endswith(("ed", "ing", "s"))


# Registry of bespoke seed predicates, keyed on the seed HEAD (text before the
# ' — ' rationale). Heads with a structured "(predicate: ...)" hint are matched
# by their head text too. A seed whose head is not registered here and is not a
# cross-rule canonical raises MCBuildError (LOUD: no silent un-checkable option).

_BESPOKE: dict[str, CheckingPredicate] = {
    # surface / lexical
    "Some word appears at least twice, counting words like 'the' and 'a'":
        CheckingPredicate("any_word_repeats", _pred_some_word_repeats),
    "The input contains the word 'the' at least twice":
        CheckingPredicate("the_twice", _pred_contains_word_twice("the")),
    "The input contains the word 'and'":
        CheckingPredicate("word_and", _pred_contains_word("and")),
    "The input contains the word 'by'":
        CheckingPredicate("word_by", _pred_contains_word("by")),
    "The input contains the word 'not'":
        CheckingPredicate("word_not", lambda t: any(tok in _NOT_WORDS for tok in _toks(t))),
    "The input contains 'was' or 'were'":
        CheckingPredicate("was_were", lambda t: any(tok in ("was", "were") for tok in _toks(t))),
    "The input contains a comma":
        CheckingPredicate("has_comma", _pred_has_comma),
    # length / count
    "The input is longer than 8 words":
        CheckingPredicate("wc_gt_8", _pred_word_count_gt(8)),
    "The input is longer than 9 words":
        CheckingPredicate("wc_gt_9", _pred_word_count_gt(9)),
    "The input has at least 6 words":
        CheckingPredicate("wc_geq_6", _pred_word_count_geq(6)),
    "The input has more than 7 words":
        CheckingPredicate("wc_gt_7", _pred_word_count_gt(7)),
    "The input is longer than 40 characters":
        CheckingPredicate("chars_gt_40", _pred_char_len_gt(40)),
    "The input has more than one clause (predicate: contains 'and'/'while'/'because')":
        CheckingPredicate(
            "multi_clause", lambda t: any(tok in _COORD_CLAUSE for tok in _toks(t))
        ),
    "The input mentions more than two things (predicate: >= 3 nouns)":
        CheckingPredicate("nouns_geq_3", _pred_distinct_nouns_geq(3)),
    "The input mentions two or more people or things doing something "
    "(predicate: >= 2 distinct nouns)":
        CheckingPredicate("nouns_geq_2", _pred_distinct_nouns_geq(2)),
    "The input mentions at least two things (predicate: >= 2 nouns)":
        CheckingPredicate("nouns_geq_2", _pred_distinct_nouns_geq(2)),
    "The input has at least 8 words":
        CheckingPredicate("wc_geq_8", _pred_word_count_geq(8)),
    "The input has exactly 8 words":
        CheckingPredicate("wc_eq_8", lambda t: len(_words(t)) == 8),
    "The input has fewer than 7 words":
        CheckingPredicate("wc_lt_7", lambda t: len(_words(t)) < 7),
    "The input has fewer than 8 words":
        CheckingPredicate("wc_lt_8", lambda t: len(_words(t)) < 8),
    "The input contains an even digit":
        CheckingPredicate("even_digit", lambda t: any(ch in "02468" for ch in t)),
    "The input has an even number of letters":
        CheckingPredicate("even_letters", lambda t: sum(ch.isalpha() for ch in t) % 2 == 0),
    # digits / numbers
    "The input contains a number greater than 10":
        CheckingPredicate("num_gt_10", _pred_number_gt(10)),
    "The input contains exactly one digit character":
        CheckingPredicate("one_digit", _pred_n_digits_eq(1)),
    "The input mentions a number or quantity, in digits or words "
    "(checking predicate: contains a digit OR a NUMBER_WORDS-lexicon word; "
    "'several/many/some' do NOT count)":
        CheckingPredicate("digit_or_numword", _pred_digit_or_number_word),
    "The input mentions a measurement or amount of money":
        CheckingPredicate(
            "measurement",
            lambda t: bool(re.search(r"\$|\d+\s?(kg|km|cm|m|ml|g|lb|oz|%)", t.lower())),
        ),
    # capitalization / names / proper nouns
    "The input contains a capitalized word after the first word":
        CheckingPredicate("cap_after_first", _pred_capitalized_after_first),
    "The input contains a capitalized word other than the first word":
        CheckingPredicate("cap_after_first", _pred_capitalized_after_first),
    "The second word is a person's name":
        CheckingPredicate("second_is_name", _pred_second_word_is_name),
    "The input contains a proper noun":
        CheckingPredicate("proper_noun", _pred_contains_proper_noun),
    "The first noun names a person":
        CheckingPredicate("first_noun_person", _pred_first_noun_is_person),
    "The input mentions a person":
        CheckingPredicate("mentions_person", _pred_first_noun_is_person),
    "The input mentions other people":
        CheckingPredicate("mentions_person", _pred_first_noun_is_person),
    "The input mentions a woman's name":
        CheckingPredicate("woman_name", lambda t: any(tok in _FEMALE_NAMES for tok in _toks(t))),
    "The input starts with an adverb":
        CheckingPredicate("starts_adverb", _pred_starts_with_adverb),
    "The input contains an adjective directly before a noun":
        CheckingPredicate("adj_before_noun", _pred_adj_before_noun),
    "The first word is a noun":
        CheckingPredicate("first_noun", lambda t: _first_word(t) in _noun_words()),
    "The first word starts with 'a' or 'the'":
        CheckingPredicate("first_a_the", lambda t: _first_word(t) in {"a", "the"}),
    "The first word is shorter than 6 letters":
        CheckingPredicate("first_len_lt_6", lambda t: _alpha_len(_first_word(t)) < 6),
    "The sentence starts with a plural word":
        CheckingPredicate("starts_plural", lambda t: _is_plural(_first_word(t))),
    "The input starts with a plural word":
        CheckingPredicate("starts_plural", lambda t: _is_plural(_first_word(t))),
    "The last word ends with a vowel":
        CheckingPredicate("last_ends_vowel", _last_word_ends_vowel),
    "The last word ends with the letter e":
        CheckingPredicate("last_ends_e", lambda t: _last_word(t).endswith("e")),
    "The last word ends with a vowel sound":
        CheckingPredicate("last_ends_vowel_sound", _last_word_ends_vowel),
    "The last word is a noun":
        CheckingPredicate("last_noun", lambda t: _last_word(t) in _noun_words()),
    "The last word ends with the letter y":
        CheckingPredicate("last_ends_y", lambda t: _last_word(t).endswith("y")),
    "The last word is short (5 or fewer letters)":
        CheckingPredicate("last_len_le_5", lambda t: _alpha_len(_last_word(t)) <= 5),
    "The first word has 7 or more letters":
        CheckingPredicate("first_len_ge_7", lambda t: _alpha_len(_first_word(t)) >= 7),
    "The last word has 4 or fewer letters":
        CheckingPredicate("last_len_le_4", lambda t: _alpha_len(_last_word(t)) <= 4),
    "The first word has more syllables than the last":
        CheckingPredicate(
            "first_syllables_gt_last",
            lambda t: _count_syllables(_first_word(t)) > _count_syllables(_last_word(t)),
        ),
    "The first word's initial letter is in the first half of the alphabet (a-m)":
        CheckingPredicate("first_initial_a_m", lambda t: "a" <= _first_initial(t) <= "m"),
    "The first word is shorter than the second word":
        CheckingPredicate(
            "first_len_lt_second",
            lambda t: len(_words(t)) >= 2
            and _alpha_len(_words(t)[0]) < _alpha_len(_words(t)[1]),
        ),
    "The first two words start with the same letter":
        CheckingPredicate(
            "first_two_same_initial",
            lambda t: len(_words(t)) >= 2
            and _alpha_word(_words(t)[0])[:1] == _alpha_word(_words(t)[1])[:1],
        ),
    "The first two words start with adjacent letters of the alphabet":
        CheckingPredicate(
            "first_two_adjacent_initials",
            lambda t: len(_words(t)) >= 2
            and abs(
                ord((_alpha_word(_words(t)[0])[:1] or "\0"))
                - ord((_alpha_word(_words(t)[1])[:1] or "\0"))
            )
            == 1,
        ),
    "The input starts with an adjective":
        CheckingPredicate("starts_adj", lambda t: _first_word(t) in (_bank_set("ADJ_PLAIN") | _VISUAL_ADJ | _bank_set("COLORS"))),
    "The second word is a plural noun":
        CheckingPredicate(
            "second_plural_noun",
            lambda t: len(_words(t)) >= 2
            and _is_plural(_alpha_word(_words(t)[1]))
            and _alpha_word(_words(t)[1]) in _noun_words(),
        ),
    "Every word has more than 4 letters":
        CheckingPredicate(
            "all_words_len_gt_4",
            lambda t: bool(_words(t)) and all(_alpha_len(tok) > 4 for tok in _words(t)),
        ),
    "The input contains the word 'the'":
        CheckingPredicate("word_the", _pred_contains_word("the")),
    "The input contains no articles (a/an/the)":
        CheckingPredicate("no_articles", lambda t: not any(tok in _ARTICLES for tok in _toks(t))),
    "The average word length is at least 5 letters":
        CheckingPredicate("avg_word_len_ge_5", lambda t: _avg_alpha_len(t) >= 5),
    "Every word is a plural noun or a verb":
        CheckingPredicate(
            "all_plural_noun_or_verb",
            lambda t: bool(_words(t))
            and all(
                (_is_plural(_alpha_word(tok)) and _alpha_word(tok) in _noun_words())
                or _is_verb_proxy(_alpha_word(tok))
                for tok in _words(t)
            ),
        ),
    "The input begins with 'The'":
        CheckingPredicate("begins_The", lambda t: bool(_words(t)) and _words(t)[0] == "The"),
    "The input contains at least two articles (counting 'a' and 'the')":
        CheckingPredicate("articles_ge_2", lambda t: sum(tok in _ARTICLES for tok in _toks(t)) >= 2),
    # bank-membership semantic seeds
    "The input mentions a city":
        _membership("city", "NONNAME_PROPER"),
    "The input mentions a month":
        CheckingPredicate(
            "month",
            lambda t: any(
                tok in {
                    "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november",
                    "december",
                }
                for tok in _toks(t)
            ),
        ),
    "The input mentions a pet":
        _lexicon(
            "pet",
            frozenset({"cat", "dog", "hamster", "rabbit", "goldfish", "parrot",
                       "kitten", "puppy", "pony", "canary"}),
        ),
    "The input mentions a farm animal":
        _lexicon(
            "farm_animal",
            frozenset({"cow", "pig", "sheep", "goat", "horse", "chicken", "hen",
                       "duck", "goose", "donkey", "rooster", "lamb", "calf"}),
        ),
    "The input mentions a living thing":
        CheckingPredicate(
            "living_thing",
            lambda t: any(
                tok in (_bank_set("ANIMALS") | _PLANTS) for tok in _toks(t)
            ),
        ),
    "The input mentions something that can move on its own":
        CheckingPredicate(
            "self_moving",
            lambda t: any(tok in (_bank_set("ANIMALS") | _VEHICLES) for tok in _toks(t)),
        ),
    "The input mentions the color red":
        CheckingPredicate(
            "color_red",
            lambda t: any(tok in {"red", "crimson", "scarlet"} for tok in _toks(t)),
        ),
    "The input mentions clothing":
        _lexicon(
            "clothing",
            frozenset({"shirt", "dress", "hat", "coat", "shoes", "socks",
                       "jacket", "scarf", "gloves", "skirt", "trousers",
                       "sweater", "tie", "boots", "jeans"}),
        ),
    "The input mentions something bright or shiny":
        _lexicon(
            "bright_shiny",
            frozenset({"bright", "shiny", "glowing", "sparkling", "gleaming",
                       "glittering", "radiant", "luminous"}),
        ),
    # food / eating
    "The input mentions eating or drinking":
        CheckingPredicate(
            "eat_drink", lambda t: any(tok in _EAT_DRINK_VERBS for tok in _toks(t))
        ),
    "The input mentions a meal time (breakfast/lunch/dinner)":
        CheckingPredicate("meal_time", lambda t: any(tok in _MEAL_TIMES for tok in _toks(t))),
    "The input is about a restaurant":
        _lexicon(
            "restaurant",
            frozenset({"restaurant", "cafe", "diner", "bistro", "menu", "waiter",
                       "chef", "kitchen", "table", "reservation"}),
        ),
    "The input is about food or restaurants":
        CheckingPredicate(
            "food_or_restaurant",
            lambda t: groundtruth.label_of("food_topic", t)
            if "food_topic" in groundtruth.RULE_PREDICATES
            and groundtruth.RULE_PREDICATES["food_topic"].label_of is not None
            else any(
                tok in _FOOD_WORDS for tok in _toks(t)
            ),
        ),
    "The input mentions something you can buy":
        _lexicon(
            "buyable",
            frozenset({"buy", "bought", "shop", "store", "price", "cost", "sale",
                       "market", "money", "pay", "paid", "purchase"}),
        ),
    # sentiment / topic free seeds (best-effort lexical; check decides)
    "The input expresses a positive sentiment":
        CheckingPredicate(
            "positive_lex", lambda t: any(tok in _POSITIVE_WORDS for tok in _toks(t))
        ),
    "The input describes a past event":
        CheckingPredicate("past_event", lambda t: groundtruth.label_of("past_tense", t)),
    "The input is in the past tense":  # bespoke fallback (also a cross-rule head)
        CheckingPredicate("past_event", lambda t: groundtruth.label_of("past_tense", t)),
    "The input describes how something looks "
    "(predicate: contains any visual adjective incl. non-color)":
        CheckingPredicate(
            "visual_adj",
            lambda t: any(
                tok in (_VISUAL_ADJ | _bank_set("COLORS") | _bank_set("ADJ_PLAIN"))
                for tok in _toks(t)
            ),
        ),
    # broad topic seeds expected to land near 50% (kept deterministic; the
    # per-context check is what admits or globally drops them)
    "The input is about a hobby or pastime":
        _lexicon("hobby", frozenset(
            {"hobby", "game", "play", "music", "paint", "draw", "read", "garden",
             "hike", "swim", "run", "dance", "sing", "craft", "puzzle"})),
    "The input is about the outdoors":
        _lexicon("outdoors", frozenset(
            {"outdoors", "outside", "park", "forest", "mountain", "river", "lake",
             "field", "garden", "trail", "beach", "hill", "woods"})),
    "The input is about a letter or message":
        _lexicon("letter_msg", frozenset(
            {"letter", "message", "note", "email", "wrote", "write", "sent",
             "postcard", "envelope", "reply", "card"})),
    "The input is about household objects":
        _lexicon("household", frozenset(
            {"table", "chair", "lamp", "door", "window", "shelf", "cup", "plate",
             "spoon", "bowl", "vase", "clock", "mirror", "rug"})),
    "The input is about magic or fantasy":
        _lexicon("magic", frozenset(
            {"magic", "wizard", "dragon", "spell", "fairy", "witch", "unicorn",
             "potion", "enchanted", "ghost", "monster"})),
    "The input is grammatically incorrect":
        CheckingPredicate("ungrammatical", lambda t: False),  # data is all grammatical
    "The input mentions something dangerous":
        _lexicon("dangerous", frozenset(
            {"dangerous", "danger", "fire", "knife", "gun", "poison", "cliff",
             "storm", "accident", "fall", "burn", "drown"})),
    "The input mentions an object doing a human action":
        # REAL predicate (was a constant-False fallback because
        # physically_impossible is validator-derived with no recomputable
        # label_of — which made the >=25% disagreement gate VACUOUS and let this
        # near-synonym distractor through). "An object doing a human action" is
        # an inanimate/object subject that is the agent of a verb; reuse the
        # step-3 subject-inanimacy reading.
        CheckingPredicate("object_human_action", _art_physically_impossible),
}

_PLANTS = frozenset(
    {"tree", "flower", "rose", "grass", "bush", "fern", "oak", "pine", "tulip",
     "daisy", "ivy", "moss", "plant", "leaf", "vine", "shrub", "weed", "lily"}
)
_VEHICLES = frozenset(
    {"car", "truck", "bus", "train", "bike", "bicycle", "boat", "ship", "plane",
     "van", "scooter", "tram", "taxi", "motorcycle"}
)
_FOOD_WORDS = frozenset(
    {"bread", "soup", "rice", "pasta", "cheese", "apple", "cake", "meal", "pizza",
     "salad", "meat", "fruit", "egg", "sandwich", "coffee", "tea", "sugar",
     "butter", "chicken", "fish", "potato", "tomato", "chocolate"}
)
_POSITIVE_WORDS = frozenset(
    {"good", "great", "love", "loved", "wonderful", "excellent", "happy",
     "delightful", "best", "amazing", "enjoyed", "perfect", "beautiful", "nice",
     "pleased", "fantastic", "brilliant", "lovely", "glad", "favorite"}
)


# --- seed -> predicate resolution ----------------------------------------------

_CROSS_RULE_RE = re.compile(r"cross-rule canonical \((r\d+)\)")


def _seed_head(seed: str) -> str:
    """Text before the ' — ' rationale dash (em dash); the predicate key."""
    return seed.split(" — ")[0].strip()


def _rn_to_rule_id(extract: dict[str, Any]) -> dict[str, str]:
    """Map 'r9' -> 'passive_voice' via plan_number in the spec extract."""
    out: dict[str, str] = {}
    for rule_id, body in extract["rules"].items():
        out[f"r{body['plan_number']}"] = rule_id
    return out


def resolve_predicate(seed: str, rn_map: dict[str, str]) -> CheckingPredicate:
    """The checking predicate for one seed string.

    Cross-rule canonical seeds ("... cross-rule canonical (rN) ...") reuse the
    cited rule's groundtruth.label_of. Otherwise the seed HEAD must be in the
    bespoke registry. Raises MCBuildError (LOUD) for an un-checkable seed."""
    head = _seed_head(seed)
    m = _CROSS_RULE_RE.search(seed)
    if m:
        rn = m.group(1)
        if rn not in rn_map:
            raise MCBuildError(f"seed cites unknown cross-rule {rn!r}: {seed!r}")
        rule_id = rn_map[rn]
        if groundtruth.RULE_PREDICATES[rule_id].label_of is not None:
            return _cross_rule(rule_id)
        # cited rule is validator-derived (LLM-judged, e.g. r15
        # positive_sentiment): there is NO recomputable canonical predicate, so
        # fall back to a bespoke lexical checker for the seed head. The seed's
        # rationale (e.g. "labels ~everything False") is preserved by a lexical
        # proxy, and the per-context >= 25% check still decides admissibility.
        if head not in _BESPOKE:
            raise MCBuildError(
                f"seed cites {rn} ({rule_id}) which is validator-derived "
                f"(no recomputable predicate) and head {head!r} has no bespoke "
                f"fallback predicate: {seed!r}"
            )
        return _BESPOKE[head]
    if head not in _BESPOKE:
        raise MCBuildError(
            f"no checking predicate for distractor seed head {head!r} "
            f"(register it in mc._BESPOKE or tag it cross-rule)"
        )
    return _BESPOKE[head]


# --- option set assembly -------------------------------------------------------


@dataclass(frozen=True)
class Option:
    """One multiple-choice option (the true rule or a distractor)."""

    text: str
    is_true: bool
    predicate_key: str  # for duplicate-predicate detection across options
    seed: str | None  # the raw seed string for a distractor; None for the true option


@dataclass
class MCOptionSet:
    """The 8 unordered options for a rule, plus the per-context disagreement
    audit. ``disagreement`` holds ONLY the 7 CHOSEN distractors (seed ->
    per-context fractions, each >= 0.25). ``evaluated`` records every seed the
    builder scored (chosen + below-floor), and ``rejected`` why each skipped
    seed was dropped — for a transparent paid-run config."""

    rule_id: str
    options: list[Option]  # length 8; exactly one is_true
    context_seeds: list[int]
    disagreement: dict[str, list[float]] = field(default_factory=dict)  # CHOSEN only
    evaluated: dict[str, list[float]] = field(default_factory=dict)  # all scored seeds
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def true_option(self) -> Option:
        return next(o for o in self.options if o.is_true)


def _disagreement(pred: Callable[[str], bool], context: list[dict[str, Any]]) -> float:
    """Fraction of the shown examples where pred(text) != shown label."""
    n = len(context)
    return sum(1 for it in context if pred(it["text"]) != it["label"]) / n


def build_option_set(
    rule_id: str,
    extract: dict[str, Any],
    contexts: list[list[dict[str, Any]]],
    context_seeds: list[int],
) -> MCOptionSet:
    """Assemble the 8-option set for one rule against its 3 actual contexts.

    The TRUE option is the canonical_articulation. Distractors are drawn from
    mc_distractor_seeds in spec order, skipping any whose head is a
    banned_distractor, any seed extensionally equal to the true rule on EVERY
    context, and any that fails the >= 25% disagreement check in ANY context
    (global replacement: a failing seed is dropped for all contexts and the next
    pool seed is tried). Duplicate checking predicates are forbidden.
    """
    if len(contexts) != len(context_seeds):
        raise MCBuildError("contexts and context_seeds length mismatch")
    base_rule_id = canonical_rule_id(rule_id)
    rule = extract["rules"][base_rule_id]
    rn_map = _rn_to_rule_id(extract)

    # the TRUE option is rendered as a terse, distractor-register paraphrase of
    # the canonical_articulation — NOT the raw canonical text — so it cannot be
    # identified by surface form (see _TRUE_OPTION_TEXT / true_option_text).
    true_text = true_option_text(base_rule_id, rule)
    true_option = Option(true_text, True, f"true:{base_rule_id}", None)

    banned_heads = {b.strip() for b in rule.get("banned_distractors", [])}
    banned_heads |= BEHAVIORAL_NEAR_SYNONYM_BANS.get(base_rule_id, set())
    # the true rule's own label on each context, taken from the SHOWN labels
    # (verified to equal groundtruth where the rule is recomputable).
    _verify_true_labels(base_rule_id, contexts)

    chosen: list[Option] = []
    used_keys = {true_option.predicate_key}
    disagreement: dict[str, list[float]] = {}  # CHOSEN distractors only
    evaluated: dict[str, list[float]] = {}  # every seed we actually scored
    rejected: list[dict[str, Any]] = []

    for seed in rule["mc_distractor_seeds"]:
        if len(chosen) >= N_DISTRACTORS:
            break
        head = _seed_head(seed)
        if head in banned_heads:
            rejected.append({"seed": seed, "reason": "banned"})
            continue
        pred = resolve_predicate(seed, rn_map)
        if pred.key in used_keys:
            rejected.append({"seed": seed, "reason": f"duplicate predicate {pred.key}"})
            continue
        per_ctx = [_disagreement(pred.label_of, ctx) for ctx in contexts]
        evaluated[seed] = per_ctx
        if min(per_ctx) < DISAGREEMENT_FLOOR:
            rejected.append(
                {"seed": seed, "reason": "below 25% in a context", "per_context": per_ctx}
            )
            continue
        # display text is the normalized (terse) head; the seed (and thus the
        # checking predicate via resolve_predicate) is preserved unchanged.
        chosen.append(Option(normalize_display(head), False, pred.key, seed))
        used_keys.add(pred.key)
        disagreement[seed] = per_ctx  # only the CHOSEN distractor's audit

    if len(chosen) < N_DISTRACTORS:
        raise MCBuildError(
            f"rule {rule_id!r}: only {len(chosen)} of {N_DISTRACTORS} distractors "
            f"survived the >= {DISAGREEMENT_FLOOR:.0%} per-context check across "
            f"{len(contexts)} contexts (need a deeper seed pool). "
            f"rejected={rejected}"
        )

    options = [true_option, *chosen]
    if len(options) != N_OPTIONS:
        raise MCBuildError(f"rule {rule_id!r}: assembled {len(options)} options, need {N_OPTIONS}")
    # paranoia: the returned audit must contain exactly the 7 chosen distractors
    # and every entry must clear the floor in every context (the core guarantee).
    if len(disagreement) != N_DISTRACTORS or any(
        min(v) < DISAGREEMENT_FLOOR for v in disagreement.values()
    ):
        raise MCBuildError(
            f"rule {rule_id!r}: internal audit inconsistency "
            f"(disagreement dict {disagreement})"
        )
    return MCOptionSet(
        rule_id=rule_id,
        options=options,
        context_seeds=list(context_seeds),
        disagreement=disagreement,
        evaluated=evaluated,
        rejected=rejected,
    )


# --- format-tell audit (construct validity) ------------------------------------
#
# Proves the true option is NOT a surface-form outlier among the 8 options, so a
# model can only identify it by CONTENT. Three independent checks:
#   1. length band: true-option char length is within [min, max] of the
#      distractors, OR within FORMAT_TELL_LEN_FACTOR x the distractor mean (and
#      never the strict longest/shortest by a wide margin);
#   2. no giveaway token ('iff', 'labeled True', ...) appears in ANY option;
#   3. shared opening frame: the true option's opening frame ("The input ...")
#      is also carried by >= 1 distractor, so the true option is not the UNIQUE
#      carrier of its frame (distractors legitimately vary their opener — "The
#      word"/"Some word"/"Every word"/"The first" — so requiring ALL 8 to match
#      would be a property of the distractor pool, not a tell on the true option;
#      what matters is that the true option blends into the majority frame).

FORMAT_TELL_LEN_FACTOR = 1.5  # true length must be <= 1.5x distractor mean (and >= 1/1.5x)
_OPENING_FRAME_RE = re.compile(r"^the input(?:'s)?\b", re.IGNORECASE)


def _opening_frame(text: str) -> str:
    """The two-word opener used to group options ('the input', 'the word',
    'some word', 'every word', 'the first', ...), lower-cased."""
    toks = text.lower().split()
    if not toks:
        return ""
    return " ".join(toks[:2]) if len(toks) >= 2 else toks[0]


@dataclass(frozen=True)
class FormatTellReport:
    """Per-rule surface-form audit of an option set (see format_tell_report)."""

    rule_id: str
    true_len: int
    distractor_lens: list[int]
    len_min: int
    len_max: int
    len_mean: float
    true_in_band: bool          # min <= true_len <= max
    true_within_factor: bool    # 1/F * mean <= true_len <= F * mean
    is_longest_outlier: bool    # true is strictly longest AND > F * longest distractor
    is_shortest_outlier: bool   # true is strictly shortest AND < shortest distractor / F
    giveaway_options: list[str]  # option texts carrying a giveaway token
    true_frame: str              # the true option's opening frame
    true_frame_shared: bool      # >= 1 distractor shares the true option's frame
    true_starts_with_the_input: bool  # true option opens with "The input ..."

    @property
    def ok(self) -> bool:
        """True iff the true option is not a surface-form tell."""
        len_ok = (self.true_in_band or self.true_within_factor) and not (
            self.is_longest_outlier or self.is_shortest_outlier
        )
        return (
            len_ok
            and not self.giveaway_options
            and self.true_frame_shared
            and self.true_starts_with_the_input
        )


def format_tell_report(option_set: "MCOptionSet") -> FormatTellReport:
    """Audit one rule's 8 options for a residual format tell on the TRUE option."""
    true_opt = option_set.true_option()
    distractors = [o for o in option_set.options if not o.is_true]
    d_lens = sorted(len(o.text) for o in distractors)
    t_len = len(true_opt.text)
    d_min, d_max = d_lens[0], d_lens[-1]
    d_mean = sum(d_lens) / len(d_lens)

    # "longest by a wide margin": strictly longer than every distractor AND more
    # than FORMAT_TELL_LEN_FACTOR x the next-longest option's length.
    is_longest = t_len > d_max and t_len > FORMAT_TELL_LEN_FACTOR * d_max
    is_shortest = t_len < d_min and t_len * FORMAT_TELL_LEN_FACTOR < d_min

    lowered = [(o.text, o.text.lower()) for o in option_set.options]
    giveaways = [
        text for text, low in lowered if any(tok in low for tok in _GIVEAWAY_TOKENS)
    ]
    true_frame = _opening_frame(true_opt.text)
    frame_shared = any(_opening_frame(o.text) == true_frame for o in distractors)

    return FormatTellReport(
        rule_id=option_set.rule_id,
        true_len=t_len,
        distractor_lens=d_lens,
        len_min=d_min,
        len_max=d_max,
        len_mean=d_mean,
        true_in_band=d_min <= t_len <= d_max,
        true_within_factor=(d_mean / FORMAT_TELL_LEN_FACTOR) <= t_len <= (d_mean * FORMAT_TELL_LEN_FACTOR),
        is_longest_outlier=is_longest,
        is_shortest_outlier=is_shortest,
        giveaway_options=giveaways,
        true_frame=true_frame,
        true_frame_shared=frame_shared,
        true_starts_with_the_input=bool(_OPENING_FRAME_RE.match(true_opt.text)),
    )


def _verify_true_labels(rule_id: str, contexts: list[list[dict[str, Any]]]) -> None:
    """Where the true rule is recomputable, the SHOWN labels must equal
    groundtruth.label_of (the few-shot block must be self-consistent). Validator-
    derived rules (LLM-judged) are skipped — their shown label IS the truth."""
    base_rule_id = canonical_rule_id(rule_id)
    entry = groundtruth.RULE_PREDICATES.get(base_rule_id)
    if entry is None or entry.label_of is None:
        return
    for ci, ctx in enumerate(contexts):
        bad = [it["item_id"] for it in ctx if entry.label_of(it["text"]) != it["label"]]
        if bad:
            raise MCBuildError(
                f"rule {rule_id!r} context {ci}: shown labels disagree with "
                f"groundtruth.label_of for items {bad} — corrupt few-shot block"
            )


def load_contexts(
    rule_id: str,
    data_dir: str | Path,
    context_seeds: list[int],
    k: int = K_FEW_SHOT,
) -> list[list[dict[str, Any]]]:
    """The SAME 32-example few-shot contexts step-1 used for this rule
    (contexts.sample_context with the given seeds)."""
    items = load_items(Path(data_dir) / rule_id / "items.jsonl")
    return [sample_context(items, k=k, seed=s) for s in context_seeds]


# --- per-query option orders ---------------------------------------------------


@dataclass(frozen=True)
class MCQuery:
    """One multiple-choice query: a rule x context x option order. The option list is the
    SHUFFLED 8 options; ``true_letter`` is the letter of the true option."""

    rule_id: str
    context_index: int
    context_seed: int
    order_index: int
    order_seed: int
    options: list[Option]  # shuffled
    true_letter: str

    def lettered(self) -> list[tuple[str, Option]]:
        return list(zip(LETTERS, self.options))


def shuffle_options(options: list[Option], order_seed: int) -> tuple[list[Option], str]:
    """Deterministic shuffle of the 8 options; returns (shuffled, true_letter)."""
    shuffled = list(options)
    Random(order_seed).shuffle(shuffled)
    true_idx = next(i for i, o in enumerate(shuffled) if o.is_true)
    return shuffled, LETTERS[true_idx]


def build_queries(
    option_set: MCOptionSet, order_seed_base: int
) -> list[MCQuery]:
    """The 12 multiple-choice queries for a rule: 3 contexts x 4 orders. Order ``j`` uses
    seed ``order_seed_base + 10*context_index + j`` (distinct per context so the
    same rule's contexts never share a permutation by accident)."""
    queries: list[MCQuery] = []
    for ci, cseed in enumerate(option_set.context_seeds):
        for j in range(N_ORDERS):
            order_seed = order_seed_base + 10 * ci + j
            shuffled, true_letter = shuffle_options(option_set.options, order_seed)
            queries.append(
                MCQuery(
                    rule_id=option_set.rule_id,
                    context_index=ci,
                    context_seed=cseed,
                    order_index=j,
                    order_seed=order_seed,
                    options=shuffled,
                    true_letter=true_letter,
                )
            )
    return queries


def load_extract(path: str | Path = "data/spec_extract.json") -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))
