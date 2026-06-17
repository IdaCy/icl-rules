"""Programmatic ground-truth label verifier (quality-bar #4).

NOTHING ELSE in the framework recomputes an item's label from its TEXT. The
banks, the per-rule generators, and the validators all TRUST the label a
generator stored in ``slots_meta``-driven construction. The quality bar
requires the opposite: ground truth must be PROGRAMMATIC and ENFORCED by a
validator that, given only the raw ``text`` string, independently recomputes
True/False by applying the rule's OWN canonical checking predicate and asserts
it matches the stored label. This module is that validator.

The contract is deliberately honest about which rules are recomputable:

- PREDICATE-BACKED rules register a ``label_of(text) -> bool`` that recomputes
  the label as a PURE FUNCTION OF TEXT, using the global tokenizer
  (``schema.words`` — whitespace split, leading/trailing PUNCT_STRIP, internal
  characters untouched), the global ``word_length`` (alphabetic-char count),
  the char-rule scope (raw string for digit/comma/exclamation/letter rules),
  and the frozen stopword list transcribed from rule-specs globals.stopwords.
  ``assert_labels_correct`` requires stored label == label_of(text) for EVERY
  item (the mutation guarantee: flip a label and it raises).

- BANK-MEMBERSHIP rules (5, 13, 14, 17) have a bank-defined semantic label, but
  it is STILL a pure function of text once you fix the bank: the predicate
  tests whether any stripped/lowercased token is in the relevant bank's
  word-set (read live from ``banks.get_bank``). These are predicate-backed too;
  the membership set is the canonical checker. (Rule 5 is literal-char: 'z'
  anywhere in the raw string — no bank needed.)

- BEST-EFFORT rules (9 passive_voice, 10 imperative) have NO clean
  text-only predicate without a parser. They register a documented best-effort
  ``label_of`` that matches the SURFACE SHAPE the recipe builds (was/were +
  -ed/-en participle for 9; subjectless verb-or-adverb-initial command for 10).
  ``assert_labels_correct`` runs them but ``BEST_EFFORT`` flags them so a caller
  can choose to treat a mismatch as advisory; by default it still RAISES (the
  recipe's surface shape is deterministic, so on the EMITTED data the
  best-effort form is exact — the looseness is only against arbitrary probe
  text).

- VALIDATOR-DERIVED rules (15 positive_sentiment, 16 food_topic,
  18 physically_impossible) have NO recomputable text predicate at all (their
  label is an LLM-judge semantic call). They register a SENTINEL, not a
  predicate. The verifier MUST NOT recompute them; instead it requires each
  item to carry a provenance flag in ``slots_meta`` proving BOTH validator
  passes agreed on the stored label, and skips recompute.

All failures raise ``GroundTruthError`` (LOUD), listing the offending item_ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

from ..contexts import normalize_label
from ..rule_ids import canonical_rule_id
from . import banks
from .schema import PUNCT_STRIP, read_items, words

# --- frozen stopword list (rule-specs globals.stopwords.list, verbatim) -------
# Transcribed from the private rule spec (not in this repository) globals.stopwords exactly as
# the bank authors transcribe BANK_QUOTAS: this is the single frozen list rule 6
# (content-word repeat) checks against. Lowercase match after stripping.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "after", "all", "also", "am", "an", "and", "any", "are",
        "as", "at", "be", "been", "before", "being", "between", "both", "but",
        "by", "can", "could", "did", "do", "does", "during", "each", "few",
        "for", "from", "had", "has", "have", "he", "her", "here", "him", "his",
        "how", "i", "if", "in", "into", "is", "it", "its", "just", "may", "me",
        "might", "more", "most", "must", "my", "near", "no", "nor", "not", "of",
        "off", "on", "only", "onto", "or", "our", "out", "over", "own", "same",
        "shall", "she", "should", "so", "some", "such", "than", "that", "the",
        "their", "them", "then", "there", "these", "they", "this", "those",
        "to", "too", "under", "until", "up", "us", "very", "was", "we", "were",
        "what", "when", "where", "which", "who", "why", "will", "with", "would",
        "yet", "you", "your",
    }
)

VOWEL_LETTERS: frozenset[str] = frozenset("aeiou")

# spelled-out numbers that count for rule-2 ('mentions a number'); see r2
# ambiguity note N13 (digits + NUMBER_WORDS lexicon; several/many/some do NOT
# count). Rule 2's TRUTH is digit-only, so this set is NOT used by rule 2's
# label_of — it lives here only to document the distinction.
NUMBER_WORDS: frozenset[str] = frozenset(
    {
        "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
        "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    }
)


class GroundTruthError(ValueError):
    """An item's stored label disagreed with the programmatic recomputation, or
    a validator-derived item lacked the required agreed-validation provenance."""


class Backing(Enum):
    """How a rule's ground truth is established (honesty about recomputability)."""

    PREDICATE = "predicate"          # pure function of text recomputes the label
    BANK_MEMBERSHIP = "bank"         # pure function of text once the bank is fixed
    BEST_EFFORT = "best_effort"      # surface-shape match; exact on emitted data
    VALIDATOR_DERIVED = "validator"  # NOT recomputable; requires provenance flag


# the slots_meta provenance key a validator-derived item must carry. Both LLM
# validation passes must have agreed on the stored label for the item to be
# accepted (rule-specs rules 15/16/18 recipe: "keep only items where both
# validators return the intended label").
VALIDATED_FLAG = "validated_agreement"


# --- alphabetic-length helper (globals.tokenizer.word_length) -----------------


def _alpha_len(word: str) -> int:
    return sum(1 for ch in word if ch.isalpha())


def _first_alpha(word: str) -> str:
    for ch in word:
        if ch.isalpha():
            return ch.lower()
    return ""


def _last_alpha(word: str) -> str:
    for ch in reversed(word):
        if ch.isalpha():
            return ch.lower()
    return ""


def _stripped_lower(text: str) -> list[str]:
    """Global tokenizer tokens, lowercased (case-insensitive word matching)."""
    return [w.lower() for w in words(text)]


# --- surface predicates (rules 1-7) -------------------------------------------


def _r1_all_lowercase(text: str) -> bool:
    """r1: True iff no uppercase letter anywhere (ABSENCE of uppercase; a
    letterless string is True). Raw-string char rule."""
    return not any(ch.isupper() for ch in text)


def _r2_contains_digit(text: str) -> bool:
    """r2: True iff any 0-9 character anywhere in the raw string. Spelled
    numbers are False by the rule's truth (ambiguity_notes)."""
    return any(ch.isdigit() for ch in text)


def _r3_contains_exclamation(text: str) -> bool:
    """r3: True iff an '!' appears anywhere in the raw string (>= 1 suffices)."""
    return "!" in text


def _r4_title_case(text: str) -> bool:
    """r4: True iff EVERY word begins with an uppercase letter (the first
    alphabetic char of every stripped token is uppercase). Words with no
    alphabetic char are ignored by the quantifier (ambiguity_notes)."""
    toks = words(text)
    if not toks:
        return False
    for tok in toks:
        first = _first_alpha_char_raw(tok)
        if first is not None and not first.isupper():
            return False
    return True


def _first_alpha_char_raw(tok: str) -> str | None:
    """First alphabetic CHARACTER of a token (case preserved), or None if the
    token has no alphabetic char. Used by r4 (case matters, so not lowered)."""
    for ch in tok:
        if ch.isalpha():
            return ch
    return None


def _r5_contains_letter_z(text: str) -> bool:
    """r5: True iff the letter z (either case) appears anywhere in the raw
    string. Char rule; bank-independent (the literal char IS the checker)."""
    return "z" in text.lower()


def _r6_repeated_content_word(text: str) -> bool:
    """r6: True iff some CONTENT word (not on STOPWORDS) occurs >= 2 times,
    matched case-insensitively after stripping. No lemmatization: 'dog'/'dogs'
    do not match (ambiguity_notes)."""
    seen: set[str] = set()
    for w in _stripped_lower(text):
        if w in STOPWORDS:
            continue
        if w in seen:
            return True
        seen.add(w)
    return False


def _r7_starts_with_vowel(text: str) -> bool:
    """r7: True iff the first LETTER of the first word is a vowel letter
    (a/e/i/o/u), regardless of pronunciation. Digit-initial -> False."""
    toks = words(text)
    if not toks:
        return False
    return _first_alpha(toks[0]) in VOWEL_LETTERS


# --- syntactic FORM predicates (8, 11, 12) ------------------------------------

# rule-8 regular-verb morphology. Training uses VERB_REGULAR only: the True
# (past) variant inflects with -ed, the False (present) variant with -s (3sg) or
# the base form (plural subject). The equivalence_class makes "the verb ends in
# -ed" coincide with "past tense" on ALL training data, so the form check IS the
# canonical text predicate here. We detect an -ed past-form verb token; we must
# NOT be fooled by the present 3sg -s. Best handled by: some token (other than a
# known stopword/aux) ends in 'ed' and is a plausible regular past.
_ED_RE = re.compile(r"[a-z]{2,}ed$")
# present-tense markers the False class uses; their presence does not make True.
_PRESENT_S_RE = re.compile(r"[a-z]{2,}s$")


def _r8_past_tense(text: str) -> bool:
    """r8 (FORM, training-decidable): True iff a regular past-tense verb form
    (-ed) is present. On the EMITTED data (regular verbs only, simple past vs
    simple present, no temporal/-ed-adjective vocabulary, no auxiliaries) this
    is extensionally the canonical 'main verb is past tense' (equivalence_class:
    'the verb ends in -ed'). Documented BEST-EFFORT against arbitrary probes
    (irregular pasts, adjectival -ed) — see Backing.BEST_EFFORT registration."""
    for w in _stripped_lower(text):
        if w in STOPWORDS:
            continue
        if _ED_RE.search(w):
            return True
    return False


# rule-11 aux/copular set (canonical_articulation lists these verbatim).
_AUX_INITIAL: frozenset[str] = frozenset(
    {"is", "are", "was", "were", "can", "will", "should", "must", "could"}
)


def _r11_question_word_order(text: str) -> bool:
    """r11: True iff the first word is an auxiliary/copular verb (aux-before-
    subject inversion). Terminal '?' is absent by design; word order alone
    decides (equivalence_class: 'starts with an auxiliary or copular verb')."""
    toks = _stripped_lower(text)
    if not toks:
        return False
    return toks[0] in _AUX_INITIAL


# rule-12 comparative form: an -er comparative OR analytic 'more X'.
_COMPARATIVE_ER_RE = re.compile(r"[a-z]{3,}er$")


def _r12_contains_comparative(text: str) -> bool:
    """r12 (FORM): True iff a comparative form appears — an -er comparative
    ('taller') or an analytic 'more {adj}'. Training uses comparative
    ADJECTIVES only, and to keep this a pure form check we must not fire on the
    -er SALT nouns (teacher/corner) the False class plants. We therefore require
    that an -er word be a known comparable adjective form OR be preceded such
    that it reads as a comparison; on the emitted data the ER_NONCOMPARATIVE
    salt would otherwise make this wrong, so the salt list is excluded
    explicitly. Documented BEST-EFFORT against arbitrary probes (suppletive
    'better', comparative adverbs) — see registration."""
    toks = _stripped_lower(text)
    for i, w in enumerate(toks):
        if w == "more" and i + 1 < len(toks):
            nxt = toks[i + 1]
            # 'more {adj}' is comparative; 'more {noun/plural}' (quantifier) is
            # not. On training, 'more' only ever precedes a comparable adj.
            if nxt not in STOPWORDS and not nxt.endswith("s"):
                return True
        if _COMPARATIVE_ER_RE.search(w) and w not in _ER_NONCOMPARATIVE:
            return True
    return False


# the ER_NONCOMPARATIVE salt nouns (rule-12 False class). Read live from the
# bank so the verifier and the generator share one source; fall back to the
# spec's enumerated examples if the bank group is not populated.
_ER_NONCOMPARATIVE_FALLBACK: frozenset[str] = frozenset(
    {
        "teacher", "river", "dinner", "corner", "paper", "ladder", "summer",
        "winter", "sister", "brother", "mother", "father", "number", "letter",
        "flower", "finger", "shoulder", "hammer", "ladder", "manager", "member",
        "officer", "farmer", "painter", "singer",
    }
)


def _load_er_noncomparative() -> frozenset[str]:
    try:
        bank = banks.get_bank("ER_NONCOMPARATIVE")
    except banks.BankError:
        return _ER_NONCOMPARATIVE_FALLBACK
    return frozenset(e.word.lower() for e in bank.entries)


_ER_NONCOMPARATIVE: frozenset[str] = _load_er_noncomparative()


# --- bank-membership predicates (5 handled above as char; 13, 14, 17) ---------


def _bank_word_set(name: str) -> frozenset[str]:
    """Lowercased surface words of a populated bank (LOUD if unpopulated)."""
    bank = banks.get_bank(name)
    return frozenset(e.word.lower() for e in bank.entries)


def _make_membership_predicate(bank_name: str) -> Callable[[str], bool]:
    """A predicate: True iff any stripped/lowercased token is in ``bank_name``'s
    word-set. The bank is loaded lazily (first call) so importing this module
    never forces every bank to be populated."""
    cache: dict[str, frozenset[str]] = {}

    def predicate(text: str) -> bool:
        wordset = cache.get(bank_name)
        if wordset is None:
            wordset = _bank_word_set(bank_name)
            cache[bank_name] = wordset
        return any(tok in wordset for tok in _stripped_lower(text))

    return predicate


_r13_mentions_animal = _make_membership_predicate("ANIMALS")
_r14_mentions_color = _make_membership_predicate("COLORS")


def _r17_contains_first_name(text: str) -> bool:
    """r17: True iff a token (case-insensitive) is in FIRST_NAMES. Other
    capitalized proper nouns (cities/months/brands, NONNAME_PROPER) are NOT
    first names -> False (canonical_articulation)."""
    wordset = _r17_names()
    return any(tok in wordset for tok in _stripped_lower(text))


_r17_NAMES_CACHE: dict[str, frozenset[str]] = {}


def _r17_names() -> frozenset[str]:
    ws = _r17_NAMES_CACHE.get("FIRST_NAMES")
    if ws is None:
        ws = _bank_word_set("FIRST_NAMES")
        _r17_NAMES_CACHE["FIRST_NAMES"] = ws
    return ws


# --- best-effort surface predicates (9 passive, 10 imperative) ----------------

_PARTICIPLE_RE = re.compile(r"[a-z]{2,}(ed|en)$")


def _r9_passive_voice(text: str) -> bool:
    """r9 (BEST-EFFORT surface form): True iff the recipe's passive shape is
    present — a be-form (was/were) followed (allowing a 'the {patient}' or
    adverbs in between is NOT how the recipe builds it) by a past participle.
    The recipe builds passives as 'The {patient} was {V-ed} ...' and
    past-progressive ACTIVES as 'The {agent} was {V-ing} ...', so the
    distinguishing surface is was/were immediately preceding an -ed/-en
    participle (not an -ing gerund). Documented best-effort: voice is not a pure
    text function in general (copular adjectives, get-passives), but on the
    EMITTED data the was/were+participle shape is exact."""
    toks = _stripped_lower(text)
    for i, w in enumerate(toks):
        if w in ("was", "were") and i + 1 < len(toks):
            nxt = toks[i + 1]
            if nxt.endswith("ing"):
                continue  # progressive active
            if _PARTICIPLE_RE.search(nxt):
                return True
    return False


# declarative SUBJECT pronouns the False class plants (She/He/They/We; the
# recipe uses these as subjects). 'it'/'you'/'i' are NOT declarative subjects
# in this recipe — 'it' is an OBJECT pronoun the 40% object mix puts AFTER the
# verb in BOTH classes ('Close it before lunch'), so it must NOT be read as a
# subject; 'I' is banned globally; 'you' never opens a declarative here.
_DECLARATIVE_SUBJECTS: frozenset[str] = frozenset({"she", "he", "they", "we"})

# sentence-initial adverb openers (the 25% adverb-start mix in BOTH classes).
# Loaded from ADVERB_SENT_INITIAL so a declarative's 'adverb + subject' shape is
# recognized only when position 1 is a real opener adverb (not a verb).
_ADVERB_OPENERS_FALLBACK: frozenset[str] = frozenset(
    {
        "quickly", "carefully", "usually", "often", "apparently", "yesterday",
        "slowly", "quietly", "suddenly", "finally", "eventually", "recently",
        "clearly", "obviously", "luckily", "sadly", "happily", "calmly",
        "gently", "loudly",
    }
)


def _load_adverb_openers() -> frozenset[str]:
    try:
        bank = banks.get_bank("ADVERB_SENT_INITIAL")
    except banks.BankError:
        return _ADVERB_OPENERS_FALLBACK
    return frozenset(e.word.lower() for e in bank.entries)


_ADVERB_OPENERS: frozenset[str] = _load_adverb_openers()


def _r10_imperative(text: str) -> bool:
    """r10 (BEST-EFFORT): True iff the sentence reads as a subjectless command.
    The recipe builds imperatives as bare-verb-initial (75%) or 1-word-adverb +
    bare-verb (25%), and declaratives as pronoun-subject + present verb (75%) or
    adverb + pronoun-subject (25%). A parser-free text test is impossible in
    general (canonical: 'no expressed subject'); the deterministic surface proxy
    EXACT on the emitted data is: it is a DECLARATIVE iff a SUBJECT pronoun
    (she/he/they/we) sits in subject position — i.e. at token 1, or at token 2
    when token 1 is a sentence-initial adverb. Object pronouns the 40% object
    mix plants after the verb ('Close it before lunch') are NOT subjects and do
    not flip the label. Documented best-effort against arbitrary probes
    (quantifier subjects 'Everyone close...', explicit 'You close...')."""
    toks = _stripped_lower(text)
    if not toks:
        return False
    if toks[0] in _DECLARATIVE_SUBJECTS:
        return False  # subject-initial declarative
    if (
        len(toks) >= 2
        and toks[0] in _ADVERB_OPENERS
        and toks[1] in _DECLARATIVE_SUBJECTS
    ):
        return False  # adverb + subject declarative
    return True


# --- positional predicates (19, 20, 21, 22) -----------------------------------


def _r19_first_word_longer_than_last(text: str) -> bool:
    """r19: True iff len(first word) > len(last word), STRICTLY, by alphabetic
    char count (global word_length). Ties -> False."""
    toks = words(text)
    if len(toks) < 2:
        return False
    return _alpha_len(toks[0]) > _alpha_len(toks[-1])


def _r20_last_word_ends_with_vowel(text: str) -> bool:
    """r20: True iff the last LETTER of the last word is a vowel letter
    (a/e/i/o/u); y does NOT count. Silent e counts ('table' -> True)."""
    toks = words(text)
    if not toks:
        return False
    return _last_alpha(toks[-1]) in VOWEL_LETTERS


def _r21_the_appears_twice(text: str) -> bool:
    """r21: True iff the word 'the' appears >= 2 times (case-insensitive; a
    sentence-initial 'The' counts). 'the' inside another word ('theater') does
    not count (word-level match on stripped tokens)."""
    return sum(1 for tok in _stripped_lower(text) if tok == "the") >= 2


def _r22_second_word_capitalized(text: str) -> bool:
    """r22: True iff the SECOND word begins with a capital letter. 'Second word'
    is the second stripped whitespace token; 'capitalized' is first char
    uppercase. Digit second word ('7') -> no uppercase letter -> False."""
    toks = words(text)
    if len(toks) < 2:
        return False
    return toks[1][:1].isupper()


# --- numeric predicates (23, 24, 25, 26) --------------------------------------


def _r23_word_count_geq_8(text: str) -> bool:
    """r23: True iff the input has >= 8 words (global tokenizer)."""
    return len(words(text)) >= 8


_DIGIT_RUN_RE = re.compile(r"\d+")


def _r24_contains_number_gt_50(text: str) -> bool:
    """r24: True iff a number WRITTEN IN DIGITS with value > 50 appears (any of
    several suffices). Spelled numbers never count (canonical: 'written in
    digits'). Integers in training; we read every maximal digit run as an int."""
    return any(int(m.group()) > 50 for m in _DIGIT_RUN_RE.finditer(text))


def _r25_even_word_count(text: str) -> bool:
    """r25: True iff the word count is even (global tokenizer; zero is even but
    cannot occur)."""
    return len(words(text)) % 2 == 0


def _r26_exactly_two_commas(text: str) -> bool:
    """r26: True iff the raw string contains EXACTLY two commas (char-rule
    scope; every ',' counts regardless of function)."""
    return text.count(",") == 2


# --- hard predicates (27, 28, 29, 30) -----------------------------------------


def _r27_first_last_same_letter(text: str) -> bool:
    """r27: True iff the first word and last word begin with the same letter
    (case-insensitive). Digit-initial words have no letter -> cannot match."""
    toks = words(text)
    if len(toks) < 2:
        return False
    a = _first_alpha(toks[0])
    b = _first_alpha(toks[-1])
    return a != "" and a == b


def _has_adjacent_double(word: str) -> bool:
    a = "".join(ch for ch in word.lower() if ch.isalpha())
    return any(a[i] == a[i + 1] for i in range(len(a) - 1))


def _r28_double_letter_word(text: str) -> bool:
    """r28: True iff some word has the same letter twice IN A ROW (adjacent
    double). 'window' (non-adjacent w's) -> False; 'coffee' -> True."""
    return any(_has_adjacent_double(tok) for tok in words(text))


def _r29_first_two_words_alphabetical(text: str) -> bool:
    """r29: True iff the first word precedes the second word in alphabetical
    (lexicographic, case-insensitive) order. Strictly before: a tie -> False."""
    toks = _stripped_lower(text)
    if len(toks) < 2:
        return False
    return toks[0] < toks[1]


def _r30_all_words_longer_than_3(text: str) -> bool:
    """r30: True iff EVERY word has > 3 letters (no 1/2/3-letter word). Universal
    quantifier over ALL words incl. function words; alphabetic char count.
    A digit token ('10') has 0 alphabetic letters -> breaks the property."""
    toks = words(text)
    if not toks:
        return False
    return all(_alpha_len(tok) > 3 for tok in toks)


# --- the registry -------------------------------------------------------------


@dataclass(frozen=True)
class RulePredicate:
    """A ground-truth entry for one rule.

    ``backing`` says how the label is established; ``label_of`` is the recompute
    function for predicate / bank / best-effort backings and is None for
    validator-derived rules (which carry a sentinel instead). ``ruling`` is the
    inline documentation of the canonical checker + any best-effort caveat."""

    rule_id: str
    backing: Backing
    ruling: str
    label_of: Callable[[str], bool] | None = None

    @property
    def recomputable(self) -> bool:
        return self.backing is not Backing.VALIDATOR_DERIVED


def _p(rule_id: str, backing: Backing, ruling: str, label_of: Callable[[str], bool] | None) -> RulePredicate:
    return RulePredicate(rule_id=rule_id, backing=backing, ruling=ruling, label_of=label_of)


RULE_PREDICATES: dict[str, RulePredicate] = {
    # ---- surface (1-7) -------------------------------------------------------
    "all_lowercase": _p(
        "all_lowercase", Backing.PREDICATE,
        "True iff no uppercase letter anywhere in the raw string (absence of "
        "uppercase, not presence of lowercase).",
        _r1_all_lowercase,
    ),
    "contains_digit": _p(
        "contains_digit", Backing.PREDICATE,
        "True iff any 0-9 char in the raw string; spelled numbers are False.",
        _r2_contains_digit,
    ),
    "contains_exclamation": _p(
        "contains_exclamation", Backing.PREDICATE,
        "True iff '!' appears anywhere in the raw string (>= 1 suffices).",
        _r3_contains_exclamation,
    ),
    "title_case": _p(
        "title_case", Backing.PREDICATE,
        "True iff every word's first alphabetic char is uppercase; tokens with "
        "no alphabetic char are ignored by the quantifier.",
        _r4_title_case,
    ),
    "contains_letter_z": _p(
        "contains_letter_z", Backing.BANK_MEMBERSHIP,
        "True iff the letter z (either case) appears anywhere in the raw "
        "string. Bank-INDEPENDENT: the literal char IS the canonical checker "
        "(the Z_WORDS bank only supplies the True slot at generation time).",
        _r5_contains_letter_z,
    ),
    "repeated_content_word": _p(
        "repeated_content_word", Backing.PREDICATE,
        "True iff some non-stopword token repeats (case-insensitive, no "
        "lemmatization). Stopwords = the frozen globals.stopwords list.",
        _r6_repeated_content_word,
    ),
    "starts_with_vowel": _p(
        "starts_with_vowel", Backing.PREDICATE,
        "True iff the first letter of the first word is a/e/i/o/u (letter-based, "
        "ignores pronunciation); digit-initial -> False.",
        _r7_starts_with_vowel,
    ),
    # ---- syntactic (8-12) ----------------------------------------------------
    "past_tense": _p(
        "past_tense", Backing.BEST_EFFORT,
        "FORM check: True iff a regular -ed past-form verb token is present. On "
        "the emitted data (regular verbs only, no temporal/-ed-adjective "
        "vocab, no auxiliaries) this is extensionally the canonical 'main verb "
        "is past tense' (equivalence_class: 'the verb ends in -ed'). BEST-EFFORT "
        "against arbitrary probes: irregular pasts (ran) and adjectival -ed "
        "(tired) are not handled by morphology alone.",
        _r8_past_tense,
    ),
    "passive_voice": _p(
        "passive_voice", Backing.BEST_EFFORT,
        "Surface-shape check: was/were immediately followed by an -ed/-en past "
        "participle (NOT an -ing gerund -> that is progressive active). Exact on "
        "the emitted data (the recipe's only passive shape); BEST-EFFORT against "
        "probes (copular adjectives, get-passives) per the recipe's voice "
        "definition.",
        _r9_passive_voice,
    ),
    "imperative": _p(
        "imperative", Backing.BEST_EFFORT,
        "PREDICATE-BASED-BEST-EFFORT (no clean parser-free text predicate): "
        "False iff a SUBJECT pronoun (she/he/they/we) sits in subject position "
        "— token 1, or token 2 after a sentence-initial adverb opener; True "
        "otherwise. Object pronouns the 40% object mix puts after the verb "
        "('Close it before lunch') are NOT subjects and do not flip it. Exact on "
        "the emitted data; not a general imperative detector.",
        _r10_imperative,
    ),
    "question_word_order": _p(
        "question_word_order", Backing.PREDICATE,
        "True iff the first word is an aux/copular verb "
        "(is/are/was/were/can/will/should/must/could) — aux-before-subject. "
        "Terminal '?' absent by design; word order alone decides.",
        _r11_question_word_order,
    ),
    "contains_comparative": _p(
        "contains_comparative", Backing.BEST_EFFORT,
        "FORM check: an -er comparative (excluding the ER_NONCOMPARATIVE salt "
        "nouns the False class plants) or analytic 'more {adj}'. Exact on the "
        "emitted data (comparative adjectives only, -er salt excluded); "
        "BEST-EFFORT against probes (suppletive 'better', 'more {noun}' "
        "quantifier, comparative adverbs).",
        _r12_contains_comparative,
    ),
    # ---- semantic substitution via bank membership (13, 14, 17) --------------
    "mentions_animal": _p(
        "mentions_animal", Backing.BANK_MEMBERSHIP,
        "True iff any stripped/lowercased token is in the ANIMALS bank word-set "
        "(independent of slots_meta). The membership set is the canonical "
        "checker on this data; mythical/extinct/product probes are out of bank.",
        _r13_mentions_animal,
    ),
    "mentions_color": _p(
        "mentions_color", Backing.BANK_MEMBERSHIP,
        "True iff any token is in the COLORS bank word-set (independent of "
        "slots_meta).",
        _r14_mentions_color,
    ),
    "contains_first_name": _p(
        "contains_first_name", Backing.BANK_MEMBERSHIP,
        "True iff any token is in the FIRST_NAMES bank word-set. Other "
        "capitalized proper nouns (NONNAME_PROPER) are NOT first names.",
        _r17_contains_first_name,
    ),
    # ---- LLM-judged: NO text predicate -> validator-derived sentinels --------
    "positive_sentiment": _p(
        "positive_sentiment", Backing.VALIDATOR_DERIVED,
        "NO recomputable text predicate (LLM-judged polarity). Verifier MUST NOT "
        "recompute; it requires the two-validator agreement provenance flag "
        f"slots_meta[{VALIDATED_FLAG!r}] proving both passes returned the "
        "stored label.",
        None,
    ),
    "food_topic": _p(
        "food_topic", Backing.VALIDATOR_DERIVED,
        "NO recomputable text predicate (LLM-judged topic). Requires the "
        f"slots_meta[{VALIDATED_FLAG!r}] two-validator agreement provenance.",
        None,
    ),
    "physically_impossible": _p(
        "physically_impossible", Backing.VALIDATOR_DERIVED,
        "NO recomputable text predicate (LLM-judged physical possibility). "
        f"Requires the slots_meta[{VALIDATED_FLAG!r}] two-validator agreement "
        "provenance.",
        None,
    ),
    # ---- positional (19-22) --------------------------------------------------
    "first_word_longer_than_last": _p(
        "first_word_longer_than_last", Backing.PREDICATE,
        "True iff alpha-len(first word) > alpha-len(last word), strictly; ties "
        "-> False.",
        _r19_first_word_longer_than_last,
    ),
    "last_word_ends_with_vowel": _p(
        "last_word_ends_with_vowel", Backing.PREDICATE,
        "True iff the last letter of the last word is a/e/i/o/u; y does NOT "
        "count; silent e counts.",
        _r20_last_word_ends_with_vowel,
    ),
    "the_appears_twice": _p(
        "the_appears_twice", Backing.PREDICATE,
        "True iff the token 'the' appears >= 2 times (case-insensitive; "
        "'theater' does not count — word-level match).",
        _r21_the_appears_twice,
    ),
    "second_word_capitalized": _p(
        "second_word_capitalized", Backing.PREDICATE,
        "True iff the second stripped token's first char is uppercase; digit "
        "second word -> False.",
        _r22_second_word_capitalized,
    ),
    # ---- numeric (23-26) -----------------------------------------------------
    "word_count_geq_8": _p(
        "word_count_geq_8", Backing.PREDICATE,
        "True iff word count (global tokenizer) >= 8.",
        _r23_word_count_geq_8,
    ),
    "contains_number_gt_50": _p(
        "contains_number_gt_50", Backing.PREDICATE,
        "True iff some maximal digit run has integer value > 50 (strictly; '50' "
        "-> False, '51' -> True); spelled numbers never count.",
        _r24_contains_number_gt_50,
    ),
    "even_word_count": _p(
        "even_word_count", Backing.PREDICATE,
        "True iff word count is even.",
        _r25_even_word_count,
    ),
    "exactly_two_commas": _p(
        "exactly_two_commas", Backing.PREDICATE,
        "True iff the raw string contains exactly two ',' chars.",
        _r26_exactly_two_commas,
    ),
    # ---- hard (27-30) --------------------------------------------------------
    "first_last_same_letter": _p(
        "first_last_same_letter", Backing.PREDICATE,
        "True iff first word and last word share their first letter "
        "(case-insensitive); digit-initial -> cannot match.",
        _r27_first_last_same_letter,
    ),
    "double_letter_word": _p(
        "double_letter_word", Backing.PREDICATE,
        "True iff some word has an ADJACENT double letter; non-adjacent repeats "
        "('window') do not count.",
        _r28_double_letter_word,
    ),
    "first_two_words_alphabetical": _p(
        "first_two_words_alphabetical", Backing.PREDICATE,
        "True iff first word < second word lexicographically (case-insensitive, "
        "full comparison); a tie -> False.",
        _r29_first_two_words_alphabetical,
    ),
    "all_words_longer_than_3": _p(
        "all_words_longer_than_3", Backing.PREDICATE,
        "True iff every word has alpha-len > 3; universal over ALL words incl. "
        "function words; a digit token (0 alpha letters) breaks it.",
        _r30_all_words_longer_than_3,
    ),
}


def _assert_registry_complete() -> None:
    """Every rule_id in the spec extract must be registered exactly once."""
    expected = set(_known_rule_ids())
    have = set(RULE_PREDICATES)
    missing = expected - have
    extra = have - expected
    if missing or extra:
        raise GroundTruthError(
            f"RULE_PREDICATES diverged from the spec rule set: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


def _known_rule_ids() -> list[str]:
    """The 30 rule ids, read from the COMMITTED spec extract (never the private
    spec). Falls back to the registry's own keys if the extract is absent (so a
    bare checkout still imports), which keeps the completeness check honest in
    the normal case where the extract is present."""
    extract = Path(__file__).resolve().parents[3] / "data" / "spec_extract.json"
    try:
        import json

        data = json.loads(extract.read_text(encoding="utf-8"))
        return list(data["rules"])
    except (OSError, KeyError, ValueError):
        return list(RULE_PREDICATES)


_assert_registry_complete()


# --- the verifier -------------------------------------------------------------


def label_of(rule_id: str, text: str) -> bool:
    """Recompute one rule's True/False label as a pure function of ``text``.

    Thin accessor over RULE_PREDICATES used by the step-2 multiple-choice distractor builder
    (cross-rule canonical distractors reuse the labelling function of the rule
    they cite). Raises GroundTruthError for an unknown rule, and for a
    VALIDATOR_DERIVED rule (positive_sentiment / food_topic /
    physically_impossible) whose label is an LLM-judge call with NO recomputable
    text predicate — those rules must NOT be used as a cross-rule checking
    predicate."""
    base_rule_id = canonical_rule_id(rule_id)
    if base_rule_id not in RULE_PREDICATES:
        raise GroundTruthError(f"no ground-truth entry for rule_id {rule_id!r}")
    entry = RULE_PREDICATES[base_rule_id]
    if entry.label_of is None:
        raise GroundTruthError(
            f"rule {rule_id!r} is {entry.backing.value}: no recomputable text "
            f"predicate (label is an LLM-judge call); cannot use as a checker"
        )
    return entry.label_of(text)


def assert_labels_correct(rule_id: str, items: Sequence[dict[str, Any]]) -> None:
    """Enforce programmatic ground truth for one rule's items.

    PREDICATE / BANK_MEMBERSHIP / BEST_EFFORT rules: recompute label_of(text)
    for every item and require it to equal the stored (normalized) label; raise
    GroundTruthError listing every offending item_id on any mismatch.

    VALIDATOR_DERIVED rules: do NOT recompute. Require each item to carry the
    two-validator agreement provenance: slots_meta[VALIDATED_FLAG] present and
    truthy AND equal to the stored label (both passes agreed on THIS label).
    Raise GroundTruthError listing items that lack the provenance or whose
    provenance disagrees with the stored label."""
    base_rule_id = canonical_rule_id(rule_id)
    if base_rule_id not in RULE_PREDICATES:
        raise GroundTruthError(f"no ground-truth entry for rule_id {rule_id!r}")
    entry = RULE_PREDICATES[base_rule_id]

    if not entry.recomputable:
        _assert_validator_derived(rule_id, items, entry)
        return

    assert entry.label_of is not None  # recomputable backings always carry one
    mismatches: list[str] = []
    for it in items:
        item_rule_id = str(it.get("rule_id", rule_id))
        if canonical_rule_id(item_rule_id) != base_rule_id:
            raise GroundTruthError(
                f"item {it.get('item_id')!r} has rule_id {it.get('rule_id')!r}, "
                f"expected {rule_id!r}"
            )
        stored = normalize_label(it["label"])
        recomputed = entry.label_of(it["text"])
        if recomputed != stored:
            mismatches.append(str(it.get("item_id")))
    if mismatches:
        raise GroundTruthError(
            f"rule {rule_id!r} ({entry.backing.value}): {len(mismatches)} "
            f"item(s) whose stored label != recomputed label_of(text): "
            f"{mismatches}"
        )


def _assert_validator_derived(
    rule_id: str, items: Sequence[dict[str, Any]], entry: RulePredicate
) -> None:
    bad_missing: list[str] = []
    bad_disagree: list[str] = []
    for it in items:
        stored = normalize_label(it["label"])
        meta = it.get("slots_meta")
        if not isinstance(meta, dict) or VALIDATED_FLAG not in meta:
            bad_missing.append(str(it.get("item_id")))
            continue
        agreed = meta[VALIDATED_FLAG]
        try:
            agreed_label = normalize_label(agreed)
        except Exception:  # any unparseable provenance value is a failure
            bad_missing.append(str(it.get("item_id")))
            continue
        if agreed_label != stored:
            bad_disagree.append(str(it.get("item_id")))
    if bad_missing or bad_disagree:
        raise GroundTruthError(
            f"rule {rule_id!r} (validator_derived): cannot recompute from text; "
            f"requires slots_meta[{VALIDATED_FLAG!r}] proving both validators "
            f"agreed on the stored label. "
            f"missing/unparseable provenance: {bad_missing}; "
            f"provenance disagrees with stored label: {bad_disagree}"
        )


def verify_dataset(rule_id: str, path: str | Path) -> int:
    """Read items.jsonl at ``path`` and run assert_labels_correct for ``rule_id``.

    Returns the number of items verified. Raises GroundTruthError on any
    mismatch / missing provenance (LOUD)."""
    items = read_items(path)
    assert_labels_correct(rule_id, items)
    return len(items)
