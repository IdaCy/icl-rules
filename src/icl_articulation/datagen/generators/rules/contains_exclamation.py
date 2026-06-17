"""Rule 3: contains_exclamation (category surface).

Canonical articulation: True iff the input contains an exclamation mark ('!').

The two variants of a base are CHARACTER-IDENTICAL except for a single trailing
'!' appended to the True variant:

    base sentence  = sentence-case sentence, 5-10 words, NO terminal punctuation,
                     drawn from a FRAME_NEUTRAL template over NOUN_CONCRETE (the
                     same bank family as all_lowercase; VERB/ADJ/ADVERB content
                     is embedded in the frames + the appended ADVERB_PLACE
                     adjunct used to equalize the word count).
    False variant  = the base sentence unchanged                  -> label False
    True  variant  = base + '!'                                   -> label True
    base_id        = the base sentence (both variants share it)

Because the '!' is the ONLY systematic difference between the True and the False
variants of a base (and '!' is stripped by the global tokenizer, so the word
count is identical), NO generic battery predicate that ignores '!' can separate
the classes: on the few_shot_pool (both variants present) every such predicate
sits at exactly 0.5, and on the one-variant splits the True/False assignment is a
seeded balanced partition over bases that is independent of base content.

COMMA SALT (rule-specs distribution_guards): EXACTLY 50% of bases carry one
internal comma via a leading temporal adjunct ('Later that day, the lamp stood
quietly near the window'). The comma is in the BASE (before the transform), so it
appears in BOTH the True and the False variant of a salted base, and salted bases
are class-balanced -> the 'contains a comma' predicate sits at ~0.5 (it would be
~75% if only one class carried commas). This is what stops 'contains any
punctuation' from being extensionally equivalent to the rule (it disagrees on the
comma-bearing False items).

Word counts are EQUALIZED to a single fixed target (``_TARGET_WORDS``) for every
base by appending 0-3-word ADVERB_PLACE adjuncts, so mean_wc(True) == mean_wc(
False) exactly (length-matching is 0.0, well under the 0.2 tolerance) and the
word-count battery predicates are constant -> agreement ~= P(label) ~= 0.5.

Base wording is NON-exclamatory by construction (FRAME_NEUTRAL declaratives over
concrete nouns): no interjections (wow/hey/oh), no 'What a / How + adj' openings,
so content cannot proxy for the mark (rule-specs distribution_guards).

This module conforms to the GENERATOR INTERFACE documented in ``base``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import (
    Gen,
    adjunct_word_lengths,
    equalize_word_count,
    fill_frame,
    to_sentence_case,
)
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

# Same bank family as all_lowercase: FRAME_NEUTRAL single-slot templates over
# NOUN_CONCRETE (VERB_REGULAR / ADJ_PLAIN content is embedded in the frames). The
# spec's banks list [NOUN_CONCRETE, VERB_REGULAR, ADJ_PLAIN, ADVERB_PLACE] is
# satisfied: nouns fill the slot, the frames carry the verbs/adjectives, and
# ADVERB_PLACE supplies the word-count-equalizing adjuncts.
_FRAME_BANK = "FRAME_NEUTRAL"
_NOUN_BANK = "NOUN_CONCRETE"
_ADJUNCT_BANK = "ADVERB_PLACE"

# Comfortably over the 340-base floor so the by-base split (100 + 120 + 100 +
# >= 20 spare = >= 340) has headroom; built distinct and 50/50 salted/unsalted.
_N_BASES = 420

# Every base is equalized to EXACTLY this many words (within the recipe's 5-10
# window and the global [4,14] cap). A single fixed count makes mean_wc(True) ==
# mean_wc(False) regardless of the split's label partition (length-matching 0.0)
# and renders the word-count battery predicates constant (agreement ~= 0.5).
_TARGET_WORDS = 10

# Comma-salt prefixes: leading temporal/scene adjuncts, each carrying EXACTLY one
# comma and NO other punctuation (apostrophes are banned internal punctuation).
# Half the bases get one prepended (so the comma lands in the base, on BOTH
# variants). Word counts vary (1 / 2 / 3 words); the trailing comma strips off in
# the tokenizer, so a prefix's word count is just the words before the comma. All
# are NON-exclamatory declarative scene-setters (no interjections).
_SALT_PREFIXES: tuple[str, ...] = (
    "later,",
    "today,",
    "outside,",
    "afterwards,",
    "meanwhile,",
    "later that day,",
    "early that morning,",
    "after the meeting,",
    "by late afternoon,",
    "earlier that week,",
    "without any warning,",
    "on a quiet morning,",
    "during the long afternoon,",
    "near the end of summer,",
)


@dataclass(frozen=True)
class Base:
    """A contains_exclamation base spec: the sentence-case base sentence (no
    terminal punctuation) + provenance.

    ``base_id`` is the base sentence itself (rule-spec: 'base_id = base'); both
    variants share it. ``salted`` records whether the comma-salt prefix is
    present so instantiate can write provenance without re-deriving it."""

    base_id: str   # == sentence (the sentence-case surface string, no terminal punct)
    sentence: str
    frame: str
    noun: str
    salted: bool
    salt_prefix: str  # "" when not salted
    adjuncts: tuple[str, ...]  # the ADVERB_PLACE phrases appended to equalize


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct base sentences (the GENERATOR INTERFACE).

    Deterministic given ``gen``: enumerate every (frame, noun) pair, seeded-
    shuffle, and walk it assigning the comma salt to EXACTLY 50% of accepted
    bases (alternating, so the rate is exact). Every base is equalized to exactly
    ``_TARGET_WORDS`` words by appending ADVERB_PLACE adjuncts; the salt prefix
    (with its comma) is prepended FIRST so its words count toward the target.
    Surface strings are kept distinct. Raises if it cannot reach the floor with
    an exact 50/50 salt split (loud — no quiet short / unbalanced dataset)."""
    frames = list(banks.BANKS[_FRAME_BANK])
    nouns = banks.get_bank(_NOUN_BANK).words()
    adjuncts_by_len = adjunct_word_lengths(banks.get_bank(_ADJUNCT_BANK).words())

    pairs = [(f, n) for f in frames for n in nouns]
    gen.shuffle(pairs)

    # independent seeded streams so the salt-prefix / adjunct draws do not perturb
    # the (frame, noun) enumeration order.
    salt_gen = gen.derive("salt")
    eq_gen = gen.derive("equalize")

    bases: list[Base] = []
    seen: set[str] = set()
    # exact 50/50 salt: assign salt to alternate accepted bases (index parity),
    # so for an even _N_BASES the rate is exactly 0.5.
    salt_prefixes = list(_SALT_PREFIXES)

    for frame, noun in pairs:
        if len(bases) >= _N_BASES:
            break
        salted = (len(bases) % 2 == 0)
        prefix = salt_gen.choice(salt_prefixes) if salted else ""

        core = fill_frame(frame, {"X": noun})
        if salted:
            # prepend the comma-bearing prefix, then sentence-case the whole.
            raw = f"{prefix} {core}"
        else:
            raw = core
        sentence = to_sentence_case(raw)

        # a salted prefix + the longest frame must still leave room to equalize
        # UP to the target (never down): skip any combination already over target.
        if word_count(sentence) > _TARGET_WORDS:
            continue

        # equalize to the fixed target with appended place adjuncts.
        equalized = equalize_word_count(
            sentence, _TARGET_WORDS, adjuncts_by_len, eq_gen.derive(sentence)
        )
        if equalized in seen:
            continue
        # record which adjuncts were appended (the tail beyond the core/prefix).
        appended = tuple(equalized.split()[len(sentence.split()):])

        seen.add(equalized)
        bases.append(
            Base(
                base_id=equalized,
                sentence=equalized,
                frame=frame,
                noun=noun,
                salted=salted,
                salt_prefix=prefix,
                adjuncts=appended,
            )
        )

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"contains_exclamation: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    n_salted = sum(1 for b in bases if b.salted)
    if n_salted * 2 != len(bases):
        raise ValueError(
            f"contains_exclamation: comma salt not exactly 50% "
            f"({n_salted}/{len(bases)} salted)"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True variant  = base sentence + '!'  (contains '!'      -> rule labels True).
    False variant = the base sentence    (no '!' anywhere   -> rule labels False).
    Deterministic; ``gen`` is unused (the transform carries no randomness) but
    kept in the signature to match the interface. The comma salt (if any) is
    already in ``spec.sentence``, so it is present on BOTH variants."""
    if label:
        text = spec.sentence + "!"
        transform = "append_exclamation"
    else:
        text = spec.sentence
        transform = "identity"
    meta = {
        "frame": spec.frame,
        "noun": spec.noun,
        "transform": transform,
        "base_sentence": spec.sentence,
        "salted": spec.salted,
        "salt_prefix": spec.salt_prefix,
        "adjuncts": list(spec.adjuncts),
        "target_words": _TARGET_WORDS,
    }
    return text, meta
