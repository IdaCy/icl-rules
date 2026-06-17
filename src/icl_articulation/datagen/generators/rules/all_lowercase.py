"""Reference rule: all_lowercase (plan rule 1).

Canonical articulation: True iff the input contains no uppercase letter
anywhere. The two variants of a base are CHARACTER-IDENTICAL except for casing:

    base sentence  = sentence-case sentence (5-10 words) from a FRAME_NEUTRAL
                     template over NOUN_CONCRETE; no proper nouns, no 'I'.
    False variant  = the base sentence (default sentence case)  -> label False
    True  variant  = base.lower()                                 -> label True
    base_id        = the base sentence (both variants share it)

Because the variants differ only in the first letter's case, NO length / vocab /
style / token feature can separate the classes — every generic battery predicate
sits at exactly 50% (its value is identical on both variants of every base),
EXCEPT the ``all_lowercase`` predicate itself (the rule), which is exempt via the
equivalence class. This is the planned "casing is the only signal" property and
makes the rule the clean end-to-end proof for the gated pipeline.

This module is the REFERENCE implementation of the GENERATOR INTERFACE; the 26
fan-out rules mirror its ``build_bases`` / ``instantiate`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, fill_frame, to_lower, to_sentence_case
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

# The rule draws its bases from FRAME_NEUTRAL templates over NOUN_CONCRETE
# (VERB_REGULAR / ADJ_PLAIN / ADVERB_PLACE content is already embedded in the
# frames — they are "FRAME_NEUTRAL-style templates over" those banks). No proper
# nouns, no 'I', sentence case: all true of the FRAME_NEUTRAL bank by construction.
_FRAME_BANK = "FRAME_NEUTRAL"
_NOUN_BANK = "NOUN_CONCRETE"

# build comfortably more than the 340-base floor so the by-base split (100 +
# 120 + 100 + >= 20 spare) has headroom; 30 frames x 131 nouns is ~3900 distinct
# (frame, noun) pairs, so 360 is easily disjoint.
_N_BASES = 360

# the base sentences are 5-10 words by frame design; assert the window the spec
# names (and the global [4,14] cap the schema validator re-checks).
_MIN_WORDS, _MAX_WORDS = 5, 10


@dataclass(frozen=True)
class Base:
    """A reference-rule base spec: the sentence-case base sentence + provenance.

    ``base_id`` is the base sentence itself (rule-spec: 'base_id = base
    sentence'); ``frame`` / ``noun`` are recorded so instantiate can write
    provenance without re-deriving them."""

    base_id: str   # == sentence (the sentence-case surface string)
    sentence: str
    frame: str
    noun: str


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct sentence-case base sentences (the GENERATOR INTERFACE).

    Deterministic given ``gen``: enumerate every (frame, noun) pair, seeded-shuffle,
    and take the first _N_BASES whose word count is in [5, 10] and whose surface
    string is distinct. Raises if it cannot reach the floor (loud — no quiet
    short dataset)."""
    frames = list(banks.BANKS[_FRAME_BANK])
    nouns = banks.get_bank(_NOUN_BANK).words()

    pairs = [(f, n) for f in frames for n in nouns]
    gen.shuffle(pairs)

    bases: list[Base] = []
    seen: set[str] = set()
    for frame, noun in pairs:
        sentence = to_sentence_case(fill_frame(frame, {"X": noun}))
        if sentence in seen:
            continue
        wc = word_count(sentence)
        if not (_MIN_WORDS <= wc <= _MAX_WORDS):
            continue
        seen.add(sentence)
        bases.append(Base(base_id=sentence, sentence=sentence, frame=frame, noun=noun))
        if len(bases) >= _N_BASES:
            break

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"all_lowercase: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True variant = base.lower() (no uppercase anywhere -> rule labels True).
    False variant = the sentence-case base (its first letter is uppercase ->
    rule labels False). Deterministic; ``gen`` is unused (the transform carries
    no randomness) but kept in the signature to match the interface."""
    if label:
        text = to_lower(spec.sentence)
        transform = "lower"
    else:
        text = spec.sentence  # already sentence case
        transform = "sentence_case"
    meta = {
        "frame": spec.frame,
        "noun": spec.noun,
        "transform": transform,
        "base_sentence": spec.sentence,
    }
    return text, meta
