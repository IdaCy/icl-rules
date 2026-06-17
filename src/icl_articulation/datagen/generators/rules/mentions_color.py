"""Rule 14: mentions_color (category: semantic substitution).

Canonical articulation: True iff the input mentions a specific color term (red,
blue, green, beige, ...). Ground truth (groundtruth._r14_mentions_color) is BANK
MEMBERSHIP: True iff any stripped/lowercased token is in the COLORS bank.

CONSTRUCTION (rule-specs id: mentions_color, generation.recipe + the matched
substitution scheme). Each base is a (frame, noun) pair; the two variants are
SURFACE-IDENTICAL except for ONE adjective token sitting directly before the
noun:

    True  variant = frame with the {ADJ} slot filled from COLORS         -> True
    False variant = frame with the {ADJ} slot filled from               -> False
                    ADJ_NONCOLOR_MATCHED (length-matched to the color)
    base_id       = frame + noun  (both variants share it)

Because the variants differ in exactly one one-token adjective, every base's
True and False surfaces have the SAME word count (the length-matching gate is
satisfied by construction, |mean_wc(T) - mean_wc(F)| == 0). The False adjective
is drawn LENGTH-MATCHED to the base's color, so per-item char counts stay close
too and no char_count battery bucket separates the classes. Frames are
chromatically NEUTRAL (no painting / dyeing / coloring verbs, no chromatic nouns
like paint / dye / rainbow anywhere — they would leak the rule), the adjective
is NEVER sentence-initial (so no first-word POS / length predicate keys on it),
and the >= 16 frames vary the slot position (subject / object / post-preposition).

Distribution guards honored:
  * Shared frames; nouns identical across classes (the same NOUN_CONCRETE bank
    feeds both variants of every base).
  * Adjective banks matched for length/frequency (COLORS <-> ADJ_NONCOLOR_MATCHED
    is a declared matched_pair in banks.py); the per-base False adjective is
    additionally picked to length-match that base's color.
  * Color spread: colors are assigned by cycling a seeded base order, so each of
    the 16 colors covers ~1/16 (~6.25%) of bases and stays well under the 15%
    cap among True items for any seeded split assignment.
  * 'orange' appears only as a PRE-NOUN adjective (an {ADJ} filler), never as the
    fruit noun (the {N} slot is NOUN_CONCRETE, which has no fruit named 'orange'
    and no color words).

This module exposes the GENERATOR INTERFACE (build_bases / instantiate) and is
dispatched by registry.run through the shared gated pipeline (base.emit_rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, fill_frame, to_sentence_case
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

_COLOR_BANK = "COLORS"
_NONCOLOR_BANK = "ADJ_NONCOLOR_MATCHED"
_NOUN_BANK = "NOUN_CONCRETE"

# >= 12 frames; slot position varies (subject vs. object vs. post-preposition).
# Each frame has exactly one {ADJ} directly before its {N}. Chromatically neutral:
# no paint/dye/color/tint/shade verbs or nouns, no 'rainbow'. Filling {ADJ} and
# {N} with one-token words each yields 7-9 words (inside the spec's 6-10 and the
# global [4, 14]). The adjective is never the first word, so first-word POS /
# length / letter-bucket predicates never key on the color signal.
_FRAMES: tuple[str, ...] = (
    "The {ADJ} {N} stood near the entrance",
    "She bought a {ADJ} {N} at the market",
    "They left the {ADJ} {N} beside the door",
    "He noticed a {ADJ} {N} across the room",
    "We found the {ADJ} {N} under the table",
    "A {ADJ} {N} rested against the wall outside",
    "The visitors admired the {ADJ} {N} by the window",
    "Someone placed a {ADJ} {N} on the shelf",
    "You can see the {ADJ} {N} from the road",
    "The {ADJ} {N} remained in the hallway all morning",
    "She kept a {ADJ} {N} inside the wooden cabinet",
    "They carried the {ADJ} {N} toward the open gate",
    "Everyone walked past the {ADJ} {N} without stopping",
    "He described the {ADJ} {N} to the curious visitor",
    "A {ADJ} {N} sat quietly in the corner today",
    "The neighbors mentioned a {ADJ} {N} near the fence",
)

# build well past the 340-base floor (16 frames x 120 nouns = 1920 pairs).
_N_BASES = 480

_MIN_WORDS, _MAX_WORDS = 6, 10  # the spec's per-frame window


@dataclass(frozen=True)
class Base:
    """A mentions_color base: a (frame, noun) pair plus the pre-chosen color
    (True variant adjective) and length-matched non-color adjective (False
    variant adjective). ``base_id`` == frame + noun (rule-spec)."""

    base_id: str
    frame: str
    noun: str
    color: str        # COLORS filler -> True variant
    noncolor: str     # ADJ_NONCOLOR_MATCHED filler -> False variant


def _nearest_noncolor(color_len: int, pool: list[str], lens: dict[str, int]) -> str:
    """Pick the non-color adjective whose length is closest to ``color_len``
    (ties -> the first in ``pool`` order, which the caller has seeded-shuffled).
    Used to length-match the False adjective to the base's color so per-item
    char counts track and no char_count battery bucket separates the classes."""
    return min(pool, key=lambda w: (abs(lens[w] - color_len), pool.index(w)))


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct (frame, noun) bases (the GENERATOR INTERFACE).

    Deterministic given ``gen``: enumerate every (frame, noun) pair, seeded-
    shuffle, then walk the shuffled order assigning each base the next color in a
    cycled, evenly-spread color order (so no color exceeds ~1/16 of bases) and a
    length-matched non-color adjective. Keep the first _N_BASES pairs whose
    filled True/False surfaces are 6-10 words and distinct."""
    frames = list(_FRAMES)
    nouns = banks.get_bank(_NOUN_BANK).words()
    colors = banks.get_bank(_COLOR_BANK).words()
    noncolors = banks.get_bank(_NONCOLOR_BANK).words()
    noncolor_len = {w: len(w) for w in noncolors}

    # seeded, even color spread: a shuffled color order cycled across bases.
    color_order = list(colors)
    gen.shuffle(color_order)
    noncolor_pool = list(noncolors)
    gen.shuffle(noncolor_pool)

    pairs = [(f, n) for f in frames for n in nouns]
    gen.shuffle(pairs)

    bases: list[Base] = []
    seen_ids: set[str] = set()
    for idx, (frame, noun) in enumerate(pairs):
        bid = f"{frame}||{noun}"
        if bid in seen_ids:
            continue
        color = color_order[len(bases) % len(color_order)]
        noncolor = _nearest_noncolor(len(color), noncolor_pool, noncolor_len)
        # word-count window check on the TRUE surface (False is the same length,
        # both adjectives are a single token).
        true_text = to_sentence_case(fill_frame(frame, {"ADJ": color, "N": noun}))
        if not (_MIN_WORDS <= word_count(true_text) <= _MAX_WORDS):
            continue
        seen_ids.add(bid)
        bases.append(Base(base_id=bid, frame=frame, noun=noun, color=color, noncolor=noncolor))
        if len(bases) >= _N_BASES:
            break

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"mentions_color: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True  -> {ADJ} filled from COLORS (a color token -> rule labels True).
    False -> {ADJ} filled from ADJ_NONCOLOR_MATCHED (no color token -> False).
    Deterministic; ``gen`` is unused (the adjective is pinned on the base) but
    kept in the signature to match the interface."""
    adj = spec.color if label else spec.noncolor
    bank = _COLOR_BANK if label else _NONCOLOR_BANK
    text = to_sentence_case(fill_frame(spec.frame, {"ADJ": adj, "N": spec.noun}))
    meta = {
        "frame": spec.frame,
        "noun": spec.noun,
        "adjective": adj,
        "adjective_bank": bank,
        "transform": "color_substitution" if label else "noncolor_substitution",
        "color": spec.color,
        "noncolor": spec.noncolor,
    }
    return text, meta
