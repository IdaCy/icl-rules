"""Rule 5: contains_letter_z (category surface).

Canonical articulation: True iff the input contains the letter z (upper or
lower case) anywhere, including inside a word.

CONSTRUCTION (per the spec's generation.recipe + distribution_guards)
=====================================================================
A base is a (frame, matched-pair) combination. The two variants of a base are
CHARACTER-IDENTICAL except for the single content slot:

    True  variant  = frame with the slot filled by a Z_WORDS word        -> True
    False variant  = frame with the SAME slot filled by that z-word's
                     Z_FREE_MATCHED counterpart (same POS, length +/- 2,
                     same frequency_tier, NO z)                            -> False

The (z-word, counterpart) pair is fixed per base (the bank's ``pair`` key). Every
OTHER word in every frame is z-free (build_bases asserts this for the frame
templates, and instantiate asserts the emitted surface has exactly one z-bearing
word in the True variant and none in the False variant).

Why this passes the four gates with NOTHING to tune
---------------------------------------------------
The discriminating feature (a 'z' inside a word token) is ORTHOGONAL to all 40
frozen battery predicates: none of them inspects the letter z. The True and
False variant of a base differ ONLY in the slot word, and:

  * word count is identical (the slot is exactly one word in both variants), so
    the confound length-match (|mean_T - mean_F| over token counts) is 0;
  * the slot is NEVER sentence-initial and NEVER sentence-final (recipe +
    ambiguity_notes: z-words live at slot positions >= 2 so casing never
    interacts), so every first-word / last-word / first-letter-bucket / casing /
    first-word-POS predicate is IDENTICAL on both variants -> exactly 50%;
  * Z_WORDS and Z_FREE_MATCHED are a MATCHED PAIR (same POS, |len| <= 2, same
    tier), so 'rare word' / 'long word' / POS proxies cannot separate the
    classes -- the bank-level mean char-length difference is ~0;
  * char_count>=k is the only predicate the slot-word length can nudge; because
    the pair lengths are matched +/- 2 and symmetric (bank mean diff ~ -0.08),
    and we BALANCE pair selection per class, it sits far below 0.75;
  * no digits and no commas are introduced (default sentence_style for this
    rule), so contains_digit / contains_comma are 50%.

Distribution guards honored
---------------------------
  * EXACTLY ONE z-word per True item (z count is not a secondary signal vs.
    zz-words: pizza/jazz/buzz/puzzle are allowed -- the ruling is >= 1 z). The
    frame templates are all z-free; instantiate asserts the True surface has
    exactly one z-bearing token.
  * Topic mix cap: <= 25% animals, <= 25% food among z-words. The Z_WORDS bank
    already enforces this (0 animals, 1 food = pizza); we additionally cap the
    food share of the z-words USED across bases at <= 25% so no semantic-class
    proxy can arise even after sampling.
  * Slot grammatical role VARIES (subject / object / adjective / verb), realised
    by POS-segmented FRAME_NEUTRAL-style frame banks (noun, adjective, verb)
    so the matched pair (which shares POS) is grammatical in BOTH variants.

This module exposes the GENERATOR INTERFACE (build_bases / instantiate) and is
run through the shared gated pipeline (base.emit_rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, fill_frame, to_sentence_case
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

# the rule's own id (used for the per-rule sentence_style policy lookup, which
# for this rule is the strict default: no terminal punctuation, no comma).
RULE_ID = "contains_letter_z"

_Z_WORDS_BANK = "Z_WORDS"
_Z_FREE_BANK = "Z_FREE_MATCHED"

# Build comfortably above the 340-base floor: 40 matched pairs x POS-appropriate
# frames gives well over 340 distinct (frame, pair) bases. 400 leaves headroom
# for the 100 + 120 + 100 + >= 20 by-base split.
_N_BASES = 400

# global schema cap is [4, 14]; the frames below are 6-10 words with a one-word
# slot, so every surface lands comfortably inside the window.
_MIN_WORDS, _MAX_WORDS = 4, 14

# food share cap on the z-words actually USED across bases (distribution_guards:
# <= 25% food so no food semantic-class proxy). Animals are already 0 in the bank.
_MAX_FOOD_FRAC = 0.25


# --- POS-segmented FRAME_NEUTRAL-style frame banks ----------------------------
# Each frame has exactly one '{X}' slot, never at position 1 (so z-words are
# never sentence-initial) and never at the final position (so the last word is
# always a frame word -> last-word battery predicates are class-invariant). All
# frame words are z-FREE (build_bases asserts). Slots realise distinct
# grammatical roles per POS: subject/object (noun), predicate (adjective),
# matrix verb (verb).

# Noun slot (subject or object), reading naturally for any concrete/abstract noun.
_NOUN_FRAMES: list[str] = [
    "The {X} was near the old fence",
    "They found the {X} behind the shed",
    "She placed the {X} on the kitchen table",
    "We left the {X} beside the front door",
    "He carried the {X} across the wide room",
    "Someone moved the {X} into the far corner",
    "The {X} stood quietly near the tall window",
    "Children gathered around the {X} after lunch",
    "Workers placed the {X} beside the muddy road",
    "Everyone walked past the {X} without stopping",
    "We discovered the {X} inside the dusty attic",
    "He noticed the {X} on the office desk",
    "They photographed the {X} at the busy market",
    "She wrote a long note about the {X}",
    "The {X} remained inside the locked metal cabinet",
]

# Adjective slot (predicate adjective), reading naturally for any plain adjective.
_ADJ_FRAMES: list[str] = [
    "The garden looked {X} after the heavy rain",
    "Her new bicycle seemed {X} from the start",
    "The old machine became {X} during the night",
    "Their little dog stayed {X} all afternoon",
    "The empty hallway felt {X} without the lights",
    "His latest drawing looked {X} to the teacher",
    "The wooden bridge appeared {X} under the load",
    "The morning sky turned {X} above the harbour",
]

# Verb slot. Two sub-banks so the matched pair's verb FORM fits grammatically:
#   * bare/infinitive frames for stem verbs (freeze, organize, ...);
#   * past-tense frames for the -ed pairs (realized<->admitted, seized<->grabbed).
_VERB_BARE_FRAMES: list[str] = [
    "They wanted to {X} the heavy boxes quickly",
    "We tried to {X} the old wooden door",
    "She decided to {X} the broken garden gate",
    "He began to {X} the dusty paper files",
    "The workers had to {X} the narrow front path",
    "Nobody wanted to {X} the fragile glass jars",
    "They hoped to {X} the small metal parts",
]
_VERB_PAST_FRAMES: list[str] = [
    "The careful builders {X} the heavy front gate",
    "Several quiet workers {X} the broken wooden fence",
    "The two students {X} the difficult exam questions",
]

# verbs whose surface is a PAST form (end in -ed); routed to the past frames.
_PAST_VERB_PAIRS = frozenset({"realized", "seized"})


@dataclass(frozen=True)
class Base:
    """A contains_letter_z base: a frame + a fixed matched (z-word, counterpart).

    ``base_id`` is a stable, distinct id for the (frame, pair) combination. Both
    variants of the base share it. ``frame`` is the template (one '{X}' slot),
    ``z_word`` the True filler, ``z_free`` its matched False filler, ``pos`` and
    ``pair`` record the matched-pair provenance, ``slot_role`` the grammatical
    role of the slot."""

    base_id: str
    frame: str
    frame_idx: int
    z_word: str
    z_free: str
    pos: str
    pair: str
    slot_role: str


def _assert_z_free(text: str, where: str) -> None:
    if "z" in text.lower():
        raise ValueError(f"contains_letter_z: frame text {where} unexpectedly contains a 'z': {text!r}")


def _frames_for(pos: str, pair_word: str) -> tuple[list[str], str]:
    """Return (frame templates, slot grammatical role) for a matched pair."""
    if pos == "noun":
        return _NOUN_FRAMES, "noun"
    if pos == "adjective":
        return _ADJ_FRAMES, "adjective"
    if pos == "verb":
        if pair_word in _PAST_VERB_PAIRS:
            return _VERB_PAST_FRAMES, "verb_past"
        return _VERB_BARE_FRAMES, "verb_bare"
    raise ValueError(f"contains_letter_z: unexpected matched-pair POS {pos!r}")


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct (frame, matched-pair) bases (GENERATOR INTERFACE).

    Deterministic given ``gen``: enumerate every (matched-pair, POS-appropriate
    frame) combination, seeded-shuffle, and take the first _N_BASES with a
    distinct base_id whose BOTH variants land in [4, 14] words, subject to the
    food-share cap on the z-words used. Raises (loud) if the floor or any guard
    cannot be met."""
    zw = banks.get_bank(_Z_WORDS_BANK).entries
    zf_by_pair = {e.pair: e for e in banks.get_bank(_Z_FREE_BANK).entries}

    # assert every frame template is z-free up front (one z-bearing token only,
    # ever, and it must be the slot filler).
    for fr in (*_NOUN_FRAMES, *_ADJ_FRAMES, *_VERB_BARE_FRAMES, *_VERB_PAST_FRAMES):
        if fr.count("{X}") != 1:
            raise ValueError(f"contains_letter_z: frame {fr!r} must have exactly one slot")
        _assert_z_free(fr.replace("{X}", ""), f"template {fr!r}")

    # enumerate all candidate (pair, frame) combos
    candidates: list[tuple[Any, int, str, str]] = []  # (z_entry, frame_idx, frame, role)
    for ze in zw:
        if ze.pair not in zf_by_pair:
            raise ValueError(f"contains_letter_z: z-word {ze.word!r} has no Z_FREE_MATCHED mate")
        frames, role = _frames_for(ze.pos, ze.pair)
        for fi, fr in enumerate(frames):
            candidates.append((ze, fi, fr, role))

    gen.shuffle(candidates)

    bases: list[Base] = []
    seen_ids: set[str] = set()
    n_food = 0
    for ze, fi, fr, role in candidates:
        zfree = zf_by_pair[ze.pair]
        true_surface = to_sentence_case(fill_frame(fr, {"X": ze.word}))
        false_surface = to_sentence_case(fill_frame(fr, {"X": zfree.word}))
        wc_t, wc_f = word_count(true_surface), word_count(false_surface)
        if not (_MIN_WORDS <= wc_t <= _MAX_WORDS and _MIN_WORDS <= wc_f <= _MAX_WORDS):
            continue
        # ground-truth sanity: True has a z, False has none (instantiate re-checks).
        if "z" not in true_surface.lower() or "z" in false_surface.lower():
            continue

        base_id = f"{role}|f{fi}|{ze.pair}"
        if base_id in seen_ids:
            continue

        # food-share cap on z-words actually used (distribution_guards)
        is_food = ze.subtype == "food"
        if is_food and (n_food + 1) > _MAX_FOOD_FRAC * (len(bases) + 1):
            # would push food share over the cap at this size; skip for now.
            continue

        seen_ids.add(base_id)
        if is_food:
            n_food += 1
        bases.append(
            Base(
                base_id=base_id,
                frame=fr,
                frame_idx=fi,
                z_word=ze.word,
                z_free=zfree.word,
                pos=ze.pos,
                pair=ze.pair,
                slot_role=role,
            )
        )
        if len(bases) >= _N_BASES:
            break

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"contains_letter_z: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    # final food-share assertion over the bases actually built (<= 25%)
    food_frac = n_food / len(bases)
    if food_frac > _MAX_FOOD_FRAC:
        raise ValueError(
            f"contains_letter_z: food z-word share {food_frac:.3f} > {_MAX_FOOD_FRAC}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (GENERATOR INTERFACE).

    True  -> frame slot filled with the z-word  (contains 'z' -> rule True).
    False -> frame slot filled with the matched z-free counterpart (no 'z').
    Both variants are sentence-cased; the slot is interior so the filler keeps
    its lowercase form. Deterministic; ``gen`` is unused (no per-variant
    randomness) but kept to match the interface."""
    filler = spec.z_word if label else spec.z_free
    surface = to_sentence_case(fill_frame(spec.frame, {"X": filler}))

    # ground-truth + exactly-one-z guards (LOUD: never emit a mislabeled item).
    z_tokens = [tok for tok in surface.split() if "z" in tok.lower()]
    if label:
        if len(z_tokens) != 1:
            raise ValueError(
                f"contains_letter_z: True variant of {spec.base_id!r} must have exactly "
                f"one z-bearing token, has {len(z_tokens)}: {surface!r}"
            )
    else:
        if z_tokens:
            raise ValueError(
                f"contains_letter_z: False variant of {spec.base_id!r} must have no "
                f"z-bearing token, found {z_tokens}: {surface!r}"
            )

    meta = {
        "frame": spec.frame,
        "frame_idx": spec.frame_idx,
        "pos": spec.pos,
        "pair": spec.pair,
        "slot_role": spec.slot_role,
        "z_word": spec.z_word,
        "z_free": spec.z_free,
        "filler": filler,
        "transform": "z_word_slot" if label else "z_free_slot",
    }
    return surface, meta
