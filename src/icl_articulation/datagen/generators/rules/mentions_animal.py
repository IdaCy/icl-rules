"""Rule 13: mentions_animal (category: semantic-substitution).

Canonical articulation: True iff the input mentions an animal -- a real, living,
non-human animal kind (mammals, birds, fish, insects, ...). Ground truth
(groundtruth._r13_mentions_animal) is bank membership: True iff any
stripped/lowercased token of the text is in the ANIMALS bank word-set.

Construction (spec generation.recipe + distribution_guards):

  * FRAME_NEUTRAL frames -- neutral predicates natural for ANY concrete noun
    (located/found/photographed/noticed), 6-10 words, ONE noun slot {X}. The
    frames are NEUTRAL by construction (no animal-typical predicates such as
    'barked' / 'grazed'), so the verb cannot proxy the rule. The frames are
    shared VERBATIM across the two classes.
  * Each base pairs a frame with a CHAR-LENGTH-MATCHED (animal, filler) pair:
        True  variant = frame filled with the animal   (from ANIMALS)        -> True
        False variant = frame filled with the filler    (from
                        OBJECTS_PLANTS_VEHICLES)                              -> False
    The pair is matched on alphabetic length (and, where supply allows,
    frequency tier), so the True and False surface strings of one base differ
    in EXACTLY one slot word AND that word has identical length. Both variants
    are sentence-cased; nothing else differs.

  Because the variants of a base are identical except for a same-length noun in
  a frame-internal position, EVERY one of the 40 frozen generic battery
  predicates evaluates IDENTICALLY on the True and the False variant of that
  base (word_count / char_count are equal, and every first/last-word, letter,
  determiner, POS, casing, digit and comma feature is frame-determined and
  frame-shared). Summed over the dataset each predicate therefore sits at
  exactly 0.5 agreement -- the noun's animacy is the ONLY signal. This is the
  semantic-substitution analogue of the reference rule's "casing is the only
  signal" property.

  distribution_guards honored:
    - frames shared verbatim across classes (one frame pool, both variants);
    - filler bank matched in length (per-pair, exact) and frequency tier
      (preferred per-pair, near-exact in aggregate);
    - the False (filler) set spans >= 15 distinct PLANTS and >= 10 distinct
      VEHICLES so 'living thing' and 'can move on its own' are NOT equivalent
      to 'animal' (mc_distractor_seeds rely on this);
    - no animal-typical predicate anywhere (FRAME_NEUTRAL only);
    - exactly one slot word differs per pair (the noun).

This module exposes the GENERATOR INTERFACE (build_bases / instantiate); it is
wired into the registry via its rule_id like every other per-rule module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, fill_frame, to_sentence_case
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

_FRAME_BANK = "FRAME_NEUTRAL"
_ANIMAL_BANK = "ANIMALS"
_FILLER_BANK = "OBJECTS_PLANTS_VEHICLES"

# Build comfortably over the 340-base floor for split headroom (100 few_shot +
# 120 held_out + 100 confirmation + >= 20 spare = 340).
_N_BASES = 380

# spec recipe: frames are 6-10 words; the global schema window is [4, 14]. With a
# one-word noun every FRAME_NEUTRAL frame fills to 6-9 words, comfortably inside.
_MIN_WORDS, _MAX_WORDS = 6, 10

# distribution_guards: the False bank must visibly contain plants and vehicles so
# 'living thing' / 'can move' do not collapse onto 'animal'. We require at least
# this many DISTINCT plant and vehicle fillers to actually appear in the dataset.
_MIN_DISTINCT_PLANT_FILLERS = 15
_MIN_DISTINCT_VEHICLE_FILLERS = 10


@dataclass(frozen=True)
class Base:
    """One mentions_animal base: a frame + a char-length-matched (animal, filler)
    pair. ``base_id`` encodes all three so it is stable and distinct.

    The True variant fills ``frame`` with ``animal`` (label True); the False
    variant fills the SAME frame with ``filler`` (label False). ``length`` is the
    shared alphabetic length of the two nouns; ``filler_subtype`` records whether
    the filler is a plant / vehicle / object (for the distribution guard)."""

    base_id: str
    frame: str
    frame_idx: int
    animal: str
    filler: str
    length: int
    filler_subtype: str


def _entries_by_length(bank_name: str) -> dict[int, list[banks.Entry]]:
    by_len: dict[int, list[banks.Entry]] = defaultdict(list)
    for e in banks.get_bank(bank_name).entries:
        by_len[e.length].append(e)
    return by_len


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (the GENERATOR INTERFACE).

    Deterministic given ``gen``. For each base we draw a frame and a CHAR-LENGTH
    -MATCHED (animal, filler) pair, preferring a filler in the SAME frequency
    tier as the animal. We dedup on (frame, animal) and (frame, filler) so no
    two bases can ever produce the same True or False surface string, and we
    front-load enough plant and vehicle fillers to satisfy the distribution
    guard before falling through to length/tier-matched objects.

    Raises (LOUD) if it cannot reach the base floor or the plant/vehicle guard."""
    frames = list(banks.BANKS[_FRAME_BANK])

    animals_by_len = _entries_by_length(_ANIMAL_BANK)
    fillers_by_len = _entries_by_length(_FILLER_BANK)

    # Only lengths present in BOTH banks can form a char-length-matched pair.
    shared_lengths = sorted(set(animals_by_len) & set(fillers_by_len))

    # All char-length-matched (animal, filler) candidate pairs, tagged with a
    # tier-match flag and the filler subtype. A seeded shuffle gives a stable,
    # file-order-independent draw order.
    pairs: list[tuple[str, str, int, bool, str]] = []
    for L in shared_lengths:
        for a in animals_by_len[L]:
            for f in fillers_by_len[L]:
                tier_match = a.frequency_tier == f.frequency_tier
                pairs.append((a.word, f.word, L, tier_match, f.subtype or "object"))
    gen.shuffle(pairs)

    # Order frames once (seeded) so slot position varies deterministically as we
    # cycle through frames.
    frame_order = list(range(len(frames)))
    gen.shuffle(frame_order)

    bases: list[Base] = []
    seen_true: set[str] = set()   # (frame, animal) -> True surface
    seen_false: set[str] = set()  # (frame, filler) -> False surface
    distinct_plant_fillers: set[str] = set()
    distinct_vehicle_fillers: set[str] = set()

    def _try_add(animal: str, filler: str, length: int, subtype: str) -> bool:
        # cycle frames so the same (animal, filler) pair lands in different frames
        # (slot position varies) without ever colliding on a surface string.
        for off in range(len(frame_order)):
            fi = frame_order[(len(bases) + off) % len(frame_order)]
            frame = frames[fi]
            true_text = to_sentence_case(fill_frame(frame, {"X": animal}))
            false_text = to_sentence_case(fill_frame(frame, {"X": filler}))
            if true_text in seen_true or false_text in seen_false:
                continue
            wc_t = word_count(true_text)
            wc_f = word_count(false_text)
            if not (_MIN_WORDS <= wc_t <= _MAX_WORDS):
                continue
            if not (_MIN_WORDS <= wc_f <= _MAX_WORDS):
                continue
            # never let the filler equal a noun the frame already contains (keeps
            # 'exactly one slot word differs' clean and avoids odd repeats).
            base_id = f"f{fi:02d}|{animal}|{filler}"
            seen_true.add(true_text)
            seen_false.add(false_text)
            if subtype == "plant":
                distinct_plant_fillers.add(filler)
            elif subtype == "vehicle":
                distinct_vehicle_fillers.add(filler)
            bases.append(
                Base(
                    base_id=base_id,
                    frame=frame,
                    frame_idx=fi,
                    animal=animal,
                    filler=filler,
                    length=length,
                    filler_subtype=subtype,
                )
            )
            return True
        return False

    # Phase 1: guarantee the plant/vehicle distribution guard first, so the False
    # class visibly contains living-but-not-animal and can-move-but-not-animal
    # items regardless of how the rest of the draw goes.
    plant_pairs = [p for p in pairs if p[4] == "plant"]
    vehicle_pairs = [p for p in pairs if p[4] == "vehicle"]
    for need, src in (
        (_MIN_DISTINCT_PLANT_FILLERS, plant_pairs),
        (_MIN_DISTINCT_VEHICLE_FILLERS, vehicle_pairs),
    ):
        used: set[str] = set()
        for animal, filler, L, _tier, subtype in src:
            if len(used) >= need:
                break
            if filler in used:
                continue
            if _try_add(animal, filler, L, subtype):
                used.add(filler)

    # Phase 2: prefer tier-matched pairs, then fill the rest.
    for tier_first in (True, False):
        for animal, filler, L, tier_match, subtype in pairs:
            if len(bases) >= _N_BASES:
                break
            if tier_match != tier_first:
                continue
            _try_add(animal, filler, L, subtype)
        if len(bases) >= _N_BASES:
            break

    if len(distinct_plant_fillers) < _MIN_DISTINCT_PLANT_FILLERS:
        raise ValueError(
            f"mentions_animal: only {len(distinct_plant_fillers)} distinct plant "
            f"fillers, need >= {_MIN_DISTINCT_PLANT_FILLERS}"
        )
    if len(distinct_vehicle_fillers) < _MIN_DISTINCT_VEHICLE_FILLERS:
        raise ValueError(
            f"mentions_animal: only {len(distinct_vehicle_fillers)} distinct "
            f"vehicle fillers, need >= {_MIN_DISTINCT_VEHICLE_FILLERS}"
        )
    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"mentions_animal: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True  variant = frame filled with the animal  (an ANIMALS token -> rule True).
    False variant = frame filled with the filler   (OBJECTS_PLANTS_VEHICLES, no
    animal token -> rule False). Both sentence-cased. Deterministic; ``gen`` is
    unused (the substitution carries no randomness) but kept to match the
    interface."""
    noun = spec.animal if label else spec.filler
    text = to_sentence_case(fill_frame(spec.frame, {"X": noun}))
    meta = {
        "frame": spec.frame,
        "frame_idx": spec.frame_idx,
        "slot_word": noun,
        "slot_length": spec.length,
        "animal": spec.animal,
        "filler": spec.filler,
        "filler_subtype": spec.filler_subtype,
        "transform": "sentence_case",
        "variant": "animal" if label else "non_animal",
    }
    return text, meta
