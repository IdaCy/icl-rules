"""Rule 21: the_appears_twice (category: positional).

Canonical articulation (groundtruth._r21_the_appears_twice): True iff the word
"the" appears at least twice (case-insensitive, so a sentence-initial "The"
counts); "the" inside another word ("theater") does NOT count — it is a
word-level match on the stripped/lowercased tokens.

CONSTRUCTION (rule-specs id: the_appears_twice, generation.recipe +
distribution_guards, honored verbatim):

  * Frames with 4 NP determiner slots {D1,D2,D3,D4} (the spec allows 2-4), each
    determiner immediately before a NOUN_CONCRETE slot, joined by a VERB_REGULAR
    slot (rendered in its 3rd-person-singular -s form, since the subject NP
    "{D1} {N1}" is always singular) and prepositions. 9-11 words (inside the
    spec's 6-11 and the global
    [4,14]). Determiners are drawn from {the, a, his, her, their, this}.
    Crucially the frame BODIES contain NO literal "the"/"a"/"and" word — EVERY
    occurrence of "the" is a determiner slot the generator controls, so the
    per-text "the" count is exactly what the recipe pins.

  * base_id = frame + noun fillers (+ verb). The TWO variants of a base differ
    ONLY in determiner identity (determiner swaps only): same frame, same nouns,
    same verb, same NP count, same word count. So no length / vocab / frame /
    positional feature except the "the" count can separate the classes.

  * Class "the"-count distribution (recipe):
        True : exactly 2 "the" (80%) or 3 (20%).
        False: exactly 1 "the" (60%) or 0 (40%).
    This 1-the-in-60%-of-False split is load-bearing: it pins the generic
    ``contains_the`` predicate's agreement to 0.5 + 0.5*P(False has 0 the) =
    0.5 + 0.5*0.40 = 0.70 <= 0.75 in any balanced split (the ``count_the>=2``
    predicate IS the rule and is equiv-exempt via equiv_keys).

  * Sentence-initial "The" in 50% of items in EACH class: D1's "the"-status is
    FIXED PER BASE (identical in the True and the False variant). 50% of bases
    have D1 = "the" (so both their variants open with "The"), 50% open with a
    non-"the" determiner. Because D1 is label-independent, the first word — and
    every first-word battery predicate (POS, length, letter bucket, vowel) —
    takes the SAME value on both variants of a base and so sits at 50% by base
    balance. The True/False the-count gap therefore lives ENTIRELY in the
    interior swing slots D2..D4. (initial_the=True forces False-count = 1, since
    the fixed opening "The" already supplies one; the 0-the False items come
    only from initial_the=False bases — the joint distribution below honors the
    50% initial-The and 60/40 False split simultaneously.)

  * Zero per-base char delta: whenever a swing slot is "the" in the True variant
    and non-"the" in the False variant, its False filler is a 3-LETTER determiner
    ("his"/"her") — the same length as "the" — so the two variants are
    CHARACTER-LENGTH identical, not just word-count identical, and no char_count
    battery bucket can separate the classes (they take the same value on both
    variants of every base).

  * "the"-position spread + determiner balance (distribution_guards): which
    swing slots carry "the" is seeded-randomized so "the" occurs in every NP
    slot across the dataset; the non-"the" determiners are drawn from
    {a, his, her, their, this} the same way in both classes so total determiner
    counts (and per-token a/his/her/their/this rates) match across classes (the
    confound report tracks the top skewed tokens).

A FIXED seed threads the whole build (the pipeline logs it into slots_meta as
``seed``; provenance below also records the per-base determiner assignment).

This module exposes the GENERATOR INTERFACE (build_bases / instantiate) and runs
through the shared gated pipeline (base.emit_rule). Style policy: rule 21 falls
through to the strict global default (no terminal, no comma), so no
STYLE_RULE_ID alias is exported; ``base_id`` lives on the dataclass so no
``base_id_of`` is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...banks import _regular_verb_forms  # the framework's regular-inflection helper
from ...genutils import Gen, base_id as _base_id, fill_frame, to_sentence_case
from ...groundtruth import _r21_the_appears_twice
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count, words

_NOUN = "NOUN_CONCRETE"
_VERB = "VERB_REGULAR"

# the determiner slots, in surface order (D1 is the sentence-initial slot).
_DET_SLOTS = ("D1", "D2", "D3", "D4")
# the interior "swing" slots whose "the"-status differs between the two variants.
_SWING_SLOTS = ("D2", "D3", "D4")

# determiners. "the" is THE rule token; the rest are the non-"the" set the recipe
# names. THREE_CHAR are the 3-letter non-"the" determiners used to fill a slot
# that flips from "the" (True) to non-"the" (False), so the swap is char-neutral
# (len("the") == len("his") == len("her") == 3).
_THE = "the"
_NON_THE = ("a", "his", "her", "their", "this")
_THREE_CHAR_NON_THE = ("his", "her")  # used at flip slots -> zero char delta

# Frames: 4 NP determiner slots, each {D}->{N}; {V} from VERB_REGULAR (emitted in
# its 3sg -s form to agree with the always-singular subject NP); no literal
# "the"/"a"/"and" in any body so EVERY "the" is a controlled slot. Word counts
# (4 dets + 4 nouns + 1 verb + connectives) land in [9, 11].
_FRAMES: tuple[str, ...] = (
    "{D1} {N1} {V} {D2} {N2} near {D3} {N3} behind {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} beside {D3} {N3} under {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} past {D3} {N3} toward {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} above {D3} {N3} below {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} inside {D3} {N3} near {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} along {D3} {N3} beside {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} beyond {D3} {N3} behind {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} under {D3} {N3} above {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} outside {D3} {N3} near {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} toward {D3} {N3} past {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} below {D3} {N3} beside {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} behind {D3} {N3} under {D4} {N4}",
)

# build well past the 340-base floor (12 frames x 131 nouns gives ample room).
_N_BASES = 480

_MIN_WORDS, _MAX_WORDS = 6, 11  # the spec's per-frame window (subset of [4,14])

# the recipe's class-conditional "the"-count distribution.
_TRUE_COUNT_HI_SHARE = 0.20   # 20% of True items have 3 "the" (else 2)
_FALSE_COUNT_ZERO_SHARE = 0.40  # 40% of False items have 0 "the" (else 1)
_INITIAL_THE_SHARE = 0.50     # 50% of bases open with "The" (both variants)


@dataclass(frozen=True)
class Base:
    """One the_appears_twice base.

    A base fixes a frame, its four noun fillers, a verb, AND the full determiner
    assignment for BOTH variants (determiner swaps only). ``the_slots_true`` /
    ``the_slots_false`` are the slot names that carry "the" in the True / False
    variant (the False set is a SUBSET of the True set; D1 is in both or neither).
    ``non_the`` pins the non-"the" determiner for every slot (the flip slots use a
    3-letter determiner so the variants are char-length identical). ``base_id`` is
    hashed from frame + noun fillers + verb (rule-spec: base_id = frame + noun
    fillers); the determiner assignment is provenance, not identity."""

    base_id: str
    frame: str
    nouns: tuple[str, str, str, str]   # fillers for N1..N4
    verb: str
    the_slots_true: tuple[str, ...]
    the_slots_false: tuple[str, ...]
    non_the: tuple[tuple[str, str], ...]  # (slot, determiner) for non-"the" slots
    initial_the: bool
    true_count: int
    false_count: int


def _frame_index(frame: str) -> int:
    return _FRAMES.index(frame)


def _build_profiles(gen: Gen) -> list[tuple[bool, int, int]]:
    """Deterministic (initial_the, true_count, false_count) profile per base.

    Honors all three distribution targets jointly:
      * initial_the True for exactly 50% of bases.
      * true_count = 3 for 20% of bases (else 2), independent of initial_the.
      * false_count = 0 for 40% of bases (else 1); ALL 0-the False bases fall in
        the initial_the=False half (initial_the=True forces false_count=1, since
        the fixed opening "The" already supplies one occurrence). Within the
        initial_the=False half (50% of bases) that means 80% are false_count=0
        and 20% are false_count=1, which yields the global 40% / 60% split.
    """
    n = _N_BASES
    n_initial_the = round(n * _INITIAL_THE_SHARE)
    initial_flags = [True] * n_initial_the + [False] * (n - n_initial_the)

    # true_count: 20% -> 3, else 2.
    n_true_hi = round(n * _TRUE_COUNT_HI_SHARE)
    true_counts = [3] * n_true_hi + [2] * (n - n_true_hi)
    gen.shuffle(true_counts)

    # false_count among the initial_the=False half: enough 0s to make the GLOBAL
    # share 40%. All initial_the=True bases are forced to false_count=1.
    n_false_zero = round(n * _FALSE_COUNT_ZERO_SHARE)  # global count-0 target
    n_false_half = n - n_initial_the                   # size of initial_the=False half
    if n_false_zero > n_false_half:
        raise ValueError(
            "the_appears_twice: cannot place all 0-the False items in the "
            "initial_the=False half (raise _INITIAL_THE_SHARE complement)"
        )

    gen.shuffle(initial_flags)
    profiles: list[tuple[bool, int, int]] = []
    zeros_left = n_false_zero
    # assign false_count: every initial_the=False base takes a 0 until the global
    # 0-quota is met, then 1; every initial_the=True base is 1.
    for i, init in enumerate(initial_flags):
        if init:
            fc = 1
        else:
            if zeros_left > 0:
                fc = 0
                zeros_left -= 1
            else:
                fc = 1
        profiles.append((init, true_counts[i], fc))
    if zeros_left != 0:
        raise ValueError("the_appears_twice: false_count=0 quota not exhausted")
    return profiles


def _assign_the_slots(
    initial_the: bool, true_count: int, false_count: int, gen: Gen
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Pick which slots carry "the" in the True and False variant (False subset
    of True; D1 in/out per ``initial_the``; swing the's spread over D2..D4).

    Returns (the_slots_true, the_slots_false) as sorted slot-name tuples."""
    base_the = ["D1"] if initial_the else []
    init_n = 1 if initial_the else 0

    # swing "the" counts for each variant (D1 already accounts for init_n).
    swing_true_n = true_count - init_n
    swing_false_n = false_count - init_n
    if not (0 <= swing_false_n <= swing_true_n <= len(_SWING_SLOTS)):
        raise ValueError(
            f"the_appears_twice: unreachable counts initial_the={initial_the} "
            f"true={true_count} false={false_count}"
        )

    swing = list(_SWING_SLOTS)
    gen.shuffle(swing)
    swing_true = swing[:swing_true_n]
    # the False swing "the"s are a SUBSET of the True swing "the"s (so True->False
    # only turns some "the"s into non-"the"; D1 and the kept swing "the"s persist).
    swing_false = swing_true[:swing_false_n]

    the_true = tuple(sorted(base_the + swing_true))
    the_false = tuple(sorted(base_the + swing_false))
    return the_true, the_false


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (the GENERATOR INTERFACE).

    For each base: pick a frame, draw four distinct nouns and a verb, derive the
    base's distribution profile (initial_the, true/false the-counts), and fix the
    "the"-slot sets for both variants plus the per-slot non-"the" determiners
    (flip slots get a 3-letter determiner -> char-neutral swap). Verify the word
    count window and that BOTH variant surfaces are globally unique; verify the
    True/False the-counts recompute correctly (defensive — the pipeline re-checks
    ground truth at GATE B)."""
    nouns_bank = banks.get_bank(_NOUN).words()
    verbs_bank = banks.get_bank(_VERB).words()

    profiles = _build_profiles(gen.derive("profiles"))

    bases: list[Base] = []
    seen_ids: set[str] = set()
    seen_surfaces: set[str] = set()

    attempts = 0
    max_attempts = _N_BASES * 800
    fi = 0
    pi = 0
    while len(bases) < _N_BASES and attempts < max_attempts:
        attempts += 1
        frame = _FRAMES[fi % len(_FRAMES)]
        fi += 1
        initial_the, tc, fc = profiles[pi % len(profiles)]
        pi += 1

        # four distinct nouns + a verb
        n1, n2, n3, n4 = gen.sample(nouns_bank, 4)
        verb = gen.choice(verbs_bank)

        the_true, the_false = _assign_the_slots(initial_the, tc, fc, gen.derive(f"slots:{attempts}"))

        # pin non-"the" determiners per slot. Flip slots (in True-the set but NOT
        # in False-the set) MUST use a 3-letter determiner so the False variant is
        # char-length identical to the True variant; other non-"the" slots (non-
        # "the" in BOTH variants) may use any non-"the" determiner. Each slot's
        # non-"the" determiner is fixed so a slot reads identically wherever it is
        # non-"the" across the two variants (no spurious per-variant token).
        true_set = set(the_true)
        false_set = set(the_false)
        non_the: dict[str, str] = {}
        sg = gen.derive(f"det:{attempts}")
        for slot in _DET_SLOTS:
            if slot in true_set and slot not in false_set:
                non_the[slot] = sg.choice(_THREE_CHAR_NON_THE)  # flip slot
            elif slot not in true_set:
                non_the[slot] = sg.choice(_NON_THE)             # non-"the" in both
            # slots in both true_set and false_set are always "the": no non-the.

        spec = Base(
            base_id=_base_id("r21", _frame_index(frame), n1, n2, n3, n4, verb),
            frame=frame,
            nouns=(n1, n2, n3, n4),
            verb=verb,
            the_slots_true=the_true,
            the_slots_false=the_false,
            non_the=tuple(sorted(non_the.items())),
            initial_the=initial_the,
            true_count=tc,
            false_count=fc,
        )
        if spec.base_id in seen_ids:
            continue

        true_text, _ = _render(spec, True)
        false_text, _ = _render(spec, False)
        wc_t = word_count(true_text)
        wc_f = word_count(false_text)
        if not (_MIN_WORDS <= wc_t <= _MAX_WORDS):
            continue
        if wc_t != wc_f:  # determiner swaps only -> must be identical
            continue
        if true_text in seen_surfaces or false_text in seen_surfaces:
            continue
        if true_text == false_text:
            continue
        # defensive ground-truth self-check (GATE B re-checks).
        if not _r21_the_appears_twice(true_text) or _r21_the_appears_twice(false_text):
            continue

        seen_ids.add(spec.base_id)
        seen_surfaces.add(true_text)
        seen_surfaces.add(false_text)
        bases.append(spec)

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"the_appears_twice: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def _render(spec: Base, label: bool) -> tuple[str, dict[str, str]]:
    """Render the surface string for one variant and the per-slot determiner map."""
    the_slots = set(spec.the_slots_true if label else spec.the_slots_false)
    non_the = dict(spec.non_the)
    dets: dict[str, str] = {}
    for slot in _DET_SLOTS:
        dets[slot] = _THE if slot in the_slots else non_the[slot]
    # The subject NP ({D1} {N1}) is always SINGULAR (N1 is a singular NOUN_CONCRETE
    # and is never pluralized), so the present-tense verb takes the regular 3rd-
    # person-singular -s form. Inflecting decide->decides is a single token in
    # both the base and the variant: it is label-neutral (it touches neither the
    # determiner slots nor the count of the word "the", and does not change the
    # word count), so the True/False the-count construction and the base/variant
    # pairing are untouched.
    verb_surface = _regular_verb_forms(spec.verb)[1]  # [1] == 3sg -s form
    fillers = {
        "D1": dets["D1"], "D2": dets["D2"], "D3": dets["D3"], "D4": dets["D4"],
        "N1": spec.nouns[0], "N2": spec.nouns[1], "N3": spec.nouns[2], "N4": spec.nouns[3],
        "V": verb_surface,
    }
    text = to_sentence_case(fill_frame(spec.frame, fillers))
    return text, dets


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True  -> the determiner assignment with 2 or 3 "the" (>= 2 -> rule True).
    False -> the determiner assignment with 1 or 0 "the" (< 2 -> rule False).
    The two variants are word- AND character-length identical (every "the"<->
    non-"the" swap is a 3-letter-for-3-letter substitution). Deterministic;
    ``gen`` is unused (all draws happened in build_bases) but kept to match the
    interface."""
    text, dets = _render(spec, label)
    n_the = sum(1 for tok in words(text) if tok.lower() == _THE)

    # defensive: the count must match the intended class (GATE B re-checks).
    if (n_the >= 2) != label:
        raise ValueError(
            f"the_appears_twice: built text with {n_the} 'the' but label={label}: "
            f"{text!r}"
        )

    meta = {
        "frame_index": _frame_index(spec.frame),
        "frame": spec.frame,
        "nouns": list(spec.nouns),
        "verb": spec.verb,                          # the VERB_REGULAR base (provenance)
        "verb_surface": _regular_verb_forms(spec.verb)[1],  # 3sg -s form actually emitted
        "determiners": dets,
        "the_count": n_the,
        "initial_the": spec.initial_the,
        "transform": "the2plus" if label else ("the1" if spec.false_count == 1 else "the0"),
        "true_count": spec.true_count,
        "false_count": spec.false_count,
    }
    return text, meta
