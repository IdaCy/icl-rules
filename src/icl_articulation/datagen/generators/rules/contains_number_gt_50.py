"""Rule 24: contains_number_gt_50 (category: numeric).

Canonical articulation: True iff the input contains a number WRITTEN IN DIGITS
whose value is greater than 50 (strictly: '50' is False, '51' is True). Spelled
numbers never count and never occur in training.

Construction (rule-specs `id: contains_number_gt_50` generation.recipe):
  * Quantity frames as in ``contains_digit``: every item — BOTH classes —
    contains EXACTLY ONE digit number in a single ``{Q}`` slot. >= 15 frames,
    5-10 words, sentence case, the ``{Q}`` slot NEVER the initial token (so
    sentence case keeps token 1 alphabetic and digit position is not a
    positional confound). ``{N}`` = a NOUN_CONCRETE filler; verb-bearing frames
    take a ``{V}`` = a VERB_REGULAR past-tense filler.
  * A base = a frame + its non-slot content (noun [+ verb]). The base_id hashes
    that INVARIANT content; the True and False variants of a base SHARE every
    token EXCEPT the number filling ``{Q}`` (the recipe: "variants differ only
    in the number value").
  * True  variant: ``{Q}`` = a digit string uniform from 51-98 -> value > 50.
  * False variant: ``{Q}`` = a digit string uniform from 10-50 -> value <= 50
    ('50' is False per the strict threshold; it appears in the data).
  * ALL numbers are two-digit (10-98) -> the digit-string LENGTH (and so the
    item's character count, word count, and every token boundary) is IDENTICAL
    across the two classes. The number's VALUE is the one and only signal.

Why every gate passes by construction:
  * The two variants of a base are token-identical except for a 2-char digit run
    that is 2 chars in BOTH classes. Therefore EVERY one of the 40 frozen battery
    predicates returns the SAME truth value on a base's True and False variant,
    so each predicate agrees with the label on exactly half the (balanced) data:
    its agreement is exactly 0.5 and its battery score 0.5 (<= 0.75). This rule's
    own discriminating feature (number VALUE > 50) is NOT one of the 40 frozen
    predicates, so nothing in the battery can lock onto it — the same "the rule's
    cue is the only signal" property the reference rule (all_lowercase) has.
    contains_number_gt_50 has NO equiv_keys / battery_exemptions (spec extract):
    none is needed, because no frozen predicate exceeds 0.75.
  * Word count identical across classes -> the confound length-match
    (|mean_wc(T) - mean_wc(F)| <= 0.2) is satisfied EXACTLY (diff 0.0).
  * Boundary density: the False range 10-50 and True range 51-98 jointly put the
    41-60 band (41-50 False, 51-60 True) on both sides of the threshold; with
    uniform draws the band's share is well over the recipe's >= 20% target
    (verified at build time, asserted loudly).
  * No number WORDS, no units skew: the only numerics are the ``{Q}`` digit run,
    and the unit noun ``{N}`` is shared between a base's two variants.

groundtruth.assert_labels_correct recomputes r24 from text by reading every
maximal digit run as an int and testing ``any(value > 50)``; the True variant's
sole run is 51-98 (> 50 -> True), the False variant's sole run is 10-50
(<= 50 -> False), so the recompute matches the stored label for every item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, base_id as make_base_id, fill_frame, to_sentence_case
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

_NOUN_BANK = "NOUN_CONCRETE"
_VERB_BANK = "VERB_REGULAR"

# the strict threshold and the two class-conditional UNIFORM digit ranges. Both
# ranges are entirely two-digit (10..98), so the digit token is always exactly 2
# characters in either class -> digit-string length carries no signal.
_THRESHOLD = 50
_TRUE_LO, _TRUE_HI = 51, 98     # value > 50
_FALSE_LO, _FALSE_HI = 10, 50   # value <= 50 (50 is False, strictly-greater rule)


@dataclass(frozen=True)
class _Frame:
    template: str        # contains exactly one {Q}, one {N}, optionally one {V}
    needs_verb: bool


# >= 15 frames, each with exactly one quantity slot {Q}, 5-10 words, the slot in
# VARIED, NON-INITIAL positions so the digit position is not a positional
# confound (sentence case keeps token 1 alphabetic). {N} = NOUN_CONCRETE; some
# frames add {V} = VERB_REGULAR (past tense) so bases stay distinct / not
# monotonous. These mirror contains_digit's quantity frames (the recipe: "Quantity
# frames as in contains_digit").
_FRAMES: tuple[_Frame, ...] = (
    _Frame("The driver delivered {Q} {N} before noon", False),
    _Frame("They counted {Q} {N} in the hall", False),
    _Frame("The teacher collected {Q} {N} after the lesson", False),
    _Frame("Workers stacked {Q} {N} along the wall", False),
    _Frame("The shop sold {Q} {N} on the first day", False),
    _Frame("We packed {Q} {N} into the small van", False),
    _Frame("The farmer loaded {Q} {N} onto the truck", False),
    _Frame("Volunteers gathered {Q} {N} near the gate", False),
    _Frame("The crew unloaded {Q} {N} at the dock", False),
    _Frame("She arranged {Q} {N} across the long table", False),
    _Frame("The children found {Q} {N} under the porch", False),
    _Frame("Guards inspected {Q} {N} beside the entrance", False),
    _Frame("The clerk wrapped {Q} {N} behind the counter", False),
    _Frame("Engineers tested {Q} {N} inside the cold lab", False),
    _Frame("The cook prepared {Q} {N} during the busy shift", False),
    _Frame("They {V} {Q} {N} near the river", True),
    _Frame("The team {V} {Q} {N} over the weekend", True),
    _Frame("Neighbours {V} {Q} {N} along the road", True),
)

_MIN_WORDS, _MAX_WORDS = 5, 10

# Build comfortably above the 340-base floor (100 few_shot + 120 held_out +
# 100 confirmation + >= 20 spare = 340). 18 frames x 131 nouns is ~2360 candidate
# (frame, noun) pairs (more with verbs), so 360 is easily reached and disjoint.
_N_BASES = 360

# the 41-60 boundary band the recipe asks to keep dense (>= 20% of items). With
# True ~ U[51,98] and False ~ U[10,50], the band is 51-60 on the True side
# (10/48 ~= 20.8%) and 41-50 on the False side (10/41 ~= 24.4%); the joint share
# is ~22.6% > 20%. Asserted loudly at build time so a frame/range edit can never
# silently drop below target.
_BOUNDARY_LO, _BOUNDARY_HI = 41, 60
_BOUNDARY_MIN_FRAC = 0.20


@dataclass(frozen=True)
class Base:
    """A contains_number_gt_50 base: the frame + its fixed non-slot content + the
    two pre-drawn number values that make the variants deterministic.

    ``base_id`` hashes only the content INVARIANT across the two variants (frame
    template, noun, verb). The number filling {Q} is the ONLY thing that differs
    between True and False, so it is NOT part of the id. ``true_num`` /
    ``false_num`` are pre-drawn (uniform 51-98 / 10-50) so instantiate is a pure
    lookup with no further randomness."""

    base_id: str
    template: str
    noun: str
    verb: str          # "" if the frame takes no verb
    true_num: int      # the True-variant {Q}, uniform 51..98 (> 50)
    false_num: int     # the False-variant {Q}, uniform 10..50 (<= 50)


def _fill(template: str, noun: str, verb: str, q: str) -> str:
    fillers: dict[str, str] = {"Q": q, "N": noun}
    if "{V}" in template:
        fillers["V"] = verb
    return to_sentence_case(fill_frame(template, fillers))


def _regular_past(verb: str) -> str:
    """The regular past-tense form of a VERB_REGULAR base (simple e/y/double rules
    cover the bank). Mirrors contains_digit's helper so verb-bearing frames read
    naturally; the number value, not the verb, is the rule's signal."""
    v = verb
    if v.endswith("e"):
        return v + "d"
    if len(v) >= 2 and v[-1] == "y" and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct contains_number_gt_50 bases (the GENERATOR INTERFACE).

    Deterministic given ``gen``. Enumerate (frame, noun[, verb]) combinations,
    seeded-shuffle, and for each candidate pre-draw the True number (uniform
    51-98) and the False number (uniform 10-50). Both surface strings must be in
    [5,10] words and globally distinct, and — the recipe's core invariant — the
    two variants must have IDENTICAL word counts (guaranteed: a two-digit number
    is one token on both sides). Asserts the >= 20% boundary-band density over
    the actually-emitted numbers (loud) and raises if it cannot reach the floor."""
    nouns = banks.get_bank(_NOUN_BANK).words()
    verbs = banks.get_bank(_VERB_BANK).words()
    verb_past = [_regular_past(v) for v in verbs]

    # candidate (frame, noun, verb) tuples
    cand: list[tuple[_Frame, str, str]] = []
    for fr in _FRAMES:
        if fr.needs_verb:
            for n in nouns:
                for vp in verb_past:
                    cand.append((fr, n, vp))
        else:
            for n in nouns:
                cand.append((fr, n, ""))
    gen.shuffle(cand)

    # deterministic sub-streams for the two class-conditional uniform draws
    g_true = gen.derive("true_num")
    g_false = gen.derive("false_num")

    bases: list[Base] = []
    seen_surface: set[str] = set()
    seen_base: set[str] = set()
    for fr, noun, verb in cand:
        if len(bases) >= _N_BASES:
            break
        bid = make_base_id("contains_number_gt_50", fr.template, noun, verb)
        if bid in seen_base:
            continue

        true_num = g_true.randint(_TRUE_LO, _TRUE_HI)
        false_num = g_false.randint(_FALSE_LO, _FALSE_HI)

        true_text = _fill(fr.template, noun, verb, str(true_num))
        false_text = _fill(fr.template, noun, verb, str(false_num))
        wt, wf = word_count(true_text), word_count(false_text)
        if not (_MIN_WORDS <= wt <= _MAX_WORDS and _MIN_WORDS <= wf <= _MAX_WORDS):
            continue
        if wt != wf:
            # both numbers are two-digit (one token each) so this can never
            # trip; guard the recipe's "lengths matched" invariant loudly anyway.
            continue
        if true_text in seen_surface or false_text in seen_surface:
            continue
        seen_surface.add(true_text)
        seen_surface.add(false_text)
        seen_base.add(bid)
        bases.append(
            Base(
                base_id=bid,
                template=fr.template,
                noun=noun,
                verb=verb,
                true_num=true_num,
                false_num=false_num,
            )
        )

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"contains_number_gt_50: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )

    # verify the recipe's >= 20% boundary-band (41-60) density over the numbers
    # that will actually be emitted (both variants of every base appear in the
    # data — few_shot emits both, the one-variant splits pick one — so counting
    # both numbers per base is the right marginal estimate).
    nums = [n for b in bases for n in (b.true_num, b.false_num)]
    in_band = sum(1 for n in nums if _BOUNDARY_LO <= n <= _BOUNDARY_HI)
    frac = in_band / len(nums)
    if frac < _BOUNDARY_MIN_FRAC:
        raise ValueError(
            f"contains_number_gt_50: boundary band [{_BOUNDARY_LO},{_BOUNDARY_HI}] "
            f"density {frac:.3f} < required {_BOUNDARY_MIN_FRAC}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True variant = frame with {Q} = the pre-drawn 51-98 number (value > 50 ->
    rule labels True). False variant = frame with {Q} = the pre-drawn 10-50
    number (value <= 50 -> rule labels False). Both share every other token, so
    word/char count are identical and only the number VALUE differs.
    Deterministic; ``gen`` is unused (all randomness was resolved at build time)."""
    num = spec.true_num if label else spec.false_num
    text = _fill(spec.template, spec.noun, spec.verb, str(num))
    meta = {
        "template": spec.template,
        "noun": spec.noun,
        "verb": spec.verb,
        "q_slot": str(num),
        "q_value": num,
        "q_kind": "digit_number",
        "number_class": "gt50" if label else "le50",
        "transform": "quantity_slot",
    }
    return text, meta
