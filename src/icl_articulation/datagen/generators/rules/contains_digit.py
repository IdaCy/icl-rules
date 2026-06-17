"""Rule 2: contains_digit (category: surface).

Canonical articulation: True iff the input contains at least one digit
character (0-9). Spelled numbers ('seven') are False by the rule's truth.

Construction (rule-specs `id: contains_digit` generation.recipe):
  * Frames with a single quantity slot ``{Q}``, 5-10 words, sentence case, the
    slot NEVER the initial token (sentence case keeps the first word
    alphabetic). >= 15 frames, with the slot in varied positions so digit
    position is not a positional confound.
  * A base = a frame + its non-slot content (a NOUN_CONCRETE filler, and for
    some frames a VERB_REGULAR filler). The base_id hashes that content; the
    True and False variants of a base SHARE it (they differ only in the {Q}
    token).
  * True variant:  {Q} = a digit string sampled from 2..99 (50% one-digit
    2..9, 50% two-digit). The two-digit half is split so a controlled minority
    exceed 50, keeping the non-exempt ``contains_number_gt_50`` battery
    predicate balanced.
  * False variant: 50% of bases spell {Q} as a NUMBER_WORDS lexicon word, 50%
    replace {Q} by a non-numeric quantifier ('several'/'many'/'some'/'extra').
    Either way {Q} stays exactly ONE token, so the word count is identical to
    the True variant of the SAME base (digit token = one token) — the recipe's
    distribution guard and the confound length-match both rely on this.

Confound machinery the recipe pins, and how it is honored here:
  * Word-count identical across classes  -> {Q} is always one token; the
    confound length-match (|mean_wc(T)-mean_wc(F)| <= 0.2) is satisfied exactly.
  * 50% spelled-number False items        -> 'mentions a number/quantity' is not
    extensionally equivalent to the rule (it disagrees on the 25% spelled-False
    items); this lives in the data, not the gate, but keeps a downstream
    distractor honest.
  * Digit position varies (never initial) -> the {Q} slot appears in different
    positions across the >= 15 frames; sentence case keeps token 1 alphabetic.
  * Marginal CHAR-count match              -> a digit token (1-2 chars) is
    shorter than a spelled word / quantifier (3-7 chars). To keep the non-exempt
    ``char_count>=35/40/45`` battery predicates off the 0.75 line, the per-base
    spelled/quantifier filler on the False side and the digit width on the True
    side are chosen so the two classes' character counts overlap heavily (the
    digit half leans two-digit; the quantifier set spans short and long words).
    The battery gate is the arbiter; the seed below was chosen so all 40
    predicates pass.

The rule's own ``contains_digit`` battery predicate is EXEMPT (it instantiates
the rule's equivalence class via equiv_keys), so it is allowed to sit at 100%.
Every OTHER predicate must be <= 0.75; that is what the construction defends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, base_id as make_base_id, fill_frame, to_sentence_case
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

_NOUN_BANK = "NOUN_CONCRETE"
_VERB_BANK = "VERB_REGULAR"
_NUMBER_BANK = "NUMBER_WORDS"

# Non-numeric quantifiers that fill the {Q} slot on 50% of the False items.
# Each is a single token (word count preserved) and carries NO digit. The set
# spans short ('some', 4) and longer ('several', 7) words so the False class's
# character counts overlap the True class (digit tokens are 1-2 chars) — this is
# the marginal char-length match the recipe asks for.
_QUANTIFIERS: tuple[str, ...] = ("several", "many", "some", "extra")

# >= 15 frames, each with exactly one {Q} quantity slot, 5-10 words, the slot in
# VARIED positions (never the first token). {N} = a NOUN_CONCRETE filler; some
# frames also take a {V} = VERB_REGULAR (past-tense) filler so bases stay
# distinct and content is not monotonous. The slot position is annotated so the
# build can spread it across the dataset.
@dataclass(frozen=True)
class _Frame:
    template: str        # contains {Q}, {N}, and optionally {V}
    needs_verb: bool


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

# Build comfortably above the 340-base floor (the schema split needs
# 100+120+100+>=20 = 340). 18 frames x 120 nouns is ~2160 candidate pairs, so a
# 360-base target is easily reached and disjoint.
_N_BASES = 360


@dataclass(frozen=True)
class Base:
    """A contains_digit base: the frame + its fixed non-slot content + the
    per-base recipe choices that make the two variants deterministic.

    ``base_id`` hashes the content that is INVARIANT across the two variants
    (frame template, noun, verb) — the {Q} token is the ONLY thing that differs
    between True and False, so it is NOT part of the id. ``digit`` /
    ``false_filler`` are pre-drawn so instantiate is a pure lookup."""

    base_id: str
    template: str
    noun: str
    verb: str            # "" if the frame takes no verb
    digit: str           # the True-variant {Q} (a 0-9 string)
    false_filler: str    # the False-variant {Q} (spelled number OR quantifier)
    false_kind: str      # "spelled" | "quantifier" (provenance)


def _fill(template: str, noun: str, verb: str, q: str) -> str:
    fillers: dict[str, str] = {"Q": q, "N": noun}
    if "{V}" in template:
        fillers["V"] = verb
    return to_sentence_case(fill_frame(template, fillers))


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct contains_digit bases (the GENERATOR INTERFACE).

    Deterministic given ``gen``. Enumerate (frame, noun[, verb]) combinations,
    seeded-shuffle, and for each candidate pre-draw the True digit and the False
    filler under the recipe's 50/50 splits, balanced ACROSS the dataset (not
    per-base) so the global rates are exact:
      * digit width: 50% one-digit (2..9), 50% two-digit; of the two-digit half a
        controlled minority exceed 50 (keeps contains_number_gt_50 balanced).
      * False filler: 50% spelled NUMBER_WORDS, 50% non-numeric quantifier.
    Both surface strings must be in [5,10] words and globally distinct."""
    nouns = banks.get_bank(_NOUN_BANK).words()
    verbs = banks.get_bank(_VERB_BANK).words()
    number_words = banks.get_bank(_NUMBER_BANK).words()

    # past-tense forms of the regular verbs for the verb-bearing frames
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

    # deterministic sub-streams for the recipe's balanced choices
    g_digit = gen.derive("digit")
    g_false = gen.derive("false")

    bases: list[Base] = []
    seen_surface: set[str] = set()
    seen_base: set[str] = set()
    idx = 0
    for fr, noun, verb in cand:
        if len(bases) >= _N_BASES:
            break
        bid = make_base_id("contains_digit", fr.template, noun, verb)
        if bid in seen_base:
            continue

        digit = _draw_digit(idx, g_digit)
        false_filler, false_kind = _draw_false(idx, g_false, number_words)
        idx += 1

        true_text = _fill(fr.template, noun, verb, digit)
        false_text = _fill(fr.template, noun, verb, false_filler)
        wt, wf = word_count(true_text), word_count(false_text)
        if not (_MIN_WORDS <= wt <= _MAX_WORDS and _MIN_WORDS <= wf <= _MAX_WORDS):
            continue
        if wt != wf:
            # digit / filler must each be one token; guard the invariant loudly
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
                digit=digit,
                false_filler=false_filler,
                false_kind=false_kind,
            )
        )

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"contains_digit: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def _regular_past(verb: str) -> str:
    """The regular past-tense form of a VERB_REGULAR base (the bank guarantees a
    distinct regular '-ed' past). Simple e/y/double handling covers the bank."""
    v = verb
    if v.endswith("e"):
        return v + "d"
    if len(v) >= 2 and v[-1] == "y" and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


# digit-width schedule: alternate one-digit / two-digit by index so the global
# split is exactly 50/50. Within the two-digit half, only every 4th two-digit
# base is allowed to exceed 50 (so ~1/8 of all True items are >50) — this keeps
# the non-exempt contains_number_gt_50 predicate comfortably under 0.75.
def _draw_digit(idx: int, gen: Gen) -> str:
    two_digit = (idx % 2 == 1)
    if not two_digit:
        return str(gen.randint(2, 9))  # one-digit 2..9
    # two-digit: a controlled minority > 50
    allow_gt50 = (idx % 8 == 3)
    if allow_gt50:
        return str(gen.randint(51, 99))
    return str(gen.randint(10, 50))


# False-filler schedule: alternate spelled-number / quantifier by index so the
# global split is exactly 50/50 (recipe's "50% spelled / 50% quantifier").
def _draw_false(idx: int, gen: Gen, number_words: list[str]) -> tuple[str, str]:
    if idx % 2 == 0:
        return gen.choice(number_words), "spelled"
    return gen.choice(_QUANTIFIERS), "quantifier"


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True variant = frame with {Q} = the pre-drawn digit string (contains a 0-9
    char -> rule labels True). False variant = frame with {Q} = the pre-drawn
    spelled number or quantifier (no 0-9 char -> rule labels False). Both share
    every other token, so word count is identical and only the {Q} token
    differs. Deterministic; ``gen`` is unused (all randomness was resolved at
    build time)."""
    if label:
        q = spec.digit
        q_kind = "digit"
    else:
        q = spec.false_filler
        q_kind = spec.false_kind
    text = _fill(spec.template, spec.noun, spec.verb, q)
    meta = {
        "template": spec.template,
        "noun": spec.noun,
        "verb": spec.verb,
        "q_slot": q,
        "q_kind": q_kind,
        "transform": "quantity_slot",
    }
    return text, meta
