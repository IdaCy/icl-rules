"""Rule 19: first_word_longer_than_last (category positional).

Canonical articulation: True iff the input's FIRST word has STRICTLY more
(alphabetic) letters than its LAST word. Ties are False (and excluded from
training — see the recipe). Length is the global ``alphabetic_length`` (all bank
words are purely alphabetic, so no edge bites).

Construction (rule-specs generation.recipe + distribution_guards)
-----------------------------------------------------------------
Frame:  ``{W1} {middle...} {W2}``  with a SHARED middle across the two variants
of a base. W1 is the first word, drawn from INITIAL_BY_LENGTH; W2 is the last
word, drawn from FINAL_BY_LENGTH. Both banks span lengths 3-11 with >= 6 words
per length.

PER-BASE LENGTH SWAP (the marginal-matching machinery). Each base fixes an
unordered length pair {hi, lo}, hi > lo (ties NEVER occur in training), and a
shared middle. Its two variants are CHARACTER-DISJOINT only in W1/W2:

    True  variant : first word length = hi, last word length = lo  -> hi > lo -> True
    False variant : first word length = lo, last word length = hi  -> lo < hi -> False

So the two variants of a base re-use the SAME pair of lengths in swapped
positions. Pooled over a base's T+F variants, the multiset of first-word lengths
({hi, lo}) equals the multiset of last-word lengths ({lo, hi}); the joint ORDER
is the only thing that flips with the label. This is what the recipe means by
"False items use the SWAPPED length pairs ... so the marginal distribution of
first-word lengths and last-word lengths is IDENTICAL across classes; only the
joint (order) differs." It removes "the first word is long" / "the last word is
short" as standalone dataset-level facts.

NEAR-BOUNDARY SAMPLING. The length gap hi-lo is sampled 50% = 1, 30% = 2, 20%
>= 3 (recipe). A small gap means a fixed length threshold k cannot cleanly
separate the classes, so every battery ``first_word_len>=k`` / ``last_word_len<=k``
predicate stays under the 0.75 agreement gate (the gap-1 majority makes both
classes straddle every k).

FIRST-WORD POS MATCHING. INITIAL_BY_LENGTH mixes one noun + five adverbs at
length 3 and is all plural nouns at lengths 4-11. If a length-3 first word were
used, the (rare) adverb opener would correlate with the False class (whose first
word is the SHORTER, hence more-often length-3, word) and skew the
``first_word_pos=adverb`` / ``=noun`` battery predicates. To keep first-word POS
class-independent we draw W1 ONLY from lengths 4-11 (all plural nouns); length 3
is used only for the LAST word (W2), where POS is not a battery feature.
JUDGMENT CALL recorded in open_concerns: the recipe lists W1 lengths as 3-11; we
use the 4-11 noun subset (still within INITIAL_BY_LENGTH, >= 6 per length) so the
6 first-word-POS predicates sit at 50%.

SYLLABLE DISCORDANCE. FINAL_BY_LENGTH flags letter/syllable-discordant words
('through' = 7 letters / 1 syllable, 'idea' = 4 letters / 3 syllables). We force
a >= 25% share of items to carry a discordant W1 or W2 so the "first word has
more syllables than the last" distractor cannot track the (letter-based) labels
at >= 75%.

FIXED WORD COUNT. Every item is exactly 7 words (W1 + a 5-word middle + W2), so
the T-vs-F word-count means are identical (Gate D length matching = 0 diff) and
all counts sit inside the global [4, 14] window. Middles are shared per base, so
a base's two variants are word-count-identical regardless of which split keeps
which variant.

base_id = middle + length-pair assignment (recipe: "base_id = middle +
length-pair assignment"), hashed for a stable filesystem-safe id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, alphabetic_length, base_id as make_base_id
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

# --- banks --------------------------------------------------------------------

_INITIAL_BANK = "INITIAL_BY_LENGTH"
_FINAL_BANK = "FINAL_BY_LENGTH"

# First word (W1): plural-noun lengths only (4-11) so first-word POS is constant
# across classes (see module docstring). Last word (W2): full 3-11 range.
_W1_LENGTHS = tuple(range(4, 12))   # 4..11, all nouns in INITIAL_BY_LENGTH
_W2_LENGTHS = tuple(range(3, 12))   # 3..11

# Shared 5-word middles. Each connects a plural-noun first word (the subject) to
# a noun/adverb last word, with NO comma / terminal punctuation / 'I' / proper
# noun and exactly five whitespace tokens. Sentence-cased at instantiate time.
# Variety (>= ~40 middles) gives the by-base split plenty of distinct base_ids.
_MIDDLES: tuple[str, ...] = (
    "moved slowly toward the quiet",
    "gathered early beside the old",
    "waited calmly near the empty",
    "stayed close behind the broken",
    "drifted gently past the silent",
    "looked carefully across the wooden",
    "rested quietly under the open",
    "walked steadily along the narrow",
    "remained safely inside the small",
    "appeared briefly above the distant",
    "settled neatly around the round",
    "lingered softly beneath the heavy",
    "spread quickly through the crowded",
    "turned sharply toward the bright",
    "leaned gently against the smooth",
    "pressed firmly onto the flat",
    "floated freely over the still",
    "huddled tightly within the cold",
    "shifted slowly between the tall",
    "scattered widely across the dusty",
    "marched proudly past the grand",
    "paused briefly before the locked",
    "circled slowly around the frozen",
    "climbed steadily above the rocky",
    "wandered aimlessly toward the misty",
    "vanished quietly behind the painted",
    "emerged slowly from the shaded",
    "stretched widely across the golden",
    "gathered quietly around the warm",
    "drifted lazily along the muddy",
    "stood firmly upon the cracked",
    "moved quietly past the sleeping",
    "waited patiently near the modern",
    "rested gently against the curved",
    "spread evenly across the polished",
    "lingered long beside the ancient",
    "swept gently over the gentle",
    "settled softly onto the velvet",
    "traveled slowly toward the hidden",
    "passed quietly behind the iron",
    "drifted softly beneath the pale",
    "gathered tightly around the carved",
    "moved gently toward the silver",
    "waited quietly beside the marble",
)

# near-boundary gap distribution (recipe): 50% gap 1, 30% gap 2, 20% gap >= 3.
# Realized as exact integer quotas over the base count so the share is pinned
# (not just sampled), then shuffled.
_GAP_WEIGHTS = ((1, 0.50), (2, 0.30), (3, 0.20))  # gap 3 stands for ">= 3"; we
# spread the ">= 3" share over the gaps that the [4..11] x [3..11] grid admits.

# discordant-share floor (recipe: >= 25%).
_DISCORDANT_MIN_FRAC = 0.25

# total words per item (W1 + 5-word middle + W2).
_TARGET_WORDS = 7

# build comfortably over the 340-base floor.
_N_BASES = 360


# --- length bookkeeping -------------------------------------------------------


def _words_by_length(bank_name: str, lengths: tuple[int, ...]) -> dict[int, list[banks.Entry]]:
    bank = banks.get_bank(bank_name)
    out: dict[int, list[banks.Entry]] = {L: [] for L in lengths}
    for e in bank.entries:
        if e.length in out:
            out[e.length].append(e)
    for L in lengths:
        if len(out[L]) < 1:
            raise ValueError(f"{bank_name}: no entries of length {L}")
    return out


def _legal_pairs() -> list[tuple[int, int]]:
    """All (hi, lo) length pairs with hi from W1 lengths, lo from W2 lengths,
    hi > lo AND the swapped (lo as a legal W1 length, hi as a legal W2 length)
    also realizable — both variants must be buildable from the banks."""
    pairs: list[tuple[int, int]] = []
    for hi in _W1_LENGTHS:
        for lo in _W2_LENGTHS:
            if hi <= lo:
                continue
            # swapped variant needs: first word length = lo (must be a W1 length),
            # last word length = hi (must be a W2 length).
            if lo in _W1_LENGTHS and hi in _W2_LENGTHS:
                pairs.append((hi, lo))
    return pairs


# --- base spec ----------------------------------------------------------------


@dataclass(frozen=True)
class Base:
    """One base: a shared middle + an unordered length pair (hi > lo).

    The True variant puts the hi-length word first / lo-length word last; the
    False variant swaps them. ``force_discordant`` asks instantiate to prefer a
    syllable-discordant FINAL word for the discordance quota."""

    base_id: str
    middle: str
    hi: int
    lo: int
    force_discordant: bool


def _quota(n: int, frac: float) -> int:
    return int(round(n * frac))


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (the GENERATOR INTERFACE).

    Deterministic given ``gen``: assign each base a length gap from the pinned
    near-boundary quota (50/30/20), pick a concrete (hi, lo) pair with that gap,
    pair it with a distinct (middle, pair) combination, and flag ~25% for a
    discordant final word. Raises if the floor cannot be reached (loud)."""
    pairs = _legal_pairs()
    by_gap: dict[int, list[tuple[int, int]]] = {}
    for hi, lo in pairs:
        by_gap.setdefault(hi - lo, []).append((hi, lo))
    gaps_avail = sorted(by_gap)

    # pinned gap quotas: gap 1, gap 2, gap >= 3 (the last spread over the larger
    # gaps the grid admits).
    n = _N_BASES
    n_g1 = _quota(n, 0.50)
    n_g2 = _quota(n, 0.30)
    n_g3plus = n - n_g1 - n_g2  # remainder -> >= 3

    gap_plan: list[int] = [1] * n_g1 + [2] * n_g2
    big_gaps = [g for g in gaps_avail if g >= 3]
    for i in range(n_g3plus):
        gap_plan.append(big_gaps[i % len(big_gaps)])
    gen.shuffle(gap_plan)

    # discordant flags: >= 25% True, exact quota, shuffled.
    n_disc = _quota(n, _DISCORDANT_MIN_FRAC) + 1  # +1 cushion above the floor
    disc_plan = [True] * n_disc + [False] * (n - n_disc)
    gen.shuffle(disc_plan)

    bases: list[Base] = []
    seen_ids: set[str] = set()
    used_pair_middle: set[tuple[str, int, int]] = set()
    middle_cursor = 0
    for i, gap in enumerate(gap_plan):
        cands = list(by_gap[gap])
        gen.shuffle(cands)
        placed = False
        for hi, lo in cands:
            # find a (middle, hi, lo) combo not yet used (keeps base_ids + the
            # surface strings distinct).
            for _ in range(len(_MIDDLES)):
                middle = _MIDDLES[middle_cursor % len(_MIDDLES)]
                middle_cursor += 1
                key = (middle, hi, lo)
                if key in used_pair_middle:
                    continue
                bid = make_base_id("first_word_longer_than_last", middle, hi, lo)
                if bid in seen_ids:
                    continue
                used_pair_middle.add(key)
                seen_ids.add(bid)
                bases.append(
                    Base(
                        base_id=bid,
                        middle=middle,
                        hi=hi,
                        lo=lo,
                        force_discordant=disc_plan[i],
                    )
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            # exhausted (middle, pair) combos for this gap; skip (we over-provision).
            continue

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"first_word_longer_than_last: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


# --- instantiation ------------------------------------------------------------


def _pick_word(
    entries_by_len: dict[int, list[banks.Entry]],
    length: int,
    gen: Gen,
    *,
    prefer_discordant: bool,
) -> banks.Entry:
    """Deterministically pick a bank entry of ``length``; prefer a discordant
    entry when asked and one exists."""
    pool = entries_by_len[length]
    if prefer_discordant:
        disc = [e for e in pool if e.discordant]
        if disc:
            return gen.choice(disc)
    return gen.choice(pool)


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True  : first word length = hi, last word length = lo  (hi > lo -> True).
    False : first word length = lo, last word length = hi  (lo < hi -> False).
    The two variants share the middle; only W1/W2 differ (swapped lengths).
    Deterministic given (spec, label, gen)."""
    w1_by_len = _words_by_length(_INITIAL_BANK, _W1_LENGTHS)
    w2_by_len = _words_by_length(_FINAL_BANK, _W2_LENGTHS)

    # per-(base,label) sub-stream so word choice is stable and order-independent.
    g = gen.derive(f"{spec.base_id}:{'T' if label else 'F'}")

    if label:
        first_len, last_len = spec.hi, spec.lo
    else:
        first_len, last_len = spec.lo, spec.hi

    # honor the discordance flag on whichever side has a discordant option; the
    # FINAL bank carries the discordant words, so prefer it for the last word.
    last_disc = spec.force_discordant
    w1 = _pick_word(w1_by_len, first_len, g.derive("w1"), prefer_discordant=False)
    w2 = _pick_word(w2_by_len, last_len, g.derive("w2"), prefer_discordant=last_disc)

    raw = f"{w1.word} {spec.middle} {w2.word}"
    # sentence case (style: first letter capitalized; case is irrelevant to the
    # letter-count rule per the spec's ambiguity_notes).
    text = raw[:1].upper() + raw[1:]

    # provenance: the recipe machinery that produced this surface.
    meta = {
        "rule": "first_word_longer_than_last",
        "middle": spec.middle,
        "hi": spec.hi,
        "lo": spec.lo,
        "first_word": w1.word,
        "first_len": alphabetic_length(w1.word),
        "last_word": w2.word,
        "last_len": alphabetic_length(w2.word),
        "first_pos": w1.pos,
        "last_pos": w2.pos,
        "last_discordant": bool(w2.discordant),
        "gap": spec.hi - spec.lo,
        "transform": "hi_first" if label else "lo_first",
        "word_count": word_count(text),
    }
    return text, meta
