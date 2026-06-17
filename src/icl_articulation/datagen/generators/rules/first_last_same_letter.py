"""Rule 27: first_last_same_letter (category hard).

Canonical articulation: True iff the input's FIRST word and its LAST word begin
with the SAME letter, ignoring case. First LETTER (initial character) only — not
rhyme, not the whole word, not the sentence's last *character*. Digit-initial
words have no letter and cannot match (ruled False; never occur in training).

Construction (rule-specs generation.recipe + distribution_guards)
-----------------------------------------------------------------
Frame: ``{W1} {middle...} {W2}``. W1 (the first word) is a sentence-initial,
capitalized plural noun from NOUN_PLURAL_BY_LETTER; W2 (the genuine last word) is
a sentence-final singular place noun from NOUN_FINAL_BY_LETTER. Both banks are
indexed by the 14 shared initial letters {b,c,d,f,g,h,l,m,n,p,r,s,t,w} with >= 5
entries per letter. The middle is shared VERBATIM between the two variants of a
base, so every middle-derived feature (commas, 'the'/'a'/'and', word/char count
contributed by the middle) is identical across a base's True and False variants.

PER-BASE LETTER DERANGEMENT (the marginal-matching machinery)
-------------------------------------------------------------
Each base fixes a shared middle, a first-word letter ``L1`` (so W1 begins with
L1), and a last-word length ``LL``. The True variant's last word also begins with
L1; the False variant's last word begins with a DIFFERENT letter ``L2`` (L2 !=
L1). The two variants of a base re-use the SAME W1 and the SAME last-word LENGTH;
only the last word's INITIAL LETTER flips with the label:

    True  variant : first letter = L1, last word begins with L1 -> MATCH  -> True
    False variant : first letter = L1, last word begins with L2 -> no match -> False

L2 is produced by DERANGING the first-letter assignment WITHIN each last-word
length group (a permutation of the L1's with no fixed point). Two invariants fall
out, exactly as the recipe demands:

  * first-word initial marginal IDENTICAL across classes — W1 is the same word in
    both variants of a base, so the multiset of first-word initials is the same
    in the True and the False class (each base contributes its L1 to both).
  * last-word initial marginal IDENTICAL across classes — the True class's
    last-word initials are {L1_i}; the False class's are {L2_i}, a derangement
    (permutation) of {L1_i} restricted to each length group, hence the SAME
    multiset overall. Only the JOINT (first==last vs first!=last) differs.

Because the derangement stays WITHIN a length group, L2 has a last word of the
SAME length LL as the True last word, so the two variants of every base share an
identical word count AND character count (W1 shared, middle shared, last word
length shared). Gate D length matching is therefore 0.0 and every
``word_count>=k`` / ``char_count>=k`` / ``first_word_len>=k`` / ``last_word_len<=k``
battery predicate is constant across a base's variants -> sits at the ~0.5 base
rate. The per-letter / per-position predicates (``first_letter_bucket_*``,
``first_starts_*``, ``last_ends_*``) sit at ~0.5 too, because the marginal letter
distributions match across classes (this is the recipe's "<= 60% agreement"
distribution guard, comfortably under the 0.75 gate).

ALLITERATION GUARD (salience knob). The recipe forbids the middle from making an
item alliterative: an item is REJECTED (in BOTH classes) if >= 3 of its words
share an initial letter. Middles are hand-written so their words have mutually
distinct initials and avoid the bank letters; instantiate re-checks the realized
True AND False surfaces and the base is dropped if either alliterates, so
'the sentence is alliterative' is never a proxy for the rule (it is constant
False) and the matching pair stays perceptually salient.

NO REPEATED FIRST/LAST WORD. W1 is always a PLURAL noun and W2 a SINGULAR place
noun, so the first and last words are never the same token (the spec's "same
word -> True, excluded from training" edge cannot arise here; it is left for
step-3 probes).

base_id = middle + the (L1, LL, L2) letter-pair/length slot (recipe: "base_id =
middle + letter-pair slot"), hashed for a stable, filesystem-safe id.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, base_id as make_base_id
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count, words

# --- banks --------------------------------------------------------------------

_W1_BANK = "NOUN_PLURAL_BY_LETTER"   # first word: sentence-initial plural noun
_W2_BANK = "NOUN_FINAL_BY_LETTER"    # last word: sentence-final singular place noun

# The 14 shared initial letters both by-letter banks are indexed on.
_LETTERS: tuple[str, ...] = ("b", "c", "d", "f", "g", "h", "l", "m", "n", "p", "r", "s", "t", "w")

# Last-word lengths whose per-letter coverage is rich enough to DERANGE the letter
# assignment within the length group (each of these lengths is carried by >= 6 of
# the 14 letters in NOUN_FINAL_BY_LETTER, so a fixed-point-free permutation of the
# group's letters always exists). Lengths 9/10/12 are single-letter -> unusable.
_LAST_LENGTHS: tuple[int, ...] = (4, 5, 6, 7, 8)

# Vowel-initial word pool for the middles, grouped by initial vowel. CRITICAL
# alliteration property: EVERY middle word begins with a vowel (a/e/i/o/u) — none
# of the 14 consonant bank letters — and ``_build_middles`` composes each middle
# so NO vowel initial appears 3+ times in it. Because W1 and W2 are consonant-
# initial, the only initial that can ever reach a count of 2 across a full item
# is the bank letter shared by W1 and a matching W2 (the rule signal); it can
# NEVER reach 3. So no item is alliterative in EITHER class (the guard's reject
# set is empty by construction), keeping the matching pair perceptually salient.
# Middle words are intransitive verbs / place adverbs / prepositions that read as
# a plausible predicate after a plural-noun subject; no comma, no terminal
# punctuation, no 'I', no proper noun, no digits.
_VOWEL_WORDS: dict[str, tuple[str, ...]] = {
    "a": ("ambled", "arrived", "appeared", "advanced", "around", "ashore", "aside", "away", "aloft", "again"),
    "e": ("eased", "edged", "entered", "emerged", "escaped", "early", "everywhere", "elsewhere", "eagerly", "evenly"),
    "i": ("idled", "inched", "inside", "indoors", "instead", "into", "idly", "inward", "intently", "initially"),
    "o": ("onward", "outdoors", "outside", "over", "onto", "openly", "often", "overhead", "outward", "obliquely"),
    "u": ("upward", "underground", "uphill", "uneasily", "upstream", "usefully", "unusually", "upwind", "urgently", "underneath"),
}

# middle word counts to use (total item length = 1 + middle_len + 1 -> 5..8).
_MIDDLE_LENS: tuple[int, ...] = (3, 4, 5, 6)

# distinct middles to compose per length (plenty for >= 396 distinct base slots).
_MIDDLES_PER_LEN = 60

# total bases to build (comfortably over the 340 floor; 100+120+100+spare).
_N_BASES = 396


def _build_middles() -> dict[int, list[str]]:
    """Compose vowel-initial middles, grouped by word count, such that within
    every middle no initial vowel occurs 3+ times (the alliteration guarantee).

    Deterministic (fixed seed, independent of the run seed: the middle pool is a
    rule constant). Cycles vowels round-robin so each middle uses <= 2 of any one
    vowel, and de-duplicates surface strings."""
    g = Gen(270270)
    out: dict[int, list[str]] = {}
    vowels = ("a", "e", "i", "o", "u")
    for ln in _MIDDLE_LENS:
        seen: set[str] = set()
        middles: list[str] = []
        attempts = 0
        while len(middles) < _MIDDLES_PER_LEN and attempts < _MIDDLES_PER_LEN * 200:
            attempts += 1
            # choose ln initial-vowels with each vowel used at most twice
            vorder = list(vowels)
            g.shuffle(vorder)
            chosen_vowels: list[str] = []
            cap = collections.Counter()
            vi = 0
            while len(chosen_vowels) < ln:
                v = vorder[vi % len(vorder)]
                vi += 1
                if cap[v] < 2:
                    chosen_vowels.append(v)
                    cap[v] += 1
                if vi > len(vorder) * 4:  # safety (ln <= 6, 5 vowels x2 = 10 slots)
                    break
            if len(chosen_vowels) < ln:
                continue
            g.shuffle(chosen_vowels)
            wordset: list[str] = []
            ok = True
            used_words: set[str] = set()
            for v in chosen_vowels:
                pool = [w for w in _VOWEL_WORDS[v] if w not in used_words]
                if not pool:
                    ok = False
                    break
                w = g.choice(sorted(pool))
                used_words.add(w)
                wordset.append(w)
            if not ok:
                continue
            middle = " ".join(wordset)
            if middle in seen:
                continue
            seen.add(middle)
            middles.append(middle)
        if len(middles) < _MIDDLES_PER_LEN:
            raise ValueError(
                f"could not compose {_MIDDLES_PER_LEN} middles of length {ln} "
                f"(got {len(middles)})"
            )
        out[ln] = middles
    return out


_MIDDLES_BY_LEN: dict[int, list[str]] = _build_middles()


# --- bank bookkeeping ---------------------------------------------------------


def _w1_by_letter() -> dict[str, list[str]]:
    """First-word (plural noun) candidates per initial letter."""
    out: dict[str, list[str]] = {L: [] for L in _LETTERS}
    for e in banks.get_bank(_W1_BANK).entries:
        if e.initial in out:
            out[e.initial].append(e.word)
    for L in _LETTERS:
        if not out[L]:
            raise ValueError(f"{_W1_BANK}: no first-word candidate for letter {L!r}")
    return out


def _w2_by_letter_len() -> dict[tuple[str, int], list[str]]:
    """Last-word (place noun) candidates per (initial letter, length)."""
    out: dict[tuple[str, int], list[str]] = collections.defaultdict(list)
    for e in banks.get_bank(_W2_BANK).entries:
        if e.initial in set(_LETTERS) and e.length in _LAST_LENGTHS:
            out[(e.initial, e.length)].append(e.word)
    return out


def _letters_with_last(length: int, w2: dict[tuple[str, int], list[str]]) -> list[str]:
    """Letters that have at least one last word of ``length`` (sorted, stable)."""
    return sorted(L for L in _LETTERS if w2.get((L, length)))


def _derange(seq: list[str], gen: Gen) -> list[str]:
    """Return a fixed-point-free permutation of the MULTISET ``seq`` (out[i] !=
    seq[i] for every i, while ``collections.Counter(out) == Counter(seq)``).

    Deterministic and guaranteed when no single value occupies more than half the
    slots (the classic multiset-derangement condition): order the positions by
    value so equal values are contiguous, then rotate that order by the size of
    the LARGEST value-group. Rotating by >= max-group-size moves every value past
    its own block, so no position keeps its value. ``gen`` seeds a tie-break
    shuffle inside each value-group for variety without breaking the invariant."""
    n = len(seq)
    if n < 2:
        raise ValueError(f"cannot derange a sequence of length {n}")
    counts = collections.Counter(seq)
    max_group = max(counts.values())
    if max_group * 2 > n:
        raise ValueError(
            f"cannot derange {dict(counts)}: a value occupies more than half the slots"
        )
    # positions grouped by value (seeded tie-break order within each value).
    by_value: dict[str, list[int]] = collections.defaultdict(list)
    for i, v in enumerate(seq):
        by_value[v].append(i)
    ordered_positions: list[int] = []
    for v in sorted(by_value):
        block = list(by_value[v])
        gen.shuffle(block)
        ordered_positions.extend(block)
    # rotate the value sequence (read in this block order) by max_group, then map
    # each position to the rotated value.
    vals_in_order = [seq[p] for p in ordered_positions]
    shift = max_group
    rotated = vals_in_order[shift:] + vals_in_order[:shift]
    out = [""] * n
    for p, v in zip(ordered_positions, rotated):
        out[p] = v
    if any(out[i] == seq[i] for i in range(n)) or collections.Counter(out) != counts:
        raise ValueError(f"derangement failed invariant for {seq!r}")
    return out


# --- base spec ----------------------------------------------------------------


@dataclass(frozen=True)
class Base:
    """One base: a shared middle, the first-word letter L1, the last-word length
    LL, and the deranged FALSE last-word letter L2 (L2 != L1, same length group).

    True variant: last word begins with L1 (match). False variant: last word
    begins with L2 (no match). Both variants share W1 and the last-word length."""

    base_id: str
    middle: str
    first_letter: str   # L1 (also W1's and the True last word's initial)
    last_len: int       # LL (length of the last word in BOTH variants)
    false_letter: str   # L2 (False last word's initial; L2 != L1, has a word of LL)


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (the GENERATOR INTERFACE).

    Plan a balanced number of bases per last-word length group; within each
    group, spread the first-word letters L1 uniformly over the letters that have
    a last word of that length, then DERANGE the L1 assignment to obtain each
    base's False last-word letter L2 (L2 != L1, same length). Pair each (L1, LL,
    L2) slot with a distinct middle so base_ids and surfaces stay distinct.
    Deterministic given ``gen``; raises if the floor cannot be reached (loud)."""
    w2 = _w2_by_letter_len()

    # letters available per length group, and how many bases to put in each group.
    groups = [(LL, _letters_with_last(LL, w2)) for LL in _LAST_LENGTHS]
    groups = [(LL, ls) for LL, ls in groups if len(ls) >= 2]  # derangeable only
    if not groups:
        raise ValueError("no derangeable last-word length group available")

    # distribute _N_BASES over the groups proportionally to letter coverage, so
    # every group is comfortably derangeable (balanced letter multiset).
    total_letters = sum(len(ls) for _, ls in groups)
    plan: list[tuple[int, list[str]]] = []  # (count_for_group, letters)
    assigned = 0
    for i, (LL, ls) in enumerate(groups):
        if i == len(groups) - 1:
            count = _N_BASES - assigned
        else:
            count = round(_N_BASES * len(ls) / total_letters)
        assigned += count
        plan.append((count, ls))

    # alphabetical middle pool (de-duplicated across the whole rule) so a (middle)
    # is used by at most one base -> distinct surfaces.
    middle_pool: list[str] = []
    for ln in sorted(_MIDDLES_BY_LEN):
        middle_pool.extend(_MIDDLES_BY_LEN[ln])
    gen.shuffle(middle_pool)

    bases: list[Base] = []
    seen_ids: set[str] = set()
    used_middle_slots: set[tuple[str, str, int, str]] = set()
    middle_cursor = 0

    # build group by group
    for group_idx, ((count, letters), LL) in enumerate(
        zip(plan, [LL for LL, _ in groups])
    ):
        if count <= 0:
            continue
        # L1 assignment: cycle the group's letters so the multiset is balanced
        # (no letter > half the slots) -> always derangeable.
        l1_seq = [letters[i % len(letters)] for i in range(count)]
        g_group = gen.derive(f"group:{LL}")
        g_group.shuffle(l1_seq)
        l2_seq = _derange(l1_seq, g_group.derive("derange"))

        for k in range(count):
            L1 = l1_seq[k]
            L2 = l2_seq[k]
            # sanity: same-length last words exist for both letters in this group.
            if not w2.get((L1, LL)) or not w2.get((L2, LL)):
                continue
            # find a distinct middle for this (L1, LL, L2) slot.
            placed = False
            for _ in range(len(middle_pool)):
                middle = middle_pool[middle_cursor % len(middle_pool)]
                middle_cursor += 1
                slot = (middle, L1, LL, L2)
                if slot in used_middle_slots:
                    continue
                bid = make_base_id("first_last_same_letter", middle, L1, LL, L2)
                if bid in seen_ids:
                    continue
                used_middle_slots.add(slot)
                seen_ids.add(bid)
                bases.append(
                    Base(
                        base_id=bid,
                        middle=middle,
                        first_letter=L1,
                        last_len=LL,
                        false_letter=L2,
                    )
                )
                placed = True
                break
            if not placed:
                continue

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"first_last_same_letter: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


# --- instantiation ------------------------------------------------------------


def _alliterative(text: str) -> bool:
    """True iff >= 3 words of ``text`` share an initial letter (the recipe's
    alliteration guard; case-insensitive, alphabetic initials only)."""
    initials = collections.Counter()
    for tok in words(text):
        for ch in tok:
            if ch.isalpha():
                initials[ch.lower()] += 1
                break
    return any(c >= 3 for c in initials.values())


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True  : last word begins with spec.first_letter (== W1's letter) -> match.
    False : last word begins with spec.false_letter (!= W1's letter) -> no match.
    Both variants share W1 and the last-word length; only the last word's initial
    flips. Deterministic given (spec, label, gen). Raises if the realized surface
    is alliterative (the recipe forbids it in BOTH classes)."""
    w1_pool = _w1_by_letter()
    w2_pool = _w2_by_letter_len()

    # per-(base) sub-stream: W1 is shared across the two variants (same word), so
    # draw it from a label-INDEPENDENT stream; the last word is drawn per label.
    g_base = gen.derive(f"{spec.base_id}")
    w1 = g_base.derive("w1").choice(sorted(w1_pool[spec.first_letter]))

    last_letter = spec.first_letter if label else spec.false_letter
    last_pool = sorted(w2_pool[(last_letter, spec.last_len)])
    w2 = g_base.derive(f"w2:{'T' if label else 'F'}").choice(last_pool)

    raw = f"{w1} {spec.middle} {w2}"
    text = raw[:1].upper() + raw[1:]  # sentence case (case is irrelevant to the rule)

    if _alliterative(text):
        # the recipe rejects alliterative items in BOTH classes; surface it loudly
        # so build_bases / the pinned middles can be adjusted rather than emitting
        # a salience-breaking item.
        raise ValueError(
            f"first_last_same_letter: alliterative surface (>= 3 shared initials) "
            f"for base {spec.base_id!r} label={label}: {text!r}"
        )

    meta = {
        "rule": "first_last_same_letter",
        "middle": spec.middle,
        "first_word": w1,
        "first_letter": spec.first_letter,
        "last_word": w2,
        "last_letter": last_letter,
        "false_letter": spec.false_letter,
        "last_len": spec.last_len,
        "match": label,
        "transform": "match" if label else "derange",
        "word_count": word_count(text),
    }
    return text, meta
