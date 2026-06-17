"""Rule 25: even_word_count (category: numeric).

Canonical articulation: True iff the input contains an EVEN number of words
(global tokenizer; zero is even but cannot occur). This is the pre-specified
expected-FAILURE rule -- parity counting without chain-of-thought -- so the
whole point of the construction is that PARITY IS THE ONLY SIGNAL: no length
threshold, vocabulary, template, first/last-word, or char-count proxy may track
the label. The recipe earns this by matched class-conditional length.

Construction (recipe, rule-specs id: even_word_count)
-----------------------------------------------------
"Same core+adjunct machinery as word_count_geq_8." A base fixes a content core
(subject noun, regular verb, object noun, and -- for 6+ word variants -- an
adjective) plus a per-base pair of EXACT target word counts:

    True  target  wT in {4, 6, 8, 10}   (even)
    False target  wF in {5, 7, 9}       (odd)

The True and False variants of a base SHARE the same content core and the same
ADVERB_PLACE adjunct vocabulary; they differ ONLY in how many words the core +
place adjuncts add up to (its parity), i.e. exactly the rule. The surface is

    <= 5 words:  "The {N1} {VERBed} {adjuncts...} {N2}"
    >= 6 words:  "The {N1} {VERBed} {adjuncts...} the {ADJ} {N2}"

Adjuncts are inserted in the INTERIOR (after the verb, before the object tail),
so the FIRST word is always "The" and the LAST word is always the object noun
N2 from the shared NOUN_CONCRETE pool -- independent of the count/parity. That
pins every first-word and last-word battery predicate at ~50% by construction;
parity changes nothing about the boundary words. The count-equalizer
(genutils.solve_adjuncts / ADVERB_PLACE phrases of 1/2/3 words) closes the
deficit to the EXACT target.

Length matching (distribution_guards: "Equal mean length, interleaved supports")
--------------------------------------------------------------------------------
True counts are drawn {4: 20%, 6: 30%, 8: 30%, 10: 20%} (mean 7.0); False counts
{5, 7, 9} uniformly (mean 7.0). The two supports fully interleave (4 < 5 < 6 <
7 < 8 < 9 < 10), so NO word-count threshold separates the classes -- every
``word_count>=k`` battery predicate sits well under 0.75 (peaks ~0.60). To hold
the gated |mean_T - mean_F| <= 0.2 length-match TIGHTLY even after the pipeline
picks ONE balanced variant per base on the one-variant splits, the per-base
counts are paired ANTI-CORRELATED (the True multiset sorted ascending against
the False multiset sorted descending): a base with a small wT carries a large
wF and vice versa, so whatever 50/50 subset of variants the pipeline emits, the
realized True-count mean and False-count mean both stay ~7.0.

base_id = base_id("even_word_count", N1, VERB, ADJ, N2, wT, wF) -- the core
clause + the count pair; shared by both variants, distinct across bases.

This module is a two-callable generator conforming to the GENERATOR INTERFACE
(see ``..base``); it is driven through the shared four-gate emit pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import (
    Gen,
    GenError,
    adjunct_word_lengths,
    base_id,
)
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

# --- banks (rule-specs generation.banks) --------------------------------------
_NOUN_CONCRETE = "NOUN_CONCRETE"
_VERB_REGULAR = "VERB_REGULAR"
_ADJ_PLAIN = "ADJ_PLAIN"
_ADVERB_PLACE = "ADVERB_PLACE"
# Single-word manner/degree adverbs used to close any word-count deficit BEYOND
# the one place phrase, so the interior never stacks two place adjuncts ("in the
# garden nearby", "at the market downtown"). All 1 word, comma-free, and they read
# as ordinary VP adverbs after the verb; they change neither the first word ("The")
# nor the last word (object noun N2), so every boundary battery predicate is
# untouched and only the (parity of the) word count moves.
_MANNER = "ADVERB_SENT_INITIAL"

# Build comfortably more than the 340-base floor (100 + 120 + 100 + >= 20 spare).
# 420 keeps every count stratum well populated and the anti-correlated pairing
# exact (420 divisible by 3 for the False counts and clean for the True weights).
_N_BASES = 420

# recipe class-conditional count distributions (both mean 7.0, supports
# interleaved 4<5<6<7<8<9<10 so no length threshold is a proxy).
_TRUE_COUNTS = (4, 6, 8, 10)          # even
_TRUE_WEIGHTS = (0.20, 0.30, 0.30, 0.20)
_FALSE_COUNTS = (5, 7, 9)             # odd, uniform

# the global [4, 14] word-count window the schema validator re-checks; both
# supports sit inside it.
_MIN_WORDS, _MAX_WORDS = 4, 10


@dataclass(frozen=True)
class Base:
    """An even_word_count base: the content core (subject/verb/adjective/object)
    plus the EXACT True (even) and False (odd) target word counts. ``base_id``
    mixes the core fillers and the count pair (shared by both variants, distinct
    across bases). Both variants are built from these fields with NO further
    per-variant content randomness -- only the place-adjunct padding differs,
    and that padding is what realises the word-count (parity)."""

    base_id: str
    n1: str       # subject NOUN_CONCRETE
    verb: str     # VERB_REGULAR base form (surfaced past-tense, regular -ed)
    adj: str      # ADJ_PLAIN used only by the >= 6-word frame; harmless otherwise
    n2: str       # object NOUN_CONCRETE (always the LAST word -> shared pool)
    w_true: int   # even target word count for the True variant
    w_false: int  # odd target word count for the False variant


def _verb_ed(verb: str) -> str:
    """Regular past-tense surface of a VERB_REGULAR base form.

    All VERB_REGULAR entries are regular: append 'ed', or 'd' if the base ends
    in 'e' (e.g. 'close' -> 'closed'). (Used only to give the core a natural
    finite clause; the rule is parity, so the exact tense is immaterial and is
    NOT a battery predicate here.)"""
    return f"{verb}d" if verb.endswith("e") else f"{verb}ed"


def _adjuncts_by_len() -> dict[int, list[str]]:
    """ADVERB_PLACE phrases grouped by word count {1: [...], 2: [...], 3: [...]}.

    A 1-word phrase is present, so the count-equalizer can reach EVERY
    non-negative deficit exactly."""
    by_len = adjunct_word_lengths(e.word for e in banks.get_bank(_ADVERB_PLACE).entries)
    if 1 not in by_len:
        raise ValueError("even_word_count: no 1-word ADVERB_PLACE adjunct available")
    return by_len


def _manner_1word() -> list[str]:
    """The 1-word manner/degree adverbs used to fill the deficit BEYOND the single
    place phrase (never a second place phrase). All exactly one word."""
    out = [e.word for e in banks.get_bank(_MANNER).entries if word_count(e.word) == 1]
    if not out:
        raise ValueError("even_word_count: no 1-word manner adverb available")
    return out


def _solve_one_place(
    deficit: int, by_len: dict[int, list[str]], manner_1word: list[str], gen: Gen
) -> list[str]:
    """Close ``deficit`` words with AT MOST ONE place phrase + 1-word manner adverbs.

    Replaces the old ``solve_adjuncts`` multiset (which stacked two ADVERB_PLACE
    phrases, e.g. "in the garden nearby"). We take a SINGLE place phrase as long as
    possible (<= 3 words and <= the deficit) and close the rest with DISTINCT
    single-word manner adverbs, so the interior carries exactly one place adjunct.
    Returns the interior tokens in surface order (manner adverbs first, then the
    one place phrase, so the place phrase sits adjacent to the object tail exactly
    as the single-phrase case always did). Deterministic given ``gen``."""
    if deficit <= 0:
        return []
    place_len = min(deficit, 3)
    while place_len >= 1 and place_len not in by_len:
        place_len -= 1
    interior_manner: list[str] = []
    if place_len >= 1:
        place = gen.choice(by_len[place_len])
    else:
        place = ""
    remaining = deficit - place_len
    if remaining > 0:
        if remaining > len(manner_1word):
            raise GenError(
                f"even_word_count: deficit {deficit} needs {remaining} manner "
                f"adverbs but only {len(manner_1word)} available"
            )
        adverbs = list(manner_1word)
        gen.shuffle(adverbs)
        interior_manner = adverbs[:remaining]
    # manner adverbs first (right after the verb), then the single place phrase
    interior = list(interior_manner)
    if place:
        interior.append(place)
    return interior


def _build_count_pairs(gen: Gen) -> list[tuple[int, int]]:
    """The per-base (w_true, w_false) target pairs (anti-correlated; deterministic).

    The True multiset realises {4:20%, 6:30%, 8:30%, 10:20%} and the False
    multiset realises {5,7,9} uniformly -- both exactly mean 7.0 over the
    _N_BASES bases. The two multisets are paired ANTI-CORRELATED (True ascending
    vs False descending) so a small even target rides with a large odd target
    and vice versa; this keeps the realised class-conditional means within the
    0.2 length-match tolerance no matter which single variant the pipeline emits
    on the one-variant splits."""
    n = _N_BASES
    true_counts: list[int] = []
    for c, w in zip(_TRUE_COUNTS, _TRUE_WEIGHTS):
        true_counts.extend([c] * round(w * n))
    # pad/trim to exactly n with the central even count (6) -- weights sum to 1.0
    # so this rarely fires; it only absorbs rounding.
    while len(true_counts) < n:
        true_counts.append(6)
    true_counts = true_counts[:n]

    false_counts: list[int] = list(_FALSE_COUNTS) * (n // len(_FALSE_COUNTS))
    while len(false_counts) < n:
        false_counts.append(7)
    false_counts = false_counts[:n]

    # anti-correlated pairing: True ascending against False descending.
    true_counts.sort()
    false_counts.sort(reverse=True)
    pairs = list(zip(true_counts, false_counts))
    gen.shuffle(pairs)
    return pairs


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (GENERATOR INTERFACE).

    Deterministic given ``gen``: draw the per-base content core from the four
    banks, attach an anti-correlated (even, odd) target-count pair, reject
    duplicate base_ids, and raise (loud) if the floor cannot be reached."""
    nouns = [e.word for e in banks.get_bank(_NOUN_CONCRETE).entries]
    verbs = [e.word for e in banks.get_bank(_VERB_REGULAR).entries]
    adjs = [e.word for e in banks.get_bank(_ADJ_PLAIN).entries]
    if len(nouns) < 2:
        raise ValueError("even_word_count: need >= 2 NOUN_CONCRETE entries")

    pairs = _build_count_pairs(gen.derive("counts"))

    g_core = gen.derive("core")
    bases: list[Base] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = _N_BASES * 80
    while len(bases) < _N_BASES and attempts < max_attempts:
        attempts += 1
        n1, n2 = g_core.sample(nouns, 2)  # distinct subject/object nouns
        verb = g_core.choice(verbs)
        adj = g_core.choice(adjs)
        w_true, w_false = pairs[len(bases)]
        bid = base_id("even_word_count", n1, verb, adj, n2, w_true, w_false)
        if bid in seen:
            continue
        seen.add(bid)
        bases.append(
            Base(
                base_id=bid,
                n1=n1,
                verb=verb,
                adj=adj,
                n2=n2,
                w_true=w_true,
                w_false=w_false,
            )
        )

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"even_word_count: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def _build_text(spec: Base, target: int, gen: Gen) -> str:
    """Surface for ``spec`` at EXACTLY ``target`` words.

    First word is always 'The'; last word is always the object noun N2 (shared
    NOUN_CONCRETE pool) -- so no boundary-word battery predicate tracks parity.
    A SINGLE place adjunct (1/2/3 words) plus, if more words are still needed,
    distinct 1-word manner adverbs are inserted in the interior to close the
    deficit exactly -- never two stacked place phrases. The <= 5-word frame drops
    the adjective (no room); the >= 6 frame carries it. Deterministic given
    (spec, target, gen)."""
    if not (_MIN_WORDS <= target <= _MAX_WORDS):
        raise ValueError(f"even_word_count: target {target} outside [{_MIN_WORDS},{_MAX_WORDS}]")
    by_len = _adjuncts_by_len()
    manner_1word = _manner_1word()
    ved = _verb_ed(spec.verb)

    if target <= 5:
        head = f"The {spec.n1} {ved}"          # 3 words
        tail = spec.n2                          # 1 word
        core_wc = 4
    else:
        head = f"The {spec.n1} {ved}"          # 3 words
        tail = f"the {spec.adj} {spec.n2}"      # 3 words
        core_wc = 6

    deficit = target - core_wc
    interior = _solve_one_place(deficit, by_len, manner_1word, gen)
    mid = " ".join(interior)

    text = f"{head} {mid} {tail}" if mid else f"{head} {tail}"
    if word_count(text) != target:
        raise ValueError(
            f"even_word_count: built {word_count(text)} words, expected {target}: {text!r}"
        )
    return text


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (GENERATOR INTERFACE).

    True  -> the even-word-count surface (w_true words; groundtruth labels True).
    False -> the odd-word-count surface  (w_false words; groundtruth labels False).
    Both share the content core and adjunct vocabulary; only the (parity of the)
    word count differs. Deterministic given (spec, label, gen)."""
    target = spec.w_true if label else spec.w_false
    # derive a per-(base,label) adjunct stream so the two variants pad
    # independently but reproducibly (and never collide on a surface string).
    sub = gen.derive(f"{spec.base_id}:{'T' if label else 'F'}")
    text = _build_text(spec, target, sub)
    meta = {
        "n1": spec.n1,
        "verb": spec.verb,
        "verb_surface": _verb_ed(spec.verb),
        "adj": spec.adj,
        "n2": spec.n2,
        "target_word_count": target,
        "w_true": spec.w_true,
        "w_false": spec.w_false,
        "parity": "even" if label else "odd",
        "transform": "interior_place_adjuncts",
    }
    return text, meta
