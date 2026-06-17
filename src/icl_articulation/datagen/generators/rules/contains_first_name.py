"""Rule 17: contains_first_name (category semantic / substitution).

Canonical articulation: True iff the input contains a person's FIRST name (Anna,
David, ...), and False otherwise — INCLUDING sentences that contain some OTHER
capitalized proper noun (a city, month, country or brand). The hard part of the
rule is exactly that: a capitalized mid-sentence proper noun is present in BOTH
classes, so 'contains a capitalized word' is a ~50% distractor, not the rule.

CONSTRUCTION (per the spec's generation.recipe + distribution_guards)
=====================================================================
A base is a (FRAME_PROPER frame, matched name/non-name pair). Each FRAME_PROPER
frame has EXACTLY ONE proper-noun slot ``{X}``; every other word is a lowercase
function/content word (no proper nouns, no 'I'). The two variants of a base are
SURFACE-IDENTICAL except for that single slot word:

    True  variant  = frame with {X} filled by a FIRST_NAMES word        -> True
    False variant  = frame with {X} filled by the SAME base's matched
                     NONNAME_PROPER word (city / month / country / brand) -> False

Both variants therefore carry EXACTLY ONE capitalized proper noun in the SAME
position (frames are shared), so the capitalization rate is identical across the
two classes BY CONSTRUCTION (review MUST-FIX #7). base derives from frame +
the matched pair.

Why this passes the four gates
------------------------------
The discriminating feature ('the slot proper noun is a person's first name') is
NOT one of the 40 frozen battery predicates, so it is not exempt — every battery
predicate must independently sit <= 0.75. The label correlates with a predicate
only when that predicate's value differs between a base's name filler and its
non-name filler. Two things kill every such correlation:

  * POSITION SYMMETRY. For the 65% MID-slot frames and the 30% slot-FINAL frames,
    the first word (and, for mid frames, the last word too) is a shared FRAME
    word, so every first-word / first-letter / first-POS predicate is identical
    on the two variants. Frame-only predicates (word_count>=k, even_word_count,
    contains_the, count_the>=2, the_appears_twice, all_words_longer_than_3, ...)
    depend on the shared frame alone -> their value is independent of the label.

  * POSITION-AWARE MATCHED PAIRS. The name and non-name of a base are matched on
    the features its slot position exposes to the battery:
        - MID frames     : (alpha length, adjacent-double-letter) -> guards
                           char_count>=k and double_letter_word.
        - FINAL frames   : the above + final-letter vowelness + initial-letter
                           bucket -> also guards last_ends_vowel, last_word_len,
                           first_last_same_letter, first_word_longer_than_last.
        - INITIAL frame  : length + double + initial letter + final letter (the
                           slot IS word 1) -> guards every first-word predicate;
                           both fillers are proper nouns so first_word_pos=noun
                           is identical anyway.
    Because both members of a base's pair share the position-exposed features,
    whichever one a single-variant split (held_out / confirmation / spare) emits
    carries the SAME features -> the marginal feature distribution of the True
    items equals that of the False items, so no length / letter / double / POS
    predicate can separate the classes (each sits at ~50%).

No digit, no comma, no terminal punctuation and no 'I' are ever introduced
(default sentence_style for this rule), so contains_digit / contains_comma /
exclamation / two-commas all label everything False -> 50%. Word count is the
frame's word count (the slot is one word in BOTH variants), so the confound
length-match |mean_T - mean_F| is exactly 0.

Distribution targets the recipe pins (NOT battery gates, honoured for step-2/3):
  * cities are ~50% of the False (non-name) fillers ('mentions a city' -> 25%
    disagreement);
  * the True (name) fillers are a 50/50 gender split ("mentions a woman's name"
    -> 25% disagreement);
  * months are kept well under 25% of False fillers (the 'mentions a month'
    distractor is a spare candidate only).

A documented residual tension: FRAME_PROPER (frozen at B0) supplies only 1 of 20
frames with a sentence-INITIAL slot (5%), so the recipe's "sentence-initial in
30% of frames" cannot be met from the fixed frame bank. This does NOT threaten
any gate (capitalization rates stay class-identical, and the
nonfirst_word_capitalized / second_word_capitalized distractors stay ~50%
because each frame contributes equally to both classes); it is recorded as an
open concern rather than worked around by editing the shared bank.

This module exposes the GENERATOR INTERFACE (``build_bases`` / ``instantiate``);
it is run through the shared gated pipeline (``base.emit_rule``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, base_id as make_base_id, fill_frame
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

# the rule's three banks (rule-specs generation.banks)
_FRAME_BANK = "FRAME_PROPER"
_NAME_BANK = "FIRST_NAMES"
_NONNAME_BANK = "NONNAME_PROPER"

# Build comfortably past the 340-base floor (100 + 120 + 100 + >=20 spare). 20
# frames x ~24 matched pairs is plenty distinct; we target this many and stop.
_N_BASES = 460

# city share of the FALSE (non-name) fillers the recipe pins (~50%), and the
# month cap (kept low so 'mentions a month' stays a spare-only distractor).
_CITY_TARGET_FRAC = 0.50
_MONTH_MAX_FRAC = 0.20

_VOWELS = frozenset("aeiou")


# --- slot-word feature helpers (alphabetic-char scope, matching banks.py) -----


def _alpha_len(word: str) -> int:
    return sum(1 for ch in word if ch.isalpha())


def _initial(word: str) -> str:
    for ch in word:
        if ch.isalpha():
            return ch.lower()
    return ""


def _final(word: str) -> str:
    for ch in reversed(word):
        if ch.isalpha():
            return ch.lower()
    return ""


def _has_double(word: str) -> bool:
    a = "".join(ch for ch in word.lower() if ch.isalpha())
    return any(a[i] == a[i + 1] for i in range(len(a) - 1))


def _bucket(letter: str) -> str:
    if letter in "abcdef":
        return "a-f"
    if letter in "ghijklm":
        return "g-m"
    if letter in "nopqrs":
        return "n-s"
    return "t-z"


class _SlotKind:
    MID = "mid"
    FINAL = "final"
    INITIAL = "initial"


def _slot_kind(frame: str) -> str:
    toks = frame.split()
    pos = next(i for i, t in enumerate(toks) if "{X}" in t)
    if pos == 0:
        return _SlotKind.INITIAL
    if pos == len(toks) - 1:
        return _SlotKind.FINAL
    return _SlotKind.MID


def _match_key(word: str, kind: str) -> tuple:
    """The feature tuple a base must match on for a slot of ``kind``.

    Only the features the slot position EXPOSES to the 40 battery predicates are
    matched, so the matched-pair pools stay large while every exposed feature is
    class-balanced."""
    base = (_alpha_len(word), _has_double(word))
    if kind == _SlotKind.MID:
        return base
    if kind == _SlotKind.FINAL:
        return base + (_final(word) in _VOWELS, _bucket(_initial(word)))
    # INITIAL: the slot is word 1 -> match every first-word feature tightly
    return base + (_initial(word), _final(word))


# --- the base spec ------------------------------------------------------------


@dataclass(frozen=True)
class Base:
    """A rule-17 base: a frame + the matched (name, non-name) slot pair.

    The True variant fills the frame's {X} with ``name``; the False variant fills
    it with ``nonname``. ``base_id`` is a stable hash of (frame, name, nonname)
    so both variants share it and it is distinct across bases."""

    base_id: str
    frame: str
    slot_kind: str
    name: str
    name_gender: str
    nonname: str
    nonname_subtype: str


# --- pair pools ---------------------------------------------------------------


def _bank_words_by(bank_name: str) -> list[tuple[str, str]]:
    """(word, subtype) for a proper-noun bank (subtype = gender for names)."""
    bank = banks.get_bank(bank_name)
    return [(e.word, e.subtype or "") for e in bank.entries]


def _grouped(words: list[tuple[str, str]], kind: str) -> dict[tuple, list[tuple[str, str]]]:
    out: dict[tuple, list[tuple[str, str]]] = {}
    for w, sub in words:
        out.setdefault(_match_key(w, kind), []).append((w, sub))
    return out


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct (frame, matched name/non-name pair) bases.

    Deterministic given ``gen``. Frames are cycled so each is used about equally
    (frame-only battery predicates then stay label-balanced); within each frame's
    slot kind, name/non-name pairs are drawn from the position-matched pools with
    the recipe's gender (50/50 names) and city (~50% non-names, months capped)
    distribution targets. Raises (loud) if the floor cannot be reached."""
    frames = list(banks.BANKS[_FRAME_BANK])
    names = _bank_words_by(_NAME_BANK)
    nonnames = _bank_words_by(_NONNAME_BANK)

    # frames grouped by slot kind, each kind's name/non-name match pools
    frames_by_kind: dict[str, list[str]] = {}
    for fr in frames:
        frames_by_kind.setdefault(_slot_kind(fr), []).append(fr)

    name_pools: dict[str, dict[tuple, list[tuple[str, str]]]] = {}
    nonname_pools: dict[str, dict[tuple, list[tuple[str, str]]]] = {}
    for kind in (_SlotKind.MID, _SlotKind.FINAL, _SlotKind.INITIAL):
        name_pools[kind] = _grouped(names, kind)
        nonname_pools[kind] = _grouped(nonnames, kind)

    # round-robin the frames so each is used ~equally (every frame appears in
    # both classes equally, keeping frame-only predicates label-balanced).
    frame_cycle = list(frames)
    gen.shuffle(frame_cycle)

    # per-kind shuffled list of candidate (name, nonname) matched pairs, with the
    # recipe distribution targets enforced as we go.
    def _candidate_pairs(kind: str) -> list[tuple[tuple[str, str], tuple[str, str]]]:
        npool = name_pools[kind]
        nnpool = nonname_pools[kind]
        keys = sorted(set(npool) & set(nnpool))
        pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
        for k in keys:
            for nm in npool[k]:
                for nn in nnpool[k]:
                    pairs.append((nm, nn))
        gen.shuffle(pairs)
        return pairs

    pair_cycles = {k: _candidate_pairs(k) for k in frames_by_kind}
    pair_idx = {k: 0 for k in frames_by_kind}

    def _next_pair(kind: str) -> tuple[tuple[str, str], tuple[str, str]]:
        pairs = pair_cycles[kind]
        if not pairs:
            raise ValueError(f"contains_first_name: no matched pairs for slot kind {kind!r}")
        p = pairs[pair_idx[kind] % len(pairs)]
        pair_idx[kind] += 1
        return p

    bases: list[Base] = []
    seen_ids: set[str] = set()
    # Both emitted surfaces must be globally distinct (Gate A no-duplicate): the
    # True variant is (frame, name) and the False variant is (frame, nonname), so
    # two bases sharing a (frame, name) — even with different non-names — would
    # collide on the True surface. Track each filled surface independently.
    seen_true_surface: set[tuple[str, str]] = set()
    seen_false_surface: set[tuple[str, str]] = set()

    # distribution counters (over the FALSE non-name fillers / TRUE name fillers)
    n_city = n_month = n_nonname = 0
    n_fem = n_masc = 0

    fi = 0
    guard = 0
    max_guard = len(frame_cycle) * 200
    while len(bases) < _N_BASES and guard < max_guard:
        guard += 1
        frame = frame_cycle[fi % len(frame_cycle)]
        fi += 1
        kind = _slot_kind(frame)

        # draw a position-matched pair, steering the non-name subtype mix toward
        # ~50% cities and few months, and the name gender toward 50/50.
        chosen = None
        for _ in range(len(pair_cycles[kind]) + 1):
            (nm, ng), (nn, nsub) = _next_pair(kind)
            # subtype steering for the FALSE filler
            # city steering: drive the non-name fillers toward ~50% cities. When
            # BELOW the target prefer a city (skip non-cities this draw); when AT
            # or OVER it prefer a non-city. The bounded scan + unconditional
            # fallback below guarantee progress when the preferred subtype is
            # momentarily unavailable.
            below_city = (n_nonname == 0) or (n_city / max(n_nonname, 1) < _CITY_TARGET_FRAC)
            if below_city and nsub != "city":
                continue
            if (not below_city) and nsub == "city":
                continue
            if nsub == "month" and (n_month + 1) / max(n_nonname + 1, 1) > _MONTH_MAX_FRAC:
                continue  # month cap
            # gender steering for the TRUE filler (keep |fem-masc| small)
            if ng == "feminine" and n_fem > n_masc + 2:
                continue
            if ng == "masculine" and n_masc > n_fem + 2:
                continue
            chosen = ((nm, ng), (nn, nsub))
            break
        if chosen is None:
            # relax steering: take the next available pair unconditionally
            chosen = _next_pair(kind)

        (nm, ng), (nn, nsub) = chosen
        true_key = (frame, nm)
        false_key = (frame, nn)
        if true_key in seen_true_surface or false_key in seen_false_surface:
            continue
        bid = make_base_id(_FRAME_BANK, frame, nm, nn)
        if bid in seen_ids:
            continue

        seen_true_surface.add(true_key)
        seen_false_surface.add(false_key)
        seen_ids.add(bid)
        bases.append(
            Base(
                base_id=bid,
                frame=frame,
                slot_kind=kind,
                name=nm,
                name_gender=ng,
                nonname=nn,
                nonname_subtype=nsub,
            )
        )
        n_nonname += 1
        if nsub == "city":
            n_city += 1
        elif nsub == "month":
            n_month += 1
        if ng == "feminine":
            n_fem += 1
        elif ng == "masculine":
            n_masc += 1

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"contains_first_name: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True  variant = frame with {X} := the base's FIRST_NAMES word  -> rule True.
    False variant = frame with {X} := the base's NONNAME_PROPER word -> rule False
                    (a capitalized city/month/country/brand, NOT a first name).
    Deterministic; ``gen`` is unused (the matched pair is fixed on the base) but
    kept in the signature to match the interface."""
    if label:
        filler = spec.name
        slot_bank = _NAME_BANK
        slot_class = spec.name_gender
    else:
        filler = spec.nonname
        slot_bank = _NONNAME_BANK
        slot_class = spec.nonname_subtype
    text = fill_frame(spec.frame, {"X": filler})
    meta = {
        "frame": spec.frame,
        "slot_kind": spec.slot_kind,
        "slot_word": filler,
        "slot_bank": slot_bank,
        "slot_class": slot_class,
        "name": spec.name,
        "nonname": spec.nonname,
        "transform": "fill_frame",
    }
    return text, meta
