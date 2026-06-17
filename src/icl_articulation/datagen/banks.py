"""Shared word-bank infrastructure: the entry contract, computed tags, the
BANK_QUOTAS contract, and the self-check the bank authors must satisfy.

============================ THE ENTRY CONTRACT ==============================

A bank is either a WORD bank (a list of word-entry dicts) or a FRAME bank (a
list of template strings). The two are handled distinctly.

WORD ENTRY (a dict). The bank-group AUTHOR supplies ONLY these keys:
    word            (str)  the lowercase surface word, alphabetic-only unless a
                           bank note says otherwise; for proper nouns the
                           capitalized surface (and proper=True, see below).
    pos             (str)  one of POS_TAGS: noun, verb, adjective, adverb,
                           pronoun, determiner, numeral, preposition, other.
    frequency_tier  (int)  1 (wordfreq top 2000) or 2 (wordfreq top 10000).
                           Nothing rarer than tier 2 is allowed anywhere.
  Optional author keys (only where a bank's quota references them):
    proper          (bool) proper noun (capitalized surface); default False.
    discordant      (bool) letter/syllable-discordant (FINAL_BY_LENGTH); default False.
    final_y         (bool) ends in letter y (TERMINAL_CONSONANT tag); default False.
    pair            (str)  matched-pair key linking an entry to its counterpart
                           in the paired bank (Z_WORDS<->Z_FREE_MATCHED).
    subtype         (str)  free-text semantic subtag (e.g. 'plant', 'vehicle',
                           'animal', 'food', 'city', 'month', 'country', 'brand')
                           used by topic-share / composition quotas.

banks.py COMPUTES these tags from ``word`` (authors never write them; if an
author DOES supply one it must match the computed value or the check raises):
    length              (int)  alphabetic-char count, per globals.word_length.
    initial             (str)  first alphabetic char, lowercased.
    final               (str)  last alphabetic char, lowercased.
    has_adjacent_double (bool) some letter immediately repeats (e.g. 'coffee').
    has_nonadjacent_repeat (bool) a letter repeats NON-adjacently (e.g. 'banana').

``Entry`` is a frozen view that merges the authored dict with the computed
tags; ``Bank`` wraps a name + a list of Entries (or, for frame banks, frames).

FRAME ENTRY: a template string with exactly one '{X}' slot marker and no
proper nouns (FRAME_NEUTRAL) / natural for names and places (FRAME_PROPER).
Frame banks carry NO word-level tags; their quota is size + slot-marker shape.

The BANK_QUOTAS table below transcribes every spec'd quota (size, per-letter /
per-length minimums, POS mixes, matched-pair constraints, special flags) from
the rule-specs banks block and the rule recipes that constrain a bank. It is
the contract the authors satisfy and ``check_bank`` enforces; violations raise
BankQuotaError (LOUD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .banks_data import (
    core,
    g1_general,
    g2_zdouble,
    g3_semantic,
    g4a_byletter,
    g4b_bylength,
    g5_numhard,
)

POS_TAGS = frozenset(
    {"noun", "verb", "adjective", "adverb", "pronoun", "determiner", "numeral", "preposition", "other"}
)

# the shared by-letter index set (rule-specs: NOUN_PLURAL_BY_LETTER et al.)
BY_LETTER_SET = ("b", "c", "d", "f", "g", "h", "l", "m", "n", "p", "r", "s", "t", "w")

# author keys banks.py never lets the author set (it computes them)
_COMPUTED_KEYS = frozenset(
    {"length", "initial", "final", "has_adjacent_double", "has_nonadjacent_repeat"}
)
_AUTHOR_REQUIRED = ("word", "pos", "frequency_tier")
_AUTHOR_OPTIONAL = frozenset({"proper", "discordant", "final_y", "pair", "subtype"})


class BankError(ValueError):
    """A bank entry violated the entry contract (bad/missing tag, etc.)."""


class BankQuotaError(ValueError):
    """A populated bank violated its BANK_QUOTAS contract (LOUD self-check)."""


# --- computed tags ------------------------------------------------------------


def alphabetic_length(word: str) -> int:
    """globals.word_length: alphabetic-char count of the (already stripped) word."""
    return sum(1 for ch in word if ch.isalpha())


def _alpha_only(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalpha())


def has_adjacent_double(word: str) -> bool:
    a = _alpha_only(word)
    return any(a[i] == a[i + 1] for i in range(len(a) - 1))


def has_nonadjacent_repeat(word: str) -> bool:
    """A letter appears >= 2 times with at least one NON-adjacent pair.

    'banana' -> True (a..a..a), 'coffee' -> the ff is adjacent but 'ee' too;
    a word can have BOTH. This is True iff some letter occurs at two indices
    that are not adjacent."""
    a = _alpha_only(word)
    positions: dict[str, list[int]] = {}
    for i, ch in enumerate(a):
        positions.setdefault(ch, []).append(i)
    for idxs in positions.values():
        if len(idxs) >= 2:
            # any pair non-adjacent?
            for j in range(len(idxs)):
                for k in range(j + 1, len(idxs)):
                    if idxs[k] - idxs[j] > 1:
                        return True
    return False


def initial_letter(word: str) -> str:
    a = _alpha_only(word)
    return a[0] if a else ""


def final_letter(word: str) -> str:
    a = _alpha_only(word)
    return a[-1] if a else ""


# --- Entry / Bank -------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """A validated word-bank entry: authored keys + banks.py-computed tags."""

    word: str
    pos: str
    frequency_tier: int
    length: int
    initial: str
    final: str
    has_adjacent_double: bool
    has_nonadjacent_repeat: bool
    proper: bool = False
    discordant: bool = False
    final_y: bool = False
    pair: str | None = None
    subtype: str | None = None


def build_entry(raw: Mapping[str, Any]) -> Entry:
    """Validate an authored dict against the entry contract and compute tags."""
    missing = [k for k in _AUTHOR_REQUIRED if k not in raw]
    if missing:
        raise BankError(f"entry {dict(raw)!r} missing required keys {missing}")
    unknown = set(raw) - set(_AUTHOR_REQUIRED) - _AUTHOR_OPTIONAL - _COMPUTED_KEYS
    if unknown:
        raise BankError(f"entry {raw.get('word')!r} has unknown keys {sorted(unknown)}")
    word = raw["word"]
    if not isinstance(word, str) or not word:
        raise BankError(f"entry word must be a non-empty string, got {word!r}")
    pos = raw["pos"]
    if pos not in POS_TAGS:
        raise BankError(f"entry {word!r} has pos {pos!r} not in {sorted(POS_TAGS)}")
    tier = raw["frequency_tier"]
    if tier not in (1, 2):
        raise BankError(f"entry {word!r} frequency_tier must be 1 or 2, got {tier!r}")

    computed = {
        "length": alphabetic_length(word),
        "initial": initial_letter(word),
        "final": final_letter(word),
        "has_adjacent_double": has_adjacent_double(word),
        "has_nonadjacent_repeat": has_nonadjacent_repeat(word),
    }
    # if an author redundantly supplied a computed key, it MUST agree
    for k, v in computed.items():
        if k in raw and raw[k] != v:
            raise BankError(
                f"entry {word!r} {k}={raw[k]!r} disagrees with computed {v!r}"
            )
    return Entry(
        word=word,
        pos=pos,
        frequency_tier=tier,
        proper=bool(raw.get("proper", False)),
        discordant=bool(raw.get("discordant", False)),
        final_y=bool(raw.get("final_y", False)),
        pair=raw.get("pair"),
        subtype=raw.get("subtype"),
        **computed,
    )


@dataclass(frozen=True)
class Bank:
    """A named word bank: its validated entries."""

    name: str
    entries: tuple[Entry, ...]

    def __len__(self) -> int:
        return len(self.entries)

    def words(self) -> list[str]:
        return [e.word for e in self.entries]


def build_bank(name: str, raw_entries: list[Any]) -> Bank:
    return Bank(name=name, entries=tuple(build_entry(e) for e in raw_entries))


# --- frame banks --------------------------------------------------------------

FRAME_BANKS = frozenset({"FRAME_NEUTRAL", "FRAME_PROPER"})
_SLOT_MARK = "{X}"


def _check_frame_bank(name: str, frames: list[Any], min_size: int) -> None:
    if len(frames) < min_size:
        raise BankQuotaError(f"{name}: {len(frames)} frames < required {min_size}")
    for fr in frames:
        if not isinstance(fr, str):
            raise BankQuotaError(f"{name}: frame {fr!r} is not a string")
        if fr.count(_SLOT_MARK) != 1:
            raise BankQuotaError(
                f"{name}: frame {fr!r} must contain exactly one '{_SLOT_MARK}' slot"
            )


# --- BANK_QUOTAS contract -----------------------------------------------------
#
# Each quota is a dict. ``size`` is the minimum entry count. Optional keys:
#   pos_required        set of POS that must each appear at least once
#   pos_mix             {pos: (lo, hi)} fractional bounds on a POS share
#   max_alpha_len       every entry length <= this
#   min_alpha_len       every entry length >= this
#   by_letter_min       (letters, k): >= k entries per initial in ``letters``
#   by_length_min       (lengths, k): >= k entries per computed length in ``lengths``
#   all_double          every entry has_adjacent_double True
#   all_double_free     every entry has_adjacent_double False
#   all_nonadj_repeat   every entry has_nonadjacent_repeat True
#   subtype_min         {subtype: k}: >= k entries with that subtype
#   subtype_max_frac    {subtype: f}: <= f fraction with that subtype
#   final_letter_in     every entry's final letter in this set (e.g. vowels)
#   final_letter_not_in every entry's final letter NOT in this set
#   not_final_e_min     >= this fraction NOT ending in 'e'
#   silent_e_min        >= this fraction ending in 'e' (silent-e proxy)
#   discordant_min      >= this fraction with discordant=True
#   initial_bucket_max  no a-f/g-m/n-s/t-z bucket exceeds this fraction; >=4 buckets used
#   phrase_word_counts  set of whitespace-token counts that must each appear
#   max_phrase_words    every entry's whitespace-token count <= this
#   matched_pair        (other_bank, [tags...]): cross-bank matched-pair check.
#                       SYMMETRIC: registering it on bank A means check_bank(A)
#                       AND check_bank(B) both run the cross-validation, so the
#                       documented check_bank(name) API is self-sufficient on
#                       either side of a pair (a tier flip on the dependent bank
#                       is caught even when only the dependent is checked).
#   custom              name (str) of a module-level callable in _CUSTOM_CHECKS,
#                       called custom(bank, BANKS) -> None for extra asserts
#
# Values transcribed faithfully from rule-specs globals.banks + rule recipes.

VOWELS = frozenset("aeiou")


def _bucket(letter: str) -> str:
    if letter in "abcdef":
        return "a-f"
    if letter in "ghijklm":
        return "g-m"
    if letter in "nopqrs":
        return "n-s"
    return "t-z"


BANK_QUOTAS: dict[str, dict[str, Any]] = {
    # ---- reference bank (fully authored at B0) -------------------------------
    "NUMBER_WORDS": {
        "size": 20,
        "pos_required": {"numeral"},
        "min_alpha_len": 3,
    },
    # ---- G1 general ----------------------------------------------------------
    "NOUN_CONCRETE": {"size": 120, "pos_required": {"noun"}},
    "VERB_REGULAR": {
        "size": 60,
        "pos_required": {"verb"},
        # rule-specs line 321 + this group's docstring: every verb must yield 4
        # DISTINCT regular forms (base / 3sg-s / past-ed / gerund-ing; past !=
        # base) and must NOT be a morphologically ambiguous (zero-past) verb.
        "custom": "_check_verb_regular_forms",
    },
    "ADJ_PLAIN": {"size": 80, "pos_required": {"adjective"}},
    "ADJ_COMPARABLE": {
        "size": 60,
        "pos_required": {"adjective"},
        # rule-specs ADJ_COMPARABLE note + rule-12 70/30 '-er'/'more X' mix: lock
        # the recorded subtype split so a future edit cannot silently drift it.
        "subtype_min": {"er": 40, "more": 20},
    },
    "ER_NONCOMPARATIVE": {
        "size": 25,
        "pos_required": {"noun"},
        # nouns ending in -er that are not comparatives
        "custom_final_er": True,
    },
    "ADVERB_PLACE": {
        "size": 30,
        # ADVERB_PLACE entries are PHRASE word-entries: 'word' may contain spaces
        # (a phrase). rule-specs line 326 ('fixed word counts (1, 2, 3 words)')
        # and the rule-9 recipe (line 799, '0-3-word ADVERB_PLACE adjunct slots')
        # allow ONLY 1/2/3-word phrases — the count-equalizer assumes a 0-3-word
        # adjunct space, so a 4-word phrase would break it. phrase_word_counts
        # asserts {1,2,3} are PRESENT; max_phrase_words=3 asserts none is larger
        # (a 4-word phrase raises).
        "phrase_word_counts": {1, 2, 3},
        "max_phrase_words": 3,
    },
    "ADVERB_SENT_INITIAL": {"size": 20, "pos_required": {"adverb"}},
    "FRAME_NEUTRAL": {"size": 30, "frame": True},
    "FRAME_PROPER": {"size": 20, "frame": True},
    # ---- G2 z / double -------------------------------------------------------
    "Z_WORDS": {
        "size": 40,
        "every_contains_z": True,
        "subtype_max_frac": {"animal": 0.25, "food": 0.25},
        "matched_pair": ("Z_FREE_MATCHED", ["pos", "length_pm2", "frequency_tier", "no_z"]),
    },
    "Z_FREE_MATCHED": {"size": 40, "every_no_z": True},
    "DOUBLE_WORDS": {
        "size": 80,
        "all_double": True,
        "subtype_max_frac": {"food": 0.25},
        "double_letter_types_min": 10,
    },
    "NONADJ_REPEAT_WORDS": {
        "size": 40,
        "all_double_free": True,
        "all_nonadj_repeat": True,
    },
    # ---- G3 semantic ---------------------------------------------------------
    "ANIMALS": {
        "size": 60,
        "pos_required": {"noun"},
        "min_alpha_len": 3,
        "max_alpha_len": 8,
        "matched_pair": ("OBJECTS_PLANTS_VEHICLES", ["length_dist", "frequency_tier"]),
    },
    "OBJECTS_PLANTS_VEHICLES": {
        "size": 60,
        "subtype_min": {"plant": 15, "vehicle": 10},
    },
    "COLORS": {"size": 16, "pos_required": {"adjective"}},
    "ADJ_NONCOLOR_MATCHED": {
        "size": 30,
        "pos_required": {"adjective"},
        "matched_pair": ("COLORS", ["length_dist", "frequency_tier"]),
    },
    "FIRST_NAMES": {
        "size": 60,
        "all_proper": True,
        "subtype_min": {"feminine": 30, "masculine": 30},
    },
    "NONNAME_PROPER": {
        "size": 60,
        "all_proper": True,
        "subtype_min": {"city": 30, "month": 12, "country": 10, "brand": 8},
    },
    # ---- G4a by-letter -------------------------------------------------------
    "NOUN_PLURAL_BY_LETTER": {
        "size": 70,
        "pos_required": {"noun"},
        "by_letter_min": (BY_LETTER_SET, 5),
    },
    "ADJ_BY_LETTER": {
        "size": 70,
        "pos_required": {"adjective"},
        "by_letter_min": (BY_LETTER_SET, 5),
    },
    "NOUN_FINAL_BY_LETTER": {
        "size": 70,
        "by_letter_min": (BY_LETTER_SET, 5),
    },
    # ---- G4b by-length / terminal --------------------------------------------
    "INITIAL_BY_LENGTH": {
        "size": 54,
        "by_length_min": (tuple(range(3, 12)), 6),
    },
    "FINAL_BY_LENGTH": {
        "size": 54,
        "by_length_min": (tuple(range(3, 12)), 6),
        "discordant_min": 0.25,
    },
    "VOWEL_INITIAL": {
        "size": 40,
        "initial_in": VOWELS,
        "pos_mix": {"noun": (0.6, 0.8), "adverb": (0.2, 0.4)},
        "exclude_words": {"one"},
    },
    "CONSONANT_INITIAL": {
        "size": 40,
        "initial_not_in": VOWELS,
        "pos_mix": {"noun": (0.6, 0.8), "adverb": (0.2, 0.4)},
        "initial_bucket_max": 0.40,
        "matched_pair": ("VOWEL_INITIAL", ["length_dist"]),
    },
    "TERMINAL_VOWEL": {
        "size": 50,
        "pos_required": {"noun"},
        "final_letter_in": VOWELS,
        "not_final_e_min": 0.50,
        "silent_e_min": 0.25,
        "matched_pair": ("TERMINAL_CONSONANT", ["length_dist"]),
    },
    "TERMINAL_CONSONANT": {
        "size": 50,
        "final_letter_not_in": VOWELS,
    },
    # ---- G5 num/hard ---------------------------------------------------------
    "SHORT_WORDS_BY_POS": {
        "size": 45,
        "max_alpha_len": 3,
        "pos_required": {"noun", "verb", "adjective", "numeral"},
    },
    "LONG_ONLY_VOCAB": {
        "size": 150,
        "min_alpha_len": 4,
        "exactly_four_min": 0.40,
        "no_articles": True,
    },
}


# --- registry: merge all groups + reference + frame banks ---------------------


def _raw_registry() -> dict[str, list[Any]]:
    """Merge every group's BANKS dict + core into one raw {name: entries} map.

    Raises if two groups define the same bank name (each bank has ONE owner)."""
    merged: dict[str, list[Any]] = {}
    sources = [
        ("core", core.BANKS),
        ("g1_general", g1_general.BANKS),
        ("g2_zdouble", g2_zdouble.BANKS),
        ("g3_semantic", g3_semantic.BANKS),
        ("g4a_byletter", g4a_byletter.BANKS),
        ("g4b_bylength", g4b_bylength.BANKS),
        ("g5_numhard", g5_numhard.BANKS),
    ]
    for src_name, banks in sources:
        for name, entries in banks.items():
            if name in merged:
                raise BankError(f"bank {name!r} defined in two groups (second: {src_name})")
            merged[name] = entries
    return merged


# the live raw registry of authored content (populated as authors fill stubs)
BANKS: dict[str, list[Any]] = _raw_registry()


def bank_names() -> list[str]:
    """All bank names that have a quota (the full target set, populated or not)."""
    return sorted(BANK_QUOTAS)


def populated_banks() -> list[str]:
    """Bank names with at least one authored entry right now."""
    return sorted(n for n, e in BANKS.items() if e)


def get_bank(name: str) -> Bank:
    """Build and return the validated Bank for ``name`` (word banks only)."""
    if name in FRAME_BANKS:
        raise BankError(f"{name} is a frame bank; use BANKS[{name!r}] for its frames")
    if name not in BANKS or not BANKS[name]:
        raise BankError(f"bank {name!r} is empty / not populated")
    return build_bank(name, BANKS[name])


# --- the quota self-check -----------------------------------------------------


def _frac(n: int, total: int) -> float:
    return n / total if total else 0.0


def check_bank(name: str) -> Bank | None:
    """Assert the BANK_QUOTAS contract for ``name``. Raise BankQuotaError on
    violation. Returns the built Bank (None for frame banks). Raises BankError
    if the name has no quota or is unpopulated (callers wanting a skip should
    consult populated_banks() first)."""
    if name not in BANK_QUOTAS:
        raise BankError(f"no quota defined for bank {name!r}")
    quota = BANK_QUOTAS[name]

    if quota.get("frame"):
        frames = BANKS.get(name) or []
        if not frames:
            raise BankQuotaError(f"{name}: frame bank is empty / not populated")
        _check_frame_bank(name, frames, quota["size"])
        return None

    if name not in BANKS or not BANKS[name]:
        raise BankQuotaError(f"{name}: bank is empty / not populated")

    bank = get_bank(name)
    entries = bank.entries
    total = len(entries)

    if total < quota["size"]:
        raise BankQuotaError(f"{name}: {total} entries < required size {quota['size']}")

    if "min_alpha_len" in quota:
        bad = [e.word for e in entries if e.length < quota["min_alpha_len"]]
        if bad:
            raise BankQuotaError(f"{name}: entries below min length {quota['min_alpha_len']}: {bad}")
    if "max_alpha_len" in quota:
        bad = [e.word for e in entries if e.length > quota["max_alpha_len"]]
        if bad:
            raise BankQuotaError(f"{name}: entries above max length {quota['max_alpha_len']}: {bad}")

    if "pos_required" in quota:
        present = {e.pos for e in entries}
        missing = set(quota["pos_required"]) - present
        if missing:
            raise BankQuotaError(f"{name}: missing required POS {sorted(missing)}")

    if "pos_mix" in quota:
        for pos, (lo, hi) in quota["pos_mix"].items():
            f = _frac(sum(1 for e in entries if e.pos == pos), total)
            if not (lo <= f <= hi):
                raise BankQuotaError(
                    f"{name}: POS {pos!r} share {f:.2f} outside [{lo}, {hi}]"
                )

    if "by_letter_min" in quota:
        letters, k = quota["by_letter_min"]
        for L in letters:
            cnt = sum(1 for e in entries if e.initial == L)
            if cnt < k:
                raise BankQuotaError(f"{name}: only {cnt} entries with initial {L!r} (need {k})")

    if "by_length_min" in quota:
        lengths, k = quota["by_length_min"]
        for L in lengths:
            cnt = sum(1 for e in entries if e.length == L)
            if cnt < k:
                raise BankQuotaError(f"{name}: only {cnt} entries of length {L} (need {k})")

    if quota.get("all_double") and any(not e.has_adjacent_double for e in entries):
        bad = [e.word for e in entries if not e.has_adjacent_double]
        raise BankQuotaError(f"{name}: entries without an adjacent double: {bad}")
    if quota.get("all_double_free") and any(e.has_adjacent_double for e in entries):
        bad = [e.word for e in entries if e.has_adjacent_double]
        raise BankQuotaError(f"{name}: entries with an adjacent double (must be double-free): {bad}")
    if quota.get("all_nonadj_repeat") and any(not e.has_nonadjacent_repeat for e in entries):
        bad = [e.word for e in entries if not e.has_nonadjacent_repeat]
        raise BankQuotaError(f"{name}: entries without a non-adjacent repeat: {bad}")

    if quota.get("every_contains_z"):
        bad = [e.word for e in entries if "z" not in e.word.lower()]
        if bad:
            raise BankQuotaError(f"{name}: entries without a z: {bad}")
    if quota.get("every_no_z"):
        bad = [e.word for e in entries if "z" in e.word.lower()]
        if bad:
            raise BankQuotaError(f"{name}: entries containing a z: {bad}")

    if quota.get("custom_final_er"):
        bad = [e.word for e in entries if not e.word.lower().endswith("er")]
        if bad:
            raise BankQuotaError(f"{name}: entries not ending in -er: {bad}")

    if "double_letter_types_min" in quota:
        types: set[str] = set()
        for e in entries:
            a = _alpha_only(e.word)
            for i in range(len(a) - 1):
                if a[i] == a[i + 1]:
                    types.add(a[i])
        if len(types) < quota["double_letter_types_min"]:
            raise BankQuotaError(
                f"{name}: only {len(types)} double-letter types (need {quota['double_letter_types_min']})"
            )

    if "subtype_min" in quota:
        for st, k in quota["subtype_min"].items():
            cnt = sum(1 for e in entries if e.subtype == st)
            if cnt < k:
                raise BankQuotaError(f"{name}: only {cnt} entries of subtype {st!r} (need {k})")
    if "subtype_max_frac" in quota:
        for st, fmax in quota["subtype_max_frac"].items():
            f = _frac(sum(1 for e in entries if e.subtype == st), total)
            if f > fmax:
                raise BankQuotaError(f"{name}: subtype {st!r} share {f:.2f} > max {fmax}")

    if quota.get("all_proper") and any(not e.proper for e in entries):
        bad = [e.word for e in entries if not e.proper]
        raise BankQuotaError(f"{name}: entries not marked proper: {bad}")

    if "initial_in" in quota:
        bad = [e.word for e in entries if e.initial not in quota["initial_in"]]
        if bad:
            raise BankQuotaError(f"{name}: entries whose initial is not in the required set: {bad}")
    if "initial_not_in" in quota:
        bad = [e.word for e in entries if e.initial in quota["initial_not_in"]]
        if bad:
            raise BankQuotaError(f"{name}: entries whose initial is in the forbidden set: {bad}")
    if "exclude_words" in quota:
        bad = [e.word for e in entries if e.word.lower() in quota["exclude_words"]]
        if bad:
            raise BankQuotaError(f"{name}: contains excluded words {bad}")

    if "final_letter_in" in quota:
        bad = [e.word for e in entries if e.final not in quota["final_letter_in"]]
        if bad:
            raise BankQuotaError(f"{name}: entries not ending in the required letter set: {bad}")
    if "final_letter_not_in" in quota:
        bad = [e.word for e in entries if e.final in quota["final_letter_not_in"]]
        if bad:
            raise BankQuotaError(f"{name}: entries ending in a forbidden letter: {bad}")
    if "not_final_e_min" in quota:
        f = _frac(sum(1 for e in entries if e.final != "e"), total)
        if f < quota["not_final_e_min"]:
            raise BankQuotaError(f"{name}: non-e ending share {f:.2f} < {quota['not_final_e_min']}")
    if "silent_e_min" in quota:
        f = _frac(sum(1 for e in entries if e.final == "e"), total)
        if f < quota["silent_e_min"]:
            raise BankQuotaError(f"{name}: e-ending share {f:.2f} < {quota['silent_e_min']}")

    if "discordant_min" in quota:
        f = _frac(sum(1 for e in entries if e.discordant), total)
        if f < quota["discordant_min"]:
            raise BankQuotaError(f"{name}: discordant share {f:.2f} < {quota['discordant_min']}")

    if "initial_bucket_max" in quota:
        buckets: dict[str, int] = {}
        for e in entries:
            buckets[_bucket(e.initial)] = buckets.get(_bucket(e.initial), 0) + 1
        if len(buckets) < 4:
            raise BankQuotaError(f"{name}: initials spread over only {len(buckets)} buckets (need >= 4)")
        for b, cnt in buckets.items():
            if _frac(cnt, total) > quota["initial_bucket_max"]:
                raise BankQuotaError(
                    f"{name}: initial bucket {b} share {_frac(cnt, total):.2f} > {quota['initial_bucket_max']}"
                )

    if "exactly_four_min" in quota:
        f = _frac(sum(1 for e in entries if e.length == 4), total)
        if f < quota["exactly_four_min"]:
            raise BankQuotaError(f"{name}: 4-letter share {f:.2f} < {quota['exactly_four_min']}")
    if quota.get("no_articles"):
        bad = [e.word for e in entries if e.word.lower() in {"a", "an", "the"}]
        if bad:
            raise BankQuotaError(f"{name}: contains articles {bad}")

    if "phrase_word_counts" in quota:
        present = {len(e.word.split()) for e in entries}
        need = quota["phrase_word_counts"]
        missing = need - present
        if missing:
            raise BankQuotaError(f"{name}: missing phrase word-counts {sorted(missing)}")
    if "max_phrase_words" in quota:
        cap = quota["max_phrase_words"]
        bad = [e.word for e in entries if len(e.word.split()) > cap]
        if bad:
            raise BankQuotaError(
                f"{name}: phrases longer than {cap} words (only 1-{cap}-word adjuncts allowed): {bad}"
            )

    # matched-pair checks need the OTHER bank populated too; skip cleanly if not.
    # SYMMETRIC: run the cross-validation whether ``name`` is the OWNER side
    # (its own quota carries matched_pair) OR the DEPENDENT side (some other
    # quota names it as the partner). Without this, check_bank("Z_FREE_MATCHED")
    # alone would miss a tier/POS flip on the dependent and only check_all_banks
    # would catch it; the documented check_bank(name) API must be self-sufficient
    # for either side. The cross-check is always run owner-first so its tag
    # semantics (e.g. the per-entry ``pair`` direction) stay well-defined.
    for owner_name, partner_name, tags in _matched_pairs_touching(name):
        if BANKS.get(owner_name) and BANKS.get(partner_name):
            _check_matched_pair(
                owner_name, get_bank(owner_name), partner_name, get_bank(partner_name), tags
            )

    if "custom" in quota:
        check = _CUSTOM_CHECKS[quota["custom"]]
        check(bank, BANKS)

    return bank


def _matched_pairs_touching(name: str) -> list[tuple[str, str, list[str]]]:
    """Every matched_pair cross-check that involves ``name`` on EITHER side,
    each returned owner-first as (owner, partner, tags). Drives the symmetric
    matched_pair check: a dependent bank's check_bank re-runs the owner's
    cross-validation."""
    out: list[tuple[str, str, list[str]]] = []
    for owner, q in BANK_QUOTAS.items():
        mp = q.get("matched_pair")
        if not mp:
            continue
        partner, tags = mp
        if name in (owner, partner):
            out.append((owner, partner, tags))
    return out


def _check_matched_pair(
    name: str, bank: Bank, other_name: str, other: Bank, tags: list[str]
) -> None:
    """Cross-bank matching. ``pair``-keyed checks (Z banks) match per linked
    entry; ``length_dist`` matches the multiset of lengths within +/- a small
    tolerance; ``frequency_tier`` matches tier composition; ``no_z`` asserts
    the counterpart has no z."""
    if "pos" in tags or "length_pm2" in tags or "no_z" in tags:
        # per-entry matched pairs via the ``pair`` key
        by_pair_other = {e.pair: e for e in other.entries if e.pair is not None}
        for e in bank.entries:
            if e.pair is None:
                raise BankQuotaError(f"{name}: entry {e.word!r} lacks a 'pair' key for matching to {other_name}")
            mate = by_pair_other.get(e.pair)
            if mate is None:
                raise BankQuotaError(f"{name}: entry {e.word!r} pair {e.pair!r} has no counterpart in {other_name}")
            if "pos" in tags and e.pos != mate.pos:
                raise BankQuotaError(f"{name}/{other_name} pair {e.pair!r}: POS {e.pos} != {mate.pos}")
            if "length_pm2" in tags and abs(e.length - mate.length) > 2:
                raise BankQuotaError(
                    f"{name}/{other_name} pair {e.pair!r}: |len {e.length}-{mate.length}| > 2"
                )
            if "frequency_tier" in tags and e.frequency_tier != mate.frequency_tier:
                raise BankQuotaError(
                    f"{name}/{other_name} pair {e.pair!r}: tier {e.frequency_tier} != {mate.frequency_tier}"
                )
            if "no_z" in tags and "z" in mate.word.lower():
                raise BankQuotaError(f"{other_name}: counterpart {mate.word!r} of {e.word!r} contains z")
        return
    # distributional matching (length / tier composition within tolerance)
    if "length_dist" in tags:
        m1 = sum(e.length for e in bank.entries) / len(bank.entries)
        m2 = sum(e.length for e in other.entries) / len(other.entries)
        if abs(m1 - m2) > 1.5:
            raise BankQuotaError(
                f"{name}/{other_name}: mean length {m1:.2f} vs {m2:.2f} differ by > 1.5"
            )
    if "frequency_tier" in tags:
        f1 = _frac(sum(1 for e in bank.entries if e.frequency_tier == 1), len(bank.entries))
        f2 = _frac(sum(1 for e in other.entries if e.frequency_tier == 1), len(other.entries))
        if abs(f1 - f2) > 0.20:
            raise BankQuotaError(
                f"{name}/{other_name}: tier-1 share {f1:.2f} vs {f2:.2f} differ by > 0.20"
            )


# --- custom per-bank checks (referenced by name from a quota's 'custom' key) --

# Zero-past / invariant-past verbs whose past tense equals the base (or whose
# -ed/-ing morphology is ambiguous / irregular): a regular-verb bank must
# exclude these so the four-distinct-forms invariant holds and so 'past' is a
# reliable surface cue for rule 8. rule-specs line 321 names put/cut/set/hit/read
# explicitly; the rest are the other common English zero-past verbs.
_AMBIGUOUS_PAST_VERBS = frozenset(
    {
        "put", "cut", "set", "hit", "read", "let", "shut", "cost", "hurt", "bet",
        "burst", "cast", "split", "spread", "thrust", "quit", "bid", "rid", "shed",
        "slit", "wed", "knit", "fit",
    }
)

_VERB_SIBILANT_ENDINGS = ("s", "x", "z", "ch", "sh")


def _regular_verb_forms(base: str) -> tuple[str, str, str, str]:
    """Derive the four regular forms (base, 3sg -s, past -ed, gerund -ing) of a
    regular base verb by the standard English spelling rules: consonant+y ->
    -ies/-ied, silent -e drop, sibilant -es, CVC consonant doubling, vowel+y
    plain. Used only to PROVE the four forms are distinct (the generator owns
    real inflection); irregular/ambiguous verbs are excluded by blacklist."""
    b = base.lower()
    vowels = "aeiou"

    def _vowel_groups(w: str) -> int:
        groups, prev = 0, False
        for ch in w:
            cur = ch in vowels
            if cur and not prev:
                groups += 1
            prev = cur
        return groups

    def _is_cvc(w: str) -> bool:
        # single final consonant preceded by a single vowel preceded by a
        # consonant (the doubling trigger); final must not be w/x/y. English
        # doubles only on stress-final bases; without a stress model we restrict
        # doubling to MONOSYLLABLES (one vowel group), the unambiguous case, so
        # multi-syllable -er/-en/-it verbs (open, order, visit) take plain -ed.
        return (
            len(w) >= 3
            and _vowel_groups(w) == 1
            and w[-1] not in vowels
            and w[-1] not in "wxy"
            and w[-2] in vowels
            and w[-3] not in vowels
        )

    # 3rd-person singular (-s)
    if any(b.endswith(s) for s in _VERB_SIBILANT_ENDINGS):
        sg = b + "es"
    elif b.endswith("y") and len(b) >= 2 and b[-2] not in vowels:
        sg = b[:-1] + "ies"
    else:
        sg = b + "s"

    # past (-ed)
    if b.endswith("e"):
        past = b + "d"
    elif b.endswith("y") and len(b) >= 2 and b[-2] not in vowels:
        past = b[:-1] + "ied"
    elif _is_cvc(b):
        past = b + b[-1] + "ed"
    else:
        past = b + "ed"

    # gerund (-ing)
    if b.endswith("e") and not b.endswith("ee") and not b.endswith("ie"):
        gerund = b[:-1] + "ing"
    elif _is_cvc(b):
        gerund = b + b[-1] + "ing"
    else:
        gerund = b + "ing"

    return b, sg, past, gerund


def _check_verb_regular_forms(bank: Bank, _registry: Mapping[str, Any]) -> None:
    """VERB_REGULAR self-check (rule-specs line 321): every base must derive four
    DISTINCT regular forms (base / 3sg-s / past-ed / gerund-ing, past != base)
    and must not be a morphologically ambiguous zero-past verb."""
    for e in bank.entries:
        base = e.word.lower()
        if base in _AMBIGUOUS_PAST_VERBS:
            raise BankQuotaError(
                f"{bank.name}: {base!r} is an ambiguous/zero-past verb "
                f"(past == base); regular verbs must have 4 distinct forms"
            )
        forms = _regular_verb_forms(base)
        b, sg, past, gerund = forms
        if past == b:
            raise BankQuotaError(f"{bank.name}: {base!r} has past == base ({past!r})")
        if len(set(forms)) != 4:
            raise BankQuotaError(
                f"{bank.name}: {base!r} does not yield 4 distinct forms "
                f"(base/3sg/past/gerund = {forms})"
            )


# registry of named custom checks (a quota's 'custom' value indexes this)
_CUSTOM_CHECKS: dict[str, Callable[[Bank, Mapping[str, Any]], None]] = {
    "_check_verb_regular_forms": _check_verb_regular_forms,
}


def check_all_banks(only_populated: bool = True) -> dict[str, str]:
    """Check every bank. With ``only_populated`` (default), skip empty banks
    (the B0-stage state where authors have not filled their groups). Returns a
    {name: 'ok'} map for the banks it checked; raises BankQuotaError on the
    FIRST violation (LOUD — a half-built bank must not pass silently). A frame
    bank counts as populated iff its frame list is non-empty."""
    result: dict[str, str] = {}
    for name in bank_names():
        present = bool(BANKS.get(name))
        if only_populated and not present:
            continue
        check_bank(name)
        result[name] = "ok"
    return result
