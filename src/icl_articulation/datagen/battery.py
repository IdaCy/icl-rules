"""The 40 FROZEN generic single-feature predicates + the battery self-test.

This is the automated hostile-reviewer check (rule-specs globals.generic_probe_
battery). For every emitted rule dataset, every one of the 40 predicates below
is applied to the items and its agreement with the labels is measured. The
criterion is:

    max(agreement, 1 - agreement) <= 75%   (INCLUSIVE; exactly 75% PASSES)

An anti-correlated cue is a confound too, hence the max() over a predicate and
its negation. A predicate is EXEMPT (allowed to exceed 75%) iff:
    (a) it OR ITS COMPLEMENT maps, via equiv_keys, to a string in the rule's
        equivalence_class, OR
    (b) the predicate is listed in the rule's battery_exemptions, OR
    (c) it meets the <= 75% criterion.

The 40 predicates are frozen here and implemented 1:1 with the spec list; the
module asserts len(PREDICATES) == 40 at import. POS predicates use the pinned
nltk averaged_perceptron_tagger; tests that exercise them use
``pytest.importorskip('nltk')`` and never download at import.

The equiv_keys table (which predicate, on which rule, instantiates or is the
complement of which equivalence-class string) is sourced from the COMMITTED
data/spec_extract.json so this module never reads the private spec. The caller
passes the rule's equivalence_class + battery_exemptions (both in the extract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .schema import words

PASS_THRESHOLD = 0.75  # inclusive: max(agreement, 1-agreement) <= 0.75 passes

VOWEL_LETTERS = frozenset("aeiou")

# coarse POS map for the nltk tagger (rule-specs implementation_pins.pos_tagger).
# "MD" (modal) -> verb so the aux/copular family lands in 'verb' (rule 11's
# True-class signal). The pinned averaged_perceptron_tagger nonetheless mis-tags
# several of the spec's aux/copular openers in isolation (e.g. 'Were'->WRB,
# 'Must'->NN, 'Can'/'Will'/'Should'/'Could'->non-MD), so MD alone is not enough:
# the closed-set override below pins exactly the spec-named words to 'verb'.
_PTB_TO_COARSE = {
    "NN": "noun", "NNS": "noun", "NNP": "noun", "NNPS": "noun",
    "VB": "verb", "VBD": "verb", "VBG": "verb", "VBN": "verb", "VBP": "verb", "VBZ": "verb",
    "MD": "verb",
    "JJ": "adjective", "JJR": "adjective", "JJS": "adjective",
    "RB": "adverb", "RBR": "adverb", "RBS": "adverb", "WRB": "adverb",
    "PRP": "pronoun", "PRP$": "pronoun", "WP": "pronoun", "WP$": "pronoun",
    "DT": "determiner", "WDT": "determiner", "PDT": "determiner",
}

# The aux/copular-verb family that rule 11's equivalence string names verbatim
# (rule-specs equivalence_class: "...auxiliary or copular verb
# (is/are/was/were/can/will/should/must/could)"). These ALWAYS coarse-map to
# 'verb' regardless of the tagger's tag — robust to nltk's isolation mis-tags
# (e.g. 'Were'->WRB->adverb, 'Must'->NN->noun, the modals falling to 'other').
# Kept tight (exactly this closed set) so no content word can be mislabeled.
_AUX_COPULAR_OVERRIDE = frozenset(
    {"is", "are", "was", "were", "can", "will", "should", "must", "could"}
)


def coarse_pos_first_word(text: str) -> str:
    """Coarse POS of the first word via the pinned nltk tagger.

    Imports nltk lazily so the module never requires it at import time. Raises
    if nltk (or its tagger data) is unavailable — callers that want to skip use
    pytest.importorskip first.

    The spec's aux/copular openers {is,are,was,were,can,will,should,must,could}
    are pinned to 'verb' via _AUX_COPULAR_OVERRIDE before consulting the tagger,
    because the pinned averaged_perceptron_tagger mis-tags several in isolation
    (notably 'Were'->WRB->adverb and 'Must'->NN->noun)."""
    import nltk  # lazy

    toks = words(text)
    if not toks:
        return "other"
    if toks[0].lower() in _AUX_COPULAR_OVERRIDE:
        return "verb"
    tagged = nltk.pos_tag(toks)
    ptb = tagged[0][1]
    return _PTB_TO_COARSE.get(ptb, "other")


# --- string-level helpers (raw string for char rules; tokenizer for words) ----


def _wc(text: str) -> int:
    return len(words(text))


def _char_count(text: str) -> int:
    return len(text)


def _first_word(text: str) -> str:
    toks = words(text)
    return toks[0] if toks else ""


def _last_word(text: str) -> str:
    toks = words(text)
    return toks[-1] if toks else ""


def _alpha_len(word: str) -> int:
    return sum(1 for ch in word if ch.isalpha())


def _first_alpha(word: str) -> str:
    for ch in word:
        if ch.isalpha():
            return ch.lower()
    return ""


def _last_alpha(word: str) -> str:
    for ch in reversed(word):
        if ch.isalpha():
            return ch.lower()
    return ""


def _contains_word(text: str, w: str) -> bool:
    return any(tok.lower() == w for tok in words(text))


def _count_word(text: str, w: str) -> int:
    return sum(1 for tok in words(text) if tok.lower() == w)


def _bucket(letter: str) -> str:
    if letter in "abcdef":
        return "a-f"
    if letter in "ghijklm":
        return "g-m"
    if letter in "nopqrs":
        return "n-s"
    return "t-z"


# --- the predicate record -----------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """One frozen battery predicate.

    ``key`` is the stable identifier used in the equiv_keys table and reports.
    ``fn(text) -> bool``. ``needs_pos`` flags the 6 first-word-POS predicates
    (they require nltk; tests importorskip)."""

    key: str
    fn: Callable[[str], bool]
    needs_pos: bool = False


def _build_predicates() -> list[Predicate]:
    preds: list[Predicate] = []

    # word_count >= k for k in 5..10 (6)
    for k in range(5, 11):
        preds.append(Predicate(f"word_count>={k}", (lambda kk: lambda t: _wc(t) >= kk)(k)))
    # char_count >= 35/40/45 (3)
    for k in (35, 40, 45):
        preds.append(Predicate(f"char_count>={k}", (lambda kk: lambda t: _char_count(t) >= kk)(k)))
    # contains 'the' / 'a' / 'and' (3)
    for w in ("the", "a", "and"):
        preds.append(Predicate(f"contains_{w}", (lambda ww: lambda t: _contains_word(t, ww))(w)))
    # count('the') >= 2 (1)
    preds.append(Predicate("count_the>=2", lambda t: _count_word(t, "the") >= 2))
    # first word starts with a vowel / consonant letter (2)
    preds.append(Predicate("first_starts_vowel", lambda t: _first_alpha(_first_word(t)) in VOWEL_LETTERS))
    preds.append(
        Predicate(
            "first_starts_consonant",
            lambda t: (_first_alpha(_first_word(t)) != "" and _first_alpha(_first_word(t)) not in VOWEL_LETTERS),
        )
    )
    # last word ends with a vowel / consonant letter (2)
    preds.append(Predicate("last_ends_vowel", lambda t: _last_alpha(_last_word(t)) in VOWEL_LETTERS))
    preds.append(
        Predicate(
            "last_ends_consonant",
            lambda t: (_last_alpha(_last_word(t)) != "" and _last_alpha(_last_word(t)) not in VOWEL_LETTERS),
        )
    )
    # contains a digit (1)
    preds.append(Predicate("contains_digit", lambda t: any(ch.isdigit() for ch in t)))
    # contains a comma (1)
    preds.append(Predicate("contains_comma", lambda t: "," in t))
    # first-word length >= k for k in 4..8 (5)
    for k in range(4, 9):
        preds.append(Predicate(f"first_word_len>={k}", (lambda kk: lambda t: _alpha_len(_first_word(t)) >= kk)(k)))
    # last-word length <= k for k in 3..6 (4)
    for k in range(3, 7):
        preds.append(Predicate(f"last_word_len<={k}", (lambda kk: lambda t: _alpha_len(_last_word(t)) <= kk)(k)))
    # first-letter bucket a-f / g-m / n-s / t-z (4)
    for b in ("a-f", "g-m", "n-s", "t-z"):
        preds.append(
            Predicate(
                f"first_letter_bucket_{b}",
                (lambda bb: lambda t: (_first_alpha(_first_word(t)) != "" and _bucket(_first_alpha(_first_word(t))) == bb))(b),
            )
        )
    # any word beyond position 1 capitalized (1)
    preds.append(
        Predicate(
            "nonfirst_word_capitalized",
            lambda t: any(tok[:1].isupper() for tok in words(t)[1:]),
        )
    )
    # all lowercase (1)
    preds.append(Predicate("all_lowercase", lambda t: t == t.lower()))
    # first-word POS in {noun, verb, adjective, adverb, pronoun, determiner} (6)
    for pos in ("noun", "verb", "adjective", "adverb", "pronoun", "determiner"):
        preds.append(
            Predicate(
                f"first_word_pos={pos}",
                (lambda pp: lambda t: coarse_pos_first_word(t) == pp)(pos),
                needs_pos=True,
            )
        )
    return preds


PREDICATES: list[Predicate] = _build_predicates()

# frozen at exactly 40 (rule-specs: "Exactly 40, frozen here")
assert len(PREDICATES) == 40, f"battery must have exactly 40 predicates, has {len(PREDICATES)}"
PREDICATES_BY_KEY: dict[str, Predicate] = {p.key: p for p in PREDICATES}


# --- agreement + exemption ----------------------------------------------------


def predicate_agreement(pred: Predicate, items: Sequence[dict]) -> float:
    """Fraction of items where pred(text) equals the (bool) label."""
    if not items:
        raise ValueError("no items to score the battery on")
    n_agree = sum(1 for it in items if pred.fn(it["text"]) == bool(it["label"]))
    return n_agree / len(items)


def predicate_score(pred: Predicate, items: Sequence[dict]) -> float:
    """max(agreement, 1 - agreement) — the confound magnitude of a predicate."""
    a = predicate_agreement(pred, items)
    return max(a, 1.0 - a)


def is_exempt(
    pred: Predicate,
    *,
    equiv_keys: dict[str, list[str]],
    equivalence_class: Sequence[str],
    battery_exemptions: Sequence[str],
) -> bool:
    """Exemption (a) equiv-class membership of the predicate OR its complement,
    or (b) explicit battery_exemptions listing. (c) — meeting the threshold —
    is decided by the caller (battery_report), not here.

    ``equiv_keys`` maps a predicate key to the verbatim equivalence_class
    strings it (or its complement) instantiates FOR THIS RULE (already filtered
    to this rule by the caller, from data/spec_extract.json). The predicate is
    equiv-exempt iff any of those strings is in this rule's equivalence_class.
    ``battery_exemptions`` is the rule's list of exempt predicate keys."""
    if pred.key in set(battery_exemptions):
        return True
    eq_set = set(equivalence_class)
    for s in equiv_keys.get(pred.key, []):
        if s in eq_set:
            return True
    return False


@dataclass(frozen=True)
class BatteryResult:
    key: str
    agreement: float
    score: float          # max(agreement, 1-agreement)
    exempt: bool          # via (a) equiv-class or (b) battery_exemptions
    passes: bool          # exempt OR score <= 0.75
    skipped: bool = False  # POS predicate skipped (nltk unavailable)


def battery_report(
    items: Sequence[dict],
    *,
    equiv_keys: dict[str, list[str]] | None = None,
    equivalence_class: Sequence[str] = (),
    battery_exemptions: Sequence[str] = (),
    run_pos: bool = True,
) -> list[BatteryResult]:
    """Score every predicate over ``items``. Each result records the agreement,
    the confound score, whether the predicate is exempt, and whether it passes
    (exempt OR score <= 0.75 INCLUSIVE).

    ``run_pos``: when False (or nltk unavailable), the 6 POS predicates are
    marked skipped rather than scored — never downloads, never raises for a
    missing optional dep."""
    equiv_keys = equiv_keys or {}
    pos_ok = run_pos
    if run_pos:
        # Eagerly probe the tagger: nltk may be installed while its averaged_
        # perceptron_tagger DATA is missing (the fresh-instance state), in which
        # case the first POS predicate would raise LookupError mid-battery.
        # Catch BOTH the package-absent (ImportError) and data-absent
        # (LookupError) cases and skip the 6 POS predicates — never download,
        # never raise for a missing optional dep.
        try:
            from nltk import pos_tag

            pos_tag(["ok"])  # forces the tagger data load now
        except (ImportError, LookupError):
            pos_ok = False

    out: list[BatteryResult] = []
    for pred in PREDICATES:
        if pred.needs_pos and not pos_ok:
            out.append(BatteryResult(pred.key, float("nan"), float("nan"), False, True, skipped=True))
            continue
        agreement = predicate_agreement(pred, items)
        score = max(agreement, 1.0 - agreement)
        exempt = is_exempt(
            pred,
            equiv_keys=equiv_keys,
            equivalence_class=equivalence_class,
            battery_exemptions=battery_exemptions,
        )
        passes = exempt or score <= PASS_THRESHOLD
        out.append(BatteryResult(pred.key, agreement, score, exempt, passes))
    return out


def battery_violations(results: Sequence[BatteryResult]) -> list[BatteryResult]:
    """The non-skipped results that FAIL (block dataset acceptance)."""
    return [r for r in results if not r.skipped and not r.passes]
