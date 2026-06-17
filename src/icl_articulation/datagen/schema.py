"""The locked Item record, jsonl IO, the seeded by-base split assigner, validators.

The item schema is the one frozen in rule-specs.yaml globals.dataset_construction:
``[item_id, base_id, rule_id, label, text, slots_meta, split]``. Anything emitted
here MUST round-trip through the existing ``icl_articulation.contexts`` loader,
so this module imports that loader's constants (REQUIRED_FIELDS, SPLITS,
BALANCED_SPLITS, normalize_label) directly rather than re-declaring them — the
loader is the single source of truth for the runner-facing contract, and a
divergence would only surface on a paid run. ``write_items`` -> ``load_items``
is the integration test.

Split assignment is BY base_id (rule-specs MUST-FIX #3): a base and ALL its
variants land in exactly one split. Two split schemes are pinned:

- programmatic_rules: bases are partitioned into few_shot_pool / held_out /
  confirmation / spare with the spec's base counts (100 / 120 / 100 / >=20).
  few_shot_pool keeps BOTH variants per base; held_out and confirmation keep
  exactly ONE variant per base (60T/60F and 50T/50F respectively); spare is
  unconstrained.
- llm_rules (15, 16): no transform pairing, base_id == item_id, item-level
  splits 120 / 120 / 100, each balanced 50/50.

All violations raise SchemaError (a DatasetError subclass, so callers that
catch the loader's error type also catch ours). LOUD, never a quiet fallback.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any, Iterable, Sequence

from ..contexts import (
    BALANCED_SPLITS,
    DatasetError,
    REQUIRED_FIELDS,
    SPLITS,
    normalize_label,
    validate_dataset,
)

# the locked field order (rule-specs globals.dataset_construction.item_schema)
ITEM_SCHEMA: tuple[str, ...] = (
    "item_id",
    "base_id",
    "rule_id",
    "label",
    "text",
    "slots_meta",
    "split",
)

# spec'd word-count window (globals.sentence_style.word_count_range)
WORD_COUNT_MIN = 4
WORD_COUNT_MAX = 14

# globals.tokenizer.punct_strip — leading/trailing only
PUNCT_STRIP = ".,!?;:\"'()[]-–—…"

# programmatic split base counts (globals.dataset_construction.programmatic_rules)
PROGRAMMATIC_SPLIT_BASES: dict[str, int] = {
    "few_shot_pool": 100,
    "held_out": 120,
    "confirmation": 100,
}
PROGRAMMATIC_SPARE_MIN = 20
PROGRAMMATIC_N_BASES_MIN = 340

# llm split item counts (globals.dataset_construction.llm_rules)
LLM_SPLIT_ITEMS: dict[str, int] = {
    "few_shot_pool": 120,
    "held_out": 120,
    "confirmation": 100,
}
LLM_N_ITEMS_MIN = 340

# Per-rule sentence-style policy, keyed by the spec's rule_id (the string id
# stored in each item's rule_id field, e.g. "contains_exclamation"). This is the
# single authoritative rule_id -> (allow_terminal, allow_internal_comma) table,
# so generators pass their rule_id and the validator SELF-SELECTS the policy
# rather than the author hand-passing flags (a footgun: wrong flags would silently
# admit or reject the rule's legal punctuation). Derived from
# globals.sentence_style:
#   - terminal_punctuation: the ONLY rule allowed a terminal char is rule 3
#     (contains_exclamation), whose True items end with '!'.
#   - internal_punctuation: commas are legal only for rule 26 (exactly_two_commas,
#     the rule itself) and the LLM rules 15/16 (positive_sentiment, food_topic,
#     subject to the comma-rate audit). Rule 3 also carries comma salt in both
#     classes (globals.sentence_style.internal_punctuation (b)).
# Every other rule_id falls through to the all-default (no terminal, no comma)
# policy via _default_style_policy().
RULE_STYLE_POLICY: dict[str, tuple[str, bool]] = {
    # rule_id: (allow_terminal, allow_internal_comma)
    "contains_exclamation": ("!", True),  # rule 3: terminal '!' + comma salt
    "exactly_two_commas": ("", True),  # rule 26: commas ARE the rule
    "positive_sentiment": ("", True),  # rule 15 (llm): commas allowed
    "food_topic": ("", True),  # rule 16 (llm): commas allowed
}

# the policy applied to any rule_id NOT in RULE_STYLE_POLICY: strict global style.
_DEFAULT_STYLE_POLICY: tuple[str, bool] = ("", False)


def style_policy_for(rule_id: str) -> tuple[str, bool]:
    """Look up the (allow_terminal, allow_internal_comma) policy for ``rule_id``.

    Unknown rule_ids get the strict global default (no terminal, no comma). This
    is the central table so no generator author can pass wrong style flags."""
    return RULE_STYLE_POLICY.get(rule_id, _DEFAULT_STYLE_POLICY)


class SchemaError(DatasetError):
    """An item record or a split assignment violated the locked schema contract."""


def _assert_field_order_consistent() -> None:
    """ITEM_SCHEMA must be a permutation of the loader's REQUIRED_FIELDS."""
    if set(ITEM_SCHEMA) != set(REQUIRED_FIELDS):
        raise SchemaError(
            "ITEM_SCHEMA diverged from contexts.REQUIRED_FIELDS: "
            f"{set(ITEM_SCHEMA) ^ set(REQUIRED_FIELDS)}"
        )


_assert_field_order_consistent()


# --- tokenizer (global definition; shared by validators + genutils) -----------


def words(text: str) -> list[str]:
    """Global tokenizer: split on whitespace, strip leading/trailing PUNCT_STRIP,
    drop tokens empty after stripping. Internal characters are never touched."""
    out: list[str] = []
    for tok in text.split():
        stripped = tok.strip(PUNCT_STRIP)
        if stripped:
            out.append(stripped)
    return out


def word_count(text: str) -> int:
    """globals.tokenizer.word_count — number of non-empty stripped tokens."""
    return len(words(text))


# --- item construction --------------------------------------------------------


def make_item(
    *,
    item_id: str,
    base_id: str,
    rule_id: str,
    label: bool,
    text: str,
    slots_meta: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    """Build one schema-conformant item dict (ordered keys), validating types.

    Labels are normalized to bool (the loader accepts bool or 'True'/'False'
    strings, but we store the canonical bool form)."""
    if split not in SPLITS:
        raise SchemaError(f"unknown split {split!r} (expected one of {SPLITS})")
    if not isinstance(text, str) or not text:
        raise SchemaError(f"text must be a non-empty string, got {text!r}")
    if not isinstance(slots_meta, dict):
        raise SchemaError(f"slots_meta must be a dict, got {type(slots_meta).__name__}")
    return {
        "item_id": str(item_id),
        "base_id": str(base_id),
        "rule_id": str(rule_id),
        "label": normalize_label(label),
        "text": text,
        "slots_meta": slots_meta,
        "split": split,
    }


# --- jsonl IO -----------------------------------------------------------------


def write_items(items: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write items to <path> as jsonl (one object per line), creating parents.

    Each item is re-emitted in the locked ITEM_SCHEMA key order. This is the
    write half of the contexts.load_items round-trip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for it in items:
        missing = [f for f in ITEM_SCHEMA if f not in it]
        if missing:
            raise SchemaError(f"item {it.get('item_id')!r} missing fields {missing}")
        ordered = {f: it[f] for f in ITEM_SCHEMA}
        ordered["label"] = normalize_label(ordered["label"])
        lines.append(json.dumps(ordered, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_items(path: str | Path) -> list[dict[str, Any]]:
    """Read jsonl items without the loader's full validation (raw parse)."""
    path = Path(path)
    if not path.is_file():
        raise SchemaError(f"dataset file not found: {path}")
    out: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return out


# --- by-base split assignment -------------------------------------------------


def assign_programmatic_splits(
    base_ids: Sequence[Any],
    seed: int,
    *,
    few_shot_pool: int = PROGRAMMATIC_SPLIT_BASES["few_shot_pool"],
    held_out: int = PROGRAMMATIC_SPLIT_BASES["held_out"],
    confirmation: int = PROGRAMMATIC_SPLIT_BASES["confirmation"],
    spare_min: int = PROGRAMMATIC_SPARE_MIN,
) -> dict[Any, str]:
    """Map each base_id -> split for a programmatic rule (seeded, by base).

    Bases are sorted (stable, file-order-independent) then shuffled with
    ``seed``; the first ``few_shot_pool`` go to few_shot_pool, the next
    ``held_out`` to held_out, the next ``confirmation`` to confirmation, the
    remainder (>= ``spare_min``) to spare. The seed is the caller's to log."""
    uniq = sorted(set(base_ids), key=str)
    if len(uniq) != len(base_ids):
        raise SchemaError(
            f"duplicate base_ids passed to split assignment "
            f"({len(base_ids)} given, {len(uniq)} distinct)"
        )
    need = few_shot_pool + held_out + confirmation + spare_min
    if len(uniq) < need:
        raise SchemaError(
            f"too few bases for the programmatic split: have {len(uniq)}, "
            f"need >= {need} ({few_shot_pool}+{held_out}+{confirmation}+>={spare_min})"
        )
    rng = Random(seed)
    order = list(uniq)
    rng.shuffle(order)
    assignment: dict[Any, str] = {}
    i = 0
    for split, n in (
        ("few_shot_pool", few_shot_pool),
        ("held_out", held_out),
        ("confirmation", confirmation),
    ):
        for base in order[i : i + n]:
            assignment[base] = split
        i += n
    for base in order[i:]:
        assignment[base] = "spare"
    return assignment


def assign_llm_splits(
    item_ids: Sequence[Any],
    labels: Sequence[bool],
    seed: int,
    *,
    few_shot_pool: int = LLM_SPLIT_ITEMS["few_shot_pool"],
    held_out: int = LLM_SPLIT_ITEMS["held_out"],
    confirmation: int = LLM_SPLIT_ITEMS["confirmation"],
) -> dict[Any, str]:
    """Map each item_id -> split for an llm rule (seeded, item-level, balanced).

    base_id == item_id for llm rules, so this is an item-level partition. Each
    balanced split draws an equal number of True and False items; the remainder
    goes to spare. Raises if either class is too small to fill the balanced
    splits 50/50."""
    if len(item_ids) != len(labels):
        raise SchemaError("item_ids and labels length mismatch")
    norm = [normalize_label(v) for v in labels]
    by_label: dict[bool, list[Any]] = {True: [], False: []}
    for iid, lab in zip(item_ids, norm):
        by_label[lab].append(iid)
    for lab in (True, False):
        by_label[lab].sort(key=str)
    # Two INDEPENDENT seeded shuffles, one per label, seeded distinctly so the
    # True and False orderings are uncorrelated. (A shared-seed shuffle here
    # would apply the SAME permutation to both lists — a latent correlation
    # trap — and is intentionally absent.)
    rng_t = Random(seed * 2 + 1)
    rng_f = Random(seed * 2 + 2)
    rng_t.shuffle(by_label[True])
    rng_f.shuffle(by_label[False])

    assignment: dict[Any, str] = {}
    cursors = {True: 0, False: 0}
    for split, n in (
        ("few_shot_pool", few_shot_pool),
        ("held_out", held_out),
        ("confirmation", confirmation),
    ):
        if n % 2 != 0:
            raise SchemaError(f"balanced split {split!r} size {n} must be even")
        half = n // 2
        for lab in (True, False):
            avail = by_label[lab][cursors[lab] : cursors[lab] + half]
            if len(avail) < half:
                raise SchemaError(
                    f"too few {lab} items for split {split!r}: need {half}, "
                    f"have {len(avail)}"
                )
            for iid in avail:
                assignment[iid] = split
            cursors[lab] += half
    for lab in (True, False):
        for iid in by_label[lab][cursors[lab] :]:
            assignment[iid] = "spare"
    return assignment


# --- validators ----------------------------------------------------------------


def assert_sentence_style(
    items: Iterable[dict[str, Any]],
    *,
    rule_id: str | None = None,
    allow_terminal: str = "",
    allow_internal_comma: bool = False,
) -> None:
    """Style conformance, parameterized so rule-driven deviations are allowed.

    Defaults enforce the global style (globals.sentence_style): ASCII only, no
    'I' pronoun, no terminal punctuation, no internal punctuation. Per-rule
    deviations are opted in two ways:
      - the SELF-SELECTING path (preferred for generators): pass ``rule_id`` and
        the policy is looked up in RULE_STYLE_POLICY (style_policy_for). This
        removes the footgun of an author hand-passing the wrong flags.
      - the EXPLICIT-FLAGS path (for tests): pass ``allow_terminal`` /
        ``allow_internal_comma`` directly. ``rule_id`` and the explicit flags are
        mutually exclusive.

    ``allow_terminal`` = terminal characters a rule may append (rule 3 -> '!');
    ``allow_internal_comma`` = rules whose data carries commas (3, 26, 15, 16).
    Raises SchemaError on the first violation (LOUD)."""
    if rule_id is not None:
        if allow_terminal or allow_internal_comma:
            raise SchemaError(
                "pass either rule_id (self-selecting policy) OR explicit flags, "
                "not both"
            )
        allow_terminal, allow_internal_comma = style_policy_for(rule_id)
    for it in items:
        text = it["text"]
        if not text.isascii():
            raise SchemaError(f"non-ASCII text: {text!r}")
        # The banned-'I' check runs over the STRIPPED tokenizer output (the same
        # words() logic the rest of the framework uses), so punctuation-adjacent
        # forms like 'I,' or 'I.' strip to 'I' and are caught. A raw text.split()
        # would miss them (globals.sentence_style.no_pronoun_I bans 'I' as an
        # uncontrolled capital EVERYWHERE).
        if any(t == "I" for t in words(text)):
            raise SchemaError(f"banned pronoun 'I' in: {text!r}")
        last = text[-1]
        if not last.isalnum():
            if last not in allow_terminal:
                raise SchemaError(
                    f"unexpected terminal punctuation {last!r} in: {text!r}"
                )
        # internal punctuation: scan all but the (possibly allowed) terminal char
        body = text[:-1] if (allow_terminal and last in allow_terminal) else text
        for ch in body:
            if ch in "\"'()[];:":
                raise SchemaError(f"banned internal punctuation {ch!r} in: {text!r}")
            if ch in "-–—":
                raise SchemaError(f"hyphenated word / dash in: {text!r}")
            if ch == "," and not allow_internal_comma:
                raise SchemaError(f"unexpected comma in: {text!r}")


def assert_word_count_window(items: Iterable[dict[str, Any]]) -> None:
    """Every item's word count in the global [4, 14] window."""
    for it in items:
        n = word_count(it["text"])
        if not (WORD_COUNT_MIN <= n <= WORD_COUNT_MAX):
            raise SchemaError(
                f"word count {n} out of [{WORD_COUNT_MIN}, {WORD_COUNT_MAX}] "
                f"in: {it['text']!r}"
            )


def assert_no_duplicate_surface(items: Iterable[dict[str, Any]]) -> None:
    """No duplicate surface string anywhere in a rule's dataset (global)."""
    seen: set[str] = set()
    for it in items:
        if it["text"] in seen:
            raise SchemaError(f"duplicate surface string: {it['text']!r}")
        seen.add(it["text"])


def assert_split_base_disjoint(items: Iterable[dict[str, Any]]) -> None:
    """A base and all its variants live in exactly one split (by-base)."""
    base_split: dict[Any, str] = {}
    for it in items:
        prev = base_split.setdefault(it["base_id"], it["split"])
        if prev != it["split"]:
            raise SchemaError(
                f"base_id {it['base_id']!r} spans splits {prev} and {it['split']}"
            )


def assert_balance(items: Sequence[dict[str, Any]]) -> None:
    """Exact 50/50 True/False in every balanced split (few_shot_pool, held_out,
    confirmation). Mirrors contexts.validate_dataset's check."""
    for split in BALANCED_SPLITS:
        group = [it for it in items if it["split"] == split]
        if not group:
            continue
        n_true = sum(normalize_label(it["label"]) for it in group)
        n_false = len(group) - n_true
        if n_true != n_false:
            raise SchemaError(
                f"split {split!r} imbalanced: {n_true} True vs {n_false} False"
            )


def validate_full(
    items: Sequence[dict[str, Any]],
    *,
    rule_id: str | None = None,
    allow_terminal: str = "",
    allow_internal_comma: bool = False,
    check_word_count: bool = True,
) -> None:
    """Run every schema-level validator, then the loader's own validate_dataset.

    The sentence-style policy is selected the same two ways as
    ``assert_sentence_style``: pass ``rule_id`` to SELF-SELECT the policy from
    RULE_STYLE_POLICY (the path generators should use), or pass the explicit
    ``allow_terminal`` / ``allow_internal_comma`` flags (for tests). The two
    paths are mutually exclusive.

    ``check_word_count`` is on by default; pass False only if a rule's recipe
    legitimately needs counts the global window would forbid (none currently
    do — the window IS the global cap)."""
    assert_no_duplicate_surface(items)
    assert_split_base_disjoint(items)
    assert_balance(items)
    assert_sentence_style(
        items,
        rule_id=rule_id,
        allow_terminal=allow_terminal,
        allow_internal_comma=allow_internal_comma,
    )
    if check_word_count:
        assert_word_count_window(items)
    # delegate to the runner-facing loader contract (the integration guarantee)
    validate_dataset(list(items))


def class_balance(items: Iterable[dict[str, Any]]) -> Counter:
    """Counter of label -> count (helper for reports/tests)."""
    return Counter(normalize_label(it["label"]) for it in items)
