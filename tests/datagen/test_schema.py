"""schema.py tests: the locked item record, jsonl IO, by-base splits, validators,
and the CRITICAL round-trip through the existing contexts.py loader."""

from __future__ import annotations

import pytest

from icl_articulation.contexts import load_items
from icl_articulation.datagen import schema
from icl_articulation.datagen.schema import (
    ITEM_SCHEMA,
    SchemaError,
    assign_llm_splits,
    assign_programmatic_splits,
    make_item,
    read_items,
    validate_full,
    word_count,
    words,
    write_items,
)


# --- schema constants match the loader contract ------------------------------


def test_item_schema_matches_loader_required_fields() -> None:
    from icl_articulation.contexts import REQUIRED_FIELDS

    assert set(ITEM_SCHEMA) == set(REQUIRED_FIELDS)


# --- tokenizer ---------------------------------------------------------------


def test_words_strips_edge_punct_only() -> None:
    assert words("Later, the dog ran!") == ["Later", "the", "dog", "ran"]
    assert words("don't stop") == ["don't", "stop"]  # internal apostrophe kept
    assert word_count("The cat sat on the mat") == 6
    assert word_count("They counted 10 boxes") == 4  # digit token counts


# --- make_item ---------------------------------------------------------------


def test_make_item_orders_and_normalizes() -> None:
    it = make_item(
        item_id="x-T", base_id="x", rule_id="r", label="True",
        text="The dog ran home", slots_meta={"k": 1}, split="held_out",
    )
    assert list(it) == list(ITEM_SCHEMA)
    assert it["label"] is True


def test_make_item_rejects_bad_split() -> None:
    with pytest.raises(SchemaError, match="unknown split"):
        make_item(item_id="x", base_id="x", rule_id="r", label=True,
                  text="a b c d", slots_meta={}, split="train")


# --- by-base programmatic split ----------------------------------------------


def test_assign_programmatic_splits_counts() -> None:
    bases = [f"b{i}" for i in range(340)]
    assign = assign_programmatic_splits(bases, seed=0)
    from collections import Counter

    counts = Counter(assign.values())
    assert counts["few_shot_pool"] == 100
    assert counts["held_out"] == 120
    assert counts["confirmation"] == 100
    assert counts["spare"] == 20


def test_assign_programmatic_splits_deterministic() -> None:
    bases = [f"b{i}" for i in range(360)]
    a = assign_programmatic_splits(bases, seed=5)
    b = assign_programmatic_splits(list(reversed(bases)), seed=5)
    assert a == b  # file-order independent, seed-deterministic


def test_assign_programmatic_splits_too_few_raises() -> None:
    with pytest.raises(SchemaError, match="too few bases"):
        assign_programmatic_splits([f"b{i}" for i in range(100)], seed=0)


def test_assign_llm_splits_balanced() -> None:
    ids = [f"i{i}" for i in range(400)]
    labels = [i % 2 == 0 for i in range(400)]  # 200 T / 200 F
    assign = assign_llm_splits(ids, labels, seed=1)
    from collections import Counter

    counts = Counter(assign.values())
    assert counts["few_shot_pool"] == 120
    assert counts["held_out"] == 120
    assert counts["confirmation"] == 100
    # each balanced split is exactly 50/50
    for split, n in (("few_shot_pool", 120), ("held_out", 120), ("confirmation", 100)):
        members = [i for i, s in assign.items() if s == split]
        n_true = sum(1 for m in members if labels[ids.index(m)])
        assert n_true == n // 2


# --- validators ---------------------------------------------------------------


def _toy_programmatic_items():
    """A tiny but fully schema-valid programmatic dataset (40 bases).

    few_shot_pool keeps both variants; held_out/confirmation keep one variant
    per base with EXACT 50/50 balance assigned by within-split index (so the
    label does not depend on which bases the seeded assigner happens to place
    in each split)."""
    bases = [f"b{i:03d}" for i in range(40)]
    assign = assign_programmatic_splits(
        bases, seed=0, few_shot_pool=10, held_out=12, confirmation=10, spare_min=2
    )
    items = []
    eval_seen: dict[str, int] = {}  # within-split running index for balancing
    for base in sorted(bases):
        split = assign[base]
        if split == "few_shot_pool":
            items.append(make_item(item_id=f"{base}-T", base_id=base, rule_id="toy",
                                   label=True, text=f"sentence {base} true alpha words here",
                                   slots_meta={}, split=split))
            items.append(make_item(item_id=f"{base}-F", base_id=base, rule_id="toy",
                                   label=False, text=f"sentence {base} false beta words here",
                                   slots_meta={}, split=split))
        else:
            idx = eval_seen.get(split, 0)
            eval_seen[split] = idx + 1
            label = (idx % 2 == 0)  # exact 50/50 within each split (even sizes)
            word = "alpha" if label else "beta"
            items.append(make_item(item_id=f"{base}-q", base_id=base, rule_id="toy",
                                   label=label, text=f"sentence {base} {word} words here now",
                                   slots_meta={}, split=split))
    return items


def test_validate_full_accepts_clean_dataset() -> None:
    validate_full(_toy_programmatic_items())  # must not raise


def test_validate_full_word_count_window() -> None:
    items = _toy_programmatic_items()
    items[0]["text"] = "tiny"  # 1 word, below the [4,14] window
    with pytest.raises(SchemaError, match="word count"):
        validate_full(items)


def test_validate_full_duplicate_surface() -> None:
    items = _toy_programmatic_items()
    items[1]["text"] = items[0]["text"]
    with pytest.raises(SchemaError, match="duplicate surface"):
        validate_full(items)


def test_assert_sentence_style_terminal_punct() -> None:
    items = [make_item(item_id="a", base_id="a", rule_id="r", label=True,
                       text="The dog ran home.", slots_meta={}, split="held_out")]
    with pytest.raises(SchemaError, match="terminal punctuation"):
        schema.assert_sentence_style(items)
    # rule 3 allows a trailing '!'
    items3 = [make_item(item_id="a", base_id="a", rule_id="r", label=True,
                        text="The dog ran home!", slots_meta={}, split="held_out")]
    schema.assert_sentence_style(items3, allow_terminal="!")  # must not raise


def test_assert_sentence_style_bans_I_and_comma() -> None:
    items = [make_item(item_id="a", base_id="a", rule_id="r", label=True,
                       text="Today I ran home", slots_meta={}, split="held_out")]
    with pytest.raises(SchemaError, match="pronoun 'I'"):
        schema.assert_sentence_style(items)
    comma = [make_item(item_id="b", base_id="b", rule_id="r", label=True,
                       text="Later, the dog ran", slots_meta={}, split="held_out")]
    with pytest.raises(SchemaError, match="comma"):
        schema.assert_sentence_style(comma)
    schema.assert_sentence_style(comma, allow_internal_comma=True)  # rule 26 etc.


# --- THE round-trip: schema-written data loads through contexts.py ------------


def test_round_trip_through_contexts_loader(tmp_path) -> None:
    """Data emitted by schema.write_items must load + validate through the
    EXISTING contexts.load_items unchanged (the runner-facing contract)."""
    items = _toy_programmatic_items()
    path = tmp_path / "data" / "toy" / "items.jsonl"
    write_items(items, path)

    loaded = load_items(path)  # the real loader, full validation
    assert len(loaded) == len(items)
    assert all(isinstance(it["label"], bool) for it in loaded)
    # the loaded items carry exactly the locked schema fields
    for it in loaded:
        assert set(it) == set(ITEM_SCHEMA)


def test_round_trip_raw_read_matches_written(tmp_path) -> None:
    items = _toy_programmatic_items()
    path = tmp_path / "items.jsonl"
    write_items(items, path)
    raw = read_items(path)
    assert len(raw) == len(items)
    assert all(list(r) == list(ITEM_SCHEMA) for r in raw)
