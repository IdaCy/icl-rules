"""Dataset loading/validation + few-shot context sampling tests."""

from __future__ import annotations

import json
import random

import pytest

from icl_articulation.contexts import (
    DatasetError,
    load_items,
    normalize_label,
    sample_context,
    select_queries,
    split_items,
)
from tests.conftest import make_rule_items, write_rule_dataset


def _write(tmp_path, items):
    rule_dir = tmp_path / "data" / "toy_rule"
    rule_dir.mkdir(parents=True, exist_ok=True)
    path = rule_dir / "items.jsonl"
    path.write_text("\n".join(json.dumps(it) for it in items) + "\n", encoding="utf-8")
    return path


# --- label normalization -------------------------------------------------------


def test_normalize_label_bools_and_strings() -> None:
    assert normalize_label(True) is True
    assert normalize_label(False) is False
    assert normalize_label("True") is True
    assert normalize_label("False") is False
    assert normalize_label("true") is True
    assert normalize_label(" FALSE ") is False


@pytest.mark.parametrize("bad", ["yes", "T", "", 1, 0, None, 1.0])
def test_normalize_label_rejects_garbage(bad) -> None:
    with pytest.raises(DatasetError):
        normalize_label(bad)


# --- loading + validation -------------------------------------------------------


def test_load_items_normalizes_string_labels(tmp_path) -> None:
    items = make_rule_items()
    for it in items[::2]:  # mix string and bool labels in one file
        it["label"] = "True" if it["label"] else "False"
    loaded = load_items(_write(tmp_path, items))
    assert len(loaded) == len(items)
    assert all(isinstance(it["label"], bool) for it in loaded)


def test_load_items_missing_file_raises(tmp_path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_items(tmp_path / "nope" / "items.jsonl")


def test_load_items_missing_field_raises(tmp_path) -> None:
    items = make_rule_items()
    del items[5]["slots_meta"]
    with pytest.raises(DatasetError, match="missing fields"):
        load_items(_write(tmp_path, items))


def test_load_items_unknown_split_raises(tmp_path) -> None:
    items = make_rule_items()
    items[0]["split"] = "train"
    with pytest.raises(DatasetError, match="unknown split"):
        load_items(_write(tmp_path, items))


def test_load_items_duplicate_item_id_raises(tmp_path) -> None:
    items = make_rule_items()
    items[1]["item_id"] = items[0]["item_id"]
    with pytest.raises(DatasetError, match="duplicate item_id"):
        load_items(_write(tmp_path, items))


def test_load_items_duplicate_text_raises(tmp_path) -> None:
    items = make_rule_items()
    items[3]["text"] = items[2]["text"]
    with pytest.raises(DatasetError, match="duplicate text"):
        load_items(_write(tmp_path, items))


def test_load_items_imbalanced_split_raises(tmp_path) -> None:
    items = make_rule_items()
    held = [it for it in items if it["split"] == "held_out"]
    held[1]["label"] = True  # was False -> 61/59
    with pytest.raises(DatasetError, match="imbalanced"):
        load_items(_write(tmp_path, items))


def test_load_items_base_in_two_splits_raises(tmp_path) -> None:
    items = make_rule_items()
    pool = [it for it in items if it["split"] == "few_shot_pool"]
    pool[0]["split"] = "spare"  # its sibling variant stays in few_shot_pool
    with pytest.raises(DatasetError, match="two splits"):
        load_items(_write(tmp_path, items))


def test_load_items_mixed_rule_ids_raises(tmp_path) -> None:
    items = make_rule_items()
    items[0]["rule_id"] = "other_rule"
    with pytest.raises(DatasetError, match="mixes rule_ids"):
        load_items(_write(tmp_path, items))


def test_split_items_filters_and_rejects_bad_split(tmp_path) -> None:
    items = load_items(write_rule_dataset(tmp_path / "data"))
    held = split_items(items, "held_out")
    assert len(held) == 120
    assert all(it["split"] == "held_out" for it in held)
    with pytest.raises(DatasetError):
        split_items(items, "test")


# --- context sampling -------------------------------------------------------------


def test_sample_context_balance_and_distinct_bases(tmp_path) -> None:
    items = load_items(write_rule_dataset(tmp_path / "data"))
    context = sample_context(items, k=32, seed=0)
    assert len(context) == 32
    assert sum(it["label"] for it in context) == 16  # exactly 16 True / 16 False
    assert len({it["base_id"] for it in context}) == 32  # no base twice
    assert all(it["split"] == "few_shot_pool" for it in context)
    assert len({it["text"] for it in context}) == 32


def test_sample_context_seed_determinism(tmp_path) -> None:
    items = load_items(write_rule_dataset(tmp_path / "data"))
    ids = lambda ctx: [it["item_id"] for it in ctx]  # noqa: E731
    assert ids(sample_context(items, seed=7)) == ids(sample_context(items, seed=7))
    assert ids(sample_context(items, seed=0)) != ids(sample_context(items, seed=1))


def test_sample_context_independent_of_file_order(tmp_path) -> None:
    items = load_items(write_rule_dataset(tmp_path / "data"))
    shuffled = list(items)
    random.Random(99).shuffle(shuffled)
    ids = lambda ctx: [it["item_id"] for it in ctx]  # noqa: E731
    assert ids(sample_context(items, seed=3)) == ids(sample_context(shuffled, seed=3))


def test_sample_context_insufficient_class_raises() -> None:
    # only 10 True-capable bases in the pool -> cannot fill 16 True
    items = []
    for i in range(40):
        label = i < 10
        items.append(
            {
                "item_id": f"i{i}",
                "base_id": f"b{i}",
                "rule_id": "r",
                "label": label,
                "text": f"text {i}",
                "slots_meta": {},
                "split": "few_shot_pool",
            }
        )
    with pytest.raises(DatasetError, match="balanced"):
        sample_context(items, k=32, seed=0)


def test_sample_context_mixed_dual_and_single_label_pool_fills() -> None:
    # review M1 regression (probe CASE1): 16 dual-label bases + 16 True-only
    # bases. A balanced 16/16 context exists (all duals -> False, all singles
    # -> True); the old rng.choice filler spent dual bases on True and starved
    # False on every seed. The need-aware sampler must succeed for any seed.
    items = []

    def add(item_id: str, base_id: str, label: bool, text: str) -> None:
        items.append(
            {
                "item_id": item_id,
                "base_id": base_id,
                "rule_id": "r",
                "label": label,
                "text": text,
                "slots_meta": {},
                "split": "few_shot_pool",
            }
        )

    for i in range(16):
        add(f"d{i}-T", f"dual{i:02d}", True, f"dual sentence {i} alpha")
        add(f"d{i}-F", f"dual{i:02d}", False, f"dual sentence {i} beta")
    for i in range(16):
        add(f"s{i}-T", f"single{i:02d}", True, f"single sentence {i} alpha")

    for seed in range(100):
        context = sample_context(items, k=32, seed=seed)
        assert len(context) == 32
        assert sum(it["label"] for it in context) == 16
        assert len({it["base_id"] for it in context}) == 32


def test_sample_context_odd_k_raises(tmp_path) -> None:
    items = load_items(write_rule_dataset(tmp_path / "data"))
    with pytest.raises(ValueError):
        sample_context(items, k=31, seed=0)


def test_sample_context_empty_pool_raises() -> None:
    items = [
        {
            "item_id": "i0",
            "base_id": "b0",
            "rule_id": "r",
            "label": True,
            "text": "t",
            "slots_meta": {},
            "split": "held_out",
        }
    ]
    with pytest.raises(DatasetError, match="few_shot_pool"):
        sample_context(items, k=32, seed=0)


# --- query selection -----------------------------------------------------------------


def test_select_queries_full_split(tmp_path) -> None:
    items = load_items(write_rule_dataset(tmp_path / "data"))
    queries = select_queries(items, "held_out", None)
    assert len(queries) == 120
    assert sum(it["label"] for it in queries) == 60


def test_select_queries_balanced_subset_deterministic(tmp_path) -> None:
    items = load_items(write_rule_dataset(tmp_path / "data"))
    sub = select_queries(items, "held_out", 40)
    assert len(sub) == 40
    assert sum(it["label"] for it in sub) == 20
    # deterministic, no seed: stable across calls and across file order
    shuffled = list(items)
    random.Random(5).shuffle(shuffled)
    sub2 = select_queries(shuffled, "held_out", 40)
    assert [it["item_id"] for it in sub] == [it["item_id"] for it in sub2]


def test_select_queries_bad_n_raises(tmp_path) -> None:
    items = load_items(write_rule_dataset(tmp_path / "data"))
    with pytest.raises(DatasetError, match="even"):
        select_queries(items, "held_out", 41)
    with pytest.raises(DatasetError, match="too small"):
        select_queries(items, "held_out", 400)
