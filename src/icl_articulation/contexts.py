"""Dataset loading + few-shot context sampling for step 1.

Datasets live at data/<rule_id>/items.jsonl, one JSON object per line with
fields [item_id, base_id, rule_id, label, text, slots_meta, split] (the
locked item_schema in rule-specs.yaml globals). Labels may arrive as bools or
as "True"/"False" strings; they are normalized to bool here. Splits are
few_shot_pool / held_out / confirmation / spare.

Context construction (PLAN, locked): sample k=32 DISTINCT base_ids from the
few_shot_pool split, one variant per base, exactly 16 True / 16 False, order
shuffled with the logged seed. All violations raise loudly (DatasetError) —
a quiet fallback here would corrupt a paid run.
"""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Any

SPLITS = ("few_shot_pool", "held_out", "confirmation", "spare")
REQUIRED_FIELDS = ("item_id", "base_id", "rule_id", "label", "text", "slots_meta", "split")
# splits whose class balance must be exactly 50/50 (spare is unconstrained)
BALANCED_SPLITS = ("few_shot_pool", "held_out", "confirmation")
# splits that hold exactly ONE variant per base
ONE_VARIANT_SPLITS = ("held_out", "confirmation")


class DatasetError(ValueError):
    """A dataset violated the locked schema/balance/disjointness invariants."""


def normalize_label(value: Any) -> bool:
    """true/false bool or 'True'/'False' string (any case) -> bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
    raise DatasetError(f"unparseable label {value!r} (expected bool or 'True'/'False')")


def load_items(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate one rule's items.jsonl; labels normalized to bool."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset file not found: {path}")
    items: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    seen_texts: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        missing = [f for f in REQUIRED_FIELDS if f not in obj]
        if missing:
            raise DatasetError(f"{path}:{lineno}: missing fields {missing}")
        if obj["split"] not in SPLITS:
            raise DatasetError(
                f"{path}:{lineno}: unknown split {obj['split']!r} (expected one of {SPLITS})"
            )
        obj["label"] = normalize_label(obj["label"])
        if obj["item_id"] in seen_ids:
            raise DatasetError(f"{path}:{lineno}: duplicate item_id {obj['item_id']!r}")
        seen_ids.add(obj["item_id"])
        if obj["text"] in seen_texts:
            # rule-specs globals: no duplicate surface string anywhere in a rule's dataset
            raise DatasetError(f"{path}:{lineno}: duplicate text {obj['text']!r}")
        seen_texts.add(obj["text"])
        items.append(obj)
    if not items:
        raise DatasetError(f"{path}: empty dataset")
    validate_dataset(items)
    return items


def validate_dataset(items: list[dict[str, Any]]) -> None:
    """Balance, base-level split disjointness, one-variant-per-base in eval splits."""
    rule_ids = {it["rule_id"] for it in items}
    if len(rule_ids) != 1:
        raise DatasetError(f"dataset mixes rule_ids: {sorted(map(str, rule_ids))}")
    # base-level split disjointness: a base and ALL its variants live in ONE split
    base_split: dict[Any, str] = {}
    for it in items:
        prev = base_split.setdefault(it["base_id"], it["split"])
        if prev != it["split"]:
            raise DatasetError(
                f"base_id {it['base_id']!r} appears in two splits: {prev} and {it['split']}"
            )
    for split in BALANCED_SPLITS:
        group = [it for it in items if it["split"] == split]
        if not group:
            continue  # presence requirements are mode-specific (runner checks)
        n_true = sum(it["label"] for it in group)
        n_false = len(group) - n_true
        if n_true != n_false:
            raise DatasetError(f"split {split!r} imbalanced: {n_true} True vs {n_false} False")
    for split in ONE_VARIANT_SPLITS:
        bases = [it["base_id"] for it in items if it["split"] == split]
        if len(bases) != len(set(bases)):
            raise DatasetError(f"split {split!r} has more than one variant for some base")


def split_items(items: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise DatasetError(f"unknown split {split!r} (expected one of {SPLITS})")
    return [it for it in items if it["split"] == split]


def sample_context(items: list[dict[str, Any]], k: int = 32, seed: int = 0) -> list[dict[str, Any]]:
    """One few-shot context: k items, k DISTINCT bases, k/2 True / k/2 False.

    Bases are drawn from the few_shot_pool split (one variant per chosen base);
    the final order is shuffled with ``seed``. Deterministic for a given
    (dataset, k, seed) regardless of file line order (bases sorted before the
    seeded shuffle).
    """
    if k % 2 != 0 or k <= 0:
        raise ValueError(f"k must be positive and even, got {k}")
    pool = split_items(items, "few_shot_pool")
    if not pool:
        raise DatasetError("few_shot_pool split is empty")
    by_base: dict[Any, list[dict[str, Any]]] = {}
    for it in pool:
        by_base.setdefault(it["base_id"], []).append(it)
    for variants in by_base.values():  # determinism regardless of file line order
        variants.sort(key=lambda it: str(it["item_id"]))

    rng = Random(seed)
    base_ids = sorted(by_base, key=str)
    rng.shuffle(base_ids)
    need = {True: k // 2, False: k // 2}
    # remaining supply per label: how many not-yet-visited bases offer it
    base_labels = {base: {it["label"] for it in by_base[base]} for base in base_ids}
    supply = {
        True: sum(1 for labels in base_labels.values() if True in labels),
        False: sum(1 for labels in base_labels.values() if False in labels),
    }
    chosen: list[dict[str, Any]] = []
    for base in base_ids:
        if not need[True] and not need[False]:
            break
        labels = sorted(lab for lab in base_labels[base] if need[lab] > 0)
        if labels:
            if len(labels) == 1:
                label = labels[0]
            else:
                # need-aware choice (review M1): a dual-label base goes to the
                # label that is scarcer among the remaining bases (higher
                # remaining_need / remaining_supply), so dual bases don't get
                # spent on a label that single-label bases could still cover;
                # rng.choice keeps ties seed-deterministic.
                ratios = [need[lab] / supply[lab] for lab in labels]
                if ratios[0] == ratios[1]:
                    label = rng.choice(labels)
                else:
                    label = labels[ratios.index(max(ratios))]
            variants = [it for it in by_base[base] if it["label"] == label]
            chosen.append(rng.choice(variants))
            need[label] -= 1
        for lab in base_labels[base]:
            supply[lab] -= 1
    if need[True] or need[False]:
        raise DatasetError(
            f"could not fill a balanced {k}-shot context from few_shot_pool: "
            f"still need {need[True]} True / {need[False]} False (this may be "
            f"a limitation of the greedy per-base label assignment in the "
            f"sampler rather than a defect in the dataset)"
        )
    rng.shuffle(chosen)

    # paranoia post-checks — these guard the locked invariants on every paid run
    bases = [it["base_id"] for it in chosen]
    if len(set(bases)) != k:
        raise DatasetError("context sampling produced a repeated base_id")
    if sum(it["label"] for it in chosen) != k // 2:
        raise DatasetError("context sampling produced an imbalanced context")
    if any(it["split"] != "few_shot_pool" for it in chosen):
        raise DatasetError("context sampling leaked a non-few_shot_pool item")
    return chosen


def select_queries(
    items: list[dict[str, Any]], split: str, n: int | None = None
) -> list[dict[str, Any]]:
    """The query items for one rule: the full split, or a balanced subset.

    Subsetting (pilot mode) is DETERMINISTIC with no seed: items sorted by
    item_id, first n/2 of each class — the same subset for every model/run,
    which keeps pilot results comparable across models.
    """
    group = sorted(split_items(items, split), key=lambda it: str(it["item_id"]))
    if not group:
        raise DatasetError(f"split {split!r} is empty")
    if n is None:
        return group
    if n % 2 != 0 or n <= 0:
        raise DatasetError(f"query subset size must be positive and even, got {n}")
    trues = [it for it in group if it["label"]][: n // 2]
    falses = [it for it in group if not it["label"]][: n // 2]
    if len(trues) < n // 2 or len(falses) < n // 2:
        raise DatasetError(
            f"split {split!r} too small for a balanced subset of {n} "
            f"(have {len(trues)} True / {len(falses)} False usable)"
        )
    return trues + falses
