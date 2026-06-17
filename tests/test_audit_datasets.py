from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_datasets as audit

WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mango",
    "novel",
    "opera",
    "piano",
    "quiet",
    "river",
    "solar",
    "tango",
]


def _write_dataset(root: Path, rule_id: str, rows: list[dict]) -> None:
    out = root / rule_id
    out.mkdir(parents=True)
    (out / "items.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(i: str, label: bool, text: str, split: str) -> dict:
    return {
        "item_id": i,
        "base_id": i,
        "rule_id": "contains_digit",
        "label": label,
        "text": text,
        "slots_meta": {},
        "split": split,
    }


def test_manual_bow_nb_learns_simple_token_shortcut(tmp_path: Path) -> None:
    rows = []
    for split, n in (("few_shot_pool", 20), ("held_out", 6), ("confirmation", 4)):
        for j in range(n):
            tag = WORDS[j]
            rows.append(_row(f"{split}-t{j}", True, f"alpha code {split} {tag}", split))
            rows.append(_row(f"{split}-f{j}", False, f"beta code {split} {tag}", split))

    result = audit.multinomial_nb(rows, lambda text: audit.token_set(text))

    assert result["train_acc"] == 1.0
    assert result["eval_acc"] == 1.0


def test_audit_rule_reports_groundtruth_mismatch(tmp_path: Path) -> None:
    rows = []
    for split, n in (("few_shot_pool", 20), ("held_out", 6), ("confirmation", 4)):
        for j in range(n):
            tag = WORDS[j]
            rows.append(_row(f"{split}-t{j}", True, f"{split} {tag} has 7", split))
            rows.append(_row(f"{split}-f{j}", False, f"{split} {tag} has none", split))
    rows[0]["label"] = False
    rows[1]["label"] = True
    _write_dataset(tmp_path, "contains_digit", rows)

    report = audit.audit_rule(tmp_path, "contains_digit")

    assert report["groundtruth"]["recomputable"] is True
    assert report["groundtruth"]["n_mismatch"] == 2
    assert report["shortcut_baselines"]["word_bow_nb"]["eval_acc"] == 1.0


def test_deconfounded_alias_uses_base_groundtruth(tmp_path: Path) -> None:
    rows = []
    for split, n in (("few_shot_pool", 20), ("held_out", 6), ("confirmation", 4)):
        for j in range(n):
            tag = WORDS[j]
            rows.append(_row(f"{split}-t{j}", True, f"{split} {tag} has 7", split) | {"rule_id": "contains_digit_deconfounded"})
            rows.append(_row(f"{split}-f{j}", False, f"{split} {tag} has none", split) | {"rule_id": "contains_digit_deconfounded"})
    _write_dataset(tmp_path, "contains_digit_deconfounded", rows)

    report = audit.audit_rule(tmp_path, "contains_digit_deconfounded")

    assert report["canonical_rule_id"] == "contains_digit"
    assert report["groundtruth"]["n_mismatch"] == 0
