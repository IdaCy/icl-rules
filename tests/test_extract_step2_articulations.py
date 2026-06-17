"""Tests for Step-2 -> Step-3 articulation extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import extract_step2_articulations as ext


def test_extract_selects_modal_direct_with_examples_candidate() -> None:
    rows = [
        {
            "kind": "generation",
            "rule_id": "food_topic_deconfounded",
            "variant": "direct",
            "has_examples": True,
            "context_index": 2,
            "phrasing": 1,
            "candidate": "True if the input is about cooking.",
        },
        {
            "kind": "generation",
            "rule_id": "food_topic_deconfounded",
            "variant": "direct",
            "has_examples": True,
            "context_index": 0,
            "phrasing": 0,
            "candidate": "True if the input is about food.",
        },
        {
            "kind": "generation",
            "rule_id": "food_topic_deconfounded",
            "variant": "direct",
            "has_examples": True,
            "context_index": 1,
            "phrasing": 1,
            "candidate": " true  if the input is about food. ",
        },
        {
            "kind": "generation",
            "rule_id": "food_topic_deconfounded",
            "variant": "think-then-state",
            "has_examples": True,
            "context_index": 0,
            "phrasing": 0,
            "candidate": "Ignored",
        },
        {
            "kind": "generation",
            "rule_id": "food_topic_deconfounded",
            "variant": "direct",
            "has_examples": False,
            "context_index": -1,
            "phrasing": 0,
            "candidate": "Ignored control",
        },
    ]

    payload = ext.extract(rows)

    assert payload["articulations"] == {
        "food_topic_deconfounded": "True if the input is about food."
    }
    assert payload["selection"]["food_topic_deconfounded"]["modal_count"] == 2


def test_main_writes_articulation_sidecar(tmp_path) -> None:
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        json.dumps(
            {
                "kind": "generation",
                "rule_id": "word_count_geq_8_deconfounded",
                "variant": "direct",
                "has_examples": True,
                "context_index": 0,
                "phrasing": 0,
                "candidate": "True if there are at least eight words.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "articulations.json"

    assert ext.main([str(responses), "--output", str(out)]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["articulations"]["word_count_geq_8_deconfounded"].startswith("True if")
