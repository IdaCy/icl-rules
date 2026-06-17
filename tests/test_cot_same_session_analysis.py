"""CoT same-session analysis tests: the CoT label parser, the sandbox consistency scorer
(real LOCAL worker on trusted synthetic code), and the no-API analyzer's
aggregation on a synthetic run dir. No network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_cot_same_session as cot
import analyze_cot_same_session as ana


# --- parser -------------------------------------------------------------------


def test_parse_clean_answer_block() -> None:
    text = "Let me reason.\nItem 1 short, item 2 long.\nAnswer 1: True\nAnswer 2: False\nAnswer 3: True"
    assert cot.parse_cot_labels(text, 3) == [True, False, True]


def test_parse_takes_last_occurrence_per_index() -> None:
    text = "draft Answer 1: False\nfinal:\nAnswer 1: True\nAnswer 2: False"
    assert cot.parse_cot_labels(text, 2) == [True, False]


def test_parse_missing_item_is_none() -> None:
    assert cot.parse_cot_labels("Answer 1: True\nAnswer 3: False", 3) == [True, None, False]


def test_parse_numbered_line_fallback() -> None:
    assert cot.parse_cot_labels("1) yes it is True\n2) nope False\n3) True", 3) == [True, False, True]


def test_parse_positional_fallback_only_when_exact() -> None:
    # exactly n tokens in the tail third -> positional map
    assert cot.parse_cot_labels("blah blah\nverdicts: True False True", 3) == [True, False, True]
    # wrong count -> all None (never guess)
    assert cot.parse_cot_labels("True", 3) == [None, None, None]


# --- consistency scorer (real local sandbox, trusted code) --------------------


def test_run_worker_scores_trusted_predicate() -> None:
    code = "def rule(text):\n    return len(text.split()) >= 8"
    items = [
        {"text": "a b c d", "label": False},
        {"text": "a b c d e f g h i", "label": True},
        {"text": "one two three", "label": False},
    ]
    res = cot.run_worker(code, items)
    assert res["ok"] is True and res["accuracy"] == 1.0 and res["n"] == 3


def test_extract_code_requires_def_rule() -> None:
    assert cot.extract_code("```python\ndef rule(text): return True\n```") == "def rule(text): return True"
    assert cot.extract_code("no code here") is None


# --- analyzer aggregation on a synthetic run dir ------------------------------


def _write_run(tmp_path: Path, model: str, cot_mode: str) -> Path:
    run_dir = tmp_path / f"cot-same-session-{model}-20260101T000000Z"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"model": model, "cot_mode": cot_mode}))
    rows = [
        # turn-1: 4 items, 3 correct (1 parse fail -> None)
        {"kind": "cot_turn1", "rule_id": "word_count_geq_8", "model": model, "context_index": 0,
         "context_seed": 0, "n_items": 4, "query_texts": ["a", "b", "c", "d"],
         "gold_labels": [True, False, True, False],
         "predictions": [True, False, True, None],
         "accuracy": {"n": 4, "n_parsed": 3, "n_correct": 3, "accuracy": 0.75, "n_parse_fail": 1},
         "finish_reason": "stop"},
        {"kind": "cot_turn2", "rule_id": "word_count_geq_8", "model": model, "context_index": 0,
         "context_seed": 0, "candidate": "the input has at least 8 words", "finish_reason": "stop"},
        {"kind": "grade", "rule_id": "word_count_geq_8", "model": model, "context_index": 0,
         "candidate": "the input has at least 8 words", "grade": 2,
         "extensionally_equivalent": True, "rationale": "ok"},
        {"kind": "compile", "rule_id": "word_count_geq_8", "model": model, "context_index": 0,
         "coder": "gpt-4.1", "candidate": "...", "code": "def rule(text): return True",
         "result_vs_gold": {"ok": True, "accuracy": 0.9, "n": 4},
         "result_vs_self": {"ok": True, "accuracy": 0.95, "n": 3}},
    ]
    (run_dir / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run_dir


def test_analyze_run_aggregates(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "gpt-4.1", "prompted_cot")
    out = ana.analyze_run(run_dir, baselines={})
    rule = out["rules"]["word_count_geq_8"]
    assert rule["cot_classification"]["rate"] == 0.75
    assert rule["cot_classification"]["n_parse_fail"] == 1
    assert rule["articulation"]["grade2_names_true_rate"] == 1.0
    assert rule["consistency"]["accuracy_vs_self"] == 0.95
    assert rule["consistency"]["accuracy_vs_gold"] == 0.9
    assert "RECOVER" in rule["verdict"].upper() or "grade-2" in rule["verdict"]


def test_verdict_flags_rationalisation() -> None:
    # grade-2 but vs_self low -> rationalisation flag
    cons = {"accuracy_vs_self": 0.4, "accuracy_vs_gold": 0.9}
    v = ana._verdict(cot_acc=0.9, grade2_rate=1.0, cons=cons)
    assert "RATIONALIS" in v.upper()


def test_verdict_flags_dissociation_survives() -> None:
    cons = {"accuracy_vs_self": None, "accuracy_vs_gold": None}
    v = ana._verdict(cot_acc=0.9, grade2_rate=0.0, cons=cons)
    assert "did NOT recover" in v
