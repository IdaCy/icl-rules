"""in-session in-session per-item runner tests. No network (FakeAPI + FakeAnthropic).
The headline guard is test_grows_one_item_per_call: the conversation must grow ONE
item per call (N classify calls then 1 articulate) — the regression test that the
CoT same-session batching is gone."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_insession_articulation as ins
from icl_articulation.prompts import insession_template_hash
from tests.conftest import FakeAPI, fake_response_data

RULE = "contains_digit"


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines()]


def _run_dir(tmp_path) -> Path:
    dirs = list((tmp_path / "results").iterdir())
    assert len(dirs) == 1
    return dirs[0]


def _argv(tmp_path, model, cot_flag, *extra):
    return [cot_flag, "--model", model, "--rules", RULE, "--n-items", "4", "--n-contexts", "1",
            "--results-dir", str(tmp_path / "results"), "--cache-dir", str(tmp_path / "cache"), *extra]


def _is_articulate(user: str) -> bool:
    return "single rule" in user or "What rule did you use" in user


class CannedAPI(FakeAPI):
    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        last = kwargs["messages"][-1]["content"]
        if "CANDIDATE ARTICULATION TO GRADE" in last:
            cand = last.split("CANDIDATE ARTICULATION TO GRADE:", 1)[1]
            grade = 2 if "digit" in cand else 0
            text = json.dumps({"grade": grade, "extensionally_equivalent": grade == 2, "rationale": "x"})
        elif _is_articulate(last):
            text = "reasoning...\nRULE: the input contains a digit"
        elif "Reason" in last or "Think step by step" in last:  # CoT classify
            text = "let me check\nAnswer: True"
        else:  # no-CoT classify
            text = "True"
        return fake_response_data(text=text, model=kwargs.get("model"))


# --- fake anthropic ------------------------------------------------------------
class _Block:
    def __init__(self, type: str, text: str = "") -> None:
        self.type, self.text = type, text


class _Usage:
    input_tokens = 100
    output_tokens = 20


class _Resp:
    def __init__(self, content):
        self.content, self.usage, self.stop_reason = content, _Usage(), "end_turn"


class _Messages:
    def __init__(self, owner):
        self._owner = owner

    async def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        last = kwargs["messages"][-1]["content"]
        if isinstance(last, str) and _is_articulate(last):
            return _Resp([_Block("thinking", ""), _Block("text", "RULE: the input contains a digit")])
        return _Resp([_Block("thinking", ""), _Block("text", "Answer: True")])


class FakeAnthropic:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.messages = _Messages(self)

    async def close(self):
        pass


# --- gpt-4.1 path --------------------------------------------------------------
def test_grows_one_item_per_call(tmp_path) -> None:
    """REGRESSION GUARD: 4 classify calls (growing by 2 messages) then 1 articulate."""
    api = CannedAPI()
    assert ins.main(_argv(tmp_path, "gpt-4.1", "--no-cot"), api=api) == 0
    subject_calls = [c for c in api.calls if "CANDIDATE ARTICULATION TO GRADE" not in c["messages"][-1]["content"]]
    classify = [c for c in subject_calls if not _is_articulate(c["messages"][-1]["content"])]
    articulate = [c for c in subject_calls if _is_articulate(c["messages"][-1]["content"])]
    assert len(classify) == 4 and len(articulate) == 1   # NOT one batched call
    # the conversation grows by 2 messages each classify turn: 2,4,6,8
    assert [len(c["messages"]) for c in classify] == [2, 4, 6, 8]
    assert len(articulate[0]["messages"]) == 10          # +assistant(ans4)+user(articulate)


def test_turn1_is_exact_step1_prompt(tmp_path) -> None:
    api = CannedAPI()
    assert ins.main(_argv(tmp_path, "gpt-4.1", "--no-cot"), api=api) == 0
    first = [c for c in api.calls if len(c["messages"]) == 2][0]
    user = first["messages"][1]["content"]
    assert "Here are labeled examples" in user
    assert "Classify the next input. Answer with exactly True or False." in user
    assert user.rstrip().endswith("Label:")
    assert "Label: True" in user and "Label: False" in user


def test_rows_classify_articulate_grade(tmp_path) -> None:
    api = CannedAPI()
    assert ins.main(_argv(tmp_path, "gpt-4.1", "--no-cot"), api=api) == 0
    rows = _read(_run_dir(tmp_path) / "responses.jsonl")
    clf = [r for r in rows if r["kind"] == "classify"]
    assert len(clf) == 4 and all(r["predicted"] is True for r in clf)
    assert [r["turn_index"] for r in clf] == [0, 1, 2, 3]
    art = [r for r in rows if r["kind"] == "articulate"]
    assert len(art) == 1 and art[0]["candidate"] == "the input contains a digit"
    grade = [r for r in rows if r["kind"] == "grade"]
    assert len(grade) == 1 and grade[0]["grade"] == 2


def test_cot_path_parses_answer_line(tmp_path) -> None:
    api = CannedAPI()
    assert ins.main(_argv(tmp_path, "gpt-4.1", "--cot"), api=api) == 0
    rows = _read(_run_dir(tmp_path) / "responses.jsonl")
    clf = [r for r in rows if r["kind"] == "classify"]
    assert len(clf) == 4 and all(r["predicted"] is True for r in clf)  # from 'Answer: True'
    cfg = json.loads((_run_dir(tmp_path) / "config.json").read_text())
    assert cfg["experiment"] == "exp2-cot" and cfg["cot"] is True


def test_cost_gate_aborts_before_any_call(tmp_path) -> None:
    api = CannedAPI()
    assert ins.main(_argv(tmp_path, "gpt-4.1", "--no-cot", "--max-cost", "1e-9"), api=api) == 1
    assert api.calls == []
    assert not (tmp_path / "results").exists()


def test_config_logs_experiment_cot_hashes(tmp_path) -> None:
    api = CannedAPI()
    assert ins.main(_argv(tmp_path, "gpt-4.1", "--no-cot"), api=api) == 0
    cfg = json.loads((_run_dir(tmp_path) / "config.json").read_text())
    assert cfg["experiment"] == "exp1-no-cot" and cfg["cot"] is False and cfg["cot_mode"] == "none"
    assert cfg["insession_template_hash"] == insession_template_hash()
    assert cfg["context_seeds"] == [0]
    assert "finished_utc" in cfg and "cost_actual_usd" in cfg


def test_odd_n_items_rejected(tmp_path) -> None:
    assert ins.main(_argv(tmp_path, "gpt-4.1", "--no-cot", "--n-items", "3"), api=CannedAPI()) == 2


def test_cot_and_no_cot_are_required(tmp_path) -> None:
    # neither flag -> argparse error (SystemExit)
    import pytest
    with pytest.raises(SystemExit):
        ins.main(["--model", "gpt-4.1", "--rules", RULE])


# --- claude path ---------------------------------------------------------------
def test_claude_no_cot_thinking_off(tmp_path) -> None:
    oai, ant = CannedAPI(), FakeAnthropic()
    assert ins.main(_argv(tmp_path, "claude-opus-4-8", "--no-cot"), api=oai, anthropic_api=ant) == 0
    # 4 classify + 1 articulate claude calls
    assert len(ant.calls) == 5
    for c in ant.calls:
        assert "thinking" not in c            # Exp1 = thinking OFF for every turn
    assert [c["max_tokens"] for c in ant.calls[:4]] == [16, 16, 16, 16]   # classify turns
    assert ant.calls[4]["max_tokens"] == ins.CLAUDE_ARTICULATE_MAX        # articulation turn
    cfg = json.loads((_run_dir(tmp_path) / "config.json").read_text())
    assert cfg["experiment"] == "exp1-no-cot"


def test_claude_cot_echoes_blocks_and_sets_thinking(tmp_path) -> None:
    oai, ant = CannedAPI(), FakeAnthropic()
    assert ins.main(_argv(tmp_path, "claude-opus-4-8", "--cot"), api=oai, anthropic_api=ant) == 0
    assert len(ant.calls) == 5
    for c in ant.calls:
        assert c["thinking"] == {"type": "adaptive"} and c["output_config"] == {"effort": "low"}
    # the 2nd classify call echoes the 1st reply's CONTENT BLOCKS as a list (unchanged)
    second = ant.calls[1]["messages"]
    assert second[1]["role"] == "assistant" and isinstance(second[1]["content"], list)
    assert second[1]["content"][1].text == "Answer: True"
