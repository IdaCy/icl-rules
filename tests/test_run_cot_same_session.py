"""CoT same-session CoT same-session runner tests. No network — OpenAI surface faked
(CannedAPI) and the anthropic surface faked (FakeAnthropic). Reads REAL target
data + the committed spec extract. The consistency sandbox runs the REAL local
worker on the canned (trusted) predicate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_cot_same_session as cot
from icl_articulation.prompts import cot_same_session_template_hash
from tests.conftest import FakeAPI, fake_response_data

RULE = "contains_digit"  # recomputable, 'digit' keyword drives the canned grade+code


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines()]


def _run_dir(tmp_path) -> Path:
    dirs = list((tmp_path / "results").iterdir())
    assert len(dirs) == 1
    return dirs[0]


def _argv(tmp_path, model: str, *extra: str) -> list[str]:
    return ["--model", model, "--rules", RULE, "--n-items", "4", "--n-contexts", "1",
            "--results-dir", str(tmp_path / "results"), "--cache-dir", str(tmp_path / "cache"), *extra]


def _n_numbered(user: str) -> int:
    import re
    return len(re.findall(r"(?m)^\d+\. ", user))


class CannedAPI(FakeAPI):
    """Routes OpenAI calls by the active (last) user message:
    judge -> grade JSON; compile -> a digit predicate; turn-2 -> RULE: line;
    turn-1 -> a CoT blob ending in 'Answer K: True' for each numbered item."""

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        msgs = kwargs["messages"]
        last = msgs[-1]["content"]
        if "CANDIDATE ARTICULATION TO GRADE" in last:
            candidate = last.split("CANDIDATE ARTICULATION TO GRADE:", 1)[1]
            grade = 2 if "digit" in candidate else 0
            text = json.dumps({"grade": grade, "extensionally_equivalent": grade == 2, "rationale": "fake"})
        elif "Write a single Python function" in last:
            text = "```python\ndef rule(text):\n    return any(c.isdigit() for c in text)\n```"
        elif "What rule did you use" in last:
            text = "I considered the inputs.\nRULE: the input contains a digit"
        else:  # turn-1 classify
            n = _n_numbered(last)
            text = "Reasoning per item.\n" + "\n".join(f"Answer {i}: True" for i in range(1, n + 1))
        return fake_response_data(text=text, model=kwargs.get("model"))


# --- fake anthropic -----------------------------------------------------------


class _Block:
    def __init__(self, type: str, text: str = "") -> None:
        self.type = type
        self.text = text


class _Usage:
    input_tokens = 120
    output_tokens = 30


class _Resp:
    def __init__(self, content: list[_Block]) -> None:
        self.content = content
        self.usage = _Usage()
        self.stop_reason = "end_turn"


class _Messages:
    def __init__(self, owner: "FakeAnthropic") -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> _Resp:
        self._owner.calls.append(kwargs)
        last = kwargs["messages"][-1]["content"]
        if isinstance(last, str) and "What rule did you use" in last:
            return _Resp([_Block("thinking", ""), _Block("text", "RULE: the input contains a digit")])
        n = _n_numbered(last if isinstance(last, str) else "")
        body = "reasoning\n" + "\n".join(f"Answer {i}: True" for i in range(1, n + 1))
        return _Resp([_Block("thinking", ""), _Block("text", body)])


class FakeAnthropic:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = _Messages(self)

    async def close(self) -> None:
        pass


# --- gpt-4.1 path -------------------------------------------------------------


def test_gpt_run_produces_all_row_kinds(tmp_path) -> None:
    api = CannedAPI()
    assert cot.main(_argv(tmp_path, "gpt-4.1"), api=api) == 0
    rows = _read(_run_dir(tmp_path) / "responses.jsonl")
    kinds = {k: [r for r in rows if r["kind"] == k] for k in ("cot_turn1", "cot_turn2", "grade", "compile")}
    assert len(kinds["cot_turn1"]) == 1  # 1 rule x 1 context
    assert len(kinds["cot_turn2"]) == 1 and len(kinds["grade"]) == 1 and len(kinds["compile"]) == 1
    t1 = kinds["cot_turn1"][0]
    assert t1["predictions"] == [True, True, True, True]
    assert t1["accuracy"]["n"] == 4
    assert kinds["cot_turn2"][0]["candidate"] == "the input contains a digit"
    assert kinds["grade"][0]["grade"] == 2
    comp = kinds["compile"][0]
    assert comp["code"] is not None
    assert comp["result_vs_gold"]["ok"] is True  # real local sandbox ran
    assert comp["result_vs_self"]["ok"] is True


def test_turn2_request_echoes_turn1_assistant(tmp_path) -> None:
    api = CannedAPI()
    assert cot.main(_argv(tmp_path, "gpt-4.1"), api=api) == 0
    # the turn-2 subject call: 4 messages, with the captured turn-1 reply as assistant
    turn2_calls = [c for c in api.calls
                   if any(isinstance(m["content"], str) and "What rule did you use" in m["content"]
                          for m in c["messages"])]
    assert len(turn2_calls) == 1
    msgs = turn2_calls[0]["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert "Answer 1: True" in msgs[2]["content"]  # turn-1 completion echoed verbatim


def test_turn1_reuses_step1_few_shot_block(tmp_path) -> None:
    api = CannedAPI()
    assert cot.main(_argv(tmp_path, "gpt-4.1"), api=api) == 0
    turn1_calls = [c for c in api.calls
                   if any(isinstance(m["content"], str) and "Apply that same rule to classify" in m["content"]
                          for m in c["messages"])]
    assert turn1_calls
    user = turn1_calls[0]["messages"][1]["content"]
    assert "Here are labeled examples" in user
    assert "Label: True" in user and "Label: False" in user


def test_cost_gate_aborts_before_any_call(tmp_path) -> None:
    api = CannedAPI()
    rc = cot.main(_argv(tmp_path, "gpt-4.1", "--max-cost", "1e-9"), api=api)
    assert rc == 1
    assert api.calls == []
    assert not (tmp_path / "results").exists()


def test_config_logs_hashes_and_cot_mode(tmp_path) -> None:
    api = CannedAPI()
    assert cot.main(_argv(tmp_path, "gpt-4.1"), api=api) == 0
    cfg = json.loads((_run_dir(tmp_path) / "config.json").read_text())
    assert cfg["cot_mode"] == "prompted_cot"
    assert cfg["cot_same_session_template_hash"] == cot_same_session_template_hash()
    assert cfg["context_seeds"] == [0]
    assert "finished_utc" in cfg and "cost_actual_usd" in cfg


def test_odd_n_items_rejected(tmp_path) -> None:
    assert cot.main(_argv(tmp_path, "gpt-4.1", "--n-items", "3"), api=CannedAPI()) == 2


# --- claude path --------------------------------------------------------------


def test_claude_path_echoes_content_blocks_unchanged(tmp_path) -> None:
    oai = CannedAPI()          # judge + compile go through gpt-4.1
    ant = FakeAnthropic()      # subject turns
    rc = cot.main(_argv(tmp_path, "claude-opus-4-8"), api=oai, anthropic_api=ant)
    assert rc == 0
    # two subject calls (turn-1, turn-2)
    assert len(ant.calls) == 2
    turn1, turn2 = ant.calls
    assert turn1["thinking"] == {"type": "adaptive"}
    assert turn1["output_config"] == {"effort": "low"}
    # turn-2 echoes the turn-1 assistant CONTENT BLOCKS unchanged (list, not string)
    assistant_msg = turn2["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert isinstance(assistant_msg["content"], list)
    assert assistant_msg["content"] is turn1_response_content(ant)  # same object echoed


def turn1_response_content(ant: FakeAnthropic) -> list:
    # reconstruct: the content the fake returned for the turn-1 (non-RULE) call
    # is a list of blocks; the runner must have passed that exact list back.
    msgs = ant.calls[1]["messages"]
    return msgs[1]["content"]


def test_claude_config_cot_mode(tmp_path) -> None:
    rc = cot.main(_argv(tmp_path, "claude-opus-4-8"), api=CannedAPI(), anthropic_api=FakeAnthropic())
    assert rc == 0
    cfg = json.loads((_run_dir(tmp_path) / "config.json").read_text())
    assert cfg["cot_mode"] == "adaptive_thinking_effort_low"
    assert cfg["claude_price_per_mtok"] == {"in": 5.0, "out": 25.0}
