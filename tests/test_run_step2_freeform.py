"""Step-2 free-form runner tests: 12 generations/rule + 4 no-examples controls,
crash-safe logging, the judge path, pre-specified metrics, cost gate.

No network — the OpenAI surface is faked. The runner reads REAL target datasets
under data/ and the committed data/spec_extract.json (the public gold), so these
tests exercise the actual context sampler + spec extract, not synthetic stubs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_step2_freeform as r2
from icl_articulation.prompts import FINAL_RULE_MARKER, freeform_template_hash
from tests.conftest import FakeAPI, fake_response_data

# a real target that is RECOMPUTABLE (so the extensional check is exercised)
RULE = "contains_digit"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _single_run_dir(tmp_path) -> Path:
    dirs = list((tmp_path / "results").iterdir())
    assert len(dirs) == 1
    return dirs[0]


def _argv(tmp_path, *extra: str) -> list[str]:
    return [
        "--model", "gpt-4.1-mini",
        "--judge-model", "gpt-4.1",
        "--rules", RULE,
        "--results-dir", str(tmp_path / "results"),
        "--cache-dir", str(tmp_path / "cache"),
        *extra,
    ]


class CannedAPI(FakeAPI):
    """Generation returns a canned articulation; the judge returns a canned grade.

    A request is a JUDGE request iff its system prompt is the grading rubric
    (contains 'CANDIDATE'); otherwise it is a generation request. Generation
    answers mention 'digit' (the candidate the judge then grades 2); the judge
    grades 2 iff 'digit' is in the candidate, else 0."""

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        system = kwargs["messages"][0]["content"]
        user = kwargs["messages"][1]["content"]
        if "CANDIDATE ARTICULATION TO GRADE" in user:  # judge call
            candidate = user.split("CANDIDATE ARTICULATION TO GRADE:", 1)[1]
            grade = 2 if "digit" in candidate else 0
            text = json.dumps({"grade": grade, "extensionally_equivalent": grade == 2,
                               "rationale": "fake"})
        else:  # generation call
            text = f"{FINAL_RULE_MARKER} the input contains a digit"
        return fake_response_data(text=text, model=kwargs.get("model"))


# --- generation grid: 12 + 4 control ------------------------------------------


def test_generation_grid_12_plus_4_control(tmp_path) -> None:
    api = CannedAPI()
    assert r2.main(_argv(tmp_path), api=api) == 0
    run_dir = _single_run_dir(tmp_path)
    rows = _read_jsonl(run_dir / "responses.jsonl")

    gens = [r for r in rows if r["kind"] == "generation"]
    assert len(gens) == 16  # 12 with-examples + 4 no-examples control
    with_ex = [g for g in gens if g["has_examples"]]
    control = [g for g in gens if not g["has_examples"]]
    assert len(with_ex) == 12  # 3 contexts x (2 variants x 2 phrasings)
    assert len(control) == 4  # 1 per (variant x phrasing), no context

    # the 12 with-examples span 3 contexts x both variants x both phrasings
    assert sorted({g["context_index"] for g in with_ex}) == [0, 1, 2]
    assert {g["variant"] for g in with_ex} == {"direct", "think-then-state"}
    assert {g["phrasing"] for g in with_ex} == {0, 1}
    # the control carries no context
    assert all(g["context_index"] == -1 and g["context_seed"] is None for g in control)
    # the one-sentence rule is extracted from the marker
    assert all(g["candidate"] == "the input contains a digit" for g in gens)


def test_with_examples_prompt_reuses_few_shot_control_omits_it(tmp_path) -> None:
    api = CannedAPI()
    assert r2.main(_argv(tmp_path), api=api) == 0
    gen_calls = [c for c in api.calls if "CANDIDATE ARTICULATION TO GRADE" not in c["messages"][1]["content"]]
    with_ex = [c for c in gen_calls if "Here are labeled examples" in c["messages"][1]["content"]]
    control = [c for c in gen_calls if "Here are labeled examples" not in c["messages"][1]["content"]]
    assert len(with_ex) == 12 and len(control) == 4
    for c in control:  # control has the articulation ask but NO example labels
        user = c["messages"][1]["content"]
        assert "Label: True" not in user and "Label: False" not in user


# --- the judge path + metrics -------------------------------------------------


def test_grading_path_and_metrics(tmp_path) -> None:
    api = CannedAPI()
    assert r2.main(_argv(tmp_path), api=api) == 0
    run_dir = _single_run_dir(tmp_path)
    rows = _read_jsonl(run_dir / "responses.jsonl")

    grades = [r for r in rows if r["kind"] == "grade"]
    assert len(grades) == 16  # one judge call per generation
    assert all(g["grade"] == 2 for g in grades)  # canned candidate mentions 'digit'
    # extensional check ran (contains_digit is recomputable)
    assert all(g["extensional_check"]["applicable"] for g in grades)
    assert all(g["extensional_check"]["n_probes"] == r2.N_PROBES for g in grades)
    assert all(g["extensional_check"]["stored_label_agreement"] == 1.0 for g in grades)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    rm = metrics["rules"][RULE]
    assert rm["primary_median_direct"] == 2.0  # median of the DIRECT variant
    assert rm["secondary_best_variant"] == 2  # best variant
    assert rm["n_generations"] == 12  # controls excluded from the primary metric
    assert rm["grade_counts"] == {"0": 0, "1": 0, "2": 12}
    assert rm["no_examples_control"]["n"] == 4  # control summarized separately
    assert metrics["overall"]["n_generations"] == 12
    assert metrics["overall"]["n_controls"] == 4


def test_judge_grade_2_for_extensional_equivalence(tmp_path) -> None:
    # an extensionally-equivalent (not verbatim-canonical) candidate still gets 2
    class EquivAPI(CannedAPI):
        def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            user = kwargs["messages"][1]["content"]
            if "CANDIDATE ARTICULATION TO GRADE" in user:
                # judge sees a paraphrase but grades 2 via the equivalence route
                return fake_response_data(
                    text=json.dumps({"grade": 2, "extensionally_equivalent": True, "rationale": "equiv"}),
                    model=kwargs.get("model"),
                )
            return fake_response_data(
                text=f"{FINAL_RULE_MARKER} the sentence has a numeral somewhere",
                model=kwargs.get("model"),
            )

    assert r2.main(_argv(tmp_path), api=EquivAPI()) == 0
    run_dir = _single_run_dir(tmp_path)
    grades = [r for r in _read_jsonl(run_dir / "responses.jsonl") if r["kind"] == "grade"]
    assert all(g["grade"] == 2 and g["extensionally_equivalent"] for g in grades)


# --- config / provenance ------------------------------------------------------


def test_config_logs_hashes_seeds_and_grid(tmp_path) -> None:
    assert r2.main(_argv(tmp_path), api=CannedAPI()) == 0
    config = json.loads((_single_run_dir(tmp_path) / "config.json").read_text())
    assert config["task"] == "step2-freeform"
    assert config["model"] == "gpt-4.1-mini" and config["judge_model"] == "gpt-4.1"
    assert config["context_seeds"] == [0, 1, 2]
    assert config["generations_per_rule"] == 12 and config["controls_per_rule"] == 4
    assert config["freeform_template_hash"] == freeform_template_hash()
    assert config["rubric_hash"] is not None
    assert config["expected_generations"] == 16
    assert "finished_utc" in config and config["cost_actual_usd"] > 0


# --- no-grade mode ------------------------------------------------------------


def test_no_grade_mode_generates_only(tmp_path) -> None:
    api = CannedAPI()
    assert r2.main(_argv(tmp_path, "--no-grade"), api=api) == 0
    run_dir = _single_run_dir(tmp_path)
    rows = _read_jsonl(run_dir / "responses.jsonl")
    assert all(r["kind"] == "generation" for r in rows)
    assert len(rows) == 16
    # no judge calls were made
    assert all("CANDIDATE ARTICULATION TO GRADE" not in c["messages"][1]["content"] for c in api.calls)
    config = json.loads((run_dir / "config.json").read_text())
    assert config["grading"] is False and config["rubric_hash"] is None


# --- cost gate ----------------------------------------------------------------


def test_cost_gate_aborts_before_any_call(tmp_path) -> None:
    api = CannedAPI()
    rc = r2.main(_argv(tmp_path, "--max-cost", "0.0000001"), api=api)
    assert rc == 1
    assert api.calls == []  # nothing hit the API
    assert not (tmp_path / "results").exists()  # no run dir created


def test_unpriced_model_fails_loudly(tmp_path) -> None:
    with pytest.raises(KeyError, match="no price known"):
        r2.main(_argv(tmp_path, "--model", "gpt-99"), api=CannedAPI())


# --- crash mid-run leaves a valid partial jsonl -------------------------------


class CrashOnGenAPI(FakeAPI):
    """Two generations succeed, then a non-retryable error."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        import httpx
        import openai

        self.n += 1
        if self.n > 2:
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            raise openai.BadRequestError("boom", response=httpx.Response(400, request=req), body=None)
        return fake_response_data(text=f"{FINAL_RULE_MARKER} a rule", model=kwargs.get("model"))


def test_crash_mid_run_leaves_partial_jsonl(tmp_path) -> None:
    import openai

    with pytest.raises(openai.BadRequestError):
        r2.main(_argv(tmp_path, "--concurrency", "1"), api=CrashOnGenAPI())
    run_dir = _single_run_dir(tmp_path)
    rows = _read_jsonl(run_dir / "responses.jsonl")  # every line valid JSON
    assert 1 <= len(rows) < 16
    config = json.loads((run_dir / "config.json").read_text())
    assert "finished_utc" not in config  # finish() never ran


# --- default targets ----------------------------------------------------------


def test_default_targets_are_the_11_survivors() -> None:
    args = r2.parse_args(["--model", "gpt-4.1-mini"])
    assert r2.resolve_rules(args) == r2.DEFAULT_TARGETS
    assert len(r2.DEFAULT_TARGETS) == 11
