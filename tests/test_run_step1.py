"""Step-1 runner tests: label parsing, logprob margins, incremental logging,
metrics (parse failures count as incorrect), cost gate, rule_given mode.

No network — the API surface is faked (conftest.FakeAPI subclasses)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_step1
from icl_articulation.prompts import rule_given_template_hash, step1_template_hash
from tests.conftest import FakeAPI, fake_response_data, write_rule_dataset


def _query_of(kwargs: dict[str, Any]) -> str:
    """The held-out sentence from a rendered prompt (trailing 'Input: X\\nLabel:')."""
    content = kwargs["messages"][-1]["content"]
    return content.rsplit("Input: ", 1)[1].removesuffix("\nLabel:")


class RuleFollowingAPI(FakeAPI):
    """Answers the synthetic toy rule perfectly: True iff the query contains 'alpha'."""

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        label = "True" if "alpha" in _query_of(kwargs) else "False"
        data = fake_response_data(text=label)
        data["model"] = kwargs.get("model")
        return data


def _argv(tmp_path, mode: str, *extra: str) -> list[str]:
    return [
        "--mode", mode,
        "--model", "gpt-4.1",
        "--rules", "toy_rule",
        "--data-dir", str(tmp_path / "data"),
        "--results-dir", str(tmp_path / "results"),
        "--cache-dir", str(tmp_path / "cache"),
        *extra,
    ]


def _single_run_dir(tmp_path) -> Path:
    dirs = list((tmp_path / "results").iterdir())
    assert len(dirs) == 1
    return dirs[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- label parsing -----------------------------------------------------------


def test_parse_label_robust() -> None:
    assert run_step1.parse_label("True") is True
    assert run_step1.parse_label("False") is False
    assert run_step1.parse_label(" True\n") is True
    assert run_step1.parse_label("True.") is True  # max_tokens=2 may append a token
    assert run_step1.parse_label("False,") is False
    assert run_step1.parse_label("TrueFalse") is True  # prefix wins
    # CASE-INSENSITIVE (review M3 option a): semantically clear answers parse
    assert run_step1.parse_label("true") is True
    assert run_step1.parse_label("FALSE") is False
    assert run_step1.parse_label("TRUE.") is True
    # parse failure = NEITHER label recognizable case-insensitively
    assert run_step1.parse_label("Maybe") is None
    assert run_step1.parse_label("Tru") is None
    assert run_step1.parse_label("") is None


def test_format_ok_tracks_exact_case() -> None:
    assert run_step1.format_ok("True") is True
    assert run_step1.format_ok(" False\n") is True
    assert run_step1.format_ok("True.") is True
    # parses (case-insensitively) but is NOT the exact-case answer format
    assert run_step1.format_ok("true") is False
    assert run_step1.format_ok("FALSE") is False
    assert run_step1.format_ok("Maybe") is False
    assert run_step1.format_ok("") is False


def _record_with_tops(text: str, tops: list[tuple[str, float]]) -> dict[str, Any]:
    return {
        "response": {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "logprobs": {
                        "content": [
                            {
                                "token": text,
                                "logprob": tops[0][1],
                                "top_logprobs": [
                                    {"token": t, "logprob": lp} for t, lp in tops
                                ],
                            }
                        ]
                    },
                }
            ]
        }
    }


def test_logprob_margin_both_labels_in_top5() -> None:
    record = _record_with_tops("True", [("True", -0.02), (" False", -3.9), ("true", -9.0)])
    tops, margin = run_step1.answer_logprobs(record, True)
    assert len(tops) == 3
    assert margin == pytest.approx(-0.02 - (-3.9))  # ' False' counts after strip
    # margin is signed from the CHOSEN label
    record = _record_with_tops("False", [("False", -0.1), ("True", -2.4)])
    _, margin = run_step1.answer_logprobs(record, False)
    assert margin == pytest.approx(-0.1 - (-2.4))


def test_logprob_margin_none_when_other_label_missing_or_unparsed() -> None:
    record = _record_with_tops("True", [("True", -0.01), ("Yes", -5.0)])
    _, margin = run_step1.answer_logprobs(record, True)
    assert margin is None
    record = _record_with_tops("Maybe", [("True", -0.5), ("False", -1.0)])
    _, margin = run_step1.answer_logprobs(record, None)  # parse failure -> no margin
    assert margin is None
    no_lp = {"response": {"choices": [{"message": {"content": "True"}, "logprobs": None}]}}
    tops, margin = run_step1.answer_logprobs(no_lp, True)
    assert tops is None and margin is None


# --- end-to-end pilot run ------------------------------------------------------


def test_pilot_run_end_to_end(tmp_path) -> None:
    write_rule_dataset(tmp_path / "data")
    api = RuleFollowingAPI()
    assert run_step1.main(_argv(tmp_path, "pilot"), api=api) == 0

    run_dir = _single_run_dir(tmp_path)
    rows = _read_jsonl(run_dir / "responses.jsonl")
    assert len(rows) == 40  # 1 rule x 1 context x 40 items
    assert sum(r["true_label"] for r in rows) == 20  # balanced pilot subset
    assert all(r["parse_ok"] and r["correct"] for r in rows)
    assert all(r["format_ok"] for r in rows)  # exact-case 'True'/'False'
    assert all(r["context_seed"] == 0 for r in rows)

    # locked call params on every wire request
    for call in api.calls:
        assert call["temperature"] == 0.0 and call["max_tokens"] == 2
        assert call["logprobs"] is True and call["top_logprobs"] == 5
        assert call["seed"] == 0
        assert "Here are labeled examples:" in call["messages"][1]["content"]

    config = json.loads((run_dir / "config.json").read_text())
    assert config["mode"] == "pilot" and config["model"] == "gpt-4.1"
    assert config["template_hash"] == step1_template_hash()
    assert config["context_seeds"] == [0] and config["k_few_shot"] == 32
    assert config["expected_total_calls"] == 40
    assert "finished_utc" in config and config["cost_actual_usd"] > 0
    ctx = config["contexts"]["toy_rule"][0]
    assert len(ctx["item_ids"]) == 32 and len(set(ctx["base_ids"])) == 32

    metrics = json.loads((run_dir / "metrics.json").read_text())
    rm = metrics["rules"]["toy_rule"]
    c0 = rm["contexts"][0]
    assert c0["n"] == 40 and c0["accuracy"] == 1.0
    assert c0["wilson_ci_95"][0] > 0.9 and c0["wilson_ci_95"][1] == 1.0
    assert c0["per_class"]["true"]["accuracy"] == 1.0
    assert c0["per_class"]["false"]["accuracy"] == 1.0
    assert c0["n_parse_failures"] == 0
    assert c0["format_ok_rate"] == 1.0
    assert rm["pooled"]["mean_accuracy"] == 1.0
    assert rm["pooled"]["cluster_bootstrap_ci_95"] is None  # single context: no bootstrap
    assert metrics["overall"]["parse_failure_rate"] == 0.0


def test_pilot_contexts_deterministic_across_runs(tmp_path) -> None:
    write_rule_dataset(tmp_path / "data")
    tasks1, meta1 = run_step1.build_tasks("pilot", ["toy_rule"], tmp_path / "data", run_seed=0)
    tasks2, meta2 = run_step1.build_tasks("pilot", ["toy_rule"], tmp_path / "data", run_seed=0)
    assert meta1 == meta2
    assert [t.messages for t in tasks1] == [t.messages for t in tasks2]
    _, meta3 = run_step1.build_tasks("pilot", ["toy_rule"], tmp_path / "data", run_seed=1)
    assert meta3 != meta1  # different run seed -> different context


# --- metrics: parse failures count as incorrect -----------------------------------


class GarbageOnBetaAPI(FakeAPI):
    """Unparseable answer for every False-class query ('beta'), 'True' otherwise."""

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        text = "True" if "alpha" in _query_of(kwargs) else "I"
        return fake_response_data(text=text)


def test_parse_failures_count_as_incorrect_and_flagged(tmp_path) -> None:
    write_rule_dataset(tmp_path / "data")
    assert run_step1.main(_argv(tmp_path, "pilot"), api=GarbageOnBetaAPI()) == 0
    run_dir = _single_run_dir(tmp_path)

    rows = _read_jsonl(run_dir / "responses.jsonl")
    failed = [r for r in rows if not r["parse_ok"]]
    assert len(failed) == 20
    assert all(r["predicted"] is None and r["correct"] is False for r in failed)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    c0 = metrics["rules"]["toy_rule"]["contexts"][0]
    assert c0["accuracy"] == 0.5  # 20 parse failures scored INCORRECT
    assert c0["n_parse_failures"] == 20
    assert c0["predictions"] == {"true": 20, "false": 0, "parse_failure": 20}
    assert c0["per_class"]["true"]["accuracy"] == 1.0
    assert c0["per_class"]["false"]["accuracy"] == 0.0
    assert metrics["overall"]["parse_failure_rate"] == 0.5


class LowercaseAPI(FakeAPI):
    """Semantically correct but lowercase answers (the review M3 scenario)."""

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        text = "true" if "alpha" in _query_of(kwargs) else "false"
        return fake_response_data(text=text)


def test_lowercase_answers_score_correct_but_flag_format(tmp_path) -> None:
    write_rule_dataset(tmp_path / "data")
    assert run_step1.main(_argv(tmp_path, "pilot"), api=LowercaseAPI()) == 0
    run_dir = _single_run_dir(tmp_path)

    rows = _read_jsonl(run_dir / "responses.jsonl")
    assert all(r["parse_ok"] and r["correct"] for r in rows)  # not scored incorrect
    assert all(r["format_ok"] is False for r in rows)  # but the format slip is recorded
    # full text still logged per row -> no-CoT/format claims stay auditable
    assert all(r["response"]["choices"][0]["message"]["content"] in ("true", "false") for r in rows)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    c0 = metrics["rules"]["toy_rule"]["contexts"][0]
    assert c0["accuracy"] == 1.0
    assert c0["n_parse_failures"] == 0
    assert c0["format_ok_rate"] == 0.0
    assert metrics["overall"]["parse_failure_rate"] == 0.0


# --- cost gate ----------------------------------------------------------------------


def test_cost_gate_aborts_before_any_call(tmp_path) -> None:
    write_rule_dataset(tmp_path / "data")
    api = RuleFollowingAPI()
    rc = run_step1.main(_argv(tmp_path, "pilot", "--max-cost", "0.000001"), api=api)
    assert rc == 1
    assert api.calls == []  # nothing hit the API
    assert not (tmp_path / "results").exists()  # no run dir created


def test_unpriced_model_fails_loudly_at_start(tmp_path) -> None:
    write_rule_dataset(tmp_path / "data")
    with pytest.raises(KeyError, match="no price known"):
        run_step1.main(
            ["--mode", "pilot", "--model", "gpt-99-turbo", "--rules", "toy_rule",
             "--data-dir", str(tmp_path / "data")],
            api=FakeAPI(),
        )


# --- incremental logging (crash mid-run leaves a partial responses.jsonl) ------------


class CrashAfterTwoAPI(FakeAPI):
    """Two successes, then a non-retryable error on every later call."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.n += 1
        if self.n > 2:
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            raise openai.BadRequestError("boom", response=httpx.Response(400, request=req), body=None)
        return fake_response_data(text="True")


def test_crash_mid_run_leaves_partial_responses_jsonl(tmp_path) -> None:
    write_rule_dataset(tmp_path / "data")
    with pytest.raises(openai.BadRequestError):
        run_step1.main(_argv(tmp_path, "pilot", "--concurrency", "1"), api=CrashAfterTwoAPI())
    run_dir = _single_run_dir(tmp_path)
    rows = _read_jsonl(run_dir / "responses.jsonl")  # every line valid JSON
    assert 1 <= len(rows) < 40  # completed tasks were logged BEFORE the crash
    config = json.loads((run_dir / "config.json").read_text())
    assert "finished_utc" not in config  # crash: finish() never ran


# --- rule_given mode --------------------------------------------------------------


def test_rule_given_renders_no_examples_and_includes_rule_text(tmp_path) -> None:
    write_rule_dataset(tmp_path / "data")
    rule_text = "The input is labeled True iff it contains the word alpha."
    rules_file = tmp_path / "canonical.json"
    rules_file.write_text(json.dumps({"toy_rule": rule_text}))
    api = RuleFollowingAPI()
    rc = run_step1.main(
        _argv(tmp_path, "rule_given", "--rules-file", str(rules_file)), api=api
    )
    assert rc == 0
    assert len(api.calls) == 120  # the same 120 held-out items, no contexts
    for call in api.calls:
        user = call["messages"][1]["content"]
        assert rule_text in user
        assert "Answer with exactly True or False." in user
        assert "Here are labeled examples" not in user  # NO examples
        assert "Label: True" not in user and "Label: False" not in user
        assert user.endswith("Label:")

    run_dir = _single_run_dir(tmp_path)
    config = json.loads((run_dir / "config.json").read_text())
    assert config["template_hash"] == rule_given_template_hash()
    assert config["context_seeds"] is None and config["k_few_shot"] == 0
    assert config["contexts"]["toy_rule"][0]["rule_text"] == rule_text
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["rules"]["toy_rule"]["contexts"][0]["accuracy"] == 1.0


def test_rule_given_variant_falls_back_to_base_rule_text(tmp_path) -> None:
    variant = "contains_digit_deconfounded"
    write_rule_dataset(tmp_path / "data", rule_id=variant)
    rule_text = "The input is labeled True iff it contains a digit."
    rules_file = tmp_path / "canonical.json"
    rules_file.write_text(json.dumps({"contains_digit": rule_text}))
    api = RuleFollowingAPI()

    rc = run_step1.main(
        [
            "--mode", "rule_given",
            "--model", "gpt-4.1",
            "--rules", variant,
            "--data-dir", str(tmp_path / "data"),
            "--results-dir", str(tmp_path / "results"),
            "--cache-dir", str(tmp_path / "cache"),
            "--rules-file", str(rules_file),
        ],
        api=api,
    )

    assert rc == 0
    assert len(api.calls) == 120
    assert all(rule_text in call["messages"][1]["content"] for call in api.calls)


def test_rule_given_requires_rules_file(tmp_path) -> None:
    write_rule_dataset(tmp_path / "data")
    assert run_step1.main(_argv(tmp_path, "rule_given"), api=FakeAPI()) == 2


def test_rules_file_rejected_in_few_shot_modes(tmp_path) -> None:
    # --rules-file is meaningless outside rule_given: error out instead of
    # silently ignoring it (an operator may believe it changed the run)
    write_rule_dataset(tmp_path / "data")
    api = FakeAPI()
    with pytest.raises(SystemExit) as exc_info:
        run_step1.main(_argv(tmp_path, "pilot", "--rules-file", "whatever.json"), api=api)
    assert exc_info.value.code == 2
    assert api.calls == []


def test_rule_given_missing_rule_text_raises(tmp_path) -> None:
    write_rule_dataset(tmp_path / "data")
    rules_file = tmp_path / "canonical.json"
    rules_file.write_text(json.dumps({"other_rule": "something"}))
    with pytest.raises(ValueError, match="no canonical rule text"):
        run_step1.main(_argv(tmp_path, "rule_given", "--rules-file", str(rules_file)), api=FakeAPI())


# --- full mode: 3 contexts, cluster bootstrap, degenerate-behavior visibility --------


class AlwaysTrueAPI(FakeAPI):
    """Degenerate single-class behavior."""

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return fake_response_data(text="True")


def test_full_mode_metrics_and_cluster_bootstrap(tmp_path) -> None:
    write_rule_dataset(tmp_path / "data")
    assert run_step1.main(_argv(tmp_path, "full"), api=AlwaysTrueAPI()) == 0
    run_dir = _single_run_dir(tmp_path)

    rows = _read_jsonl(run_dir / "responses.jsonl")
    assert len(rows) == 360  # 1 rule x 3 contexts x 120 items
    config = json.loads((run_dir / "config.json").read_text())
    assert config["context_seeds"] == [0, 1, 2]
    assert config["expected_total_calls"] == 360
    metas = config["contexts"]["toy_rule"]
    assert len(metas) == 3
    assert metas[0]["item_ids"] != metas[1]["item_ids"]  # different seeds -> different contexts

    metrics = json.loads((run_dir / "metrics.json").read_text())
    rm = metrics["rules"]["toy_rule"]
    assert len(rm["contexts"]) == 3
    for c in rm["contexts"]:
        assert c["n"] == 120
        assert c["accuracy"] == 0.5  # all-True answers on a 60/60 split
        # the per-class breakdown exposes the degenerate behavior
        assert c["per_class"]["true"]["accuracy"] == 1.0
        assert c["per_class"]["false"]["accuracy"] == 0.0
        assert c["predictions"] == {"true": 120, "false": 0, "parse_failure": 0}
    pooled = rm["pooled"]
    assert pooled["mean_accuracy"] == 0.5
    low, high = pooled["cluster_bootstrap_ci_95"]
    assert low <= 0.5 <= high
    assert 0.3 < low < 0.5 < high < 0.7  # item-level resampling, not degenerate
