"""Step-3 3-arm runner end-to-end through a FakeAPI (no network).

The fake labels each probe by the TRUE rule for arm 1 (in-context behavior) and
arm 3 (true-rule-given), and by the articulation-STUB predicate for arm 2
(self-application). With behavior == true rule and self-application ==
articulation, the divergence subset MUST show behavior tracking the TRUE rule and
diverging from self-application — the unfaithful signature.

The runner reads the REAL target datasets under data/ and the committed
data/spec_extract.json, so it exercises the actual probe builder + context
sampler, not synthetic stubs."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_step3_faithfulness as r3
from icl_articulation.faithfulness import ARM_IN_CONTEXT, ARM_SELF, ARM_TRUE
from icl_articulation.prompts import rule_given_template_hash, step1_template_hash
from icl_articulation.step3_probes import (
    TARGET_RULES,
    articulation_for,
    articulation_predict,
    build_probe_set,
)
from tests.conftest import FakeAPI, fake_response_data

RULE = "physically_impossible"  # a hand-labeled (validator-derived) rule
RULE_RECOMP = "second_word_capitalized"  # a recomputable rule


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _single_run_dir(tmp_path) -> Path:
    dirs = list((tmp_path / "results").iterdir())
    assert len(dirs) == 1
    return dirs[0]


def _argv(tmp_path, *extra: str) -> list[str]:
    return [
        "--model", "gpt-4.1-mini",
        "--rules", RULE,
        "--results-dir", str(tmp_path / "results"),
        "--cache-dir", str(tmp_path / "cache"),
        *extra,
    ]


class ThreeArmAPI(FakeAPI):
    """Label probes per arm so the divergence signature is exact.

    Arm detection from the rendered messages:
      - ARM 1 (in-context): the step-1 user prompt ("Here are labeled examples").
      - ARM 2 (self-application): rule-given prompt whose rule text is the MODEL'S
        articulation -> answer by the articulation-STUB predicate.
      - ARM 3 (true-rule-given): rule-given prompt whose rule text is the
        canonical articulation -> answer by the TRUE rule.

    The TRUE-rule answer for arm 1 / arm 3 is looked up from the probe set by the
    probe text (the only place the runner's true labels live), so the fake is the
    perfect 'model learned the true rule' oracle. Arm 2 uses articulation_predict
    (the model applies its own stated rule)."""

    def __init__(self, rules=(RULE,), model="gpt-4.1-mini") -> None:
        super().__init__()
        # text -> (rule_id, true_label) over every probe of the run's rules
        self.true_by_text: dict[str, tuple[str, bool]] = {}
        self.articulation_texts: dict[str, str] = {}
        for rule_id in rules:
            self.articulation_texts[rule_id] = articulation_for(rule_id, model)
            for p in build_probe_set(rule_id, "data", model=model):
                self.true_by_text[p.text] = (rule_id, p.true_label)

    def _classify(self, user: str) -> bool:
        # recover the probe text from the trailing "Input: <text>\nLabel:" line
        text = user.rsplit("Input:", 1)[1].rsplit("\nLabel:", 1)[0].strip()
        rule_id, true_label = self.true_by_text[text]
        if "Here are labeled examples" in user:  # ARM 1 in-context -> true rule
            return true_label
        # rule-given: arm 2 (own articulation) vs arm 3 (canonical)
        if self.articulation_texts[rule_id] in user:  # ARM 2 self -> articulation stub
            return articulation_predict(rule_id, text)
        return true_label  # ARM 3 true-rule-given -> true rule

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        user = kwargs["messages"][1]["content"]
        label = self._classify(user)
        return fake_response_data(text="True" if label else "False", model=kwargs.get("model"))


# --- all three arms logged per probe ------------------------------------------


def test_three_arms_logged_per_probe(tmp_path) -> None:
    assert r3.main(_argv(tmp_path), api=ThreeArmAPI()) == 0
    run_dir = _single_run_dir(tmp_path)
    rows = _read_jsonl(run_dir / "responses.jsonl")
    classifications = [r for r in rows if r["kind"] == "classification"]

    probes = build_probe_set(RULE, "data", model="gpt-4.1-mini")
    n_probes = len(probes)
    assert len(classifications) == n_probes * 3  # 3 arms per probe

    # every probe has exactly one row per arm
    by_probe: dict[str, set[str]] = {}
    for r in classifications:
        by_probe.setdefault(r["probe_id"], set()).add(r["arm"])
    assert len(by_probe) == n_probes
    assert all(arms == {ARM_IN_CONTEXT, ARM_SELF, ARM_TRUE} for arms in by_probe.values())
    assert all(r["parse_ok"] for r in classifications)


def test_arm_prompts_are_distinct(tmp_path) -> None:
    api = ThreeArmAPI()
    assert r3.main(_argv(tmp_path), api=api) == 0
    art = articulation_for(RULE, "gpt-4.1-mini")
    in_context = [c for c in api.calls if "Here are labeled examples" in c["messages"][1]["content"]]
    rule_given = [c for c in api.calls if "according to this rule" in c["messages"][1]["content"]]
    self_app = [c for c in rule_given if art in c["messages"][1]["content"]]
    true_given = [c for c in rule_given if art not in c["messages"][1]["content"]]
    n = len(build_probe_set(RULE, "data", model="gpt-4.1-mini"))
    assert len(in_context) == n  # arm 1 reuses the step-1 few-shot block
    assert len(self_app) == n  # arm 2 = render_rule_given(own articulation)
    assert len(true_given) == n  # arm 3 = render_rule_given(canonical)


def test_articulations_file_overrides_arm2_text(tmp_path) -> None:
    override = "True if the subject is a statue; False otherwise."
    art_file = tmp_path / "articulations.json"
    art_file.write_text(json.dumps({"articulations": {RULE: override}}), encoding="utf-8")
    api = ThreeArmAPI()

    assert r3.main(_argv(tmp_path, "--articulations-file", str(art_file)), api=api) == 0

    self_app = [
        c for c in api.calls
        if "according to this rule" in c["messages"][1]["content"]
        and override in c["messages"][1]["content"]
    ]
    n = len(build_probe_set(RULE, "data", model="gpt-4.1-mini"))
    assert len(self_app) == n
    config = json.loads((_single_run_dir(tmp_path) / "config.json").read_text())
    assert config["articulations_file"] == str(art_file)
    assert config["art_label_source"] == "legacy_static_probe_predicates"
    metrics = json.loads((_single_run_dir(tmp_path) / "metrics.json").read_text())
    assert metrics["articulation_override"] is True
    assert metrics["articulations_file"] == str(art_file)
    assert metrics["art_label_source"] == "legacy_static_probe_predicates"
    assert "fixed_designed" in metrics["faithfulness"]["rules"][RULE]


# --- the analysis computes faithfulness + divergence agreements ----------------


def test_metrics_show_unfaithful_divergence_signature(tmp_path) -> None:
    assert r3.main(_argv(tmp_path), api=ThreeArmAPI()) == 0
    metrics = json.loads((_single_run_dir(tmp_path) / "metrics.json").read_text())
    rm = metrics["faithfulness"]["rules"][RULE]
    assert rm["n_divergence"] > 0

    # KEY numbers on the divergence subset: behavior tracks TRUE, not SELF.
    h = rm["legacy_empirical_divergence"]
    assert h["behavior_tracks_true_rate"] == 1.0
    assert h["behavior_tracks_self_rate"] == 0.0
    assert h["gap_true_minus_self"] == 1.0
    assert "unfaithful" in h["interpretation"]

    # faithfulness = behavior-vs-self agreement; over the divergence subset it is
    # 0 (behavior == true, self == articulation, and they disagree there).
    assert rm["divergence"]["faithfulness_behavior_vs_self"]["rate"] == 0.0
    # The fixed designed set is not selected on arm-2 behaviour and still reports
    # actual behavior-vs-self faithfulness. It may include non-discriminating
    # designed probes, so it is not identical to the empirical-divergence rate.
    assert rm["fixed_designed"]["faithfulness_behavior_vs_self"]["rate"] is not None
    # arm-3 sanity: told the canonical rule, the fake matches the true label
    assert rm["divergence"]["true_rule_given_vs_true"]["rate"] == 1.0
    # behavior tracks the true rule perfectly over ALL probes too
    assert rm["overall"]["behavior_vs_true"]["rate"] == 1.0


class EmpiricalDivergenceAPI(FakeAPI):
    """Drive a CONTROLLED empirical-divergence subset that DIFFERS from the
    build-time proxy tag, to prove the analysis keys on ARM-2 (self-application)
    disagreeing with the true label, not on Probe.is_divergence.

    A fixed set of probe TEXTS (``diverge_texts``) is the intended empirical
    divergence subset. For those texts:
      - ARM 2 (self-application) returns NOT true_label  -> arm2 != true (the
        empirical-divergence condition), regardless of the proxy tag.
      - ARM 1 (in-context behavior) returns true_label   -> behavior~true == 1.0,
        behavior~self == 0.0 on the subset (the unfaithful signature).
    For every OTHER text: all three arms return true_label (arm2 == true -> NOT
    in the empirical subset, even if the build-time proxy tagged it)."""

    def __init__(self, diverge_texts: set[str], rules=(RULE,), model="gpt-4.1-mini") -> None:
        super().__init__()
        self.diverge_texts = diverge_texts
        self.true_by_text: dict[str, bool] = {}
        self.articulation_texts: dict[str, str] = {}
        for rule_id in rules:
            self.articulation_texts[rule_id] = articulation_for(rule_id, model)
            for p in build_probe_set(rule_id, "data", model=model):
                self.true_by_text[p.text] = p.true_label

    def _classify(self, user: str, rule_id: str = RULE) -> bool:
        text = user.rsplit("Input:", 1)[1].rsplit("\nLabel:", 1)[0].strip()
        true_label = self.true_by_text[text]
        is_self_arm = self.articulation_texts[rule_id] in user and "Here are labeled examples" not in user
        if is_self_arm and text in self.diverge_texts:
            return not true_label  # ARM 2 disagrees with true on the chosen subset
        return true_label  # arm1, arm3, and arm2-elsewhere all track the true rule

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        label = self._classify(kwargs["messages"][1]["content"])
        return fake_response_data(text="True" if label else "False", model=kwargs.get("model"))


def test_empirical_divergence_subset_is_arm2_vs_true(tmp_path) -> None:
    # Pick a controlled set of probe texts to be the empirical divergence subset.
    probes = build_probe_set(RULE, "data", model="gpt-4.1-mini")
    # choose 5 probes whose build-time proxy tag is FALSE, so the empirical subset
    # provably differs from the proxy tag.
    non_proxy = [p for p in probes if not p.is_divergence][:5]
    diverge_texts = {p.text for p in non_proxy}
    assert len(diverge_texts) == 5

    api = EmpiricalDivergenceAPI(diverge_texts)
    assert r3.main(_argv(tmp_path), api=api) == 0
    metrics = json.loads((_single_run_dir(tmp_path) / "metrics.json").read_text())
    rm = metrics["faithfulness"]["rules"][RULE]

    # The EMPIRICAL subset is exactly the 5 chosen probes (arm2 != true), NOT the
    # build-time proxy-tagged set.
    assert rm["divergence_definition"].startswith("empirical")
    assert rm["n_divergence"] == 5
    assert rm["n_divergence_proxy"] != 5  # proxy tag is a different (larger) set

    # On that subset behavior (arm1) tracks the TRUE rule, not self -> unfaithful.
    h = rm["legacy_empirical_divergence"]
    assert h["behavior_tracks_true_rate"] == 1.0
    assert h["behavior_tracks_self_rate"] == 0.0
    assert h["gap_true_minus_self"] == 1.0
    assert "unfaithful" in h["interpretation"]

    # Cross-check against the raw rows: the empirical subset is exactly the probes
    # where the logged ARM-2 prediction disagrees with the true label.
    rows = _read_jsonl(_single_run_dir(tmp_path) / "responses.jsonl")
    by_probe: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("kind") == "classification":
            by_probe.setdefault(r["probe_id"], {})[r["arm"]] = r
    empirical = {
        pid for pid, arms in by_probe.items()
        if arms[ARM_SELF]["predicted"] != arms[ARM_SELF]["true_label"]
    }
    assert len(empirical) == 5
    assert {by_probe[pid][ARM_SELF]["text"] for pid in empirical} == diverge_texts


def test_summary_row_per_rule(tmp_path) -> None:
    assert r3.main(_argv(tmp_path), api=ThreeArmAPI()) == 0
    metrics = json.loads((_single_run_dir(tmp_path) / "metrics.json").read_text())
    summary = metrics["faithfulness"]["summary"]
    assert [row["rule_id"] for row in summary] == [RULE]
    row = summary[0]
    assert row["legacy_empirical_selection_biased"] is True
    assert row["legacy_empirical_gap_true_minus_self"] == 1.0
    assert "faithfulness_overall" in row and "faithfulness_divergence" in row


# --- divergence probes are correctly tagged (true != articulation stub) --------


def test_divergence_probes_tagged_consistently_with_arms(tmp_path) -> None:
    assert r3.main(_argv(tmp_path), api=ThreeArmAPI()) == 0
    rows = _read_jsonl(_single_run_dir(tmp_path) / "responses.jsonl")
    classifications = [r for r in rows if r["kind"] == "classification"]
    # group by probe
    by_probe: dict[str, dict[str, Any]] = {}
    for r in classifications:
        by_probe.setdefault(r["probe_id"], {})[r["arm"]] = r
    for arms in by_probe.values():
        ic = arms[ARM_IN_CONTEXT]
        sa = arms[ARM_SELF]
        is_div = ic["is_divergence"]
        # the fake makes arm1 == true_label and arm2 == art_label; so a probe is
        # tagged divergence iff its two arms' PREDICTIONS disagree.
        assert is_div == (ic["predicted"] != sa["predicted"])
        assert is_div == (ic["true_label"] != ic["art_label"])


# --- config logs probes, hashes, articulations --------------------------------


def test_config_logs_probes_hashes_and_articulations(tmp_path) -> None:
    assert r3.main(_argv(tmp_path), api=ThreeArmAPI()) == 0
    config = json.loads((_single_run_dir(tmp_path) / "config.json").read_text())
    assert config["task"] == "step3-faithfulness"
    assert config["arms"] == [ARM_IN_CONTEXT, ARM_SELF, ARM_TRUE]
    assert config["step1_template_hash"] == step1_template_hash()
    assert config["rule_given_template_hash"] == rule_given_template_hash()
    pr = config["probes_per_rule"][RULE]
    assert pr["n_probes"] >= 50 and pr["n_divergence"] > 0
    assert pr["articulation"] == articulation_for(RULE, "gpt-4.1-mini")
    assert pr["canonical_articulation"].startswith("The input is labeled True")
    assert len(pr["probes"]) == pr["n_probes"]
    assert "finished_utc" in config and config["cost_actual_usd"] > 0


# --- recomputable rule path (true labels recomputed) --------------------------


def test_recomputable_rule_runs_and_diverges(tmp_path) -> None:
    api = ThreeArmAPI(rules=(RULE_RECOMP,))
    argv = [
        "--model", "gpt-4.1-mini",
        "--rules", RULE_RECOMP,
        "--results-dir", str(tmp_path / "results"),
        "--cache-dir", str(tmp_path / "cache"),
    ]
    assert r3.main(argv, api=api) == 0
    metrics = json.loads((_single_run_dir(tmp_path) / "metrics.json").read_text())
    rm = metrics["faithfulness"]["rules"][RULE_RECOMP]
    assert rm["n_divergence"] > 0
    assert rm["legacy_empirical_divergence"]["selection_biased"] is True
    assert rm["legacy_empirical_divergence"]["gap_true_minus_self"] == 1.0


# --- multiple rules at once ----------------------------------------------------


def test_default_runs_all_four_targets(tmp_path) -> None:
    api = ThreeArmAPI(rules=TARGET_RULES)
    argv = [
        "--model", "gpt-4.1-mini",
        "--results-dir", str(tmp_path / "results"),
        "--cache-dir", str(tmp_path / "cache"),
    ]
    assert r3.main(argv, api=api) == 0
    metrics = json.loads((_single_run_dir(tmp_path) / "metrics.json").read_text())
    assert set(metrics["faithfulness"]["rules"]) == set(TARGET_RULES)


# --- cost gate aborts before any call -----------------------------------------


def test_cost_gate_aborts_before_any_call(tmp_path) -> None:
    api = ThreeArmAPI()
    rc = r3.main(_argv(tmp_path, "--max-cost", "0.00000001"), api=api)
    assert rc == 1
    assert api.calls == []  # nothing hit the API
    assert not (tmp_path / "results").exists()  # no run dir created


def test_unpriced_model_fails_loudly(tmp_path) -> None:
    with pytest.raises(KeyError, match="no price known"):
        r3.main(_argv(tmp_path, "--model", "gpt-99"), api=ThreeArmAPI())


def test_bad_rule_fails_loudly(tmp_path) -> None:
    from icl_articulation.contexts import DatasetError

    with pytest.raises(DatasetError, match="not step-3 targets"):
        r3.main(_argv(tmp_path, "--rules", "contains_digit"), api=ThreeArmAPI())


def test_resolve_rules_accepts_variant_of_step3_target() -> None:
    args = r3.parse_args(["--model", "gpt-4.1-mini", "--rules", "second_word_capitalized_deconfounded"])

    assert r3.resolve_rules(args) == ["second_word_capitalized_deconfounded"]


def test_build_rule_accepts_variant_dataset_id(tmp_path) -> None:
    variant = "second_word_capitalized_deconfounded"
    data_dir = tmp_path / "data"
    shutil.copytree(Path("data") / "second_word_capitalized", data_dir / variant)
    spec_rules = r3.load_spec_extract("data/spec_extract.json")

    build = r3.build_rule(
        variant,
        data_dir,
        spec_rules,
        model="gpt-4.1-mini",
        context_seed=0,
        n_in_distribution=40,
    )

    assert build.rule_id == variant
    assert build.articulation == articulation_for(variant, "gpt-4.1-mini")
    assert build.canonical == spec_rules["second_word_capitalized"]["canonical_articulation"].strip()
    assert len(build.tasks) == len(build.probes) * 3


# --- crash mid-run leaves a valid partial jsonl -------------------------------


class CrashAfterTwoAPI(FakeAPI):
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
        return fake_response_data(text="True", model=kwargs.get("model"))


def test_crash_mid_run_leaves_partial_jsonl(tmp_path) -> None:
    import openai

    with pytest.raises(openai.BadRequestError):
        r3.main(_argv(tmp_path, "--concurrency", "1"), api=CrashAfterTwoAPI())
    run_dir = _single_run_dir(tmp_path)
    rows = _read_jsonl(run_dir / "responses.jsonl")  # every line valid JSON
    assert 1 <= len(rows)
    config = json.loads((run_dir / "config.json").read_text())
    assert "finished_utc" not in config  # finish() never ran
