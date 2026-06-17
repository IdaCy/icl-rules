"""Grading-module tests: judge prompt, 2/1/0 parse, extensional check,
pre-specified metrics (median-of-direct, best variant). No network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from icl_articulation.client import OpenAIClient
from icl_articulation.grading import (
    GRADING_RUBRIC,
    best_grade,
    extensional_probe,
    gold_for,
    grade_one,
    load_spec_extract,
    median_of_direct,
    parse_judge,
    render_judge,
    rubric_hash,
    summarize_rule,
)
from tests.conftest import FakeAPI, fake_response_data

EXAMPLES = [("the cat sat", True), ("The Dog Ran", False)]


# --- judge prompt + parsing ---------------------------------------------------


def test_render_judge_feeds_gold_candidate_examples() -> None:
    messages = render_judge(
        "the input contains a digit",
        "True iff the raw string contains a 0-9 digit.",
        ["the sentence has a numeral", "labeled False when there is no digit"],
        EXAMPLES,
    )
    assert messages[0]["content"] == GRADING_RUBRIC  # rubric is the system prompt
    user = messages[1]["content"]
    assert "True iff the raw string contains a 0-9 digit." in user  # canonical
    assert "the sentence has a numeral" in user  # equivalence class member
    assert "the input contains a digit" in user  # candidate
    assert "the cat sat" in user  # example sentences for extensional reasoning
    assert "strict json" in user.lower()


def test_parse_judge_maps_grades_and_equiv_flag() -> None:
    v = parse_judge('{"grade": 2, "extensionally_equivalent": true, "rationale": "match"}')
    assert v == {"grade": 2, "extensionally_equivalent": True, "rationale": "match"}
    assert parse_judge('{"grade": 1, "rationale": "near miss"}')["grade"] == 1
    assert parse_judge('{"grade": 0}')["grade"] == 0
    # tolerant of surrounding prose / fencing
    fenced = 'Here is my verdict:\n```json\n{"grade": 2, "extensionally_equivalent": false}\n```'
    assert parse_judge(fenced)["grade"] == 2


def test_parse_judge_raises_loudly_on_bad_output() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        parse_judge("the candidate is good")
    with pytest.raises(ValueError, match="missing 'grade'"):
        parse_judge('{"verdict": "good"}')
    with pytest.raises(ValueError, match="out of range"):
        parse_judge('{"grade": 5}')


def test_rubric_hash_stable() -> None:
    assert rubric_hash() == rubric_hash()
    assert len(rubric_hash()) == 64


# --- spec extract access (public extract only) --------------------------------


def test_load_spec_extract_and_gold_for() -> None:
    extract = Path(__file__).resolve().parent.parent / "data" / "spec_extract.json"
    rules = load_spec_extract(extract)
    gold = gold_for(rules, "contains_digit")
    assert isinstance(gold["canonical_articulation"], str) and gold["canonical_articulation"]
    assert isinstance(gold["equivalence_class"], list)


def test_gold_for_missing_rule_raises() -> None:
    with pytest.raises(KeyError, match="not in spec extract"):
        gold_for({"contains_digit": {}}, "nope")
    with pytest.raises(ValueError, match="no canonical_articulation"):
        gold_for({"contains_digit": {"equivalence_class": []}}, "contains_digit")


# --- extensional check --------------------------------------------------------


def test_extensional_probe_runs_label_of_for_recomputable_rules() -> None:
    probes = [
        {"text": "page 42 here", "label": True},
        {"text": "no numbers at all", "label": False},
    ]
    out = extensional_probe(lambda t: any(c.isdigit() for c in t), probes)
    assert out["applicable"] is True
    assert out["n_probes"] == 2
    assert out["true_rule_n_true"] == 1 and out["true_rule_n_false"] == 1
    assert out["stored_label_agreement"] == 1.0


def test_extensional_probe_not_applicable_for_validator_derived() -> None:
    out = extensional_probe(None, [{"text": "x", "label": True}])
    assert out["applicable"] is False and "no recomputable predicate" in out["reason"]


# --- one graded call through a fake judge -------------------------------------


class JudgeAPI(FakeAPI):
    """Fake judge: grades 2 iff the candidate mentions 'digit', else 0."""

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        user = kwargs["messages"][1]["content"]
        # the candidate is the block after 'CANDIDATE ARTICULATION TO GRADE:'
        candidate = user.split("CANDIDATE ARTICULATION TO GRADE:", 1)[1]
        grade = 2 if "digit" in candidate else 0
        verdict = json.dumps(
            {"grade": grade, "extensionally_equivalent": grade == 2, "rationale": "fake"}
        )
        return fake_response_data(text=verdict, model=kwargs.get("model"))


def test_grade_one_through_fake_judge() -> None:
    async def run() -> None:
        client = OpenAIClient(api=JudgeAPI(), cache_dir=_tmp_cache())
        try:
            good = await grade_one(client, "contains a digit", "canonical", [], EXAMPLES)
            assert good["grade"] == 2 and good["extensionally_equivalent"] is True
            bad = await grade_one(client, "mentions a color", "canonical", [], EXAMPLES)
            assert bad["grade"] == 0
        finally:
            await client.aclose()

    asyncio.run(run())


def _tmp_cache() -> str:
    import tempfile

    return tempfile.mkdtemp()


# --- pre-specified metrics ----------------------------------------------------


def test_median_of_direct_and_best() -> None:
    assert median_of_direct([2, 2, 1]) == 2.0  # median of the direct variant
    assert median_of_direct([0, 2]) == 1.0  # even count interpolates (NOT rounded)
    assert best_grade([0, 1, 2, 1]) == 2
    with pytest.raises(ValueError):
        median_of_direct([])


def _g(variant: str, ctx: int, grade: int, has_examples: bool = True) -> dict[str, Any]:
    return {"variant": variant, "phrasing": 0, "context_index": ctx,
            "has_examples": has_examples, "grade": grade}


def test_summarize_rule_primary_secondary_and_control() -> None:
    graded = [
        # direct variant across 3 contexts: grades 2, 2, 1 -> median 2
        _g("direct", 0, 2), _g("direct", 1, 2), _g("direct", 2, 1),
        # think-then-state lands a 2 somewhere -> best variant still 2
        _g("think-then-state", 0, 1), _g("think-then-state", 1, 2), _g("think-then-state", 2, 0),
        # no-examples control (separate summary, a-priori guessability)
        _g("direct", -1, 0, has_examples=False), _g("think-then-state", -1, 1, has_examples=False),
    ]
    out = summarize_rule(graded)
    assert out["primary_median_direct"] == 2.0  # median of [2,2,1]
    assert out["secondary_best_variant"] == 2
    assert out["n_generations"] == 6  # excludes the 2 controls
    assert out["grade_counts"] == {"0": 1, "1": 2, "2": 3}
    assert out["by_variant"]["direct"]["median"] == 2.0
    assert out["no_examples_control"]["n"] == 2
    assert out["no_examples_control"]["max"] == 1  # control is weaker than with-examples


def test_summarize_rule_extensional_equivalence_grades_2() -> None:
    # an extensionally-equivalent candidate that the judge graded 2 stays 2 in
    # the per-rule summary (no post-hoc demotion)
    graded = [
        {"variant": "direct", "phrasing": 0, "context_index": 0, "has_examples": True,
         "grade": 2, "extensionally_equivalent": True},
    ]
    out = summarize_rule(graded)
    assert out["primary_median_direct"] == 2.0
    assert out["grade_counts"]["2"] == 1
