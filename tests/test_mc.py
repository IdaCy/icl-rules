"""Step-2 multiple-choice distractor builder + runner tests.

The CORE GUARANTEE is proven on REAL data/<rule>/items.jsonl through the ACTUAL
contexts (mc.load_contexts): for >= 3 rules across categories, every one of the
7 distractors disagrees with the true rule on >= 25% of the 32 shown examples
in EACH of the rule's 3 contexts. No network — the API is faked.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from icl_articulation import mc
from tests.conftest import FakeAPI, fake_response_data

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
EXTRACT = REPO / "data" / "spec_extract.json"

# >= 3 rules across categories: surface, numeric, positional, semantic, syntactic.
CROSS_CATEGORY_RULES = [
    "contains_digit",          # surface
    "word_count_geq_8",        # numeric
    "second_word_capitalized", # positional
    "mentions_color",          # semantic
    "passive_voice",           # syntactic
]

# the 11 confirmed step-1 survivors (the articulation targets).
TARGET_RULES = [
    "passive_voice",
    "food_topic",
    "positive_sentiment",
    "mentions_animal",
    "contains_first_name",
    "second_word_capitalized",
    "physically_impossible",
    "word_count_geq_8",
    "repeated_content_word",
    "contains_digit",
    "mentions_color",
]

CONTEXT_SEEDS = [0, 1, 2]


def _extract() -> dict[str, Any]:
    return mc.load_extract(EXTRACT)


# --- the core >= 25% guarantee on REAL data -----------------------------------


@pytest.mark.parametrize("rule_id", CROSS_CATEGORY_RULES)
def test_every_distractor_disagrees_at_least_25pct_on_each_context(rule_id: str) -> None:
    contexts = mc.load_contexts(rule_id, DATA_DIR, CONTEXT_SEEDS)
    assert len(contexts) == 3 and all(len(c) == 32 for c in contexts)

    option_set = mc.build_option_set(rule_id, _extract(), contexts, CONTEXT_SEEDS)

    distractors = [o for o in option_set.options if not o.is_true]
    assert len(distractors) == mc.N_DISTRACTORS  # exactly 7

    # THE LOAD-BEARING CHECK: recompute disagreement independently here (do not
    # trust the builder's own audit dict) against each of the 3 actual contexts.
    rn_map = mc._rn_to_rule_id(_extract())
    for opt in distractors:
        pred = mc.resolve_predicate(opt.seed, rn_map)
        for ci, ctx in enumerate(contexts):
            disagree = sum(1 for it in ctx if pred.label_of(it["text"]) != it["label"]) / len(ctx)
            assert disagree >= mc.DISAGREEMENT_FLOOR, (
                f"{rule_id} distractor {opt.predicate_key!r} disagrees only "
                f"{disagree:.3f} on context {ci} (< {mc.DISAGREEMENT_FLOOR})"
            )


@pytest.mark.parametrize("rule_id", CROSS_CATEGORY_RULES)
def test_option_set_shape(rule_id: str) -> None:
    contexts = mc.load_contexts(rule_id, DATA_DIR, CONTEXT_SEEDS)
    option_set = mc.build_option_set(rule_id, _extract(), contexts, CONTEXT_SEEDS)

    # exactly 8 options, exactly one true. The true option is a TERSE
    # distractor-register paraphrase of the canonical (construct-validity fix),
    # NOT the raw canonical_articulation, so it cannot be format-matched.
    assert len(option_set.options) == mc.N_OPTIONS
    trues = [o for o in option_set.options if o.is_true]
    assert len(trues) == 1
    canonical = _extract()["rules"][rule_id]["canonical_articulation"].strip()
    assert trues[0].text == mc._TRUE_OPTION_TEXT[rule_id]
    assert trues[0].text != canonical
    # the paraphrase must NOT carry the raw-canonical format tell
    low = trues[0].text.lower()
    assert "iff" not in low and "labeled true" not in low

    # no two options share a checking predicate
    keys = [o.predicate_key for o in option_set.options]
    assert len(keys) == len(set(keys))

    # no distractor is a banned_distractor head
    banned = {b.strip() for b in _extract()["rules"][rule_id]["banned_distractors"]}
    assert all(o.text not in banned for o in option_set.options if not o.is_true)


def test_option_set_accepts_variant_dataset_id(tmp_path: Path) -> None:
    variant = "contains_digit_deconfounded"
    data_dir = tmp_path / "data"
    shutil.copytree(DATA_DIR / "contains_digit", data_dir / variant)

    contexts = mc.load_contexts(variant, data_dir, CONTEXT_SEEDS)
    option_set = mc.build_option_set(variant, _extract(), contexts, CONTEXT_SEEDS)
    true = [o for o in option_set.options if o.is_true]

    assert len(true) == 1
    assert true[0].text == mc._TRUE_OPTION_TEXT["contains_digit"]
    assert true[0].predicate_key == "true:contains_digit"
    assert option_set.rule_id == variant


# --- construct validity: the TRUE option is not a surface-form tell -----------


@pytest.mark.parametrize("rule_id", TARGET_RULES)
def test_no_residual_format_tell_on_true_option(rule_id: str) -> None:
    """For each of the 11 target rules, PROVE the true option cannot be picked by
    SURFACE FORM alone: its length is in the distractor band (or within 1.5x the
    distractor mean) and not the longest/shortest by a wide margin; NO option
    carries a giveaway token ('iff', 'labeled True', ...); and the true option
    shares its 'The input ...' opening frame with >= 1 distractor."""
    contexts = mc.load_contexts(rule_id, DATA_DIR, CONTEXT_SEEDS)
    option_set = mc.build_option_set(rule_id, _extract(), contexts, CONTEXT_SEEDS)
    rep = mc.format_tell_report(option_set)

    # length: not an outlier
    assert rep.true_in_band or rep.true_within_factor, (
        f"{rule_id}: true len {rep.true_len} outside distractor band "
        f"[{rep.len_min},{rep.len_max}] and beyond 1.5x mean {rep.len_mean:.1f}"
    )
    assert not rep.is_longest_outlier, (
        f"{rule_id}: true option is the longest by a wide margin "
        f"(len {rep.true_len} vs distractor max {rep.len_max})"
    )
    assert not rep.is_shortest_outlier, (
        f"{rule_id}: true option is the shortest by a wide margin "
        f"(len {rep.true_len} vs distractor min {rep.len_min})"
    )
    # no giveaway token in ANY option (true or distractor)
    assert rep.giveaway_options == [], (
        f"{rule_id}: giveaway token(s) leaked into options {rep.giveaway_options}"
    )
    # the true option blends into the majority opening frame
    assert rep.true_starts_with_the_input, (
        f"{rule_id}: true option opener {rep.true_frame!r} is not 'The input ...'"
    )
    assert rep.true_frame_shared, (
        f"{rule_id}: true option is the UNIQUE carrier of its opener {rep.true_frame!r}"
    )
    assert rep.ok


@pytest.mark.parametrize("rule_id", TARGET_RULES)
def test_distractor_display_text_is_normalized(rule_id: str) -> None:
    """Distractor DISPLAY text must be terse: no ' — ' rationale tail and no
    parenthetical (author-note '(predicate: ...)' or inline '(0-9)') clause —
    while the underlying checking predicate (resolved from the raw seed) is
    unchanged."""
    contexts = mc.load_contexts(rule_id, DATA_DIR, CONTEXT_SEEDS)
    option_set = mc.build_option_set(rule_id, _extract(), contexts, CONTEXT_SEEDS)
    rn_map = mc._rn_to_rule_id(_extract())
    for opt in option_set.options:
        if opt.is_true:
            continue
        assert "(" not in opt.text and ")" not in opt.text, opt.text
        assert " — " not in opt.text, opt.text
        assert "predicate" not in opt.text.lower(), opt.text
        # the seed still resolves to the SAME predicate used in the >=25% check
        assert opt.predicate_key == mc.resolve_predicate(opt.seed, rn_map).key


def test_order_shuffle_deterministic_and_records_true_letter() -> None:
    rule_id = "contains_digit"
    contexts = mc.load_contexts(rule_id, DATA_DIR, CONTEXT_SEEDS)
    option_set = mc.build_option_set(rule_id, _extract(), contexts, CONTEXT_SEEDS)

    q1 = mc.build_queries(option_set, 1000)
    q2 = mc.build_queries(option_set, 1000)
    # 3 contexts x 4 orders = 12 queries
    assert len(q1) == mc.N_CONTEXTS * mc.N_ORDERS == 12
    # deterministic: same seeds -> identical orders and true-letters
    assert [[o.text for o in q.options] for q in q1] == [[o.text for o in q.options] for q in q2]
    assert [q.true_letter for q in q1] == [q.true_letter for q in q2]

    for q in q1:
        # the recorded true_letter actually points at the true option
        idx = mc.LETTERS.index(q.true_letter)
        assert q.options[idx].is_true
        assert len(q.options) == mc.N_OPTIONS

    # the 4 orders within a context are not all identical (the shuffle moves things)
    ctx0 = [q for q in q1 if q.context_index == 0]
    orderings = {tuple(o.text for o in q.options) for q in ctx0}
    assert len(orderings) > 1


def test_build_error_when_pool_too_shallow() -> None:
    # a fabricated rule whose every seed maps to the SAME (true-everything)
    # predicate: all duplicates / below floor -> cannot fill 7 distractors.
    extract = {
        "rules": {
            "contains_digit": {
                "plan_number": 2,
                "canonical_articulation": "X",
                "banned_distractors": [],
                # only 2 viable seed heads -> < 7 distractors
                "mc_distractor_seeds": [
                    "The input contains the word 'and'",
                    "The input contains the word 'and'",  # duplicate predicate
                ],
            }
        }
    }
    contexts = mc.load_contexts("contains_digit", DATA_DIR, CONTEXT_SEEDS)
    with pytest.raises(mc.MCBuildError):
        mc.build_option_set("contains_digit", extract, contexts, CONTEXT_SEEDS)


def test_unknown_seed_head_raises() -> None:
    rn_map = mc._rn_to_rule_id(_extract())
    with pytest.raises(mc.MCBuildError):
        mc.resolve_predicate("The input is purple and humming on a Tuesday", rn_map)


# --- runner end-to-end via FakeAPI --------------------------------------------


import run_step2_mc  # noqa: E402


class _TrueLetterAPI(FakeAPI):
    """Answers the multiple-choice by reading the prompt: returns the letter in front of the
    option line whose text is the rule's true-option paraphrase. (The true option
    is NO LONGER the longest line after the construct-validity fix, so a length
    heuristic would now FAIL — we match the known true-option text instead.) The
    runner therefore scores near-perfect on the examples arm."""

    _TRUE_TEXTS = set(mc._TRUE_OPTION_TEXT.values())

    def next_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        user = kwargs["messages"][-1]["content"]
        # parse the "X) text" option lines; pick the letter of the true-option text
        best_letter = "A"
        for line in user.splitlines():
            line = line.strip()
            if len(line) >= 3 and line[0] in mc.LETTERS and line[1] == ")":
                text = line[2:].strip()
                if text in self._TRUE_TEXTS:
                    best_letter = line[0]
                    break
        data = fake_response_data(text=best_letter, model=kwargs.get("model"))
        # rebuild logprobs so the chosen letter is the top token
        data["choices"][0]["logprobs"]["content"][0] = {
            "token": best_letter,
            "logprob": -0.01,
            "top_logprobs": [
                {"token": best_letter, "logprob": -0.01},
                {"token": "Z", "logprob": -5.0},
            ],
        }
        return data


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--model", "gpt-4.1",
        "--rules", "contains_digit",
        "--arms", "examples,no_examples",
        "--data-dir", str(DATA_DIR),
        "--spec-extract", str(EXTRACT),
        "--results-dir", str(tmp_path / "results"),
        "--cache-dir", str(tmp_path / "cache"),
        *extra,
    ]


def _run_dir(tmp_path: Path) -> Path:
    dirs = list((tmp_path / "results").iterdir())
    assert len(dirs) == 1
    return dirs[0]


def test_runner_end_to_end_logs_and_scores(tmp_path: Path) -> None:
    rc = run_step2_mc.main(_argv(tmp_path), api=_TrueLetterAPI())
    assert rc == 0

    run_dir = _run_dir(tmp_path)
    # config + metrics + responses all written
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    rows = [json.loads(ln) for ln in (run_dir / "responses.jsonl").read_text().splitlines()]

    # 1 rule x 3 contexts x 4 orders x 2 arms = 24 calls
    assert config["expected_total_calls"] == 24
    assert len(rows) == 24
    assert config["template_hash"]
    assert config["disagreement_floor"] == 0.25

    # the audit recorded the per-distractor per-context disagreement >= 0.25
    meta = config["rules_meta"]["contains_digit"]
    assert meta["min_disagreement"] >= 0.25
    assert len(meta["options"]) == 8

    # the longest-option heuristic picks the canonical (true) option -> the
    # examples arm scores perfectly and the modal claim is the true rule
    ex = metrics["rules"]["contains_digit"]["examples"]
    assert ex["n_queries"] == 12
    assert ex["accuracy"] == 1.0
    assert ex["modal_is_true"] is True
    assert ex["n_parse_failures"] == 0

    # control arm exists with chance baseline recorded
    assert metrics["aggregate"]["no_examples"]["chance"] == pytest.approx(1 / 8)


def test_runner_no_examples_arm_removes_few_shot_block(tmp_path: Path) -> None:
    api = _TrueLetterAPI()
    rc = run_step2_mc.main(_argv(tmp_path, "--arms", "no_examples"), api=api)
    assert rc == 0
    # every captured no-examples prompt must NOT contain the few-shot block
    for kwargs in api.calls:
        user = kwargs["messages"][-1]["content"]
        assert "Here are labeled examples:" not in user
        assert "\nLabel: True" not in user and "\nLabel: False" not in user
        assert "Which rule best describes how the labels were assigned?" in user


def test_runner_cost_gate_aborts(tmp_path: Path) -> None:
    rc = run_step2_mc.main(_argv(tmp_path, "--max-cost", "0.0"), api=_TrueLetterAPI())
    assert rc == 1
    # aborted before any run dir was created
    assert not (tmp_path / "results").exists() or not list((tmp_path / "results").iterdir())


def test_parse_letter_robust() -> None:
    assert run_step2_mc.parse_letter("A", 8) == "A"
    assert run_step2_mc.parse_letter(" b)", 8) == "B"
    assert run_step2_mc.parse_letter("Answer: C.", 8) == "C"
    assert run_step2_mc.parse_letter("(D)", 8) == "D"
    assert run_step2_mc.parse_letter("h", 8) == "H"
    assert run_step2_mc.parse_letter("I", 8) is None  # out of range (only A-H for 8 options)
    assert run_step2_mc.parse_letter("xyz", 8) is None
    # first valid option letter in the string wins
    assert run_step2_mc.parse_letter("zzz B", 8) == "B"
    assert run_step2_mc.parse_letter("yz G then C", 8) == "G"


def test_answer_letter_cot_tolerant() -> None:
    al = run_step2_mc.answer_letter
    # identical to parse_letter on bare single-letter answers (gpt-4.1 reproduces)
    for t in ("A", " b)", "Answer: C.", "(D)", "h"):
        assert al(t, 8) == run_step2_mc.parse_letter(t, 8)
    assert al("I", 8) is None and al("xyz", 8) is None
    # CoT: a letter mentioned mid-reasoning must NOT win; the answer-cued/last does
    assert al("Let's analyze option A and option B. The best option is C.", 8) == "C"
    assert al("Between A and D, the correct answer is D.", 8) == "D"
    assert al("Option A looks plausible but actually the answer is F", 8) == "F"
    # no cue -> the concluding (last) standalone letter
    assert al("hmm B ... finally G", 8) == "G"
