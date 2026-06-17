"""Template rendering + hash stability tests."""

from __future__ import annotations

import hashlib

import pytest

from icl_articulation.prompts import (
    FINAL_RULE_MARKER,
    FREEFORM_VARIANTS,
    STEP1_EXAMPLE_TEMPLATE,
    STEP1_SYSTEM,
    STEP1_USER_TEMPLATE,
    RULE_GIVEN_USER_TEMPLATE,
    extract_rule,
    freeform_template_hash,
    render_freeform_articulation,
    render_freeform_no_examples,
    render_mc_articulation,
    render_rule_given,
    render_step1,
    rule_given_template_hash,
    step1_template_hash,
)

EXAMPLES = [("the cat sat", True), ("The Dog Ran", False), ("all quiet here", True)]


def test_template_hash_stable_and_matches_constants() -> None:
    h1 = step1_template_hash()
    h2 = step1_template_hash()
    assert h1 == h2
    assert len(h1) == 64
    blob = "\n---\n".join([STEP1_SYSTEM, STEP1_EXAMPLE_TEMPLATE, STEP1_USER_TEMPLATE])
    assert h1 == hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_template_hash_regression() -> None:
    # Pins the exact template text. If this fails, the template changed:
    # update the hash CONSCIOUSLY and note it — every run config logs it.
    assert step1_template_hash() == (
        "9e7fb5854ffa142bb2b5d97b5192d12c21e0a2d602e4ffd6b63cf9140572a643"
    )


def test_render_step1_structure() -> None:
    messages = render_step1(EXAMPLES, "a new sentence")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == STEP1_SYSTEM
    user = messages[1]["content"]
    assert "Here are labeled examples:" in user
    assert "Input: the cat sat\nLabel: True" in user
    assert "Input: The Dog Ran\nLabel: False" in user
    assert "Classify the next input. Answer with exactly True or False." in user
    assert user.endswith("Input: a new sentence\nLabel:")


def test_render_step1_preserves_example_order() -> None:
    user = render_step1(EXAMPLES, "q")[1]["content"]
    positions = [user.index(f"Input: {text}") for text, _ in EXAMPLES]
    assert positions == sorted(positions)


def test_render_step1_no_rule_leakage() -> None:
    # generic template: nothing about the rule beyond the examples themselves
    user = render_step1(EXAMPLES, "q")[1]["content"]
    for word in ("rule", "lowercase", "capital"):
        assert word not in user.lower()
    assert "rule" not in STEP1_SYSTEM.lower()


def test_render_rule_given_structure() -> None:
    messages = render_rule_given("The input contains a digit.", "page 42")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == STEP1_SYSTEM
    user = messages[1]["content"]
    assert "The input contains a digit." in user
    assert "Answer with exactly True or False." in user
    assert user.endswith("Input: page 42\nLabel:")
    # zero-shot baseline: NO examples anywhere
    assert "Here are labeled examples" not in user
    assert "Label: True" not in user and "Label: False" not in user


def test_rule_given_template_hash_stable() -> None:
    h = rule_given_template_hash()
    assert h == rule_given_template_hash()
    assert len(h) == 64
    blob = "\n---\n".join([STEP1_SYSTEM, RULE_GIVEN_USER_TEMPLATE])
    assert h == hashlib.sha256(blob.encode("utf-8")).hexdigest()
    assert h != step1_template_hash()


def test_render_mc_articulation_structure() -> None:
    messages = render_mc_articulation(EXAMPLES, ["rule A", "rule B", "rule C"])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == STEP1_SYSTEM
    user = messages[1]["content"]
    assert "Here are labeled examples:" in user
    assert "Input: the cat sat\nLabel: True" in user
    assert "Which rule best describes how the labels were assigned?" in user
    assert "A) rule A" in user and "B) rule B" in user and "C) rule C" in user
    assert user.endswith("Answer with the single letter of the best option.")


# --- step-2 free-form articulation -------------------------------------------


def test_freeform_grid_is_2x2() -> None:
    # {direct, think-then-state} x {2 phrasings}
    assert set(FREEFORM_VARIANTS) == {"direct", "think-then-state"}
    assert all(len(ph) == 2 for ph in FREEFORM_VARIANTS.values())


def test_render_freeform_reuses_few_shot_block_and_asks_for_one_sentence() -> None:
    messages = render_freeform_articulation(EXAMPLES, "direct", 0)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == STEP1_SYSTEM  # same neutral classifier system
    user = messages[1]["content"]
    # the SAME few-shot example block as step 1
    assert "Input: the cat sat\nLabel: True" in user
    assert "Input: The Dog Ran\nLabel: False" in user
    # asks to STATE the rule, one sentence, marker-prefixed
    assert "ONE sentence" in user
    assert FINAL_RULE_MARKER in user


def test_direct_forbids_reasoning_think_allows_it() -> None:
    direct = render_freeform_articulation(EXAMPLES, "direct", 0)[1]["content"]
    think = render_freeform_articulation(EXAMPLES, "think-then-state", 0)[1]["content"]
    assert "Do not explain or show your reasoning" in direct
    assert "think step by step" in think.lower() or "reasoning" in think.lower()


def test_phrasings_differ_within_a_variant() -> None:
    p0 = render_freeform_articulation(EXAMPLES, "direct", 0)[1]["content"]
    p1 = render_freeform_articulation(EXAMPLES, "direct", 1)[1]["content"]
    assert p0 != p1


def test_render_freeform_no_examples_control_has_no_examples() -> None:
    user = render_freeform_no_examples("direct", 0)[1]["content"]
    assert "Here are labeled examples" not in user
    assert "Label: True" not in user and "Label: False" not in user
    assert "Input: the cat sat" not in user
    # still the same articulation request
    assert FINAL_RULE_MARKER in user and "rule" in user.lower()


def test_freeform_invalid_variant_or_phrasing_raises() -> None:
    with pytest.raises(ValueError, match="unknown free-form variant"):
        render_freeform_articulation(EXAMPLES, "nope", 0)
    with pytest.raises(ValueError, match="phrasing index"):
        render_freeform_articulation(EXAMPLES, "direct", 5)


def test_extract_rule_handles_marker_and_fallbacks() -> None:
    # direct: marker at the start
    assert extract_rule("RULE: the input contains a digit") == "the input contains a digit"
    # think-then-state: CoT then a final marker line
    cot = "Let me think.\nThe True ones have digits.\nRULE: the input contains a digit."
    assert extract_rule(cot) == "the input contains a digit."
    # last marker wins if the word appears earlier
    assert extract_rule("The RULE: is unclear\nRULE: it mentions a color") == "it mentions a color"
    # no marker -> last non-empty line (never an empty candidate)
    assert extract_rule("some reasoning\nit is about colors") == "it is about colors"
    assert extract_rule("just one line") == "just one line"


def test_freeform_template_hash_stable_and_distinct() -> None:
    h = freeform_template_hash()
    assert h == freeform_template_hash()
    assert len(h) == 64
    assert h != step1_template_hash() and h != rule_given_template_hash()


def test_cot_same_session_renderer_and_hash() -> None:
    from icl_articulation.prompts import (
        cot_same_session_template_hash,
        cot_turn2_user,
        render_cot_turn1,
    )

    h = cot_same_session_template_hash()
    assert h == cot_same_session_template_hash() and len(h) == 64
    assert h != step1_template_hash() and h != freeform_template_hash()

    msgs = render_cot_turn1(["one two three", "four five six seven"], EXAMPLES)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == STEP1_SYSTEM
    user = msgs[1]["content"]
    # reuses the step-1 few-shot block + numbers the queries + demands the answer block
    assert "Here are labeled examples" in user
    assert "Label: True" in user and "Label: False" in user
    assert "1. one two three" in user and "2. four five six seven" in user
    assert "Answer 1: True" in user  # the parseable-block instruction

    # turn-2 ends with the RULE: marker clause, so extract_rule recovers a candidate
    t2 = cot_turn2_user()
    assert t2.rstrip().endswith("'RULE:'.")
    assert extract_rule("reasoning here\nRULE: at least eight words") == "at least eight words"
