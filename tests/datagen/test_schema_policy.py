"""Reviewer-finding regression tests for schema.py:

(1) banned-'I' detection runs over the STRIPPED tokenizer output, so
    punctuation-adjacent 'I' ('I,' / 'I.') is caught.
(2) the central RULE_STYLE_POLICY table + the rule_id self-selecting path on
    assert_sentence_style / validate_full (rule 3 -> '!', rule 26 -> commas,
    default rule -> neither).
(3) assign_llm_splits stays deterministic and exactly 50/50 after the dead
    in-loop shuffle was removed.
"""

from __future__ import annotations

from collections import Counter

import pytest

from icl_articulation.datagen import schema
from icl_articulation.datagen.schema import (
    RULE_STYLE_POLICY,
    SchemaError,
    assert_sentence_style,
    assign_llm_splits,
    make_item,
    style_policy_for,
    validate_full,
)


def _item(text: str, *, rule_id: str = "r", label: bool = True) -> dict:
    return make_item(
        item_id="x",
        base_id="x",
        rule_id=rule_id,
        label=label,
        text=text,
        slots_meta={},
        split="held_out",
    )


# --- (1) banned 'I' over the stripped tokenizer -------------------------------


def test_banned_I_caught_when_punctuation_adjacent() -> None:
    # 'I,' and 'I.' strip to 'I' under words(); a raw text.split() would miss them.
    for text in ("Later I, the dog ran home", "The dog ran home with I."):
        with pytest.raises(SchemaError, match="pronoun 'I'"):
            assert_sentence_style([_item(text)], allow_internal_comma=True)


def test_banned_I_still_caught_bare() -> None:
    with pytest.raises(SchemaError, match="pronoun 'I'"):
        assert_sentence_style([_item("Today I ran home")])


def test_capital_I_inside_word_not_flagged() -> None:
    # only the standalone pronoun is banned; 'India' must not trip the check.
    assert_sentence_style([_item("the dog visited India today")])


# --- (2) central policy table + self-selecting path ---------------------------


def test_policy_table_values() -> None:
    assert style_policy_for("contains_exclamation") == ("!", True)
    assert style_policy_for("exactly_two_commas") == ("", True)
    assert style_policy_for("positive_sentiment") == ("", True)
    assert style_policy_for("food_topic") == ("", True)
    # an unknown / default rule gets the strict global style.
    assert style_policy_for("title_case") == ("", False)
    assert style_policy_for("some_unknown_rule") == ("", False)


def test_rule3_self_selects_terminal_bang() -> None:
    # rule 3 (contains_exclamation): terminal '!' allowed, others rejected.
    assert_sentence_style(
        [_item("the dog ran home!", rule_id="contains_exclamation")],
        rule_id="contains_exclamation",
    )
    with pytest.raises(SchemaError, match="terminal punctuation"):
        assert_sentence_style(
            [_item("the dog ran home.", rule_id="contains_exclamation")],
            rule_id="contains_exclamation",
        )


def test_rule26_self_selects_commas() -> None:
    # rule 26 (exactly_two_commas): internal commas allowed, no terminal char.
    assert_sentence_style(
        [_item("apples, pears, and plums", rule_id="exactly_two_commas")],
        rule_id="exactly_two_commas",
    )
    with pytest.raises(SchemaError, match="terminal punctuation"):
        assert_sentence_style(
            [_item("apples, pears, and plums!", rule_id="exactly_two_commas")],
            rule_id="exactly_two_commas",
        )


def test_default_rule_self_selects_strict() -> None:
    # a default rule (title_case): neither terminal punctuation nor commas.
    rid = "title_case"
    assert_sentence_style([_item("The Dog Ran Home", rule_id=rid)], rule_id=rid)
    with pytest.raises(SchemaError, match="comma"):
        assert_sentence_style([_item("Later, The Dog Ran", rule_id=rid)], rule_id=rid)
    with pytest.raises(SchemaError, match="terminal punctuation"):
        assert_sentence_style([_item("The Dog Ran Home!", rule_id=rid)], rule_id=rid)


def test_rule_id_and_explicit_flags_mutually_exclusive() -> None:
    with pytest.raises(SchemaError, match="not both"):
        assert_sentence_style(
            [_item("the dog ran home!")],
            rule_id="contains_exclamation",
            allow_terminal="!",
        )


def test_validate_full_self_selects_policy() -> None:
    # validate_full threads rule_id through to the style policy. A clean rule-26
    # dataset (commas legal) passes; the same comma under a default rule fails.
    items = [
        make_item(item_id="b0-T", base_id="b0", rule_id="exactly_two_commas",
                  label=True, text="apples, pears, and ripe plums here",
                  slots_meta={}, split="few_shot_pool"),
        make_item(item_id="b0-F", base_id="b0", rule_id="exactly_two_commas",
                  label=False, text="apples, pears and ripe plums over there",
                  slots_meta={}, split="few_shot_pool"),
    ]
    validate_full(items, rule_id="exactly_two_commas")  # must not raise
    with pytest.raises(SchemaError, match="comma"):
        validate_full(items, rule_id="title_case")


# --- (3) assign_llm_splits determinism + 50/50 balance ------------------------


def test_assign_llm_splits_deterministic_and_balanced() -> None:
    ids = [f"i{i}" for i in range(400)]
    labels = [i % 2 == 0 for i in range(400)]  # 200 T / 200 F

    a = assign_llm_splits(ids, labels, seed=7)
    b = assign_llm_splits(list(reversed(ids)), list(reversed(labels)), seed=7)
    # same seed -> identical assignment regardless of input order (determinism).
    assert a == b

    counts = Counter(a.values())
    assert counts["few_shot_pool"] == 120
    assert counts["held_out"] == 120
    assert counts["confirmation"] == 100

    label_of = dict(zip(ids, labels))
    for split, n in (("few_shot_pool", 120), ("held_out", 120), ("confirmation", 100)):
        members = [iid for iid, s in a.items() if s == split]
        n_true = sum(1 for m in members if label_of[m])
        assert n_true == n // 2, f"{split} not 50/50: {n_true}/{n}"

    # the two label orderings are seeded INDEPENDENTLY (no shared-seed permutation).
    assert RULE_STYLE_POLICY  # table is populated (sanity)
