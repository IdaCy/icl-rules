"""Step-3 probe-set tests: >= 50 probes/rule, the three sources present, true
labels for recomputable rules checked against groundtruth.label_of, divergence
correctly tagged (true_label != articulation-predicate label).

No network: the probe layer is pure data + the rules' OWN datasets + the public
data/spec_extract.json."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from icl_articulation.datagen import groundtruth
from icl_articulation.step3_probes import (
    ARTICULATIONS,
    RECOMPUTABLE,
    TARGET_RULES,
    ProbeError,
    articulation_for,
    articulation_predict,
    build_all_probe_sets,
    build_probe_set,
    parse_edge_example,
)


def test_all_four_rules_have_at_least_50_probes() -> None:
    sets = build_all_probe_sets(data_dir="data", model="gpt-4.1")
    assert set(sets) == set(TARGET_RULES)
    for rule_id, probes in sets.items():
        assert len(probes) >= 50, f"{rule_id}: only {len(probes)} probes"


def test_each_rule_has_all_three_probe_sources() -> None:
    for rule_id in TARGET_RULES:
        sources = {p.source for p in build_probe_set(rule_id, "data", model="gpt-4.1")}
        assert "in_distribution" in sources
        assert "divergence" in sources
        # edge_idea is present for every target (each has parseable edge examples)
        assert "edge_idea" in sources, f"{rule_id}: no edge_idea probes"


def test_each_rule_has_a_nonempty_divergence_subset() -> None:
    for rule_id in TARGET_RULES:
        probes = build_probe_set(rule_id, "data", model="gpt-4.1")
        div = [p for p in probes if p.is_divergence]
        assert div, f"{rule_id}: no divergence probes"


def test_divergence_flag_is_true_label_ne_articulation_label() -> None:
    for rule_id in TARGET_RULES:
        for p in build_probe_set(rule_id, "data", model="gpt-4.1"):
            assert p.is_divergence == (p.true_label != p.art_label)
            # the articulation label equals the rule's articulation predicate
            assert p.art_label == articulation_predict(rule_id, p.text)


def test_recomputable_rules_true_labels_match_groundtruth() -> None:
    for rule_id in RECOMPUTABLE:
        for p in build_probe_set(rule_id, "data", model="gpt-4.1"):
            assert p.true_label_source == "recomputed"
            assert p.true_label == groundtruth.label_of(rule_id, p.text)


def test_build_probe_set_accepts_variant_dataset_id(tmp_path: Path) -> None:
    variant = "second_word_capitalized_deconfounded"
    data_dir = tmp_path / "data"
    shutil.copytree(Path("data") / "second_word_capitalized", data_dir / variant)

    probes = build_probe_set(variant, data_dir, model="gpt-4.1")

    assert len(probes) >= 50
    assert {p.rule_id for p in probes} == {variant}
    assert any(p.is_divergence for p in probes)
    for p in probes:
        assert p.true_label_source == "recomputed"
        assert p.true_label == groundtruth.label_of(variant, p.text)
        assert p.art_label == articulation_predict(variant, p.text)


def test_validator_derived_rules_have_hand_true_labels() -> None:
    for rule_id in ("physically_impossible", "food_topic"):
        probes = build_probe_set(rule_id, "data", model="gpt-4.1")
        # in-distribution items keep the dataset's stored label (flagged hand,
        # since the rule is not recomputable); constructed probes are hand too.
        assert all(p.true_label_source == "hand" for p in probes)
        # groundtruth.label_of MUST refuse these rules (no text predicate)
        with pytest.raises(groundtruth.GroundTruthError):
            groundtruth.label_of(rule_id, probes[0].text)


def test_physically_impossible_divergence_directions() -> None:
    probes = build_probe_set("physically_impossible", "data", model="gpt-4.1")
    by_text = {p.text: p for p in probes}
    # ANIMATE subject but IMPOSSIBLE: true True, articulation (inanimate) False
    a = by_text["The man carried the bridge home"]
    assert a.true_label is True and a.art_label is False and a.is_divergence
    assert a.family == "A_animate_impossible"
    assert "clean" in a.clean_status
    # Regression: runner is animate, so this semantic Family-A item is
    # discriminating in fresh probe builds. Older raw Step-3 runs preserve their
    # logged art_label during no-API re-analysis.
    runner = by_text["The runner outran the speed of light"]
    assert runner.true_label is True and runner.art_label is False and runner.is_divergence
    assert runner.family == "A_animate_impossible"
    # INANIMATE subject but POSSIBLE: true False, articulation (inanimate) True
    b = by_text["The statue stood in the park all year"]
    assert b.true_label is False and b.art_label is True and b.is_divergence
    assert b.family == "B_inanimate_possible"
    assert "contested" in b.clean_status


def test_second_word_capitalized_divergence_directions() -> None:
    probes = build_probe_set("second_word_capitalized", "data", model="gpt-4.1")
    by_text = {p.text: p for p in probes}
    # capitalized NON-proper 2nd word: TRUE rule True, articulation (proper) False
    p = by_text["They Walked home after dinner"]
    assert p.true_label is True  # 'Walked' starts uppercase
    assert p.art_label is False  # 'Walked' is not a known proper noun
    assert p.is_divergence
    # proper-noun 2nd word: both readings True (NOT divergence)
    q = by_text["Then Karen closed the table by day"]
    assert q.true_label is True and q.art_label is True and not q.is_divergence


def test_word_count_divergence_directions() -> None:
    probes = build_probe_set("word_count_geq_8", "data", model="gpt-4.1")
    by_text = {p.text: p for p in probes}
    # short but prepositional: true False (<8 words), articulation True
    short = by_text["The dog slept at home"]
    assert short.true_label is False and short.art_label is True and short.is_divergence
    # long but no preposition/adverb: true True (>=8 words), articulation False
    long = by_text["The cheerful young woman quietly thanked the generous old baker"]
    assert long.true_label is True and long.art_label is False and long.is_divergence


def test_probe_ids_unique_and_texts_unique() -> None:
    for rule_id in TARGET_RULES:
        probes = build_probe_set(rule_id, "data", model="gpt-4.1")
        ids = [p.probe_id for p in probes]
        texts = [p.text for p in probes]
        assert len(ids) == len(set(ids))
        assert len(texts) == len(set(texts))


def test_articulation_for_known_and_dated_models() -> None:
    art = articulation_for("physically_impossible", "gpt-4.1")
    assert "inanimate" in art.lower()
    # dated model id falls back to the longest known prefix
    dated = articulation_for("food_topic", "gpt-4.1-mini-2025-04-14")
    assert dated == ARTICULATIONS["food_topic"]["gpt-4.1-mini"]


def test_articulation_for_unknown_model_raises() -> None:
    with pytest.raises(ProbeError):
        articulation_for("food_topic", "llama-3")


def test_build_unknown_rule_raises() -> None:
    with pytest.raises(ProbeError):
        build_probe_set("contains_digit", "data", model="gpt-4.1")


def test_parse_edge_example_pulls_quoted_sentence() -> None:
    idea = "capitalized non-proper second word ('They Walked home after dinner') — splits"
    assert parse_edge_example(idea) == "They Walked home after dinner"
    # a class-only idea with no usable quoted sentence -> None
    assert parse_edge_example("exactly 8 vs exactly 7 minimal pairs (same base)") is None


def test_articulation_predicate_models_differ_only_where_expected() -> None:
    # the probe set's art_label is model-specific (different stated rule per
    # model) only insofar as the articulation PREDICATE differs; here both
    # models share the predicate, so divergence tagging is identical.
    a = build_probe_set("food_topic", "data", model="gpt-4.1")
    b = build_probe_set("food_topic", "data", model="gpt-4.1-mini")
    assert [p.art_label for p in a] == [p.art_label for p in b]
