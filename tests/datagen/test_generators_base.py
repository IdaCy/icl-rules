"""Tests for the shared gated emit pipeline (generators.base) + the reference
rule all_lowercase (generators.rules.all_lowercase).

Covers, with NO network and NO writes outside tmp_path:
  - load_rule_spec reads the committed extract and is loud on a bad rule_id;
  - the GENERATOR INTERFACE contract (build_bases / instantiate) is honoured by
    the reference rule (>= 340 distinct bases; True=lower, False=sentence-case);
  - the per-split variant pattern (few_shot BOTH variants; held_out / confirmation
    / spare one balanced variant) and exact 50/50 balance;
  - each of the four gates: schema (A), groundtruth (B), battery (C), confound (D)
    — both that they PASS on the reference data and that they RAISE loudly when
    their invariant is violated;
  - determinism (same seed -> byte-identical items + identical summary);
  - the registry dispatch + that nothing is written under write=False except the
    confound report.

The 6 nltk POS battery predicates run when nltk + its tagger data are present
(the CI/instance state); they are skipped via run_pos=False where a test does
not need them, so this file never downloads.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from icl_articulation.contexts import load_items
from icl_articulation.datagen import battery, groundtruth, schema
from icl_articulation.datagen.generators import base, registry
from icl_articulation.datagen.generators.base import PipelineError, emit_rule, load_rule_spec
from icl_articulation.datagen.generators.rules import all_lowercase as ref
from icl_articulation.datagen.genutils import Gen

RULE = "all_lowercase"


# --- helpers ------------------------------------------------------------------


def _emit(tmp_path, *, seed=1234, write=True, run_pos=False):
    """Run the reference rule through the pipeline into tmp_path (run_pos off by
    default to keep the body fast / nltk-free; POS-on is exercised separately)."""
    return registry.run(RULE, seed=seed, write=write, run_pos=run_pos, data_dir=str(tmp_path))


# --- spec loading -------------------------------------------------------------


def test_load_rule_spec_reads_committed_extract():
    spec = load_rule_spec(RULE)
    assert spec.rule_id == RULE
    # the rule's equivalence_class includes the lowercase string that exempts the
    # all_lowercase battery predicate
    assert "the sentence is written entirely in lowercase" in spec.equivalence_class
    assert spec.equiv_keys.get("all_lowercase") == [
        "the sentence is written entirely in lowercase"
    ]


def test_load_rule_spec_unknown_rule_is_loud():
    with pytest.raises(PipelineError):
        load_rule_spec("definitely_not_a_rule")


# --- the reference rule's GENERATOR INTERFACE ---------------------------------


def test_build_bases_floor_and_distinct():
    bases = ref.build_bases(Gen(1).derive("b"))
    assert len(bases) >= schema.PROGRAMMATIC_N_BASES_MIN
    base_ids = [b.base_id for b in bases]
    assert len(set(base_ids)) == len(base_ids)
    # base_id == the base sentence (rule-spec) and is 5-10 words, sentence case
    for b in bases:
        assert b.base_id == b.sentence
        assert 5 <= schema.word_count(b.sentence) <= 10
        assert b.sentence[0].isupper()


def test_instantiate_variants_have_correct_groundtruth():
    b = ref.build_bases(Gen(2).derive("b"))[0]
    g = Gen(2)
    t_text, t_meta = ref.instantiate(b, True, g)
    f_text, f_meta = ref.instantiate(b, False, g)
    # True variant is all-lowercase; False variant is sentence case
    assert t_text == t_text.lower()
    assert f_text != f_text.lower()
    # both reduce to the same base sentence ignoring case (character-identical)
    assert t_text.lower() == f_text.lower()
    # ground truth recomputed from text matches the intended label
    assert groundtruth.RULE_PREDICATES[RULE].label_of(t_text) is True
    assert groundtruth.RULE_PREDICATES[RULE].label_of(f_text) is False
    assert t_meta["transform"] == "lower" and f_meta["transform"] == "sentence_case"


def test_instantiate_is_deterministic():
    b = ref.build_bases(Gen(3).derive("b"))[0]
    a1 = ref.instantiate(b, True, Gen(3))
    a2 = ref.instantiate(b, True, Gen(3))
    assert a1 == a2


# --- the full pipeline: counts, balance, gates --------------------------------


def test_emit_rule_all_gates_pass_and_counts(tmp_path):
    summary = _emit(tmp_path, run_pos=False)
    assert summary.all_gates_pass
    assert summary.gate_schema and summary.gate_groundtruth
    assert summary.gate_battery and summary.gate_confound
    # the spec's programmatic split pattern, exactly
    sc = summary.split_counts
    assert sc["few_shot_pool"] == {"true": 100, "false": 100, "total": 200}
    assert sc["held_out"] == {"true": 60, "false": 60, "total": 120}
    assert sc["confirmation"] == {"true": 50, "false": 50, "total": 100}
    # spare is >= 20 bases, one variant each, balanced as parity allows
    assert sc["spare"]["total"] >= schema.PROGRAMMATIC_SPARE_MIN
    assert summary.n_items == sum(v["total"] for v in sc.values())
    # battery worst NON-exempt predicate stays well under the 0.75 floor
    assert summary.battery_max_agreement <= battery.PASS_THRESHOLD


def test_emitted_dataset_roundtrips_through_loader(tmp_path):
    summary = _emit(tmp_path, run_pos=False)
    items = load_items(summary.items_path)  # the runner-facing contract
    assert len(items) == summary.n_items
    # every item carries the run seed + transform provenance
    for it in items:
        assert it["slots_meta"]["seed"] == summary.seed
        assert it["slots_meta"]["transform"] in ("lower", "sentence_case")


def test_emit_rule_can_write_variant_dataset_id(tmp_path):
    variant = f"{RULE}_deconfounded"
    summary = registry.run(
        RULE,
        seed=1234,
        write=True,
        run_pos=False,
        data_dir=str(tmp_path),
        output_rule_id=variant,
        stored_rule_id=variant,
    )
    assert summary.rule_id == variant
    assert summary.all_gates_pass
    assert (tmp_path / variant / "items.jsonl").is_file()

    items = load_items(summary.items_path)
    assert len(items) == summary.n_items
    assert {it["rule_id"] for it in items} == {variant}
    groundtruth.assert_labels_correct(variant, items)


def test_few_shot_pool_has_both_variants_per_base(tmp_path):
    summary = _emit(tmp_path, run_pos=False)
    items = schema.read_items(summary.items_path)
    fs = [it for it in items if it["split"] == "few_shot_pool"]
    by_base: dict[str, set] = {}
    for it in fs:
        by_base.setdefault(it["base_id"], set()).add(bool(it["label"]))
    assert len(by_base) == 100
    assert all(v == {True, False} for v in by_base.values())


def test_one_variant_splits_have_one_variant_per_base(tmp_path):
    summary = _emit(tmp_path, run_pos=False)
    items = schema.read_items(summary.items_path)
    for split in ("held_out", "confirmation", "spare"):
        per_base: dict[str, int] = {}
        for it in items:
            if it["split"] == split:
                per_base[it["base_id"]] = per_base.get(it["base_id"], 0) + 1
        assert per_base and all(c == 1 for c in per_base.values())


def test_determinism_byte_identical(tmp_path):
    s1 = _emit(tmp_path / "a", run_pos=False)
    s2 = _emit(tmp_path / "b", run_pos=False)
    assert s1.as_dict() == s2.as_dict() or (
        # paths differ; compare everything except the two path fields
        {k: v for k, v in s1.as_dict().items() if "path" not in k}
        == {k: v for k, v in s2.as_dict().items() if "path" not in k}
    )
    raw_a = (tmp_path / "a" / RULE / "items.jsonl").read_text()
    raw_b = (tmp_path / "b" / RULE / "items.jsonl").read_text()
    assert raw_a == raw_b


def test_no_write_skips_items_but_writes_report(tmp_path):
    summary = _emit(tmp_path, write=False, run_pos=False)
    assert summary.all_gates_pass
    assert not (tmp_path / RULE / "items.jsonl").exists()
    assert (tmp_path / RULE / "confound_report.json").exists()


# --- gate D artifact ----------------------------------------------------------


def test_confound_report_written_and_passes(tmp_path):
    summary = _emit(tmp_path, run_pos=False)
    report = json.loads((tmp_path / RULE / "confound_report.json").read_text())
    assert report["overall_pass"] is True
    assert report["length_match_ok"] is True
    assert report["word_count_mean_abs_diff"] <= confound_tol()
    assert report["battery_violations"] == []


def confound_tol() -> float:
    from icl_articulation.datagen.confound import PROGRAMMATIC_WC_TOL

    return PROGRAMMATIC_WC_TOL


# --- gates RAISE loudly on a violated invariant -------------------------------


def test_gate_groundtruth_raises_on_label_flip(tmp_path):
    """An instantiate that mislabels (emits an all-lowercase text for the False
    variant of a UNIQUE base) must be caught — by Gate B groundtruth."""

    def lying_instantiate(spec, label, gen):
        # always emit the all-lowercase text, but report it under both labels
        return spec.sentence.lower() + (" x" if not label else ""), {"transform": "lie"}

    with pytest.raises((PipelineError, groundtruth.GroundTruthError, schema.SchemaError)):
        emit_rule(
            RULE, ref.build_bases, lying_instantiate, 5, write=False,
            run_pos=False, data_dir=str(tmp_path),
        )


def test_gate_battery_raises_on_confound(tmp_path):
    """An instantiate that injects a digit into every True variant makes the
    'contains_digit' predicate a perfect cue -> Gate C must raise."""

    def confounding_instantiate(spec, label, gen):
        if label:
            return ("7 " + spec.sentence.lower()), {"transform": "confound"}
        return spec.sentence, {"transform": "sentence_case"}

    with pytest.raises(PipelineError, match="battery"):
        emit_rule(
            RULE, ref.build_bases, confounding_instantiate, 6, write=False,
            run_pos=False, data_dir=str(tmp_path),
        )


def test_pipeline_raises_when_too_few_bases(tmp_path):
    def short_build(gen):
        return ref.build_bases(gen)[:10]

    with pytest.raises(PipelineError, match="bases"):
        emit_rule(
            RULE, short_build, ref.instantiate, 7, write=False,
            run_pos=False, data_dir=str(tmp_path),
        )


def test_pipeline_raises_on_duplicate_base_ids(tmp_path):
    def dup_build(gen):
        bs = ref.build_bases(gen)
        return bs[:-1] + [replace(bs[0])]  # a duplicate base_id

    with pytest.raises(PipelineError, match="duplicate"):
        emit_rule(
            RULE, dup_build, ref.instantiate, 8, write=False,
            run_pos=False, data_dir=str(tmp_path),
        )


# --- registry -----------------------------------------------------------------


def test_registry_lists_and_dispatches_reference_rule():
    assert RULE in registry.registered_rules()
    bb, inst = registry.get_generator(RULE)
    assert bb is ref.build_bases and inst is ref.instantiate


def test_registry_unknown_rule_is_loud():
    with pytest.raises(registry.RegistryError):
        registry.get_module("not_a_registered_rule")


# --- POS predicates run for real when nltk is present -------------------------


def test_battery_pos_predicates_run_when_nltk_present(tmp_path):
    pytest.importorskip("nltk")
    try:
        from nltk import pos_tag

        pos_tag(["ok"])  # tagger data present?
    except LookupError:
        pytest.skip("nltk tagger data not installed")
    summary = _emit(tmp_path, run_pos=True)
    assert summary.all_gates_pass
    assert summary.battery_pos_ran is True
    assert summary.battery_max_agreement <= battery.PASS_THRESHOLD
