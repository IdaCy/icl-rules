"""Faithfulness analysis tests: agreement + Wilson CI, the divergence-subset
contrast (behavior-vs-true vs behavior-vs-self), and parse-failure handling.

Pure functions, no network."""

from __future__ import annotations

from icl_articulation.faithfulness import (
    ARM_IN_CONTEXT,
    ARM_SELF,
    ARM_TRUE,
    ArmPredictions,
    agreement,
    analyze,
    analyze_rule,
    build_all_probe_sets,
    corrected_divergence_analysis,
    designed_divergence_mask,
    discriminating_mask,
)
from icl_articulation.step3_probes import Probe


def _probe(rule: str, i: int, true: bool, art: bool, divergence_source: bool = True) -> Probe:
    return Probe(
        rule_id=rule,
        probe_id=f"{rule}-{i:03d}",
        text=f"probe {i}",
        true_label=true,
        art_label=art,
        source="divergence" if divergence_source else "in_distribution",
        true_label_source="hand",
        note="",
    )


# --- agreement + Wilson CI ----------------------------------------------------


def test_agreement_perfect_and_partial() -> None:
    full = agreement([True, False, True], [True, False, True])
    assert full["rate"] == 1.0 and full["n"] == 3 and full["agree"] == 3
    half = agreement([True, True, False, False], [True, False, False, True])
    assert half["rate"] == 0.5 and half["n"] == 4 and half["agree"] == 2
    assert half["ci_low"] < 0.5 < half["ci_high"]  # Wilson CI brackets the point


def test_agreement_drops_none_pairs_from_denominator() -> None:
    a = [True, None, False, True]
    b = [True, False, None, False]
    out = agreement(a, b)
    assert out["n"] == 2  # pairs 0 and 3 are comparable
    assert out["n_unparsed"] == 2
    assert out["agree"] == 1  # pair 0 agrees, pair 3 disagrees
    assert out["rate"] == 0.5


def test_agreement_all_none_has_no_rate() -> None:
    out = agreement([None, None], [True, False])
    assert out["rate"] is None and out["n"] == 0 and out["n_unparsed"] == 2


# --- the divergence-subset contrast -------------------------------------------


def _build(rule: str, ic, sa, tg, probes) -> ArmPredictions:
    return ArmPredictions(rule, probes, ic, sa, tg)


def test_behavior_tracks_true_is_unfaithful() -> None:
    # All divergence probes; behavior == true_label, self == articulation.
    probes = [_probe("r", i, true=(i % 2 == 0), art=(i % 2 == 1)) for i in range(10)]
    ic = [p.true_label for p in probes]  # behavior follows what it LEARNED
    sa = [p.art_label for p in probes]  # self-application follows what it SAID
    tg = [p.true_label for p in probes]
    rm = analyze_rule(_build("r", ic, sa, tg, probes))
    assert rm["n_divergence"] == 10
    h = rm["legacy_empirical_divergence"]
    assert h["behavior_tracks_true_rate"] == 1.0
    assert h["behavior_tracks_self_rate"] == 0.0
    assert h["gap_true_minus_self"] == 1.0
    assert "unfaithful" in h["interpretation"]
    # faithfulness = behavior-vs-self agreement, which is 0 on a pure-divergence set
    assert rm["divergence"]["faithfulness_behavior_vs_self"]["rate"] == 0.0


def test_behavior_tracks_self_is_faithful() -> None:
    probes = [_probe("r", i, true=(i % 2 == 0), art=(i % 2 == 1)) for i in range(10)]
    ic = [p.art_label for p in probes]  # behavior follows what it SAID
    sa = [p.art_label for p in probes]
    tg = [p.true_label for p in probes]
    rm = analyze_rule(_build("r", ic, sa, tg, probes))
    h = rm["legacy_empirical_divergence"]
    assert h["behavior_tracks_true_rate"] == 0.0
    assert h["behavior_tracks_self_rate"] == 1.0
    assert h["gap_true_minus_self"] == -1.0
    assert "faithful" in h["interpretation"]


def test_no_divergence_subset_means_no_headline() -> None:
    # ARM-2 self-application agrees with true everywhere -> empty EMPIRICAL
    # divergence subset, even though some probes carry a build-time proxy tag.
    probes = [_probe("r", i, true=True, art=True, divergence_source=False) for i in range(6)]
    ic = [True] * 6
    rm = analyze_rule(_build("r", ic, ic, ic, probes))
    assert rm["n_divergence"] == 0
    assert rm["divergence"] is None
    assert rm["legacy_empirical_divergence"] is None
    assert rm["overall"]["faithfulness_behavior_vs_self"]["rate"] == 1.0


# --- the EMPIRICAL divergence subset (arm2 != true), not the build-time tag ----


def test_empirical_divergence_subset_is_arm2_disagrees_with_true() -> None:
    """The analysis divergence subset is defined by the MODEL's own ARM-2
    self-application disagreeing with true_label, NOT by the build-time proxy
    tag. Construct a set where the two DIFFER and assert the analysis picks the
    empirical one."""
    from icl_articulation.faithfulness import empirical_divergence_mask

    # 6 probes. Build-time proxy tag (true != art) is set on probes 0..2.
    # We make ARM-2 self-application disagree with true on a DIFFERENT subset:
    #   probes 3,4 -> arm2 != true (the EMPIRICAL divergence subset)
    #   probes 0,1,2,5 -> arm2 == true (not empirical divergence)
    probes = [
        _probe("r", 0, true=True, art=False),   # proxy-tagged, arm2 will == true
        _probe("r", 1, true=True, art=False),   # proxy-tagged, arm2 will == true
        _probe("r", 2, true=True, art=False),   # proxy-tagged, arm2 will == true
        _probe("r", 3, true=True, art=True),    # NOT proxy-tagged, arm2 != true
        _probe("r", 4, true=False, art=False),  # NOT proxy-tagged, arm2 != true
        _probe("r", 5, true=True, art=True),    # NOT proxy-tagged, arm2 == true
    ]
    true = [p.true_label for p in probes]
    # ARM-2 self-application: matches true except on probes 3 and 4 (flip).
    sa = list(true)
    sa[3] = not sa[3]
    sa[4] = not sa[4]
    preds = _build("r", true, sa, true, probes)

    mask = empirical_divergence_mask(preds)
    assert mask == [False, False, False, True, True, False]

    rm = analyze_rule(preds)
    # EMPIRICAL subset size is 2 (probes 3,4); the build-time proxy tagged 3.
    assert rm["n_divergence"] == 2
    assert rm["n_divergence_proxy"] == 3
    assert rm["divergence_definition"].startswith("empirical")


def test_empirical_divergence_gap_behavior_true_vs_self() -> None:
    """On the empirical-divergence subset, behavior~true and behavior~self are
    computed over exactly the arm2!=true probes, and the gap is right."""
    # 5 probes; arm2 disagrees with true on probes 0,1,2 (the empirical subset).
    probes = [_probe("r", i, true=True, art=True) for i in range(5)]
    true = [p.true_label for p in probes]
    sa = [False, False, False, True, True]  # arm2: divergence on 0,1,2
    # behavior (arm1) tracks the TRUE rule on the empirical subset (acts on what
    # it learned) -> unfaithful. On 2 of 3 subset probes behavior==true.
    ic = [True, True, False, True, True]    # probe 2 behavior != true
    preds = _build("r", ic, sa, true, probes)

    rm = analyze_rule(preds)
    assert rm["n_divergence"] == 3  # probes 0,1,2
    h = rm["legacy_empirical_divergence"]
    # behavior~true over {0,1,2}: agree on 0,1 (True==True), disagree on 2 -> 2/3
    assert h["behavior_tracks_true_rate"] == 2 / 3
    # behavior~self over {0,1,2}: ic=[T,T,F] vs sa=[F,F,F] -> agree only on 2 -> 1/3
    assert h["behavior_tracks_self_rate"] == 1 / 3
    assert abs(h["gap_true_minus_self"] - (2 / 3 - 1 / 3)) < 1e-12
    # gap > 0.1 -> unfaithful
    assert "unfaithful" in h["interpretation"]
    # Wilson CIs are attached to both rates.
    assert h["behavior_tracks_true_ci"][0] <= 2 / 3 <= h["behavior_tracks_true_ci"][1]


def test_empirical_divergence_drops_none_arm2() -> None:
    """A None ARM-2 prediction carries no divergence signal -> not in the subset."""
    from icl_articulation.faithfulness import empirical_divergence_mask

    probes = [_probe("r", i, true=True, art=True) for i in range(3)]
    # arm2: probe 0 disagrees (False!=True), probe 1 is None, probe 2 agrees.
    sa = [False, None, True]
    preds = _build("r", [True, True, True], sa, [True, True, True], probes)
    assert empirical_divergence_mask(preds) == [True, False, False]
    rm = analyze_rule(preds)
    assert rm["n_divergence"] == 1


# --- the CORRECTED designed-divergence analysis (author art_label, not arm-2) --


def test_corrected_masks_select_designed_and_discriminating() -> None:
    probes = [
        _probe("r", 0, true=True, art=False, divergence_source=True),    # designed + discriminating
        _probe("r", 1, true=True, art=True, divergence_source=True),     # designed, NOT discriminating
        _probe("r", 2, true=False, art=False, divergence_source=False),  # in_distribution
    ]
    preds = _build("r", [True] * 3, [True] * 3, [True] * 3, probes)
    assert designed_divergence_mask(preds) == [True, True, False]
    assert discriminating_mask(preds) == [True, False, False]


def test_corrected_tracks_true_is_unfaithful() -> None:
    # discriminating designed probes, behaviour follows the TRUE rule, and
    # behaviour is non-constant (true alternates) so it is not flagged constant.
    probes = [_probe("r", i, true=(i % 2 == 0), art=(i % 2 == 1)) for i in range(12)]
    ic = [p.true_label for p in probes]
    sa = [p.art_label for p in probes]
    cd = corrected_divergence_analysis(_build("r", ic, sa, ic, probes))
    assert cd["discriminating"]["n"] == 12
    assert cd["discriminating"]["behaviour_tracks_true_rate"] == 1.0
    assert cd["discriminating"]["binom_p_two_sided_vs_chance"] < 0.001
    assert not cd["behaviour_distribution"]["is_constant"]
    assert "unfaithful" in cd["verdict"]


def test_corrected_tracks_stated_is_faithful() -> None:
    probes = [_probe("r", i, true=(i % 2 == 0), art=(i % 2 == 1)) for i in range(12)]
    ic = [p.art_label for p in probes]  # behaviour follows the STATED rule
    sa = [p.art_label for p in probes]
    cd = corrected_divergence_analysis(_build("r", ic, sa, ic, probes))
    assert cd["discriminating"]["behaviour_tracks_true_rate"] == 0.0
    assert "faithful" in cd["verdict"]


def test_corrected_direction_imbalanced_default_is_not_cleanly_stated() -> None:
    # second_word_capitalized shape: most discriminating probes are one direction
    # and behaviour mostly defaults False. Low track-true is not enough to claim
    # clean stated-rule tracking.
    probes = (
        [_probe("r", i, true=True, art=False) for i in range(7)]
        + [_probe("r", 10, true=False, art=True)]
    )
    ic = [False, False, False, False, False, False, True, False]
    sa = [p.art_label for p in probes]
    cd = corrected_divergence_analysis(_build("r", ic, sa, ic, probes))
    assert cd["discriminating"]["behaviour_tracks_true_rate"] == 0.25
    assert cd["behaviour_distribution"]["majority_frac"] == 0.875
    assert "stated/default ambiguous" in cd["verdict"]


def test_corrected_constant_behaviour_tracks_neither() -> None:
    # word_count_geq_8 pattern: true balanced, behaviour constant False -> the
    # learned feature does not generalise OOD; tracks neither rule.
    probes = [_probe("r", i, true=(i % 2 == 0), art=(i % 2 == 1)) for i in range(16)]
    ic = [False] * 16   # constant
    sa = [p.art_label for p in probes]
    cd = corrected_divergence_analysis(_build("r", ic, sa, ic, probes))
    assert cd["behaviour_distribution"]["is_constant"]
    assert cd["discriminating"]["behaviour_tracks_true_rate"] == 0.5  # chance, meaningless
    assert "NEITHER" in cd["verdict"]


def test_corrected_by_direction_splits_families() -> None:
    # physically_impossible shape: Family A (true=T/stated=F) animate-impossible,
    # Family B (true=F/stated=T) inanimate-possible. behaviour tracks true in both
    # but self-application endorses the stated label only in family A here.
    probes = (
        [_probe("r", i, true=True, art=False) for i in range(4)]      # family A
        + [_probe("r", 10 + i, true=False, art=True) for i in range(3)]  # family B
    )
    ic = [p.true_label for p in probes]                 # behaviour tracks true everywhere
    sa = [p.art_label for p in probes[:4]] + [not p.art_label for p in probes[4:]]  # B self != stated
    cd = corrected_divergence_analysis(_build("r", ic, sa, ic, probes))
    bd = {d["true_label"]: d for d in cd["discriminating"]["by_direction"]}
    assert bd[True]["n"] == 4 and bd[True]["behaviour_tracks_true_n"] == 4
    assert bd[True]["self_endorses_stated_n"] == 4          # family A: self endorses stated
    assert bd[False]["n"] == 3 and bd[False]["behaviour_tracks_true_n"] == 3
    assert bd[False]["self_endorses_stated_n"] == 0          # family B: self does NOT endorse stated


def test_corrected_reports_family_breakdown() -> None:
    probes = [
        Probe(
            rule_id="r",
            probe_id=f"r-{i}",
            text=f"probe {i}",
            true_label=True,
            art_label=False,
            source="divergence",
            true_label_source="hand",
            family="A",
            clean_status="clean",
        )
        for i in range(3)
    ]
    ic = [True, True, False]
    sa = [False, False, False]
    cd = corrected_divergence_analysis(_build("r", ic, sa, ic, probes))
    assert cd["families"][0]["family"] == "A"
    assert cd["families"][0]["clean_statuses"] == ["clean"]
    assert cd["families"][0]["n"] == 3
    assert cd["families"][0]["n_discriminating"] == 3


def test_corrected_control_when_no_discriminating() -> None:
    # food_topic pattern: stated rule == true rule on the designed set.
    probes = [_probe("r", i, true=(i % 2 == 0), art=(i % 2 == 0)) for i in range(8)]
    ic = [p.true_label for p in probes]
    cd = corrected_divergence_analysis(_build("r", ic, ic, ic, probes))
    assert cd["discriminating"]["n"] == 0
    assert "control" in cd["verdict"]


def test_corrected_self_application_reliability_reported() -> None:
    probes = [_probe("r", i, true=(i % 2 == 0), art=(i % 2 == 1)) for i in range(10)]
    ic = [p.true_label for p in probes]
    sa = [p.art_label for p in probes]   # arm-2 perfectly applies the stated rule
    cd = corrected_divergence_analysis(_build("r", ic, sa, ic, probes))
    assert cd["self_application_reliability"]["self_vs_stated"]["rate"] == 1.0
    # against the true label, the same arm-2 disagrees everywhere (true != stated)
    assert cd["self_application_reliability"]["self_vs_true"]["rate"] == 0.0


def test_analyze_rule_exposes_corrected_block_as_primary() -> None:
    probes = [_probe("r", i, true=(i % 2 == 0), art=(i % 2 == 1)) for i in range(10)]
    ic = [p.true_label for p in probes]
    sa = [p.art_label for p in probes]
    rm = analyze_rule(_build("r", ic, sa, ic, probes))
    assert "corrected_designed_divergence" in rm
    assert "unfaithful" in rm["corrected_designed_divergence"]["verdict"]
    # the empirical view is still present, explicitly demoted to legacy/audit
    assert "legacy_empirical_divergence" in rm


# --- analyze() over real probe sets -------------------------------------------


def test_analyze_over_real_probe_sets_unfaithful_simulation() -> None:
    sets = build_all_probe_sets(data_dir="data", model="gpt-4.1")
    per_rule = {}
    for rid, probes in sets.items():
        ic = [p.true_label for p in probes]  # behavior = learned (true) rule
        sa = [p.art_label for p in probes]  # self = articulation
        tg = [p.true_label for p in probes]
        per_rule[rid] = ArmPredictions(rid, probes, ic, sa, tg)
    out = analyze(per_rule)
    assert set(out["rules"]) == set(sets)
    for row in out["summary"]:
        # on the divergence subset, behavior (true rule) disagrees with self
        # (articulation) by construction -> gap == 1.0, unfaithful.
        assert row["legacy_empirical_selection_biased"] is True
        assert row["legacy_empirical_gap_true_minus_self"] == 1.0
        assert "unfaithful" in row["legacy_empirical_interpretation"]
