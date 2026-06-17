"""Offline tests for the AUDIT-RATE-AWARE selection refinement (rules 15/16).

ZERO network: every test drives the shared pipeline through the deterministic,
network-free mock seam (``mock_generator`` / ``mock_labeler``), wrapped so the
candidate POOL carries a controlled distribution of the audited rate-bearing
tokens (negators). No OPENAI_API_KEY, no HTTP, no nltk download.

The headline behaviour under test (the rule-15 negator_rate=0.0511 reject):
the within-cell down-sample is NEGATOR-BLIND, so a random draw can land on a
marginal-but-failing subset even when a COMPLIANT one exists in the pool. The
refinement (``refine_rate_audits``) holds the exact topic x class balance FIXED
and swaps selected items for same-cell validated leftovers to drive the audited
rate gaps under their spec thresholds.

  * FEASIBLE pool (a negator-free False subset exists) -> the pipeline EMITS and
    the confound audit passes (refinement found the compliant subset).
  * INFEASIBLE pool (EVERY False item carries a negator, no True item does) ->
    the pipeline RAISES LLMPipelineError naming negator_rate with the achieved
    value (the binding threshold is surfaced, never silently accepted).
  * DETERMINISM: same seed + same wrapped seam -> byte-identical emitted items.
  * The refinement is PURE SELECTION over the already-validated pool (it never
    asks the generator/validators for anything) -> cache-preserving.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from icl_articulation.datagen.confound import _negator_rate, build_confound_report
from icl_articulation.datagen.generators.llm import (
    get_rule_config,
    mock_generator,
    mock_labeler,
    run_pipeline,
)
from icl_articulation.datagen.generators.llm.pipeline import (
    Candidate,
    LLMPipelineError,
    refine_rate_audits,
)
from icl_articulation.datagen.schema import read_items, words

RULE = "positive_sentiment"


# --- helpers ------------------------------------------------------------------


def _unique_id(line: str) -> int:
    """The trailing 'item <uid>' number the mock embeds (its only per-item key)."""
    toks = line.split()
    return int(toks[-1]) if toks and toks[-1].isdigit() else 0


def _inject_negator(line: str) -> str:
    """Replace the FIRST lowercase 'the' with the negator 'no'. Word count is
    preserved (a 1-token-for-1-token swap), the embedded class marker is untouched
    (so ``mock_labeler`` still recovers the class), and style stays valid."""
    toks = line.split()
    for i, t in enumerate(toks):
        if t == "the":
            toks[i] = "no"
            return " ".join(toks)
    # no lowercase 'the' (rare scaffold) -> leave unchanged; the cell still has
    # plenty of negator-free items, so the audit is unaffected.
    return line


def _negator_gen(fraction_false_carrying: float):
    """A mock generator wrapper that puts a negator on a CONTROLLED fraction of
    FALSE (negative) candidates and NONE of the True ones, keyed deterministically
    on the embedded unique id (so it is reproducible across calls and processes).
    True items never carry a negator, so the natural rate gap is wholly driven by
    the False fraction this controls."""

    def _g(rule_id, label, topic, n, *, seed):
        lines = mock_generator(rule_id, label, topic, n, seed=seed)
        if label:
            return lines  # True class: never a negator
        out = []
        for line in lines:
            # deterministic ~fraction by the unique id's last two digits
            carries = (_unique_id(line) % 100) < int(round(fraction_false_carrying * 100))
            out.append(_inject_negator(line) if carries else line)
        return out

    return _g


def _all_false_negator_gen():
    """Every FALSE item carries a negator, no True item does -> an INFEASIBLE pool
    for the negator-rate gate (the False class can never get below ~1.0 while True
    sits at 0)."""

    def _g(rule_id, label, topic, n, *, seed):
        lines = mock_generator(rule_id, label, topic, n, seed=seed)
        return lines if label else [_inject_negator(line) for line in lines]

    return _g


# --- the headline feasible / infeasible / determinism behaviour ---------------


def test_refinement_emits_when_compliant_subset_exists(tmp_path: Path) -> None:
    """A pool where ~25% of the False candidates carry a negator (so a random,
    negator-blind down-sample would land WELL over the 0.05 gap) but a negator-free
    False subset clearly exists. The rate-aware refinement must select that subset
    so the dataset EMITS and the confound audit passes."""
    gen = _negator_gen(0.25)
    result = run_pipeline(
        RULE, gen, mock_labeler, seed=0, max_candidates=600,
        data_dir=tmp_path, run_pos=False,
    )
    # it EMITTED and the whole confound audit (incl. the negator-rate hard gate)
    # passed — the refinement found the compliant subset.
    assert result.confound_overall_pass is True
    items_path = tmp_path / RULE / "items.jsonl"
    assert items_path.is_file()
    items = read_items(items_path)

    # the realized negator-rate gap is under the spec threshold (0.05)
    trues = [it for it in items if bool(it["label"])]
    falses = [it for it in items if not bool(it["label"])]
    gap = abs(_negator_rate(trues) - _negator_rate(falses))
    assert gap <= 0.05 + 1e-9, gap

    # an independent confound report agrees the negator gate passes
    topics = [it["slots_meta"]["topic"] for it in items]
    report = build_confound_report(items, is_llm_rule=True, run_pos=False, topics=topics)
    neg = report["audit_thresholds"]["checks"]["negator_rate_abs_diff"]
    assert neg["passes"] is True, neg


def test_random_selection_would_have_failed_without_refinement(tmp_path: Path) -> None:
    """Sanity that the test pool is genuinely adversarial: the un-refined,
    negator-blind selection that the same pool yields DOES violate the gate (so the
    emit success above is the refinement's doing, not a benign pool)."""
    import icl_articulation.datagen.generators.llm.pipeline as pipe

    gen = _negator_gen(0.25)
    # disable the refinement (identity) and confirm the dataset is REJECTED
    orig = pipe.refine_rate_audits
    pipe.refine_rate_audits = lambda cfg, selected, pool, seed, **kw: list(selected)
    try:
        with pytest.raises(LLMPipelineError) as ei:
            run_pipeline(RULE, gen, mock_labeler, seed=0, max_candidates=600,
                         data_dir=tmp_path, run_pos=False)
    finally:
        pipe.refine_rate_audits = orig
    assert "negator" in str(ei.value).lower()
    assert not (tmp_path / RULE / "items.jsonl").exists()


def test_refinement_raises_on_infeasible_pool(tmp_path: Path) -> None:
    """Every False item carries a negator and no True item does: NO compliant
    subset exists. The pipeline must RAISE LLMPipelineError naming negator_rate
    with the achieved value (the binding threshold is surfaced, not relaxed)."""
    gen = _all_false_negator_gen()
    with pytest.raises(LLMPipelineError) as ei:
        run_pipeline(RULE, gen, mock_labeler, seed=0, max_candidates=600,
                     data_dir=tmp_path, run_pos=False)
    msg = str(ei.value).lower()
    assert "negator" in msg
    # nothing was written
    assert not (tmp_path / RULE / "items.jsonl").exists()


def test_refinement_is_deterministic(tmp_path: Path) -> None:
    """Same seed + same wrapped seam -> byte-identical emitted texts (the swap loop
    is fully seed-driven)."""
    gen = _negator_gen(0.25)
    a = tmp_path / "a"
    b = tmp_path / "b"
    run_pipeline(RULE, gen, mock_labeler, seed=11, max_candidates=600, data_dir=a, run_pos=False)
    run_pipeline(RULE, gen, mock_labeler, seed=11, max_candidates=600, data_dir=b, run_pos=False)
    ta = [it["text"] for it in read_items(a / RULE / "items.jsonl")]
    tb = [it["text"] for it in read_items(b / RULE / "items.jsonl")]
    assert ta == tb
    assert len(ta) > 0


# --- the refinement primitive in isolation ------------------------------------


def _mk(text: str, label: bool, topic: str = "t") -> Candidate:
    return Candidate(text=text, intended_label=label, topic=topic)


def test_refine_swaps_only_within_cell_and_preserves_counts() -> None:
    """``refine_rate_audits`` holds every cell's selected COUNT fixed (so topic x
    class + 50/50 balance is preserved) and only ever swaps in same-cell leftovers.
    Build one True cell (negator-free) and one False cell whose SELECTED items all
    carry a negator while the LEFTOVERS are negator-free; the refinement must swap
    the negators out, closing the gap, without changing per-cell counts."""
    cfg = get_rule_config(RULE)
    # restrict to a single topic per class for a tight, readable cell structure
    import dataclasses

    cfg = dataclasses.replace(cfg, true_topics=("restaurants",), false_topics=("restaurants",))

    # True cell: 4 negator-free selected
    true_sel = [_mk(f"the visit there felt lovely today number {i}", True, "restaurants") for i in range(4)]
    # False cell: 4 selected ALL carry 'no', plus 6 negator-free leftovers
    false_sel = [_mk(f"no visit there felt bland today number {i}", False, "restaurants") for i in range(4)]
    false_left = [_mk(f"the visit there felt dull today number {i}", False, "restaurants") for i in range(10, 16)]

    selected = true_sel + false_sel
    pool = selected + false_left  # leftovers are in the pool but not selected

    # before: False negator rate 1.0, True 0.0 -> gap 1.0 (fails)
    assert abs(_negator_rate([{"text": c.text, "label": c.intended_label} for c in false_sel])
               - _negator_rate([{"text": c.text, "label": c.intended_label} for c in true_sel])) == 1.0

    refined = refine_rate_audits(cfg, selected, pool, seed=0)

    # same total + same per-cell counts (4 True, 4 False)
    assert len(refined) == len(selected)
    assert sum(1 for c in refined if c.intended_label) == 4
    assert sum(1 for c in refined if not c.intended_label) == 4
    # every refined item is from the original cell's pool
    pool_texts = {c.text for c in pool}
    assert all(c.text in pool_texts for c in refined)

    # the negator gap is now closed (False cell selected the negator-free leftovers)
    rt = _negator_rate([{"text": c.text, "label": c.intended_label} for c in refined if c.intended_label])
    rf = _negator_rate([{"text": c.text, "label": c.intended_label} for c in refined if not c.intended_label])
    assert abs(rt - rf) <= 0.05 + 1e-9


def test_refine_noop_when_already_compliant() -> None:
    """When the selection already passes every rate gate, refinement returns it
    unchanged (no needless churn) — same items, same order length."""
    cfg = get_rule_config(RULE)
    import dataclasses

    cfg = dataclasses.replace(cfg, true_topics=("restaurants",), false_topics=("restaurants",))
    sel = (
        [_mk(f"the visit there felt lovely today number {i}", True, "restaurants") for i in range(4)]
        + [_mk(f"the visit there felt bland today number {i}", False, "restaurants") for i in range(4)]
    )
    refined = refine_rate_audits(cfg, sel, sel, seed=0)
    assert [c.text for c in refined] == [c.text for c in sel]
