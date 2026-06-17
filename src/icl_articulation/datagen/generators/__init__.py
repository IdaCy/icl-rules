"""Per-rule dataset generators built on the B0 framework (phases B1-B4).

The B0 framework (banks, schema, battery, confound, groundtruth, genutils) is
DONE and ACCEPTED. This package adds the SHARED, GATED emit pipeline that every
per-rule generator runs through (``base.emit_rule``) plus the per-rule generator
modules under ``rules/`` and the ``registry`` that maps a rule_id to its module.

A per-rule generator is a tiny module exposing exactly two callables that
conform to the GENERATOR INTERFACE documented in ``base`` (and re-stated here):

    build_bases(gen: Gen) -> list[BaseSpec]
        Return >= PROGRAMMATIC_N_BASES_MIN (340) distinct base specs. Each
        BaseSpec carries everything needed to instantiate BOTH its True and its
        False variant deterministically (no further randomness at instantiate
        time beyond what ``gen`` provides). A BaseSpec is any object; the
        pipeline only requires that ``instantiate`` understands it and that the
        ``base_id`` it yields is stable and distinct across specs. The canonical
        shape (used by the reference rule) is a dataclass with a ``base_id``
        field and the fields the instantiate function needs.

    instantiate(spec: BaseSpec, label: bool, gen: Gen) -> tuple[str, dict]
        Return (text, slots_meta) for ONE variant (``label``) of ``spec``. The
        text MUST be the surface string whose ground-truth label (recomputed
        from text by groundtruth.assert_labels_correct) equals ``label``.
        slots_meta MUST be a JSON-serialisable dict recording provenance: at
        minimum the seed and which slots / transform produced the item. The
        function MUST be deterministic given (spec, label, gen).

Run a rule end-to-end as a CLI:

    python -m icl_articulation.datagen.generators <rule_id>

which dispatches via ``registry`` to that rule's module and runs the full
gated pipeline (Gate A schema, Gate B groundtruth, Gate C battery, Gate D
confound), writing data/<rule_id>/items.jsonl + data/<rule_id>/confound_report.json
only if ALL gates pass.
"""

from __future__ import annotations

from .base import EmitSummary, PipelineError, emit_rule, load_rule_spec

__all__ = ["EmitSummary", "PipelineError", "emit_rule", "load_rule_spec"]
