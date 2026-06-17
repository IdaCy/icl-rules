"""The single SHARED, GATED emit pipeline every per-rule generator runs through.

``emit_rule`` is the one entry point. It takes a rule's ``build_bases`` and
``instantiate`` callables (the GENERATOR INTERFACE), runs them, assigns splits
BY BASE via the schema assigner, instantiates the per-split variant pattern,
NORMALIZES indefinite articles on each item's text, then puts the full emitted
dataset through FOUR gates IN ORDER, with NO quiet fallback — any failure raises
``PipelineError`` (or the gate's own loud error) and NOTHING is written:

    NORMALIZE (pre-gate, central, label-neutral): immediately after
                        ``instantiate`` returns a text and BEFORE any gate sees
                        it, ``genutils.fix_indefinite_articles`` rewrites the
                        article "a"/"A" -> "an"/"An" when the following word
                        starts with a vowel sound (vowel-letter heuristic + a
                        small exception set). It touches ONLY the article token
                        and "an" is one whitespace token, so the WORD COUNT and
                        every other rule signal are unchanged — the rewrite is
                        label-neutral and the gates below validate the corrected
                        text. This is the ONE place article casing is fixed, so
                        all 27 rules inherit it with no per-rule edits.


    GATE A schema       schema.validate_full (balance per balanced split,
                        base-disjointness across splits, no duplicate surface
                        string, word-count in [4,14], per-rule sentence_style
                        via style_policy_for(rule_id)).
    GATE B groundtruth  groundtruth.assert_labels_correct(rule_id, all_items)
                        (recomputes each label from TEXT; raises on mismatch).
    GATE C battery      battery.battery_report over the FULL dataset; every
                        frozen predicate must have max(agreement, 1-agreement)
                        <= 0.75 UNLESS exempt (equiv_keys / battery_exemptions).
    GATE D confound     confound.build_confound_report; assert overall_pass
                        (length-matching |mean_T - mean_F| <= 0.2 etc.) and
                        write data/<rule_id>/confound_report.json. The word-count
                        rules (RuleSpec.length_match_exempt, globals.length_
                        matching.policy "EXCEPT 23 and 25") are carved out of the
                        |mean_T - mean_F| assert -- word count IS the rule there,
                        so that match is unsatisfiable -- but the diff is still
                        computed and reported in the confound audit.

Only if ALL gates pass is data/<rule_id>/items.jsonl written. ``emit_rule``
returns an ``EmitSummary`` (counts per split, battery max agreement, gate
booleans).

=============================================================================
GENERATOR INTERFACE (the contract the 26 fan-out agents MUST conform to)
=============================================================================
A per-rule generator module exposes two module-level callables:

  build_bases(gen: Gen) -> list[BaseSpec]
    * Return at least ``schema.PROGRAMMATIC_N_BASES_MIN`` (340) base specs.
    * Each spec must carry enough to instantiate BOTH its True and its False
      variant deterministically.
    * The pipeline reads a stable, distinct base_id from each spec. The spec
      may be any object; if it is not a (str) base_id itself, the generator
      must also provide ``base_id_of`` OR each spec must expose a ``base_id``
      attribute / 'base_id' key. The reference rule uses a frozen dataclass
      with a ``base_id`` field.
    * The returned base_ids must be DISTINCT (the split assigner rejects dups).

  instantiate(spec: BaseSpec, label: bool, gen: Gen) -> (text, slots_meta)
    * Return the surface string for the ``label`` variant of ``spec`` and a
      JSON-serialisable slots_meta dict (provenance: seed + slots/transform).
    * groundtruth.assert_labels_correct must recompute ``label`` from ``text``;
      i.e. instantiate(spec, True, gen) must yield a text the rule labels True
      and instantiate(spec, False, gen) a text the rule labels False.
    * Deterministic given (spec, label, gen).

Per-split variant pattern (enforced by the pipeline, matching the spec's
programmatic split scheme):
  * few_shot_pool (100 bases): BOTH variants per base -> 200 items, 100T/100F.
    The context sampler draws 16T/16F from DISTINCT bases, so both variants must
    exist.
  * held_out (120 bases): ONE balanced variant per base -> 60T/60F.
  * confirmation (100 bases): ONE balanced variant per base -> 50T/50F.
  * spare (>= 20 bases): ONE variant per base (balanced as far as parity allows;
    used for step-3 probe construction / replacements).
The True/False assignment for the one-variant splits is seeded and balanced so
each balanced split is exactly 50/50 (Gate A re-checks this).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .. import battery, confound, groundtruth, schema
from ..genutils import Gen, fix_indefinite_articles
from ...rule_ids import canonical_rule_id
from ..schema import (
    PROGRAMMATIC_N_BASES_MIN,
    PROGRAMMATIC_SPARE_MIN,
    PROGRAMMATIC_SPLIT_BASES,
    assign_programmatic_splits,
    make_item,
    style_policy_for,
    write_items,
)

# repo root: .../src/icl_articulation/datagen/generators/base.py -> parents[4]
REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_EXTRACT_PATH = REPO_ROOT / "data" / "spec_extract.json"
DATA_DIR = REPO_ROOT / "data"

# The one-variant splits whose single variant per base is chosen to balance the
# split exactly 50/50 (held_out, confirmation, spare). few_shot_pool always
# emits BOTH variants. (Mirrors schema's ONE_VARIANT_SPLITS but spare is also
# one-variant here.)
_ONE_VARIANT_SPLITS = ("held_out", "confirmation", "spare")


class PipelineError(RuntimeError):
    """The emit pipeline hit an unrecoverable condition (LOUD, no fallback)."""


# --- the public per-rule spec projection --------------------------------------


@dataclass(frozen=True)
class RuleSpec:
    """The public, machine-usable facts the pipeline needs for one rule, read
    from the COMMITTED data/spec_extract.json (never the private spec)."""

    rule_id: str
    equivalence_class: tuple[str, ...]
    battery_exemptions: tuple[str, ...]
    equiv_keys: dict[str, list[str]]
    banks: tuple[str, ...]
    length_match_exempt: bool = False


def load_rule_spec(rule_id: str, *, path: Path | str | None = None) -> RuleSpec:
    """Load one rule's public spec projection from data/spec_extract.json.

    Raises PipelineError (loud) if the extract is missing or the rule_id is
    absent — the pipeline must never run a rule it has no spec for."""
    p = Path(path) if path is not None else SPEC_EXTRACT_PATH
    if not p.is_file():
        raise PipelineError(f"spec extract not found: {p} (run scripts/extract_spec.py)")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError(f"cannot read spec extract {p}: {exc}") from exc
    rules = data.get("rules", {})
    if rule_id not in rules:
        raise PipelineError(
            f"rule_id {rule_id!r} not in spec extract {p} "
            f"(known: {sorted(rules)[:5]}...)"
        )
    r = rules[rule_id]
    gen = r.get("generation", {}) or {}
    return RuleSpec(
        rule_id=rule_id,
        equivalence_class=tuple(r.get("equivalence_class", []) or []),
        battery_exemptions=tuple(r.get("battery_exemptions", []) or []),
        equiv_keys=dict(r.get("equiv_keys", {}) or {}),
        banks=tuple(gen.get("banks", []) or []),
        length_match_exempt=bool(r.get("length_match_exempt", False)),
    )


# --- the emit summary ---------------------------------------------------------


@dataclass(frozen=True)
class EmitSummary:
    """What ``emit_rule`` returns: counts per split, battery worst case, gates."""

    rule_id: str
    seed: int
    n_items: int
    n_bases: int
    split_counts: dict[str, dict[str, int]]  # split -> {"true": n, "false": n, "total": n}
    battery_max_agreement: float            # worst score among NON-exempt predicates
    battery_max_agreement_predicate: str
    battery_pos_ran: bool                    # False if the 6 POS predicates were skipped
    gate_schema: bool
    gate_groundtruth: bool
    gate_battery: bool
    gate_confound: bool
    items_path: str
    confound_report_path: str

    @property
    def all_gates_pass(self) -> bool:
        return all(
            (self.gate_schema, self.gate_groundtruth, self.gate_battery, self.gate_confound)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "seed": self.seed,
            "n_items": self.n_items,
            "n_bases": self.n_bases,
            "split_counts": self.split_counts,
            "battery_max_agreement": self.battery_max_agreement,
            "battery_max_agreement_predicate": self.battery_max_agreement_predicate,
            "battery_pos_ran": self.battery_pos_ran,
            "gates": {
                "schema": self.gate_schema,
                "groundtruth": self.gate_groundtruth,
                "battery": self.gate_battery,
                "confound": self.gate_confound,
            },
            "all_gates_pass": self.all_gates_pass,
            "items_path": self.items_path,
            "confound_report_path": self.confound_report_path,
        }


# --- base_id extraction -------------------------------------------------------


def _base_id_of(spec: Any, base_id_of: Callable[[Any], str] | None) -> str:
    """Pull a stable base_id out of a BaseSpec (the interface allows three forms)."""
    if base_id_of is not None:
        return str(base_id_of(spec))
    if isinstance(spec, str):
        return spec
    if hasattr(spec, "base_id"):
        return str(getattr(spec, "base_id"))
    if isinstance(spec, dict) and "base_id" in spec:
        return str(spec["base_id"])
    raise PipelineError(
        f"cannot determine base_id for spec {spec!r}; provide base_id_of, or a "
        "'base_id' attribute / key, or make the spec the base_id string itself"
    )


# --- the pipeline -------------------------------------------------------------


def _balanced_variant_labels(base_ids: Sequence[str], gen: Gen) -> dict[str, bool]:
    """Assign exactly half True / half False over ``base_ids`` (seeded shuffle).

    For an odd count the extra base is True (parity-only; balanced splits are
    sized even so this never bites few_shot/held_out/confirmation, and spare is
    not a balanced split)."""
    order = sorted(base_ids)
    gen.shuffle(order)
    n_true = (len(order) + 1) // 2
    out: dict[str, bool] = {}
    for i, b in enumerate(order):
        out[b] = i < n_true
    return out


def emit_rule(
    rule_id: str,
    build_bases: Callable[[Gen], list[Any]],
    instantiate: Callable[[Any, bool, Gen], tuple[str, dict[str, Any]]],
    seed: int,
    *,
    style_rule_id: str | None = None,
    base_id_of: Callable[[Any], str] | None = None,
    spec: RuleSpec | None = None,
    data_dir: Path | str | None = None,
    output_rule_id: str | None = None,
    stored_rule_id: str | None = None,
    write: bool = True,
    run_pos: bool = True,
) -> EmitSummary:
    """Run one programmatic rule end-to-end through the four gates.

    Parameters
    ----------
    rule_id        the spec rule_id (string id stored in each item's rule_id).
    build_bases    GENERATOR INTERFACE: build_bases(gen) -> list[BaseSpec].
    instantiate    GENERATOR INTERFACE: instantiate(spec, label, gen) -> (text, meta).
    seed           the single seed threaded through generation (logged).
    style_rule_id  rule_id passed to schema's per-rule sentence_style policy;
                   defaults to ``rule_id`` (use only to alias a style policy).
    base_id_of     optional extractor if a BaseSpec is not a str / lacks base_id.
    spec           the RuleSpec; loaded from the committed extract if omitted.
    data_dir       output root (defaults to repo data/); items + report go under
                   <data_dir>/<rule_id>/.
    write          if False, run all gates but do NOT write items.jsonl (the
                   confound report is still written — it is a gate artifact).
    run_pos        passed to the battery; False skips the 6 nltk POS predicates.

    Raises on ANY gate failure (loud). Returns EmitSummary on full success.
    """
    base_rule_id = canonical_rule_id(rule_id)
    output_rule_id = output_rule_id or rule_id
    stored_rule_id = stored_rule_id or output_rule_id
    style_rule_id = style_rule_id or base_rule_id
    spec = spec or load_rule_spec(base_rule_id)
    out_root = Path(data_dir) if data_dir is not None else DATA_DIR
    out_dir = out_root / output_rule_id

    gen = Gen(seed)

    # 1) BUILD BASES -----------------------------------------------------------
    bases = build_bases(gen.derive("build_bases"))
    if not isinstance(bases, list):
        raise PipelineError(f"build_bases must return a list, got {type(bases).__name__}")
    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise PipelineError(
            f"build_bases returned {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    base_ids = [_base_id_of(b, base_id_of) for b in bases]
    if len(set(base_ids)) != len(base_ids):
        raise PipelineError("build_bases produced duplicate base_ids")
    spec_by_id = dict(zip(base_ids, bases))

    # 2) ASSIGN SPLITS BY BASE (seeded, schema's assigner) ---------------------
    assignment = assign_programmatic_splits(base_ids, gen.derive("splits").seed)
    by_split: dict[str, list[str]] = {}
    for bid, split in assignment.items():
        by_split.setdefault(split, []).append(bid)

    # 3) INSTANTIATE ITEMS -----------------------------------------------------
    inst_gen = gen.derive("instantiate")
    items: list[dict[str, Any]] = []

    def _emit(bid: str, label: bool, split: str, variant_tag: str = "") -> None:
        text, meta = instantiate(spec_by_id[bid], label, inst_gen)
        # CENTRAL indefinite-article normalizer: rewrite "a"->"an" before a
        # vowel-sound word for EVERY rule, applied here (right after instantiate,
        # BEFORE all four gates) so groundtruth/battery/confound all see the
        # corrected surface. It is label-neutral (only the article token is
        # touched; "an" is one token, so word count is unchanged) — see
        # genutils.fix_indefinite_articles.
        text = fix_indefinite_articles(text)
        if not isinstance(meta, dict):
            raise PipelineError(
                f"instantiate must return (text, dict), got meta "
                f"{type(meta).__name__} for base {bid!r}"
            )
        # always record the run seed in provenance (log seed)
        meta = {"seed": seed, **meta}
        item_id = f"{bid}-{'T' if label else 'F'}"
        if variant_tag:
            item_id = f"{item_id}-{variant_tag}"
        items.append(
            make_item(
                item_id=item_id,
                base_id=bid,
                rule_id=stored_rule_id,
                label=label,
                text=text,
                slots_meta=meta,
                split=split,
            )
        )

    # few_shot_pool: BOTH variants per base
    for bid in by_split.get("few_shot_pool", []):
        _emit(bid, True, "few_shot_pool")
        _emit(bid, False, "few_shot_pool")

    # one-variant splits: balanced single variant per base
    for split in _ONE_VARIANT_SPLITS:
        ids = by_split.get(split, [])
        if not ids:
            continue
        labels = _balanced_variant_labels(ids, inst_gen.derive(f"variant:{split}"))
        for bid in ids:
            _emit(bid, labels[bid], split)

    if not items:
        raise PipelineError("no items emitted")

    # 4) GATE A — schema -------------------------------------------------------
    schema.validate_full(items, rule_id=style_rule_id)
    gate_schema = True

    # 5) GATE B — groundtruth (recompute every label from TEXT) ----------------
    groundtruth.assert_labels_correct(stored_rule_id, items)
    gate_groundtruth = True

    # 6) GATE C — battery over the FULL dataset --------------------------------
    results = battery.battery_report(
        items,
        equiv_keys=spec.equiv_keys,
        equivalence_class=spec.equivalence_class,
        battery_exemptions=spec.battery_exemptions,
        run_pos=run_pos,
    )
    violations = battery.battery_violations(results)
    if violations:
        detail = ", ".join(f"{r.key}={r.score:.3f}" for r in violations)
        raise PipelineError(
            f"GATE C battery failed for {rule_id!r}: {len(violations)} "
            f"non-exempt predicate(s) over the 0.75 threshold: {detail}"
        )
    gate_battery = True
    # worst-case NON-exempt, non-skipped predicate (the headline battery number)
    scored = [r for r in results if not r.skipped and not r.exempt]
    if scored:
        worst = max(scored, key=lambda r: r.score)
        battery_max = worst.score
        battery_max_pred = worst.key
    else:
        battery_max = 0.0
        battery_max_pred = ""
    pos_ran = run_pos and not any(r.skipped for r in results)

    # 7) GATE D — confound (build + assert overall_pass + write report) --------
    report = confound.build_confound_report(
        items,
        is_llm_rule=False,
        equiv_keys=spec.equiv_keys,
        equivalence_class=spec.equivalence_class,
        battery_exemptions=spec.battery_exemptions,
        run_pos=run_pos,
        length_match_exempt=spec.length_match_exempt,
    )
    if not report["overall_pass"]:
        raise PipelineError(
            f"GATE D confound failed for {rule_id!r}: "
            f"length_match_ok={report['length_match_ok']} "
            f"(|mean_T-mean_F|={report['word_count_mean_abs_diff']:.3f} > "
            f"{report['length_match_tolerance']}), "
            f"length_match_exempt={report['length_match_exempt']}, "
            f"battery_ok={report['battery_ok']} "
            f"violations={report['battery_violations']}"
        )
    gate_confound = True
    report_path = confound.write_confound_report(report, out_dir / "confound_report.json")

    # 8) ALL GATES PASSED — write items.jsonl ----------------------------------
    items_path = out_dir / "items.jsonl"
    if write:
        write_items(items, items_path)

    # split counts for the summary
    split_counts: dict[str, dict[str, int]] = {}
    for it in items:
        sc = split_counts.setdefault(it["split"], {"true": 0, "false": 0, "total": 0})
        sc["true" if bool(it["label"]) else "false"] += 1
        sc["total"] += 1

    return EmitSummary(
        rule_id=output_rule_id,
        seed=seed,
        n_items=len(items),
        n_bases=len(bases),
        split_counts=split_counts,
        battery_max_agreement=battery_max,
        battery_max_agreement_predicate=battery_max_pred,
        battery_pos_ran=pos_ran,
        gate_schema=gate_schema,
        gate_groundtruth=gate_groundtruth,
        gate_battery=gate_battery,
        gate_confound=gate_confound,
        items_path=str(items_path),
        confound_report_path=str(report_path),
    )
