"""Rule 18 (physically_impossible): LLM-validated, by-base minimal-pair build.

This is the rule-18 generator. It lives in the ``generators/llm`` package, which
is DISJOINT from ``generators/rules`` (the programmatic auto-discovery path): the
registry scans only ``rules/``, so this module is never auto-run with a missing
network. It is driven explicitly via the package CLI
(``python -m icl_articulation.datagen.generators.llm physically_impossible``).

Pipeline (rule-specs rule 18 recipe):

  AUTHOR   ``frames_physically_impossible`` supplies >= 440 minimal-pair frames,
           each a (plausible, impossible) single-word swap in one shared frame,
           covering the four impossibility types with cross-class word reuse
           >= 80% (no lexical proxy). The frame bank self-audits at import.

  VALIDATE Each of a base's two variants (plausible-filled = author-intended
           POSSIBLE, impossible-filled = author-intended IMPOSSIBLE) is sent to
           TWO independent validators:
             pass A: gpt-4.1-mini, prompt A;
             pass B: gpt-4.1, a DIFFERENTLY WORDED prompt B.
           Each validator answers impossible / possible / unclear. A VARIANT
           passes iff BOTH validators return the author-intended verdict
           (impossible for the True variant, possible for the False variant);
           any 'unclear' or any disagreement DISPUTES the variant (dropped).

  SURVIVE  A BASE survives iff BOTH its variants pass (rule-18: base_id = frame,
           both variants share it, base survives only if both pass). Surviving
           bases carry slots_meta['validated_agreement'] = the agreed (stored)
           label, which groundtruth.assert_labels_correct REQUIRES for this
           validator-derived rule (it never recomputes from text).

  SPLIT    PROGRAMMATIC by-base split over the SURVIVING bases (schema's
           assign_programmatic_splits): few_shot_pool keeps BOTH variants,
           held_out / confirmation / spare keep ONE balanced variant per base.

  EMIT     data/physically_impossible/items.jsonl + confound_report.json. The
           confound report is built on the is_llm_rule path (tol 1.0); by
           construction the two variants of a base share an EXACT word count, so
           |mean_wc(T) - mean_wc(F)| is ~0 regardless of tolerance.

OFFLINE-TESTABLE: ``run_build`` takes a ``validator`` callable. In ``--mock``
mode a deterministic keyword validator is injected (no network), so the full
author -> validate -> by-base-split -> emit flow runs on the REAL frames with
zero API calls. The real path builds an OpenAIClient (client.py) and an async
two-pass validator; both share the same emit code.

All API calls go through ``icl_articulation.client`` (disk cache + cost meter +
retries). LOUD on every invariant: a malformed bank, an empty surviving set, a
split that cannot be filled, or a groundtruth/schema/confound gate failure all
raise -- nothing partial is written.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .... import client as client_mod
from .... import runlog
from ... import confound, groundtruth, schema
from ...schema import (
    PROGRAMMATIC_N_BASES_MIN,
    PROGRAMMATIC_SPARE_MIN,
    assign_programmatic_splits,
    make_item,
)
from ...genutils import fix_indefinite_articles, to_sentence_case
from . import frames_physically_impossible as bank

RULE_ID = "physically_impossible"

# repo root: .../datagen/generators/llm/physically_impossible.py -> parents[5]
REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = REPO_ROOT / "data"

# validator models (rule-18 recipe: pass A gpt-4.1-mini, pass B gpt-4.1).
PASS_A_MODEL = "gpt-4.1-mini"
PASS_B_MODEL = "gpt-4.1"

# the three verdicts a validator may return.
IMPOSSIBLE = "impossible"
POSSIBLE = "possible"
UNCLEAR = "unclear"
VERDICTS = (IMPOSSIBLE, POSSIBLE, UNCLEAR)

DEFAULT_SEED = 1234
# validators are deterministic-leaning; temperature 0 + a fixed seed keeps the
# paid run reproducible and cache-friendly.
VALIDATE_TEMPERATURE = 0.0
VALIDATE_MAX_TOKENS = 4


# =============================================================================
# the two differently-worded validation prompts
# =============================================================================

# Pass A (gpt-4.1-mini). Direct, option-listing.
_PROMPT_A_SYSTEM = (
    "You are a careful physical-reasoning judge. You decide whether a described "
    "event could actually happen in the real physical world."
)
_PROMPT_A_USER = (
    "Read the sentence and judge ONLY its literal physical plausibility.\n"
    "- Answer 'impossible' if the event could not physically happen in the real "
    "world (for example an inanimate object performing a biological action, a "
    "person lifting something far too heavy, an object that cannot fit into a "
    "much smaller container, or matter in an impossible state).\n"
    "- Answer 'possible' if the event is physically ordinary, even if unusual, "
    "rare, socially odd, or unlikely.\n"
    "- Answer 'unclear' only if you genuinely cannot decide.\n"
    "Judge the literal event, not metaphor or fiction framing.\n\n"
    "Sentence: {text}\n\n"
    "Answer with exactly one word: impossible, possible, or unclear."
)

# Pass B (gpt-4.1). Differently worded: framed as 'violates the laws of physics?'
_PROMPT_B_SYSTEM = (
    "You assess statements for physical realism. A statement is realistic if the "
    "event it literally describes is consistent with how the physical world works."
)
_PROMPT_B_USER = (
    "Consider the literal meaning of the statement below. Does the event it "
    "describes violate the physical laws of the everyday world -- that is, is it "
    "something that simply cannot occur (such as a lifeless object eating or "
    "sleeping, a human carrying an enormous structure, a large object squeezed "
    "into a tiny one, or a material behaving against its nature)?\n\n"
    "Statement: {text}\n\n"
    "Reply with a single word:\n"
    "  'impossible' -- it cannot physically occur;\n"
    "  'possible'   -- it can occur, even if strange or improbable;\n"
    "  'unclear'    -- truly undecidable.\n"
    "Do not consider metaphor, exaggeration, or story-telling -- only the literal "
    "physical event."
)


def _render_pass_a(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _PROMPT_A_SYSTEM},
        {"role": "user", "content": _PROMPT_A_USER.format(text=text)},
    ]


def _render_pass_b(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _PROMPT_B_SYSTEM},
        {"role": "user", "content": _PROMPT_B_USER.format(text=text)},
    ]


def parse_verdict(raw: str) -> str:
    """Map a raw model reply to one of the three verdicts (LOUD on garbage).

    Lenient on surrounding whitespace / punctuation / casing, but a reply that
    contains NONE of the three verdict words is an error -- we never silently
    coerce an unparseable verdict to a passing one."""
    low = raw.strip().lower()
    # check for an exact leading word first (the prompt asks for one word)
    head = low.split()[0].strip(".,!?:;'\"") if low.split() else ""
    if head in VERDICTS:
        return head
    # otherwise substring containment, but disambiguate 'impossible' vs 'possible'
    if IMPOSSIBLE in low:
        return IMPOSSIBLE
    if POSSIBLE in low:
        return POSSIBLE
    if UNCLEAR in low:
        return UNCLEAR
    raise ValueError(f"unparseable validator verdict: {raw!r}")


# =============================================================================
# bases / variants
# =============================================================================


@dataclass(frozen=True)
class Variant:
    """One filled variant of a frame (one of a base's two members)."""

    base_id: str
    label: bool  # author-intended label: True = impossible, False = plausible
    text: str
    slot_word: str
    itype: str

    @property
    def intended_verdict(self) -> str:
        return IMPOSSIBLE if self.label else POSSIBLE


@dataclass(frozen=True)
class BaseFrame:
    """A base = one frame; its True (impossible) and False (plausible) variants
    share the same base_id (rule-18: base_id = frame)."""

    base_id: str
    impossible_variant: Variant
    plausible_variant: Variant
    itype: str

    @property
    def variants(self) -> tuple[Variant, Variant]:
        return (self.impossible_variant, self.plausible_variant)


def _frame_text(template: str, word: str) -> str:
    """Sentence-case + indefinite-article-normalize one filled frame (the same
    two label-neutral transforms the programmatic emit pipeline applies)."""
    return fix_indefinite_articles(to_sentence_case(template.replace("{S}", word)))


def build_bases() -> list[BaseFrame]:
    """Turn the authored frame bank into bases (one per frame), deduping surfaces.

    base_id encodes the frame index + both slot words so it is stable, distinct,
    and human-readable. Raises if two frames would yield an identical surface
    string in either class (a duplicate surface is a schema violation)."""
    frames = bank.FRAMES
    bases: list[BaseFrame] = []
    seen_surface: set[str] = set()
    for i, fr in enumerate(frames):
        imp_text = _frame_text(fr.template, fr.impossible)
        pla_text = _frame_text(fr.template, fr.plausible)
        if imp_text in seen_surface or pla_text in seen_surface or imp_text == pla_text:
            # duplicate / degenerate surface: skip this frame (the bank has ample
            # headroom; uniqueness is enforced on the EMITTED data downstream).
            continue
        seen_surface.add(imp_text)
        seen_surface.add(pla_text)
        base_id = f"f{i:04d}|{fr.plausible}|{fr.impossible}"
        imp_v = Variant(base_id, True, imp_text, fr.impossible, fr.itype)
        pla_v = Variant(base_id, False, pla_text, fr.plausible, fr.itype)
        bases.append(BaseFrame(base_id, imp_v, pla_v, fr.itype))
    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"rule 18: only {len(bases)} distinct-surface bases from the bank, "
            f"need >= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


# =============================================================================
# validation (offline-injectable)
# =============================================================================

# A Validator takes a sentence and returns the (pass_a_verdict, pass_b_verdict)
# pair. The real validator calls the API; the mock validator is pure.
Validator = Callable[[str], tuple[str, str]]


def make_mock_pair_validator(
    *, drop_fraction: float = 0.0, drop_salt: int = 0
) -> Validator:
    """Deterministic OFFLINE validator (no network) for the offline test.

    A model validator only ever sees a sentence's surface, not which frame it came
    from -- but a USEFUL offline mock must reproduce the author's intent for valid
    minimal pairs so the survival/split/emit flow can be exercised. So the mock is
    seeded from a ``{surface_text -> intended_verdict}`` map built from the bank's
    own variants: both passes return the intended verdict (impossible for an
    impossible-filled surface, possible for a plausible-filled one), and 'unclear'
    for any surface it has never seen.

    ``drop_fraction`` (in [0, 1)) deterministically DISPUTES that fraction of
    variants (returns an 'unclear' from pass B), so the offline test can also
    exercise the by-base SURVIVAL filter (a base dies if either variant is
    disputed) and the non-zero drop-rate accounting. The dropped set is chosen by
    a stable hash of the surface + ``drop_salt`` (reproducible)."""
    if not (0.0 <= drop_fraction < 1.0):
        raise ValueError(f"drop_fraction must be in [0, 1), got {drop_fraction}")
    import hashlib

    intended: dict[str, str] = {}
    for base in build_bases():
        intended[base.impossible_variant.text] = IMPOSSIBLE
        intended[base.plausible_variant.text] = POSSIBLE

    def _dropped(text: str) -> bool:
        if drop_fraction <= 0.0:
            return False
        h = hashlib.sha256(f"{drop_salt}:{text}".encode("utf-8")).hexdigest()
        # map the first 8 hex digits to [0, 1)
        frac = int(h[:8], 16) / 0xFFFFFFFF
        return frac < drop_fraction

    def _v(text: str) -> tuple[str, str]:
        verdict = intended.get(text, UNCLEAR)
        if _dropped(text):
            # pass B disputes -> the variant fails -> its base dies (survival test)
            return (verdict, UNCLEAR)
        return (verdict, verdict)

    return _v


def make_api_validator(
    cli: "client_mod.OpenAIClient",
    *,
    seed: int,
    log: "runlog.RunLog | None" = None,
) -> Callable[[list[str]], "asyncio.Future"]:
    """Build an ASYNC batched validator over the real OpenAI client.

    Returns ``validate_all(texts) -> list[(verdict_a, verdict_b)]`` that issues
    pass-A (gpt-4.1-mini) and pass-B (gpt-4.1) calls for every text concurrently
    (the client bounds concurrency + caches + meters cost). Each response record
    is logged to ``log`` if provided."""

    async def _one(model: str, render, text: str) -> str:
        rec = await cli.complete(
            model,
            render(text),
            temperature=VALIDATE_TEMPERATURE,
            max_tokens=VALIDATE_MAX_TOKENS,
            seed=seed,
        )
        if log is not None:
            log.log_response(rec)
        return parse_verdict(client_mod.response_text(rec))

    async def validate_all(texts: list[str]) -> list[tuple[str, str]]:
        tasks = []
        for text in texts:
            tasks.append(
                asyncio.gather(
                    _one(PASS_A_MODEL, _render_pass_a, text),
                    _one(PASS_B_MODEL, _render_pass_b, text),
                )
            )
        return await asyncio.gather(*tasks)

    return validate_all


# =============================================================================
# survival + emit
# =============================================================================


@dataclass(frozen=True)
class VariantVerdict:
    """The recorded two-pass verdict for one variant (provenance + survival)."""

    variant: Variant
    verdict_a: str
    verdict_b: str

    @property
    def passed(self) -> bool:
        """Both validators returned the author-intended verdict (no dispute)."""
        intended = self.variant.intended_verdict
        return self.verdict_a == intended and self.verdict_b == intended


def survive_bases(
    bases: list[BaseFrame], verdicts: dict[str, VariantVerdict]
) -> list[BaseFrame]:
    """Keep only bases whose BOTH variants passed (rule-18 survival rule)."""
    survivors: list[BaseFrame] = []
    for base in bases:
        imp = verdicts.get(base.impossible_variant.text)
        pla = verdicts.get(base.plausible_variant.text)
        if imp is None or pla is None:
            raise ValueError(
                f"missing verdict for base {base.base_id!r} variants; validator "
                "must score every variant"
            )
        if imp.passed and pla.passed:
            survivors.append(base)
    return survivors


def _emit_items(
    survivors: list[BaseFrame], verdicts: dict[str, VariantVerdict], seed: int
) -> list[dict[str, Any]]:
    """Assign the programmatic by-base split over surviving bases and instantiate
    items, stamping slots_meta['validated_agreement'] (the agreed stored label).

    few_shot_pool keeps BOTH variants per base; held_out / confirmation / spare
    keep ONE balanced variant per base. The chosen variant's stored label is the
    author-intended (and now validator-agreed) label, so the split stays 50/50
    by base parity within each balanced split."""
    base_ids = [b.base_id for b in survivors]
    assignment = assign_programmatic_splits(base_ids, seed)
    by_id = {b.base_id: b for b in survivors}

    items: list[dict[str, Any]] = []

    def _meta(v: Variant) -> dict[str, Any]:
        vv = verdicts[v.text]
        return {
            "seed": seed,
            "frame_base_id": v.base_id,
            "slot_word": v.slot_word,
            "impossibility_type": v.itype,
            "validator_pass_a": {"model": PASS_A_MODEL, "verdict": vv.verdict_a},
            "validator_pass_b": {"model": PASS_B_MODEL, "verdict": vv.verdict_b},
            # groundtruth.assert_labels_correct REQUIRES this for rule 18:
            "validated_agreement": v.label,
        }

    def _add(v: Variant, split: str) -> None:
        items.append(
            make_item(
                item_id=f"{v.base_id}-{'T' if v.label else 'F'}",
                base_id=v.base_id,
                rule_id=RULE_ID,
                label=v.label,
                text=v.text,
                slots_meta=_meta(v),
                split=split,
            )
        )

    # group bases by split, then pick variant(s) per the programmatic scheme.
    by_split: dict[str, list[str]] = {}
    for bid, split in assignment.items():
        by_split.setdefault(split, []).append(bid)

    # few_shot_pool: BOTH variants.
    for bid in by_split.get("few_shot_pool", []):
        base = by_id[bid]
        _add(base.impossible_variant, "few_shot_pool")
        _add(base.plausible_variant, "few_shot_pool")

    # one-variant splits: balanced single variant per base (seeded by sorted order
    # + parity, so exactly 50/50 in even-sized balanced splits).
    for split in ("held_out", "confirmation", "spare"):
        ids = sorted(by_split.get(split, []))
        for rank, bid in enumerate(ids):
            base = by_id[bid]
            # alternate impossible/plausible by rank -> exact 50/50 for even n.
            v = base.impossible_variant if rank % 2 == 0 else base.plausible_variant
            _add(v, split)

    return items


@dataclass(frozen=True)
class BuildSummary:
    """What ``run_build`` returns: bank/validation/survival counts + gate flags."""

    rule_id: str
    seed: int
    n_frames: int
    n_bases: int
    n_variants_validated: int
    n_variants_passed: int
    n_bases_survived: int
    n_items: int
    split_counts: dict[str, dict[str, int]]
    drop_rate_variant: float
    drop_rate_base: float
    gate_schema: bool
    gate_groundtruth: bool
    gate_confound: bool
    items_path: str
    confound_report_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "seed": self.seed,
            "n_frames": self.n_frames,
            "n_bases": self.n_bases,
            "n_variants_validated": self.n_variants_validated,
            "n_variants_passed": self.n_variants_passed,
            "n_bases_survived": self.n_bases_survived,
            "n_items": self.n_items,
            "split_counts": self.split_counts,
            "drop_rate_variant": self.drop_rate_variant,
            "drop_rate_base": self.drop_rate_base,
            "gates": {
                "schema": self.gate_schema,
                "groundtruth": self.gate_groundtruth,
                "confound": self.gate_confound,
            },
            "items_path": self.items_path,
            "confound_report_path": self.confound_report_path,
        }


def run_build(
    validator: Validator,
    *,
    seed: int = DEFAULT_SEED,
    data_dir: Path | str | None = None,
    write: bool = True,
    run_pos: bool = True,
) -> BuildSummary:
    """Author -> validate (2-pass) -> by-base survive -> split -> gate -> emit.

    ``validator(text) -> (verdict_a, verdict_b)`` is the INJECTED two-pass
    validator (mock offline, or an API-backed one). Everything after validation
    is shared by both paths. Raises (LOUD) on any gate failure or if too few
    bases survive to fill the programmatic split."""
    out_root = Path(data_dir) if data_dir is not None else DATA_DIR
    out_dir = out_root / RULE_ID

    # 1) AUTHOR
    bases = build_bases()

    # 2) VALIDATE every variant (both passes)
    verdicts: dict[str, VariantVerdict] = {}
    n_validated = 0
    n_passed = 0
    for base in bases:
        for v in base.variants:
            va, vb = validator(v.text)
            if va not in VERDICTS or vb not in VERDICTS:
                raise ValueError(
                    f"validator returned non-verdict ({va!r},{vb!r}) for {v.text!r}"
                )
            vv = VariantVerdict(v, va, vb)
            verdicts[v.text] = vv
            n_validated += 1
            if vv.passed:
                n_passed += 1

    # 3) SURVIVE by base (both variants passed)
    survivors = survive_bases(bases, verdicts)
    need = (
        schema.PROGRAMMATIC_SPLIT_BASES["few_shot_pool"]
        + schema.PROGRAMMATIC_SPLIT_BASES["held_out"]
        + schema.PROGRAMMATIC_SPLIT_BASES["confirmation"]
        + PROGRAMMATIC_SPARE_MIN
    )
    if len(survivors) < need:
        raise ValueError(
            f"rule 18: only {len(survivors)}/{len(bases)} bases survived "
            f"validation, need >= {need} to fill the programmatic split "
            f"({len(bases) - len(survivors)} bases dropped). "
            "Author more frames or fix low-yield ones (recipe tolerates ~23% loss)."
        )

    # 4) SPLIT + INSTANTIATE
    items = _emit_items(survivors, verdicts, seed)
    if not items:
        raise ValueError("rule 18: no items emitted after survival/split")

    # 5) GATE A schema (style policy self-selected by rule_id), GATE B groundtruth
    #    (validator-derived: requires validated_agreement; never recomputes).
    schema.validate_full(items, rule_id=RULE_ID)
    gate_schema = True
    groundtruth.assert_labels_correct(RULE_ID, items)
    gate_groundtruth = True

    # 6) GATE C confound (is_llm_rule path; by construction word count identical).
    report = confound.build_confound_report(items, is_llm_rule=True, run_pos=run_pos)
    if not report["overall_pass"]:
        raise ValueError(
            f"rule 18 GATE confound failed: length_match_ok="
            f"{report['length_match_ok']} "
            f"(|mean_T-mean_F|={report['word_count_mean_abs_diff']:.3f} > "
            f"{report['length_match_tolerance']}), battery_ok={report['battery_ok']} "
            f"violations={report['battery_violations']}"
        )
    gate_confound = True
    report_path = confound.write_confound_report(report, out_dir / "confound_report.json")

    # 7) EMIT
    items_path = out_dir / "items.jsonl"
    if write:
        schema.write_items(items, items_path)

    split_counts: dict[str, dict[str, int]] = {}
    for it in items:
        sc = split_counts.setdefault(it["split"], {"true": 0, "false": 0, "total": 0})
        sc["true" if bool(it["label"]) else "false"] += 1
        sc["total"] += 1

    return BuildSummary(
        rule_id=RULE_ID,
        seed=seed,
        n_frames=len(bank.FRAMES),
        n_bases=len(bases),
        n_variants_validated=n_validated,
        n_variants_passed=n_passed,
        n_bases_survived=len(survivors),
        n_items=len(items),
        split_counts=split_counts,
        drop_rate_variant=1.0 - (n_passed / n_validated) if n_validated else 0.0,
        drop_rate_base=1.0 - (len(survivors) / len(bases)) if bases else 0.0,
        gate_schema=gate_schema,
        gate_groundtruth=gate_groundtruth,
        gate_confound=gate_confound,
        items_path=str(items_path),
        confound_report_path=str(report_path),
    )


# =============================================================================
# real (API) entry point
# =============================================================================


def run_api_build(
    *,
    seed: int = DEFAULT_SEED,
    max_candidates: int | None = None,
    data_dir: Path | str | None = None,
    write: bool = True,
    run_pos: bool = True,
    cache_dir: str | Path = "cache",
    results_dir: str | Path = "results",
    concurrency: int = client_mod.DEFAULT_CONCURRENCY,
) -> BuildSummary:
    """The REAL run: build an OpenAIClient and validate via the 2-pass API.

    Makes paid API calls (set OPENAI_API_KEY).
    It logs a run config (seed, models, cost estimate) and the actual cost via
    runlog. ``max_candidates`` caps the number of bases validated (for a smaller
    pilot); None validates all of them."""
    bases = build_bases()
    if max_candidates is not None:
        bases = bases[:max_candidates]
    n_variants = 2 * len(bases)
    est = estimate_cost(n_variants)

    cfg = {
        "rule_id": RULE_ID,
        "seed": seed,
        "pass_a_model": PASS_A_MODEL,
        "pass_b_model": PASS_B_MODEL,
        "n_frames": len(bank.FRAMES),
        "n_bases": len(bases),
        "n_variants": n_variants,
        "validate_temperature": VALIDATE_TEMPERATURE,
        "validate_max_tokens": VALIDATE_MAX_TOKENS,
        "prompt_a_hash": _prompt_hash(_PROMPT_A_SYSTEM, _PROMPT_A_USER),
        "prompt_b_hash": _prompt_hash(_PROMPT_B_SYSTEM, _PROMPT_B_USER),
        "frame_bank_audit": bank.AUDIT_SUMMARY,
    }
    log = runlog.start_run(
        f"datagen-{RULE_ID}", cfg, est["total_usd"], results_dir=results_dir
    )

    cli = client_mod.OpenAIClient(concurrency=concurrency, cache_dir=cache_dir)

    async def _drive() -> BuildSummary:
        """The WHOLE rule-18 real run under ONE event loop: validate every variant
        concurrently, run the (synchronous) survive->split->gate->emit, log
        metrics, AND ``await cli.aclose()`` — all on this single loop.

        The old code ran validation under one ``asyncio.run`` and then closed the
        client under a SECOND ``asyncio.run(cli.aclose())`` in the finally; the
        client's httpx pool was bound to the first (now-closed) loop, so the close
        raised 'RuntimeError: Event loop is closed'. Awaiting validation AND the
        close on the same loop fixes it. The per-text seeds + prompts (and so the
        disk-cache keys) are unchanged — only the loop/dispatch structure moved."""
        summary: BuildSummary | None = None
        try:
            validate_all = make_api_validator(cli, seed=seed, log=log)
            texts: list[str] = []
            for base in bases:
                texts.append(base.impossible_variant.text)
                texts.append(base.plausible_variant.text)
            pairs = await validate_all(texts)
            verdict_by_text = {t: p for t, p in zip(texts, pairs)}

            def _validator(text: str) -> tuple[str, str]:
                return verdict_by_text[text]

            summary = run_build(
                _validator,
                seed=seed,
                data_dir=data_dir,
                write=write,
                run_pos=run_pos,
            )
            return summary
        finally:
            cost = cli.cost.total_usd
            log.write_metrics(
                {
                    "client_stats": cli.stats(),
                    "cost": cli.cost.summary(),
                    "build_summary": summary.as_dict() if summary is not None else None,
                }
            )
            log.finish(
                cost,
                extra={"build_summary": summary.as_dict() if summary is not None else None},
            )
            # close the client on the SAME loop validation ran on (single-loop fix).
            await cli.aclose()

    return asyncio.run(_drive())


def _prompt_hash(system: str, user: str) -> str:
    import hashlib

    return hashlib.sha256(f"{system}\n---\n{user}".encode("utf-8")).hexdigest()


# =============================================================================
# cost estimate
# =============================================================================

# Rough per-call token sizes for the cost estimate. The validation prompts are
# ~210 (pass A) / ~230 (pass B) prompt tokens once the sentence is filled in;
# completions are a single word (<= VALIDATE_MAX_TOKENS). We round UP for a
# conservative advance estimate.
EST_PROMPT_TOKENS_A = 230
EST_PROMPT_TOKENS_B = 260
EST_COMPLETION_TOKENS = 3


def estimate_cost(n_variants: int) -> dict[str, Any]:
    """Advance cost estimate for validating ``n_variants`` items x 2 passes.

    Each variant gets one pass-A (gpt-4.1-mini) and one pass-B (gpt-4.1) call."""
    from ....prices import cost_usd

    cost_a = cost_usd(PASS_A_MODEL, EST_PROMPT_TOKENS_A * n_variants, EST_COMPLETION_TOKENS * n_variants)
    cost_b = cost_usd(PASS_B_MODEL, EST_PROMPT_TOKENS_B * n_variants, EST_COMPLETION_TOKENS * n_variants)
    return {
        "n_variants": n_variants,
        "n_calls": 2 * n_variants,
        "pass_a_model": PASS_A_MODEL,
        "pass_b_model": PASS_B_MODEL,
        "pass_a_usd": cost_a,
        "pass_b_usd": cost_b,
        "total_usd": cost_a + cost_b,
        "assumptions": {
            "prompt_tokens_a": EST_PROMPT_TOKENS_A,
            "prompt_tokens_b": EST_PROMPT_TOKENS_B,
            "completion_tokens": EST_COMPLETION_TOKENS,
        },
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m icl_articulation.datagen.generators.llm",
        description="rule-18 (physically_impossible) LLM-validated build",
    )
    p.add_argument(
        "rule_id",
        nargs="?",
        default=RULE_ID,
        help=f"rule_id to build (only {RULE_ID!r} is implemented here)",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="generation/validation seed")
    p.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="cap the number of bases validated (pilot); default all",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="OFFLINE: inject a deterministic keyword validator (no network)",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="run all gates but do not write items.jsonl (dry run)",
    )
    p.add_argument(
        "--no-pos",
        action="store_true",
        help="skip the nltk first-word-POS battery predicates in the report",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="output root (defaults to repo data/); items go under <root>/<rule_id>/",
    )
    p.add_argument(
        "--estimate-only",
        action="store_true",
        help="print the advance cost estimate and exit (no build)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rule_id != RULE_ID:
        print(
            f"error: this CLI implements only {RULE_ID!r}, got {args.rule_id!r}",
            file=sys.stderr,
        )
        return 2

    bases = build_bases()
    if args.max_candidates is not None:
        bases = bases[: args.max_candidates]

    if args.estimate_only:
        print(json.dumps(estimate_cost(2 * len(bases)), indent=2))
        return 0

    if args.mock:
        summary = run_build(
            make_mock_pair_validator(),
            seed=args.seed,
            data_dir=args.data_dir,
            write=not args.no_write,
            run_pos=not args.no_pos,
        )
    else:
        summary = run_api_build(
            seed=args.seed,
            max_candidates=args.max_candidates,
            data_dir=args.data_dir,
            write=not args.no_write,
            run_pos=not args.no_pos,
        )

    print(json.dumps(summary.as_dict(), indent=2))
    all_pass = summary.gate_schema and summary.gate_groundtruth and summary.gate_confound
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
