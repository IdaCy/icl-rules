"""The shared llm_validated pipeline: generate -> 2-pass validate -> rebalance
-> emit, for rules 15 (positive_sentiment) and 16 (food_topic).

ONE rule-agnostic flow, parameterised by an ``LLMRuleConfig`` (see ``config``).
Every step is loud on failure; nothing is written unless the emitted dataset
passes the schema + groundtruth + confound gates.

THE API SEAM (offline-testability)
----------------------------------
All model calls go through two injected callables:

    Generator: (rule_id, label, topic, n, *, seed) -> list[str]
        ask the generation model for ~n sentences of ``label``'s class on
        ``topic``; returns raw candidate strings.
    Labeler:   (rule_id, which, text, *, seed) -> bool | None
        run validation pass ``which`` ('A'=mini, 'B'=4.1) over ``text``;
        returns True (intended-True), False (intended-False), or None (drop).

The REAL run wires these to ``icl_articulation.client.OpenAIClient`` (so caching,
cost metering, retries, and run logging all apply) via ``openai_generator`` /
``openai_labeler`` in ``api.py``. ``--mock`` wires the deterministic
``mock_generator`` / ``mock_labeler`` here, which never touch the network. A test
exercises the full flow on a tiny fake corpus with the mock seam.

WHY A 3-WAY VALIDATOR LABEL
---------------------------
Each pass returns True / False / None. An item is KEPT only if BOTH passes
return the SAME non-None label AND that label equals the item's intended label
(rule-specs: "keep only items where both validators return the intended
non-neutral label"). The agreed label is stamped into
``slots_meta['validated_agreement']`` — the provenance
``groundtruth.assert_labels_correct`` REQUIRES for these validator-derived rules.

REBALANCING
-----------
Validator drops are asymmetric, so after validation the surviving items are
down-sampled at random within each (topic x class) cell to the per-cell quota,
enforcing exact 50/50 + topic balance. Short cells trigger targeted
regeneration (the real run re-calls the generator for that cell; the offline
pipeline supports it the same way through the injected Generator).
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Callable, Sequence

from .... import contexts  # noqa: F401  (kept for parity; schema imports the loader)
from ...confound import (
    LLM_CAP_RATE_TOL,
    LLM_COMMA_RATE_TOL,
    LLM_NEGATOR_RATE_TOL,
    LLM_STOPWORD_RATE_TOL,
    NEGATORS,
    STOPWORDS,
    build_confound_report,
    write_confound_report,
)
from ...groundtruth import VALIDATED_FLAG, assert_labels_correct
from ...schema import (
    LLM_N_ITEMS_MIN,
    LLM_SPLIT_ITEMS,
    assert_sentence_style,
    assign_llm_splits,
    make_item,
    validate_full,
    word_count,
    words,
    write_items,
)
from .config import STYLE_RULES, LLMRuleConfig, get_rule_config

# repo root: .../datagen/generators/llm/pipeline.py -> parents[5]
REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = REPO_ROOT / "data"

# Generation / validation models (rule-specs recipe). Real run uses these; the
# cost estimate and run log record them.
GEN_MODEL = "gpt-4.1-mini"
VALIDATOR_A_MODEL = "gpt-4.1-mini"
VALIDATOR_B_MODEL = "gpt-4.1"

# total balanced-split items needed (120 + 120 + 100 = 340); the kept set must
# reach >= this AND have enough per (topic x class) cell to rebalance to exact
# quotas. The pipeline targets a per-cell quota derived from the split sizes.
N_KEPT_MIN = LLM_N_ITEMS_MIN  # 340

# the banned characters whose presence drops a candidate in the style filter
# banned anywhere in a candidate (mirrors schema.assert_sentence_style for the
# llm rules: no terminal/exclamation/question/quote/contraction/hyphen). The
# comma is the ONE internal punct the llm rules may keep (comma-rate audited), so
# it is deliberately ABSENT here. The ASCII hyphen and the unicode dashes are all
# banned (globals.sentence_style.no_hyphenated_words).
_BANNED_PUNCT = set("!?\"';:()[]{}…–—-")
_BANNED_SUBSTRINGS = ("--",)


class LLMPipelineError(RuntimeError):
    """The llm_validated pipeline hit an unrecoverable condition (LOUD)."""


# ===========================================================================
# the API seam types
# ===========================================================================
# Generator(rule_id, label, topic, n, *, seed) -> list[str]
Generator = Callable[..., list[str]]
# Labeler(rule_id, which, text, *, seed) -> bool | None
Labeler = Callable[..., "bool | None"]


# --- batched-dispatch requests --------------------------------------------
# The pipeline collects a whole phase's requests as these immutable records and
# hands them to a Dispatcher, which issues them CONCURRENTLY (real run, bounded
# by the client's semaphore) or directly in order (mock, no event loop). Each
# record carries EXACTLY the args the old sequential callable received, so the
# request content + seed mapping (and thus the disk-cache key) is unchanged.
@dataclass(frozen=True)
class GenRequest:
    """One generation call: ``generator(rule_id, label, topic, n, seed=seed)``."""

    rule_id: str
    label: bool
    topic: str
    n: int
    seed: int


@dataclass(frozen=True)
class LabelRequest:
    """One validation call: ``labeler(rule_id, which, text, seed=seed)``."""

    rule_id: str
    which: str
    text: str
    seed: int


class _CallableDispatcher:
    """Adapts the plain ``generator`` / ``labeler`` callables (e.g. the mock seam)
    to the batch interface the pipeline drives.

    The mock callables are pure and network-free, so this just calls them in
    order — no event loop, no concurrency machinery — preserving the exact
    sequential semantics ``--mock`` had. A real seam that exposes its own
    concurrent ``generate_many`` / ``label_many`` (``ClientSeam``,
    ``is_concurrent=True``) is used directly instead of this adapter."""

    is_concurrent = False

    def __init__(self, generator: Generator, labeler: Labeler) -> None:
        self._generator = generator
        self._labeler = labeler

    def generate_many_sync(self, requests: Sequence[GenRequest]) -> list[list[str]]:
        return [
            self._generator(r.rule_id, r.label, r.topic, r.n, seed=r.seed)
            for r in requests
        ]

    def label_many_sync(self, requests: Sequence[LabelRequest]) -> list[bool | None]:
        return [
            self._labeler(r.rule_id, r.which, r.text, seed=r.seed) for r in requests
        ]


async def _run_generate(dispatcher: Any, requests: Sequence[GenRequest]) -> list[list[str]]:
    """Dispatch a batch of generation requests, concurrently when the seam
    supports it, else directly. Results stay in request order.

    This is a COROUTINE so the WHOLE run shares ONE event loop: the concurrent
    seam's ``generate_many`` is *awaited* here, never wrapped in its own
    ``asyncio.run`` — every phase (generation rounds, validation, targeted regen)
    and the final ``seam.aclose()`` run under the single loop opened once by the
    caller (``run_pipeline_async``). A fresh ``asyncio.run`` per phase would bind
    the client's httpx pool to a loop that the next phase has already closed
    ('RuntimeError: Event loop is closed'). The mock dispatcher is synchronous, so
    its branch returns immediately without touching the loop."""
    if not requests:
        return []
    if getattr(dispatcher, "is_concurrent", False):
        return await dispatcher.generate_many(list(requests))
    return dispatcher.generate_many_sync(requests)


async def _run_label(dispatcher: Any, requests: Sequence[LabelRequest]) -> list[bool | None]:
    """Dispatch a batch of validation requests, concurrently when the seam
    supports it, else directly. Results stay in request order.

    Like ``_run_generate``, this AWAITS the concurrent seam's ``label_many`` on
    the caller's single loop instead of opening its own — so validation runs on
    the same loop generation did, on the same (still-open) client connection
    pool. The mock dispatcher's sync branch returns immediately."""
    if not requests:
        return []
    if getattr(dispatcher, "is_concurrent", False):
        return await dispatcher.label_many(list(requests))
    return dispatcher.label_many_sync(requests)


def _run_coro_sync(coro: Any) -> Any:
    """Drive a NON-SUSPENDING coroutine to completion with NO event loop.

    The mock seam's dispatch (``_CallableDispatcher``) returns its results without
    awaiting anything, so the orchestrator coroutine built over it never suspends:
    a single ``.send(None)`` runs it start-to-finish. This keeps the ``--mock``
    path exactly as synchronous as before (no loop is ever created). It is a
    programming error if such a coroutine DOES suspend (it would mean a real await
    slipped into the mock path), so we surface that loudly rather than hang."""
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    coro.close()
    raise RuntimeError(
        "mock pipeline coroutine suspended on an await; the synchronous (--mock) "
        "path must not touch the async client"
    )


def _run_coro(coro: Any, *, concurrent: bool) -> Any:
    """Run a pipeline coroutine to completion from a SYNC entry point.

    The production path (``__main__._run_real`` / ``run_api_build``) awaits the
    async orchestrator directly under its own single loop, so this helper is only
    the SYNC entry point the offline callers use (``--mock`` and the tests that
    drive ``run_pipeline`` / ``generate_candidates`` / ``validate_candidates``
    synchronously).

    ``concurrent`` is False for the MOCK seam: we drive the coroutine with NO event
    loop (``_run_coro_sync``), so ``--mock`` stays fully synchronous. ``concurrent``
    is True for a real seam used ONCE from sync code (the offline FakeAPI tests):
    we open ONE ``asyncio.run`` for this single call. Nesting inside an
    already-running loop is refused — that would be the multi-loop bug this fix
    removes; the in-loop caller must use the async orchestrator instead."""
    if not concurrent:
        return _run_coro_sync(coro)

    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "pipeline sync entry point called for a concurrent seam from inside a "
        "running event loop; use the async orchestrator (run_pipeline_async) so "
        "the whole run shares one loop"
    )


def _as_dispatcher(generator_or_seam: Any, labeler: Any | None) -> Any:
    """Resolve the run_pipeline (generator, labeler) arguments into a Dispatcher.

    Three call styles are supported:
      * a concurrent seam passed as BOTH args (``run_pipeline(rid, seam, seam)``)
        or as the generator with labeler=None — used directly (fans out calls);
      * a concurrent seam as the first arg with any labeler — the seam wins
        (it owns both phases);
      * two plain callables (the mock seam) — wrapped in _CallableDispatcher,
        which calls them in order with NO event loop (keeps --mock synchronous).
    """
    if getattr(generator_or_seam, "is_concurrent", False):
        return generator_or_seam
    return _CallableDispatcher(generator_or_seam, labeler)


@dataclass
class Candidate:
    """One generated sentence with its provenance through the pipeline."""

    text: str
    intended_label: bool
    topic: str
    # validation results, filled by the two passes (None until run / on neutral)
    pass_a: bool | None = None
    pass_b: bool | None = None
    # generation-time advisory tags (e.g. rule 16's eat_verb), for the audits
    tags: dict[str, bool] = field(default_factory=dict)
    # lazily-cached audited rate features (negator/stopword/comma/cap counts), so
    # the rate-aware refinement swap loop does not re-tokenize; not provenance.
    _audit_features: dict[str, float] | None = field(default=None, repr=False, compare=False)

    @property
    def cell(self) -> tuple[str, bool]:
        """The (topic, class) rebalancing cell key."""
        return (self.topic, self.intended_label)

    @property
    def kept(self) -> bool:
        """Both passes returned the SAME non-None label equal to the intended
        label (rule-specs: both validators return the intended label)."""
        return (
            self.pass_a is not None
            and self.pass_a == self.pass_b
            and self.pass_a == self.intended_label
        )


@dataclass
class PipelineResult:
    """What ``run_pipeline`` returns (no I/O side effects captured here)."""

    rule_id: str
    seed: int
    n_generated: int
    n_kept: int
    drop_rate: float
    per_cell_kept: dict[str, int]  # "topic|class" -> surviving count (pre-rebalance)
    quota_per_cell: int
    n_emitted: int
    split_counts: dict[str, dict[str, int]]
    keyword_audit: dict[str, Any]
    confound_overall_pass: bool
    items_path: str | None
    confound_report_path: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "seed": self.seed,
            "n_generated": self.n_generated,
            "n_kept": self.n_kept,
            "drop_rate": self.drop_rate,
            "per_cell_kept": self.per_cell_kept,
            "quota_per_cell": self.quota_per_cell,
            "n_emitted": self.n_emitted,
            "split_counts": self.split_counts,
            "keyword_audit": self.keyword_audit,
            "confound_overall_pass": self.confound_overall_pass,
            "items_path": self.items_path,
            "confound_report_path": self.confound_report_path,
        }


# ===========================================================================
# style filtering (post-generation, before validation)
# ===========================================================================


def style_ok(text: str) -> bool:
    """Cheap programmatic style gate (rule-specs globals.sentence_style for the
    LLM rules): strip-then-check that the candidate could be admitted by
    schema.assert_sentence_style with the comma-allowed llm policy.

    Drops anything with banned punctuation, '!' / '?', a contraction apostrophe,
    a hyphen/dash, non-ASCII (emoji), the pronoun 'I', or a word count outside
    [4, 12]. The pipeline calls this BEFORE spending validator calls so junk
    never reaches the (paid) validators."""
    if not text or not text.isascii():
        return False
    if any(ch in _BANNED_PUNCT for ch in text):
        return False
    if any(sub in text for sub in _BANNED_SUBSTRINGS):
        return False
    if any(t == "I" for t in words(text)):
        return False
    n = word_count(text)
    if not (4 <= n <= 12):
        return False
    return True


def normalize_candidate(raw: str) -> str | None:
    """Clean a raw generation line into a candidate sentence, or None to drop.

    Strips surrounding whitespace and a single trailing period (the recipe:
    'LLM-generated items have terminal periods stripped post-hoc'), removes a
    leading list marker the model may have added despite instructions, then
    requires the result to pass ``style_ok``. Items still carrying '?'/'!' or a
    quote/contraction are dropped by ``style_ok`` (recipe: such items are
    dropped, not repaired)."""
    s = raw.strip()
    if not s:
        return None
    # strip a leading "1. " / "- " / "* " list marker if the model added one
    while s[:1] in {"-", "*", "•"}:
        s = s[1:].strip()
    # leading "<n>." numbering
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i > 0 and i < len(s) and s[i] in {".", ")"}:
        s = s[i + 1 :].strip()
    # strip a single trailing period (the ONLY terminal-punctuation rewrite)
    if s.endswith("."):
        s = s[:-1].rstrip()
    if not style_ok(s):
        return None
    return s


# ===========================================================================
# generation
# ===========================================================================


def _max_gen_calls(cfg: LLMRuleConfig, n_target: int) -> int:
    """Per-cell generation call budget (the loop bound that keeps a pathological
    generator from looping forever)."""
    return max(4, math.ceil(n_target / max(1, cfg.pairs_per_call)) * 4)


def _ingest_cell_raws(
    cfg: LLMRuleConfig,
    label: bool,
    topic: str,
    raws: list[str],
    out: list[Candidate],
    seen: set[str],
) -> None:
    """Normalize + de-dup one call's raw lines into a cell's candidate list,
    in place. Identical per-line logic to the old sequential loop."""
    for raw in raws:
        norm = normalize_candidate(raw)
        if norm is None or norm in seen:
            continue
        seen.add(norm)
        tags = {name: pred(norm) for name, pred in cfg.candidate_tags.items()}
        out.append(Candidate(text=norm, intended_label=label, topic=topic, tags=tags))


async def _generate_cells_async(
    cfg: LLMRuleConfig,
    cells: Sequence[tuple[bool, str, int]],
    n_target: int,
    dispatcher: Any,
) -> dict[int, list[Candidate]]:
    """Generate (and style-filter) >= ``n_target`` distinct candidates for each
    cell, fanning out the calls of a ROUND across all still-short cells.

    ``cells`` is ``[(label, topic, cell_seed), ...]`` keyed by index. Round ``r``
    issues call ``r`` (seed ``cell_seed + r``) for EVERY cell still under target,
    all concurrently; results are folded back per cell in input order. This issues
    the EXACT same per-cell request set ``{cell_seed + i : i in [0, k_cell)}`` the
    old per-cell sequential loop did (same early-stop at ``n_target``), so the
    cache key and the seed->call mapping are byte-identical — only the dispatch
    (across cells, within a round) is now concurrent.

    Awaits each round's dispatch on the caller's loop (no per-round
    ``asyncio.run``), so every generation round shares the run's single loop."""
    max_calls = _max_gen_calls(cfg, n_target)
    out: dict[int, list[Candidate]] = {i: [] for i in range(len(cells))}
    seen: dict[int, set[str]] = {i: set() for i in range(len(cells))}
    for call_i in range(max_calls):
        # which cells still need more candidates this round (stable order)
        active = [i for i in range(len(cells)) if len(out[i]) < n_target]
        if not active:
            break
        requests = [
            GenRequest(cfg.rule_id, cells[i][0], cells[i][1], cfg.pairs_per_call, cells[i][2] + call_i)
            for i in active
        ]
        results = await _run_generate(dispatcher, requests)
        for i, raws in zip(active, results):
            label, topic, _ = cells[i]
            _ingest_cell_raws(cfg, label, topic, raws, out[i], seen[i])
    return out


async def _generate_cell_async(
    cfg: LLMRuleConfig,
    label: bool,
    topic: str,
    n_target: int,
    dispatcher: Any,
    seed: int,
) -> list[Candidate]:
    """Async core of the single-cell generate (targeted-regen path); delegates to
    ``_generate_cells_async`` so the dispatch is identical (a one-cell round)."""
    return (await _generate_cells_async(cfg, [(label, topic, seed)], n_target, dispatcher))[0]


async def generate_candidates_async(
    cfg: LLMRuleConfig,
    per_cell_target: int,
    dispatcher: Any,
    seed: int,
) -> list[Candidate]:
    """Async core of ``generate_candidates`` (see its docstring). Runs every
    generation round on the caller's single event loop."""
    rng = Random(seed)
    # True cells over true_topics, False cells over false_topics; per-cell seed is
    # drawn in the SAME order as before so each cell sees its old seed.
    cells: list[tuple[bool, str, int]] = []
    for topic in cfg.true_topics:
        cells.append((True, topic, rng.randint(0, 2**31 - 1)))
    for topic in cfg.false_topics:
        cells.append((False, topic, rng.randint(0, 2**31 - 1)))
    per_cell = await _generate_cells_async(cfg, cells, per_cell_target, dispatcher)
    cands: list[Candidate] = []
    for i in range(len(cells)):
        cands.extend(per_cell[i])
    return cands


def generate_candidates(
    cfg: LLMRuleConfig,
    per_cell_target: int,
    dispatcher: Any,
    seed: int,
) -> list[Candidate]:
    """Generate matched pairs PER TOPIC for both classes (topic balance is exact
    by construction, rule-specs recipe). ``per_cell_target`` candidates are aimed
    for each (topic, class) cell so post-validation drops can be absorbed.

    All cells are generated together so each round's calls fan out concurrently
    (real seam) up to the client's concurrency bound; the per-cell seed sequence
    is unchanged, so the generated corpus is byte-identical to the old serial
    build.

    SYNC entry point (used by offline tests / the mock path): the mock dispatcher
    runs with NO event loop; a concurrent seam used once gets ONE asyncio.run. The
    production run uses ``generate_candidates_async`` (via ``run_pipeline_async``)
    so its loop is shared with validation and aclose."""
    return _run_coro(
        generate_candidates_async(cfg, per_cell_target, dispatcher, seed),
        concurrent=getattr(dispatcher, "is_concurrent", False),
    )


# ===========================================================================
# two-pass validation
# ===========================================================================


async def validate_candidates_async(
    cfg: LLMRuleConfig,
    cands: Sequence[Candidate],
    dispatcher: Any,
    seed: int,
) -> None:
    """Async core of ``validate_candidates`` (see its docstring). Awaits the whole
    batch on the caller's single event loop."""
    requests: list[LabelRequest] = []
    for i, c in enumerate(cands):
        requests.append(LabelRequest(cfg.rule_id, "A", c.text, seed + i))
        requests.append(LabelRequest(cfg.rule_id, "B", c.text, seed + i))
    results = await _run_label(dispatcher, requests)
    for i, c in enumerate(cands):
        c.pass_a = results[2 * i]
        c.pass_b = results[2 * i + 1]


def validate_candidates(
    cfg: LLMRuleConfig,
    cands: Sequence[Candidate],
    dispatcher: Any,
    seed: int,
) -> None:
    """Run BOTH validation passes over every candidate, in place.

    pass A = gpt-4.1-mini (validator prompt A), pass B = gpt-4.1 (differently
    worded prompt B). Each pass returns True / False / None; ``Candidate.kept``
    then requires both to equal the intended label. The per-candidate seed
    (``seed + i``) is threaded EXACTLY as the old sequential loop did, so the
    real client's cache key (and the mock) are unchanged.

    The 2*len(cands) calls (pass A and pass B for every candidate) are collected
    and dispatched CONCURRENTLY in one batch — bounded by the client's
    Semaphore(concurrency) — instead of awaiting them one at a time; the
    request->result order is preserved, so each candidate's pass_a/pass_b are
    written from its own calls.

    SYNC entry point (offline tests / mock path): the mock dispatcher runs with NO
    event loop; a concurrent seam used once gets ONE asyncio.run. The production
    run uses ``validate_candidates_async`` so its loop is shared with generation
    and aclose (the single-loop fix)."""
    _run_coro(
        validate_candidates_async(cfg, cands, dispatcher, seed),
        concurrent=getattr(dispatcher, "is_concurrent", False),
    )


# ===========================================================================
# rebalancing (down-sample to exact quota per cell; targeted regen for short)
# ===========================================================================


def _quota_per_cell(cfg: LLMRuleConfig, n_target_items: int) -> int:
    """Per (topic x class) cell quota so the emitted set is exact 50/50 + topic
    balanced. With T true-topics and F false-topics, half the items are True
    (spread over T topic cells) and half False (over F cells); to keep EVERY
    topic exactly 50/50 (rule 15) / every non-food topic <= 25% (rule 16) we use
    a single per-cell quota = ceil( (n/2) / max(T, F) ). The emitted total may
    slightly exceed n_target_items; the split assigner takes exactly the split
    sizes and the rest go to spare."""
    n_true_topics = len(cfg.true_topics)
    n_false_topics = len(cfg.false_topics)
    half = math.ceil(n_target_items / 2)
    per_true = math.ceil(half / n_true_topics)
    per_false = math.ceil(half / n_false_topics)
    return max(per_true, per_false)


def rebalance(
    cfg: LLMRuleConfig,
    kept: Sequence[Candidate],
    quota: int,
    seed: int,
) -> tuple[list[Candidate], dict[str, int], list[tuple[bool, str]]]:
    """Down-sample each (topic, class) cell at random to exactly ``quota`` items.

    Returns (balanced_candidates, per_cell_surviving_counts, short_cells).
    ``short_cells`` are cells that did not reach the quota (the caller runs
    targeted regeneration for them). Within-cell down-sampling is a seeded random
    draw (rule-specs: 'down-sample at random within each topic x class cell')."""
    by_cell: dict[tuple[str, bool], list[Candidate]] = defaultdict(list)
    for c in kept:
        by_cell[c.cell].append(c)

    rng = Random(seed)
    balanced: list[Candidate] = []
    per_cell: dict[str, int] = {}
    short: list[tuple[bool, str]] = []

    # iterate cells in a stable order (topic, label) so output is deterministic
    all_cells: list[tuple[bool, str]] = [(True, t) for t in cfg.true_topics] + [
        (False, t) for t in cfg.false_topics
    ]
    for label, topic in all_cells:
        cell = (topic, label)
        members = sorted(by_cell.get(cell, []), key=lambda c: c.text)
        per_cell[f"{topic}|{'T' if label else 'F'}"] = len(members)
        if len(members) < quota:
            short.append((label, topic))
            balanced.extend(members)  # take what we have; caller regenerates
            continue
        rng.shuffle(members)
        balanced.extend(members[:quota])
    return balanced, per_cell, short


def _enforce_exact_balance(
    cfg: LLMRuleConfig, balanced: Sequence[Candidate], seed: int
) -> list[Candidate]:
    """After down-sampling, trim to an EXACT 50/50 True/False total while keeping
    every cell as even as possible. Down-samples the larger class's cells (largest
    first) until the class counts match. Loud if either class is empty."""
    trues = [c for c in balanced if c.intended_label]
    falses = [c for c in balanced if not c.intended_label]
    if not trues or not falses:
        raise LLMPipelineError(
            f"rule {cfg.rule_id!r}: a class is empty after rebalancing "
            f"(T={len(trues)} F={len(falses)})"
        )
    target = min(len(trues), len(falses))
    rng = Random(seed)

    def _trim(items: list[Candidate]) -> list[Candidate]:
        if len(items) <= target:
            return items
        # group by topic, drop from the largest cells first, evenly
        by_topic: dict[str, list[Candidate]] = defaultdict(list)
        for c in items:
            by_topic[c.topic].append(c)
        for t in by_topic:
            rng.shuffle(by_topic[t])
        keep: list[Candidate] = []
        # round-robin take from topics so the trim stays topic-balanced
        topics = sorted(by_topic, key=lambda t: (-len(by_topic[t]), t))
        idx = {t: 0 for t in topics}
        while len(keep) < target:
            progressed = False
            for t in topics:
                if len(keep) >= target:
                    break
                if idx[t] < len(by_topic[t]):
                    keep.append(by_topic[t][idx[t]])
                    idx[t] += 1
                    progressed = True
            if not progressed:
                break
        return keep

    return _trim(trues) + _trim(falses)


# ===========================================================================
# audit-rate-aware selection refinement (rules 15/16)
# ===========================================================================
#
# The spec audit_thresholds bound the |True - False| gap on four per-class rates
# (negator / stopword / comma / capitalized-word). Validator drops are asymmetric
# AND the rate-bearing tokens are not evenly distributed, so the random cell
# down-sample can land on a marginal-but-failing subset even when a COMPLIANT one
# exists in the candidate pool (the real rule-15 negator_rate=0.0511 reject). This
# refinement holds the exact topic x class balance FIXED and SWAPS items in/out
# WITHIN each cell (selected <-> that cell's validated leftovers) to drive every
# rate gap under threshold. It is pure post-validation SELECTION: it never asks
# the generator/validators for anything, so the disk cache still fully hits on a
# re-run, and it is fully seed-deterministic.

# The four audited rate features, each (kind, numerator_fn, threshold). 'item'
# rates count items-with-the-feature / n_items; 'token' rates count
# feature-tokens / total-tokens. These MATCH confound.class_stats exactly.
_RATE_FEATURES: tuple[tuple[str, str, float], ...] = (
    ("negator", "item", LLM_NEGATOR_RATE_TOL),
    ("stopword", "token", LLM_STOPWORD_RATE_TOL),
    ("comma", "item_count", LLM_COMMA_RATE_TOL),
    ("capitalized", "token", LLM_CAP_RATE_TOL),
)


def _candidate_features(c: Candidate) -> dict[str, float]:
    """The per-candidate counts the four audited rates are built from, computed
    EXACTLY as confound.class_stats / build_audit_thresholds do (same tokenizer,
    same stopword/negator sets, same caps test). Cached on the candidate so the
    greedy swap loop does not retokenize."""
    cached = getattr(c, "_audit_features", None)
    if cached is not None:
        return cached
    toks = words(c.text)
    n_tok = len(toks)
    lows = [t.lower() for t in toks]
    feats = {
        # negator_rate: items containing >= 1 negator / n_items (per item)
        "negator_num": 1.0 if any(t in NEGATORS for t in lows) else 0.0,
        "negator_den": 1.0,
        # stopword_rate: stopword tokens / total tokens (token weighted)
        "stopword_num": float(sum(1 for t in lows if t in STOPWORDS)),
        "stopword_den": float(n_tok),
        # comma_rate: commas per item (per item; den is 1 item)
        "comma_num": float(c.text.count(",")),
        "comma_den": 1.0,
        # capitalized_word_rate: caps tokens / total tokens (token weighted)
        "capitalized_num": float(sum(1 for t in toks if t[:1].isupper())),
        "capitalized_den": float(n_tok),
    }
    c._audit_features = feats  # type: ignore[attr-defined]
    return feats


class _ClassRateAccumulator:
    """Running per-class numerators/denominators for the four audited rates, so a
    swap's effect is an O(1) add/remove rather than a full re-tokenize of the set.
    Mirrors confound.class_stats: stopword/cap rates are token-weighted, negator/
    comma rates are per item."""

    def __init__(self, members: Sequence[Candidate]) -> None:
        self.num: dict[str, float] = defaultdict(float)
        self.den: dict[str, float] = defaultdict(float)
        for c in members:
            self.add(c)

    def add(self, c: Candidate) -> None:
        f = _candidate_features(c)
        for kind, _mode, _tol in _RATE_FEATURES:
            self.num[kind] += f[f"{kind}_num"]
            self.den[kind] += f[f"{kind}_den"]

    def remove(self, c: Candidate) -> None:
        f = _candidate_features(c)
        for kind, _mode, _tol in _RATE_FEATURES:
            self.num[kind] -= f[f"{kind}_num"]
            self.den[kind] -= f[f"{kind}_den"]

    def rate(self, kind: str) -> float:
        d = self.den[kind]
        return (self.num[kind] / d) if d else 0.0


def _worst_gap(true_acc: _ClassRateAccumulator, false_acc: _ClassRateAccumulator) -> tuple[str, float, float]:
    """Return (kind, gap, slack) for the rate with the largest threshold-normalized
    violation. ``slack = gap - threshold`` (> 0 means failing). When all pass, the
    returned kind still names the tightest rate but slack is <= 0."""
    worst_kind = _RATE_FEATURES[0][0]
    worst_slack = -float("inf")
    worst_gap = 0.0
    for kind, _mode, tol in _RATE_FEATURES:
        gap = abs(true_acc.rate(kind) - false_acc.rate(kind))
        slack = gap - tol
        if slack > worst_slack:
            worst_slack = slack
            worst_kind = kind
            worst_gap = gap
    return worst_kind, worst_gap, worst_slack


def _all_rates_pass(true_acc: _ClassRateAccumulator, false_acc: _ClassRateAccumulator) -> bool:
    for kind, _mode, tol in _RATE_FEATURES:
        if abs(true_acc.rate(kind) - false_acc.rate(kind)) > tol + 1e-12:
            return False
    return True


def _sum_normalized_gap(true_acc: _ClassRateAccumulator, false_acc: _ClassRateAccumulator) -> float:
    """Total threshold-normalized gap across the four rates (the tie-break score a
    swap minimizes once the binding gap is reduced): sum(gap / tol). Lower is
    better; pushes ALL rates toward parity, not just the worst."""
    total = 0.0
    for kind, _mode, tol in _RATE_FEATURES:
        gap = abs(true_acc.rate(kind) - false_acc.rate(kind))
        total += gap / tol if tol else 0.0
    return total


def refine_rate_audits(
    cfg: LLMRuleConfig,
    selected: Sequence[Candidate],
    pool: Sequence[Candidate],
    seed: int,
    *,
    max_iters: int = 5000,
) -> list[Candidate]:
    """Audit-rate-aware refinement of the final selection (rules 15/16).

    Holds the EXACT per-cell selected count (and thus topic x class + 50/50
    balance) FIXED, and greedily SWAPS a selected item out for a same-cell
    validated leftover whenever the swap reduces the worst threshold-normalized
    rate gap (ties broken by the summed normalized gap, so all four rates move
    toward parity). Pure selection over the already-validated ``pool`` — no model
    call — so the disk cache is untouched. Deterministic in ``seed``.

    ``selected`` is the post-balance emitted set; ``pool`` is every validated
    candidate (``kept``) — the per-cell leftovers are ``pool \\ selected``. Returns
    the refined selection (same length, same cell counts). Raises nothing here; the
    caller's audit gate surfaces a genuinely infeasible pool."""
    rng = Random(seed)
    selected_ids = {id(c) for c in selected}
    # per-cell selected lists + same-cell leftovers (validated, not selected).
    by_cell_sel: dict[tuple[str, bool], list[Candidate]] = defaultdict(list)
    by_cell_left: dict[tuple[str, bool], list[Candidate]] = defaultdict(list)
    for c in selected:
        by_cell_sel[c.cell].append(c)
    for c in pool:
        if id(c) not in selected_ids:
            by_cell_left[c.cell].append(c)
    # stable, seeded leftover order so the search is deterministic
    for cell in by_cell_left:
        by_cell_left[cell].sort(key=lambda c: c.text)
        rng.shuffle(by_cell_left[cell])

    true_members = [c for c in selected if c.intended_label]
    false_members = [c for c in selected if not c.intended_label]
    true_acc = _ClassRateAccumulator(true_members)
    false_acc = _ClassRateAccumulator(false_members)

    if _all_rates_pass(true_acc, false_acc):
        return list(selected)

    # cells iterated in a stable order each pass
    cells_order = [(True, t) for t in cfg.true_topics] + [(False, t) for t in cfg.false_topics]

    for _ in range(max_iters):
        if _all_rates_pass(true_acc, false_acc):
            break
        _, _, base_slack = _worst_gap(true_acc, false_acc)
        base_sum = _sum_normalized_gap(true_acc, false_acc)
        # the current state is the baseline a swap must strictly beat, scored
        # lexicographically (worst threshold-normalized slack first, then the
        # summed normalized gap so non-binding rates also drift toward parity).
        base_score = (base_slack, base_sum)
        best_score = base_score
        best_move: tuple[tuple[str, bool], int, int] | None = None
        # search every within-cell swap (selected_i <-> leftover_j); the
        # accumulators are mutated in place and restored, so each trial is O(1).
        for label, topic in cells_order:
            cell = (topic, label)
            sel = by_cell_sel.get(cell)
            left = by_cell_left.get(cell)
            if not sel or not left:
                continue
            acc = true_acc if label else false_acc
            for si, out_c in enumerate(sel):
                acc.remove(out_c)
                for lj, in_c in enumerate(left):
                    if in_c is out_c:
                        continue
                    acc.add(in_c)
                    _, _, slack = _worst_gap(true_acc, false_acc)
                    score = (slack, _sum_normalized_gap(true_acc, false_acc))
                    # strict lexicographic improvement over the best so far
                    if score[0] < best_score[0] - 1e-12 or (
                        abs(score[0] - best_score[0]) <= 1e-12
                        and score[1] < best_score[1] - 1e-12
                    ):
                        best_score = score
                        best_move = (cell, si, lj)
                    acc.remove(in_c)
                acc.add(out_c)
        if best_move is None:
            break  # no strictly-improving swap exists; caller's gate decides
        cell, si, lj = best_move
        label = cell[1]
        acc = true_acc if label else false_acc
        out_c = by_cell_sel[cell][si]
        in_c = by_cell_left[cell][lj]
        acc.remove(out_c)
        acc.add(in_c)
        by_cell_sel[cell][si] = in_c
        by_cell_left[cell][lj] = out_c

    refined: list[Candidate] = []
    for label, topic in cells_order:
        refined.extend(by_cell_sel.get((topic, label), []))
    # include any cells not in the canonical order (defensive; should be none)
    for cell, members in by_cell_sel.items():
        if (cell[1], cell[0]) not in cells_order:
            refined.extend(members)
    return refined


# ===========================================================================
# keyword-quota audit (rule 16) + skew audit hook
# ===========================================================================


def keyword_audit(cfg: LLMRuleConfig, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Programmatic keyword-predicate agreement audit.

    For rule 16 this enforces BOTH halves of the spec gate (rule-specs MAJOR-2):
    the eat/drink/taste/cook-verb predicate must agree <= 75% with the labels
    AND >= 55% of True items must contain NO such verb (the realized no-verb
    fraction, not just the generation-prompt request). ``passes`` is the AND of
    the two; the pipeline raises if it is False, so a short no-verb fraction
    rejects the dataset rather than emitting a confounded one. For rule 15 there
    is no keyword predicate, so this reports the negator-rate gap instead
    (audit_threshold |negator rate diff| <= 0.05; also a hard gate via the
    confound audit_thresholds)."""
    if "eat_verb" in cfg.candidate_tags:
        from .config import contains_eat_verb

        n = len(items)
        agree = 0
        true_no_verb = true_total = 0
        for it in items:
            label = bool(it["label"])
            has = contains_eat_verb(it["text"])
            # predicate 'mentions eating' -> True; agreement with the food label
            if has == label:
                agree += 1
            if label:
                true_total += 1
                if not has:
                    true_no_verb += 1
        agreement = agree / n if n else 0.0
        true_no_verb_fraction = (true_no_verb / true_total) if true_total else 0.0
        # rule-specs MAJOR-2: >= 55% of True items contain NO eat/drink/taste/cook
        # verb. This is a HARD post-validation gate (not just a generation-prompt
        # request): both the agreement cap AND the no-verb quota must hold or the
        # dataset is rejected (a short no-verb fraction means 'mentions eating'
        # would be an equivalent, not a distractor).
        agreement_ok = max(agreement, 1 - agreement) <= 0.75
        quota_ok = true_no_verb_fraction >= 0.55
        return {
            "predicate": "mentions_eat_drink_taste_cook_verb",
            "agreement": agreement,
            "max_agreement": max(agreement, 1 - agreement),
            "threshold": 0.75,
            "agreement_ok": agreement_ok,
            "true_no_verb_fraction": true_no_verb_fraction,
            "no_verb_quota": 0.55,
            "no_verb_quota_ok": quota_ok,
            "passes": agreement_ok and quota_ok,
        }
    # rule 15: negator-rate gap
    negators = {"not", "never", "no", "nothing", "nobody"}

    def neg_rate(group: list[dict[str, Any]]) -> float:
        if not group:
            return 0.0
        hits = sum(
            1 for it in group if any(w.lower() in negators for w in words(it["text"]))
        )
        return hits / len(group)

    trues = [it for it in items if bool(it["label"])]
    falses = [it for it in items if not bool(it["label"])]
    rt, rf = neg_rate(trues), neg_rate(falses)
    return {
        "predicate": "negator_rate",
        "true_rate": rt,
        "false_rate": rf,
        "abs_diff": abs(rt - rf),
        "threshold": 0.05,
        "passes": abs(rt - rf) <= 0.05,
    }


# ===========================================================================
# emit
# ===========================================================================


def candidates_to_items(
    cfg: LLMRuleConfig, balanced: Sequence[Candidate], seed: int
) -> list[dict[str, Any]]:
    """Turn balanced candidates into schema items with split assignment +
    validator-agreement provenance.

    base_id == item_id (llm rule). Split assignment uses schema.assign_llm_splits
    (item-level, balanced 120/120/100, remainder -> spare). Each item carries
    slots_meta[VALIDATED_FLAG] = the agreed label (the provenance
    groundtruth.assert_labels_correct requires for validator-derived rules)."""
    # build a stable item_id per candidate
    item_ids: list[str] = []
    labels: list[bool] = []
    for i, c in enumerate(sorted(balanced, key=lambda c: (c.intended_label, c.topic, c.text))):
        item_ids.append(f"{cfg.rule_id}-{i:04d}")
        labels.append(c.intended_label)
    ordered = sorted(balanced, key=lambda c: (c.intended_label, c.topic, c.text))

    assignment = assign_llm_splits(item_ids, labels, seed)

    items: list[dict[str, Any]] = []
    for iid, c in zip(item_ids, ordered):
        meta = {
            "seed": seed,
            "topic": c.topic,
            "intended_label": c.intended_label,
            # the two-validator agreement provenance (REQUIRED by groundtruth):
            VALIDATED_FLAG: c.intended_label,
            "validator_pass_a": c.pass_a,
            "validator_pass_b": c.pass_b,
            "validator_a_model": VALIDATOR_A_MODEL,
            "validator_b_model": VALIDATOR_B_MODEL,
            "gen_model": GEN_MODEL,
            "tags": c.tags,
        }
        items.append(
            make_item(
                item_id=iid,
                base_id=iid,  # llm rule: base_id == item_id
                rule_id=cfg.rule_id,
                label=c.intended_label,
                text=c.text,
                slots_meta=meta,
                split=assignment[iid],
            )
        )
    return items


# ===========================================================================
# the orchestrator
# ===========================================================================


def run_pipeline(
    rule_id: str,
    generator: Generator,
    labeler: Labeler | None = None,
    *,
    seed: int = 0,
    max_candidates: int = 600,
    data_dir: Path | str | None = None,
    write: bool = True,
    run_pos: bool = True,
) -> PipelineResult:
    """Run the full generate -> validate -> rebalance -> emit flow for one
    llm_validated rule, with the model calls supplied by the injected seam
    (real client or mock).

    Parameters
    ----------
    rule_id          'positive_sentiment' or 'food_topic'.
    generator        EITHER a concurrent seam (``ClientSeam``, exposing
                     ``generate_many`` / ``label_many`` and ``is_concurrent``)
                     that owns BOTH phases, OR a plain Generator callable
                     (rule_id, label, topic, n, *, seed)->list[str].
    labeler          plain Labeler callable (rule_id, which, text, *, seed)->
                     bool|None; ignored when ``generator`` is a concurrent seam.
    seed             single seed threaded through generation/validation/splits.
    max_candidates   target number of candidates to generate (>= 600 for the real
                     run; the offline test passes a tiny number). Spread evenly
                     over the (topic x class) cells.
    data_dir         output root (defaults to repo data/); items + report go to
                     <data_dir>/<rule_id>/.
    write            if False, run every gate but do not write items.jsonl (the
                     confound report is still written as a gate artifact).
    run_pos          passed to the battery / confound report (False skips nltk POS).

    The real seam fans out each generation round + the whole validation set
    CONCURRENTLY (bounded by the client's semaphore); the mock callables are
    driven directly in order (no event loop). Either way the per-call seeds and
    request content are unchanged, so the emitted dataset is identical.

    SYNC entry point. With the MOCK seam there is no event loop at all (the
    callable dispatcher returns synchronously). With a concurrent seam this opens
    ONE event loop for the whole run via ``run_pipeline_async`` — generation,
    validation, targeted regen, and emit all share it. The production CLI does NOT
    call this; it awaits ``run_pipeline_async`` directly so that ``seam.aclose()``
    runs on the SAME loop (the single-loop fix — see ``__main__._run_real``).

    Raises LLMPipelineError / GroundTruthError / SchemaError on any gate failure.
    Returns a PipelineResult.
    """
    # a concurrent seam is passed as ``generator`` (it owns both phases); the mock
    # seam is two plain callables. Only the former needs an event loop.
    concurrent = getattr(generator, "is_concurrent", False)
    return _run_coro(
        run_pipeline_async(
            rule_id,
            generator,
            labeler,
            seed=seed,
            max_candidates=max_candidates,
            data_dir=data_dir,
            write=write,
            run_pos=run_pos,
        ),
        concurrent=concurrent,
    )


async def run_pipeline_async(
    rule_id: str,
    generator: Generator,
    labeler: Labeler | None = None,
    *,
    seed: int = 0,
    max_candidates: int = 600,
    data_dir: Path | str | None = None,
    write: bool = True,
    run_pos: bool = True,
) -> PipelineResult:
    """Async core of ``run_pipeline`` — the SINGLE-LOOP orchestrator.

    Generation rounds, both validation passes, targeted regeneration, and emit are
    all awaited here, so a concurrent seam's client stays bound to ONE event loop
    for the entire run. The caller (``__main__._run_real``) opens that loop once,
    awaits this, and then ``await``s ``seam.aclose()`` on the SAME loop — no second
    ``asyncio.run`` ever touches the client (which is what produced
    'RuntimeError: Event loop is closed'). The per-call seeds + request content are
    unchanged, so the emitted dataset and the disk-cache keys are identical."""
    cfg = get_rule_config(rule_id)
    dispatcher = _as_dispatcher(generator, labeler)
    out_root = Path(data_dir) if data_dir is not None else DATA_DIR
    out_dir = out_root / rule_id

    n_cells = len(cfg.true_topics) + len(cfg.false_topics)
    per_cell_target = max(1, math.ceil(max_candidates / n_cells))

    # 1) GENERATE -------------------------------------------------------------
    gen_seed = Random(seed).randint(0, 2**31 - 1)
    cands = await generate_candidates_async(cfg, per_cell_target, dispatcher, gen_seed)
    n_generated = len(cands)
    if n_generated == 0:
        raise LLMPipelineError(f"rule {rule_id!r}: generator produced no usable candidates")

    # 2) VALIDATE (2 passes) --------------------------------------------------
    val_seed = Random(seed + 1).randint(0, 2**31 - 1)
    await validate_candidates_async(cfg, cands, dispatcher, val_seed)
    kept = [c for c in cands if c.kept]
    n_kept = len(kept)
    drop_rate = 1.0 - (n_kept / n_generated) if n_generated else 1.0
    if n_kept == 0:
        raise LLMPipelineError(
            f"rule {rule_id!r}: every candidate was dropped by validation "
            f"(generated {n_generated}); raise max_candidates or check prompts"
        )

    # 3) REBALANCE (down-sample to quota; targeted regen for short cells) ------
    quota = _quota_per_cell(cfg, N_KEPT_MIN)
    balanced, per_cell_kept, short_cells = rebalance(
        cfg, kept, quota, Random(seed + 2).randint(0, 2**31 - 1)
    )

    # targeted regeneration for short cells (ONE extra round; real run loops the
    # client, the mock generator is deterministic so a wider net is requested).
    if short_cells:
        regen_seed = Random(seed + 3).randint(0, 2**31 - 1)
        extra: list[Candidate] = []
        for j, (label, topic) in enumerate(short_cells):
            existing = {c.text for c in kept if c.cell == (topic, label)}
            fresh = await _generate_cell_async(
                cfg, label, topic, quota * 3, dispatcher, regen_seed + j * 1000
            )
            fresh = [c for c in fresh if c.text not in existing]
            await validate_candidates_async(cfg, fresh, dispatcher, regen_seed + 500_000 + j * 1000)
            extra.extend([c for c in fresh if c.kept])
        if extra:
            kept = kept + extra
            n_kept = len(kept)
            balanced, per_cell_kept, short_cells = rebalance(
                cfg, kept, quota, Random(seed + 4).randint(0, 2**31 - 1)
            )

    # enforce EXACT 50/50 across the whole emitted set (topic-balanced trim)
    balanced = _enforce_exact_balance(cfg, balanced, Random(seed + 5).randint(0, 2**31 - 1))

    # AUDIT-RATE-AWARE REFINEMENT: hold the exact topic x class balance FIXED and
    # swap selected items for same-cell validated leftovers (from ``kept``) to push
    # the four audited rate gaps (negator/stopword/comma/cap) under their spec
    # thresholds when the pool permits — replacing the negator-blind random
    # down-sample that hard-failed on the marginal rule-15 negator gap. Pure
    # post-validation SELECTION (no model call), so the disk cache still fully
    # hits; deterministic in the threaded seed. A genuinely infeasible pool is left
    # failing and surfaced by the audit gate below.
    balanced = refine_rate_audits(
        cfg, balanced, kept, Random(seed + 7).randint(0, 2**31 - 1)
    )

    n_true = sum(1 for c in balanced if c.intended_label)
    n_false = len(balanced) - n_true
    # the balanced splits need 60+60+50 of each class = 170 per class minimum
    need_per_class = sum(LLM_SPLIT_ITEMS.values()) // 2  # (120+120+100)/2 = 170
    if n_true < need_per_class or n_false < need_per_class:
        raise LLMPipelineError(
            f"rule {rule_id!r}: not enough validated items to fill the "
            f"120/120/100 balanced splits after rebalancing "
            f"(have T={n_true} F={n_false}, need >= {need_per_class} each). "
            f"Raise max_candidates (real run uses >= 600) or check the validator "
            f"agreement rate. short_cells={short_cells}"
        )

    # 4) EMIT (build items + split + provenance) ------------------------------
    items = candidates_to_items(cfg, balanced, Random(seed + 6).randint(0, 2**31 - 1))

    # ---- GATE A: schema (style policy self-selected via rule_id) ----
    validate_full(items, rule_id=rule_id)

    # ---- GATE B: groundtruth (validator-derived: provenance must be present) ----
    assert_labels_correct(rule_id, items)

    # ---- keyword-quota audit (rule-relevant threshold) ----
    kw = keyword_audit(cfg, items)
    if not kw["passes"]:
        raise LLMPipelineError(
            f"rule {rule_id!r}: keyword/negator audit failed: {kw}"
        )

    # ---- GATE C+D: confound report (is_llm_rule=True -> audit_thresholds path) ----
    # Pass per-item topics so the topic-balance CONTENT HARD gate fires and the
    # judge-reported token-in-both-classes metric is computed. Rules 15/16 are
    # content rules; the STRUCTURAL audit_thresholds (5 rate gates + topic balance)
    # are "dataset REJECTED if violated", so build_confound_report folds them into
    # overall_pass and we reject loudly. The token-in-both-classes / skew lists are
    # judge-adjudicated (class-exclusive vocab is the rule's own signal for
    # sentiment/food), reported in the confound JSON, NOT auto-gated here.
    topics = [it["slots_meta"]["topic"] for it in items]
    report = build_confound_report(
        items,
        is_llm_rule=True,
        run_pos=run_pos,
        topics=topics,
    )
    report["keyword_audit"] = kw
    report["drop_rate"] = drop_rate
    report["n_generated"] = n_generated
    report["n_kept"] = n_kept
    if not report["overall_pass"]:
        failed = []
        at = report.get("audit_thresholds")
        if at is not None:
            # name each binding audit gate WITH its achieved value vs threshold, so
            # a genuinely infeasible pool surfaces the rate the refinement could not
            # close (e.g. negator_rate_abs_diff=0.071 > 0.05) rather than a bare
            # gate name. ``value`` is a float for the numeric-diff rate gates; the
            # content gates carry structured payloads, so fall back to the name.
            for k, c in at["checks"].items():
                if not (c["hard_gate"] and not c["passes"]):
                    continue
                v = c.get("value")
                if isinstance(v, (int, float)):
                    failed.append(f"{k}={float(v):.4f} > {c['threshold']}")
                else:
                    failed.append(k)
        raise LLMPipelineError(
            f"rule {rule_id!r}: confound report did not pass: "
            f"length_match_ok={report['length_match_ok']} "
            f"(|mean_T-mean_F|={report['word_count_mean_abs_diff']:.3f} > "
            f"{report['length_match_tolerance']}), "
            f"battery_ok={report['battery_ok']} "
            f"violations={report['battery_violations']}; "
            f"audit_hard_pass={report.get('audit_hard_pass')} "
            f"failed_audit_gates={failed}"
        )

    report_path = write_confound_report(report, out_dir / "confound_report.json")

    items_path: Path | None = None
    if write:
        items_path = write_items(items, out_dir / "items.jsonl")

    split_counts: dict[str, dict[str, int]] = {}
    for it in items:
        sc = split_counts.setdefault(it["split"], {"true": 0, "false": 0, "total": 0})
        sc["true" if bool(it["label"]) else "false"] += 1
        sc["total"] += 1

    return PipelineResult(
        rule_id=rule_id,
        seed=seed,
        n_generated=n_generated,
        n_kept=n_kept,
        drop_rate=drop_rate,
        per_cell_kept=per_cell_kept,
        quota_per_cell=quota,
        n_emitted=len(items),
        split_counts=split_counts,
        keyword_audit=kw,
        confound_overall_pass=bool(report["overall_pass"]),
        items_path=str(items_path) if items_path else None,
        confound_report_path=str(report_path),
    )


# ===========================================================================
# MOCK SEAM (zero network) — offline-testable generate/validate
# ===========================================================================
#
# The mocks build a tiny deterministic corpus that the mock labeler then
# validates as if both passes agreed. They are pure functions of their args, so
# the whole pipeline is reproducible and network-free under --mock.


# The mock corpus must itself PASS the new is_llm_rule audit_thresholds (the
# token-in-both-classes content gate, the topic-balance gate, and the matched
# rate diffs), exactly as a clean REAL dataset would. A real rule-15/16 dataset
# passes "every token with overall frequency >= 3% appears in both classes"
# because the SCAFFOLD vocabulary (function words + neutral frame words) is
# SHARED across both classes, while the genuinely class-skewed marker (the
# evaluative word / food word) is drawn from a LARGE pool so each individual
# marker's per-item doc-frequency stays below 3%. The mock mirrors that: every
# sentence is one SHARED neutral template instantiated with ONE rotating
# class-marker word; the marker pools are big (>= 30 each) so no marker reaches
# 3%, and the templates/connectors/tails are identical across classes.

# Shared, class-NEUTRAL sentence scaffolds (a {M} marker slot + an "item <uid>"
# tail). Used VERBATIM for both classes of a rule, so every scaffold word lands
# in BOTH classes (the only systematic True/False difference is the marker).
_MOCK_SCAFFOLDS: dict[str, list[str]] = {
    "positive_sentiment": [
        "the place we picked felt {M} to me today",
        "everyone agreed the visit there was rather {M}",
        "the whole afternoon turned out {M} in the end",
        "people kept saying the trip felt {M} this time",
        "the part we noticed most seemed {M} overall",
        "by the close the experience was clearly {M}",
        "the group came away thinking it was {M}",
        "most of the day there struck us as {M}",
    ],
    "food_topic": [
        "the {M} sat in the wide bowl near the window",
        "they left the {M} beside the long kitchen counter",
        "the {M} stayed on the shelf above the table",
        "a tray of {M} waited next to the stove",
        "the {M} rested in the basket by the door",
        "someone placed the {M} on the broad wooden board",
        "the {M} filled the dish at the back",
        "the {M} lined the rack inside the room",
    ],
}

# LARGE class-marker pools (>= 30 each). The marker is the ONLY rule-bearing
# token; with >= 30 markers rotated over ~175 items per class each marker sits
# at ~3% or below, so no SINGLE marker trips the 3% token-in-both-classes gate.
# The mock labeler recovers the class from the marker, so both passes agree.
_MOCK_MARKERS: dict[str, dict[str, list[str]]] = {
    "positive_sentiment": {
        "pos": [
            "wonderful", "delightful", "brilliant", "smooth", "lovely", "pleasant",
            "friendly", "charming", "superb", "excellent", "gorgeous", "joyful",
            "splendid", "graceful", "radiant", "cheerful", "fabulous", "elegant",
            "soothing", "uplifting", "magical", "glorious", "blissful", "vibrant",
            "refreshing", "heartening", "delicious", "stellar", "dazzling", "serene",
            "wholesome", "amazing", "marvellous", "agreeable", "satisfying", "sunny",
            "rewarding", "pleasing", "spirited", "lively", "luminous", "charismatic",
            "comforting", "memorable", "exquisite", "harmonious", "inviting",
            "cheery", "warmhearted", "thrilling",
        ],
        "neg": [
            "bland", "tiring", "dull", "clumsy", "miserable", "rude", "stressful",
            "frustrating", "careless", "dreary", "tedious", "awful", "dismal",
            "gloomy", "irritating", "shabby", "tiresome", "dreadful", "unpleasant",
            "disappointing", "lousy", "grim", "annoying", "horrible", "joyless",
            "draining", "tasteless", "sloppy", "depressing", "harsh", "tense",
            "forgettable", "abysmal", "disagreeable", "exhausting", "bleak",
            "wretched", "drab", "weary", "sour", "lacklustre", "dispiriting",
            "monotonous", "underwhelming", "graceless", "charmless", "sombre",
            "joylessly", "wearisome", "uninspiring",
        ],
    },
    "food_topic": {
        # food-denoting NOUNS, none an eat/drink/taste/cook VERB (so the no-verb
        # quota holds), each a plain food/dish/ingredient noun.
        "food": [
            "soup", "bread", "noodles", "tomatoes", "pastry", "rolls", "rice",
            "pie", "broth", "stew", "salad", "biscuits", "dumplings", "porridge",
            "muffins", "cheese", "pancakes", "lentils", "risotto", "chowder",
            "quiche", "tarts", "waffles", "scones", "casserole", "omelette",
            "curry", "ravioli", "cobbler", "custard", "crackers", "fritters",
            "gnocchi", "couscous", "polenta", "pretzels", "lasagne", "tortillas",
            "noodle", "pudding", "marmalade", "granola", "oatmeal", "burrito",
            "tamales", "kebabs", "falafel", "hummus", "guacamole", "chutney",
        ],
        "notfood": [
            # plain non-food nouns spanning the False topics (sports, weather,
            # transport, work, music, gardening flowers), each clearly off-topic.
            "racket", "hurdles", "umbrella", "drizzle", "bicycle", "tractor",
            "ledger", "stapler", "trumpet", "cello", "tulips", "daisies",
            "compass", "lantern", "saddle", "anchor", "telescope", "kettle",
            "shovel", "hammer", "violin", "drumstick", "scooter", "canoe",
            "binoculars", "wrench", "harmonica", "marigolds", "petals", "trowel",
            "gauge", "satchel", "abacus", "javelin", "skateboard", "barometer",
            "paddle", "goggles", "trophy", "raincoat", "timetable", "keyboard",
            "saxophone", "lilies", "daffodils", "sprinkler", "spanner", "helmet",
            "snowflake", "dashboard",
        ],
    },
}


def mock_generator(rule_id: str, label: bool, topic: str, n: int, *, seed: int) -> list[str]:
    """Deterministic, network-free Generator for offline tests.

    Produces ``n`` distinct, style-compliant sentences for the (label, topic)
    cell. Each is a SHARED neutral scaffold (identical structure + vocabulary
    across both classes) with ONE rotating class-marker word substituted in, plus
    a LABEL-NEUTRAL unique numeric tail ("item <N>"). Because the scaffold is
    shared and the markers are drawn from a large pool, the only token that
    systematically skews by class is the marker — and each marker stays under the
    3% doc-frequency floor — so the emitted mock corpus PASSES the is_llm_rule
    audit_thresholds (token-in-both-classes, matched rate diffs, topic balance),
    just like a clean real dataset. Pure function of its args (seed/index makes
    every sentence distinct and reproducible)."""
    scaffolds = _MOCK_SCAFFOLDS[rule_id]
    pools = _MOCK_MARKERS[rule_id]
    if rule_id == "positive_sentiment":
        markers = pools["pos"] if label else pools["neg"]
    else:
        markers = pools["food"] if label else pools["notfood"]
    rng = Random(f"{seed}:{rule_id}:{label}:{topic}")
    out: list[str] = []
    n_mark = len(markers)
    for i in range(n):
        # a STABLE per-item hash (hashlib, not the salted builtin hash) so the
        # corpus is reproducible across processes AND the generator is called
        # many times per cell (the _generate_cell call loop and the targeted-regen
        # path each pass a different ``seed``; ``i`` only runs 0..pairs_per_call-1
        # within a call). Deriving BOTH the marker index and the unique id from
        # this hash spreads markers UNIFORMLY across the whole cell — so no single
        # marker reaches the 3% token-in-both-classes floor, regardless of how
        # many calls the cell needed.
        h = int(hashlib.sha256(f"{rule_id}:{label}:{topic}:{seed}:{i}".encode()).hexdigest(), 16)
        scaffold = scaffolds[(seed + i) % len(scaffolds)]
        # marker index from a CONTIGUOUS within-cell counter. _generate_cell
        # advances ``seed`` by exactly 1 per call and always passes the same batch
        # size ``n``, with ``i`` running 0..n-1; so (seed * n + i) is a strictly
        # contiguous global position across the whole cell (up to a constant
        # offset). Taking it mod the pool size is therefore EXACT round-robin —
        # every marker is used in turn, so the kept items of a cell carry DISTINCT
        # markers and no marker can pile up past the 3% token-in-both-classes
        # floor after the rebalance down-sample. Decorrelated from the scaffold
        # axis (different multiplier + modulus).
        marker = markers[(seed * n + i) % n_mark]
        body = scaffold.replace("{M}", marker)
        # a unique numeric id, identical tail structure in both classes; digits
        # appear equally in both classes so they are not a battery confound.
        uid = h % 100000
        sentence = f"{body} item {uid}"
        s = sentence[0].upper() + sentence[1:]
        out.append(s)
    rng.shuffle(out)
    return out


def mock_labeler(rule_id: str, which: str, text: str, *, seed: int) -> bool | None:
    """Deterministic, network-free Labeler for offline tests.

    Recovers the intended class from the class-marker word the generator embedded
    (positive vs negative adjective; food vs non-food noun), so BOTH passes agree
    on the intended label — exactly the 'both validators returned the intended
    label' path the pipeline keeps. A handful of texts are deliberately sent to
    None to exercise the drop/rebalance path."""
    low = text.lower()
    toks = set(low.split())
    # deterministic ~10% neutral drops keyed on the embedded unique id, to
    # exercise the asymmetric-drop rebalancing + targeted-regeneration path:
    # drop when the trailing "item <uid>" digit run ends in 7 (about 10%).
    for tok in low.split():
        if tok.isdigit() and tok.endswith("7"):
            return None
    if rule_id == "positive_sentiment":
        pos_markers = set(_MOCK_MARKERS["positive_sentiment"]["pos"])
        neg_markers = set(_MOCK_MARKERS["positive_sentiment"]["neg"])
        if toks & pos_markers:
            return True
        if toks & neg_markers:
            return False
        return None
    # food_topic
    food_markers = set(_MOCK_MARKERS["food_topic"]["food"])
    notfood_markers = set(_MOCK_MARKERS["food_topic"]["notfood"])
    if toks & food_markers:
        return True
    if toks & notfood_markers:
        return False
    return None
