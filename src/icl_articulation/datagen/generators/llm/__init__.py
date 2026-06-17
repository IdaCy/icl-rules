"""LLM-validated generation for the ``llm_validated`` rules (15 positive_sentiment,
16 food_topic, 18 physically_impossible).

This package is DELIBERATELY DISJOINT from ``generators/rules/``: the registry
auto-discovers every module under ``rules/`` and drives it through the
PROGRAMMATIC gated pipeline (template/substitution generators with a recomputable
text predicate). The rules here have NO recomputable text predicate — their
ground truth is an LLM-judge call — so they must NOT be auto-discovered. Nothing
in this package exposes the ``build_bases`` / ``instantiate`` interface the
registry scans for; the entry point is the CLI in ``__main__``.

Rules 15 and 16 share ONE parameterized pipeline (generate -> 2-pass validate ->
rebalance -> emit). Per-rule differences (generation prompt, topic lists, label
semantics, validator prompts, audit knobs) live in ``config`` as ``LLMRuleConfig``
objects, keyed by rule_id. ``pipeline`` is rule-agnostic.

Rule 18 (``physically_impossible``) uses a DIFFERENT recipe and so has its own
module (``physically_impossible`` + the authored ``frames_physically_impossible``
bank): minimal-pair frames -> 2-pass validation -> by-base SURVIVAL (a base
survives only if BOTH its variants pass) -> PROGRAMMATIC by-base split -> emit.
It shares the same offline-testability seam (an injected ``validator``) and the
same CLI; see ``physically_impossible.run_build`` / ``run_api_build``.

OFFLINE-TESTABILITY: every API call goes through a small ``Labeler`` /
``Generator`` callable seam. ``pipeline.run_pipeline`` accepts injected fake
callables (``--mock`` on the CLI wires deterministic, no-network fakes), so the
whole generate -> validate -> rebalance -> emit flow runs on tiny fake data with
zero network. The REAL run wires the same seam to ``icl_articulation.client``.
"""

from __future__ import annotations

from .config import LLM_RULE_CONFIGS, LLMRuleConfig, get_rule_config
from .pipeline import (
    Candidate,
    PipelineResult,
    mock_generator,
    mock_labeler,
    run_pipeline,
    run_pipeline_async,
)

__all__ = [
    "LLM_RULE_CONFIGS",
    "LLMRuleConfig",
    "get_rule_config",
    "Candidate",
    "PipelineResult",
    "run_pipeline",
    "run_pipeline_async",
    "mock_generator",
    "mock_labeler",
]
