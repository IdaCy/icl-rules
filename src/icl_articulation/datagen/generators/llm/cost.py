"""Advance cost estimate for an llm_validated real run.

Costs three call streams (rule-specs recipe), priced from ``prices.py``:

  1. GENERATION  — gpt-4.1-mini, batched. One call per (topic x class) cell-batch
     of cfg.pairs_per_call sentences; total calls = ceil(max_candidates /
     pairs_per_call). Each gen call: prompt (~the constraint block) + completion
     (~pairs_per_call short sentences).
  2. VALIDATION pass A — gpt-4.1-mini, ONE call per surviving candidate.
  3. VALIDATION pass B — gpt-4.1, ONE call per surviving candidate.

Validation runs over the GENERATED candidates (every candidate gets both passes,
before the drop), so pass counts = n_candidates, not n_kept. Token counts use a
chars/4 heuristic on the actual rendered prompts plus a flat completion budget,
mirroring scripts/run_step1.py's estimator. The estimate is intentionally an
UPPER-ish bound (it does not credit the disk cache, which makes re-runs free).
"""

from __future__ import annotations

import math
from typing import Any

from ....prices import cost_usd, price_for
from .config import get_rule_config
from .pipeline import GEN_MODEL, VALIDATOR_A_MODEL, VALIDATOR_B_MODEL

# chars/4 token heuristic + a flat chat-formatting overhead per call
CHARS_PER_TOKEN = 4
CALL_OVERHEAD_TOKENS = 12
# average completion length of one short sentence (4-12 words) in tokens
GEN_TOKENS_PER_SENTENCE = 16
# a validator answers with one of three short words
VAL_COMPLETION_TOKENS = 3


def _prompt_tokens(messages: list[dict[str, str]]) -> int:
    chars = sum(len(m["content"]) for m in messages)
    return math.ceil(chars / CHARS_PER_TOKEN) + CALL_OVERHEAD_TOKENS


def estimate_cost(rule_id: str, max_candidates: int) -> dict[str, Any]:
    """Advance cost estimate for generating ~``max_candidates`` candidates for
    ``rule_id`` and validating each in two passes. Returns a breakdown dict."""
    cfg = get_rule_config(rule_id)
    # price sanity — KeyError (loud) if any model is unpriced
    for m in (GEN_MODEL, VALIDATOR_A_MODEL, VALIDATOR_B_MODEL):
        price_for(m)

    n_cells = len(cfg.true_topics) + len(cfg.false_topics)
    per_cell = max(1, math.ceil(max_candidates / n_cells))
    calls_per_cell = max(1, math.ceil(per_cell / cfg.pairs_per_call))
    gen_calls = calls_per_cell * n_cells
    n_candidates = gen_calls * cfg.pairs_per_call

    # representative prompts (token sizes are cell-independent up to topic length)
    sample_topic = cfg.true_topics[0]
    gen_msgs = cfg.generation_messages(True, sample_topic, cfg.pairs_per_call)
    gen_prompt_tok = _prompt_tokens(gen_msgs)
    gen_completion_tok = GEN_TOKENS_PER_SENTENCE * cfg.pairs_per_call

    sample_text = "the meal was wonderful and the staff felt warm"
    val_a_prompt_tok = _prompt_tokens(cfg.validator_messages("A", sample_text))
    val_b_prompt_tok = _prompt_tokens(cfg.validator_messages("B", sample_text))

    gen_cost = gen_calls * cost_usd(GEN_MODEL, gen_prompt_tok, gen_completion_tok)
    val_a_cost = n_candidates * cost_usd(VALIDATOR_A_MODEL, val_a_prompt_tok, VAL_COMPLETION_TOKENS)
    val_b_cost = n_candidates * cost_usd(VALIDATOR_B_MODEL, val_b_prompt_tok, VAL_COMPLETION_TOKENS)
    total = gen_cost + val_a_cost + val_b_cost

    return {
        "rule_id": rule_id,
        "max_candidates": max_candidates,
        "n_cells": n_cells,
        "per_cell_target": per_cell,
        "gen_calls": gen_calls,
        "n_candidates": n_candidates,
        "validation_calls": n_candidates * 2,
        "gen_model": GEN_MODEL,
        "validator_a_model": VALIDATOR_A_MODEL,
        "validator_b_model": VALIDATOR_B_MODEL,
        "gen_cost_usd": gen_cost,
        "validate_pass_a_cost_usd": val_a_cost,
        "validate_pass_b_cost_usd": val_b_cost,
        "total_usd": total,
        "assumptions": (
            f"chars/4 token heuristic + {CALL_OVERHEAD_TOKENS}-token overhead/call; "
            f"gen completion ~{GEN_TOKENS_PER_SENTENCE} tok/sentence; both passes "
            f"run on every candidate (pre-drop); disk cache NOT credited."
        ),
    }
