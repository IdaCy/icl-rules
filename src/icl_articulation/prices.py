"""Price table for cost metering.

Prices are USD per 1 MILLION tokens.

!!! VERIFY these against https://platform.openai.com/docs/pricing BEFORE every
paid run — OpenAI changes prices without notice. Last checked: 2026-06 (from
training-time knowledge, NOT a live check).
"""

from __future__ import annotations

PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "gpt-4.1": {"prompt": 2.00, "completion": 8.00},
    "gpt-4.1-mini": {"prompt": 0.40, "completion": 1.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "claude-opus-4-8": {"prompt": 5.00, "completion": 25.00},
    "deepseek-v4-flash": {
        "prompt": 0.14,
        "prompt_cache_miss": 0.14,
        "prompt_cache_hit": 0.0028,
        "completion": 0.28,
    },
}


def price_for(model: str) -> dict[str, float]:
    """Return the price entry for a model id.

    Dated snapshots like "gpt-4.1-2025-04-14" match by longest prefix, so
    "gpt-4.1-mini-..." matches gpt-4.1-mini, not gpt-4.1.
    """
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    candidates = [name for name in PRICES_PER_MTOK if model.startswith(name)]
    if not candidates:
        raise KeyError(f"no price known for model {model!r}; add it to prices.py")
    return PRICES_PER_MTOK[max(candidates, key=len)]


def cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    prompt_cache_hit_tokens: int | None = None,
    prompt_cache_miss_tokens: int | None = None,
) -> float:
    """Dollar cost of a single call's token usage."""
    p = price_for(model)
    if prompt_cache_hit_tokens is not None or prompt_cache_miss_tokens is not None:
        hit = int(prompt_cache_hit_tokens or 0)
        miss = int(prompt_cache_miss_tokens or 0)
        unknown = max(0, prompt_tokens - hit - miss)
        prompt_cost = (
            hit * p.get("prompt_cache_hit", p["prompt"])
            + miss * p.get("prompt_cache_miss", p["prompt"])
            + unknown * p["prompt"]
        )
    else:
        prompt_cost = prompt_tokens * p["prompt"]
    return (prompt_cost + completion_tokens * p["completion"]) / 1_000_000.0
