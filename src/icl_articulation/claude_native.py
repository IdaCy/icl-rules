"""Native-Anthropic call path for the otherwise OpenAI-only Step-2 runners.

The multiple-choice and free-form runners talk to ``OpenAIClient`` (OpenAI / DeepSeek
OpenAI-compatible). Claude is a different SDK with no logprobs and a different
no-CoT story, so it was only ever run through a separate native path (see
``run_insession_articulation.py``). This module factors that path out so the
multiple-choice and free-form runners can reach Claude *without* touching their gpt/mini
behaviour, and returns records shaped exactly like ``OpenAIClient.complete`` so
the existing ``analyze`` / ``response_text`` / ``extract_rule`` / grader code
works unchanged.

Conventions mirror the in-session runner: model ``claude-opus-4-8``, the same
``STEP1_SYSTEM`` system prompt, thinking left at its default in the no-CoT
regime (no ``thinking`` key passed), no logprobs. Claude truncates at
``max_tokens=2``, so callers pass a Claude-appropriate budget (~16 for a single
multiple-choice letter, sentence-room for free-form). This carries the same documented
Claude caveat as the cross-family Step-1 runs: thinking-disabled is approximate
and a small fraction of answers may still carry trailing reasoning, so trust the
relative pattern, not the absolute level.
"""
from __future__ import annotations

import asyncio
from typing import Any

from .prompts import STEP1_SYSTEM

CLAUDE_MODEL = "claude-opus-4-8"
CLAUDE_PRICE_IN, CLAUDE_PRICE_OUT = 5.00, 25.00  # USD per Mtok (opus 4.8)


def is_claude(model: str) -> bool:
    return model.startswith("claude")


def make_async_anthropic() -> Any:
    """AsyncAnthropic with .env loaded first (ANTHROPIC_API_KEY lives in .env).

    A runner that reaches Claude without first constructing an OpenAIClient
    never triggers ``load_dotenv``; do it here so every entry point is safe."""
    from dotenv import load_dotenv

    load_dotenv()
    import anthropic

    return anthropic.AsyncAnthropic(max_retries=0)


def _split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Pull a leading system message out for the Anthropic ``system`` kwarg;
    fall back to STEP1_SYSTEM (what the OpenAI path uses when none is present)."""
    system = STEP1_SYSTEM
    turns: list[dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            turns.append({"role": m["role"], "content": m["content"]})
    return system, turns


def new_meter() -> dict[str, int]:
    return {"in": 0, "out": 0}


def claude_cost_usd(meter: dict[str, int]) -> float:
    return (meter["in"] * CLAUDE_PRICE_IN + meter["out"] * CLAUDE_PRICE_OUT) / 1e6


async def claude_complete(
    ac: Any, messages: list[dict[str, str]], *, max_tokens: int, meter: dict[str, int]
) -> dict[str, Any]:
    """One Claude completion, returned in OpenAIClient.complete record shape."""
    system, turns = _split_system(messages)
    try:
        import anthropic

        retry = (
            anthropic.RateLimitError,
            anthropic.APIStatusError,
            anthropic.APIConnectionError,
        )
    except ModuleNotFoundError:
        retry = ()  # type: ignore[assignment]

    resp = None
    for attempt in range(6):
        try:
            resp = await ac.messages.create(
                model=CLAUDE_MODEL, system=system, max_tokens=max_tokens, messages=turns
            )
            break
        except retry:
            if attempt == 5:
                raise
            await asyncio.sleep(min(2 ** attempt, 30))

    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    )
    u = getattr(resp, "usage", None)
    in_tok = getattr(u, "input_tokens", 0) or 0
    out_tok = getattr(u, "output_tokens", 0) or 0
    meter["in"] += in_tok
    meter["out"] += out_tok
    response = {
        "choices": [
            {
                "message": {"content": text},
                "logprobs": None,
                "finish_reason": getattr(resp, "stop_reason", "") or "",
            }
        ],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
    }
    return {
        "cache_key": None,
        "cached": False,
        "request": {"model": CLAUDE_MODEL, "messages": messages, "max_tokens": max_tokens},
        "response": response,
    }
