"""Async OpenAI chat-completions client with disk cache, retries, and a cost meter.

The API key is read from the environment (``OPENAI_API_KEY``, loaded from .env
via python-dotenv if present). The key is never logged, never stored on any
returned record, and never written to the cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv

from .prices import cost_usd

DEFAULT_CONCURRENCY = 16
DEFAULT_MAX_RETRIES = 6


def cache_key(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    logprobs: bool,
    top_logprobs: int | None,
    seed: int | None,
) -> str:
    """sha256 over the canonical JSON of everything that determines a response."""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    """sqlite-backed response cache. One row per cache key."""

    def __init__(self, cache_dir: str | Path = "cache") -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.dir / "responses.sqlite")
        # Multi-process robustness (e.g. two sweeps sharing a cache dir):
        # WAL allows concurrent readers during a write; busy_timeout keeps a
        # paid `put` from crashing with "database is locked".
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " key TEXT PRIMARY KEY,"
            " response TEXT NOT NULL,"
            " created REAL NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, response: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, response, created) VALUES (?, ?, ?)",
            (key, json.dumps(response, ensure_ascii=False), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class CostMeter:
    """Accumulates actual token usage (fresh API calls only) and prices it."""

    def __init__(self) -> None:
        self.prompt_tokens: dict[str, int] = {}
        self.completion_tokens: dict[str, int] = {}
        self.prompt_cache_hit_tokens: dict[str, int] = {}
        self.prompt_cache_miss_tokens: dict[str, int] = {}

    def add(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        prompt_cache_hit_tokens: int | None = None,
        prompt_cache_miss_tokens: int | None = None,
    ) -> None:
        self.prompt_tokens[model] = self.prompt_tokens.get(model, 0) + prompt_tokens
        self.completion_tokens[model] = self.completion_tokens.get(model, 0) + completion_tokens
        if prompt_cache_hit_tokens is not None:
            self.prompt_cache_hit_tokens[model] = (
                self.prompt_cache_hit_tokens.get(model, 0) + prompt_cache_hit_tokens
            )
        if prompt_cache_miss_tokens is not None:
            self.prompt_cache_miss_tokens[model] = (
                self.prompt_cache_miss_tokens.get(model, 0) + prompt_cache_miss_tokens
            )

    @property
    def total_usd(self) -> float:
        return sum(
            cost_usd(
                m,
                self.prompt_tokens[m],
                self.completion_tokens.get(m, 0),
                prompt_cache_hit_tokens=self.prompt_cache_hit_tokens.get(m),
                prompt_cache_miss_tokens=self.prompt_cache_miss_tokens.get(m),
            )
            for m in self.prompt_tokens
        )

    def summary(self) -> dict[str, Any]:
        return {
            "prompt_tokens": dict(self.prompt_tokens),
            "completion_tokens": dict(self.completion_tokens),
            "prompt_cache_hit_tokens": dict(self.prompt_cache_hit_tokens),
            "prompt_cache_miss_tokens": dict(self.prompt_cache_miss_tokens),
            "total_usd": self.total_usd,
        }


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, asyncio.TimeoutError))


def _provider_for_model(model: str) -> str:
    if model.startswith("deepseek-"):
        return "deepseek"
    return "openai"


class OpenAIClient:
    """Bounded-concurrency chat-completions client with cache, retries, cost meter.

    ``api`` lets tests inject a fake with the AsyncOpenAI surface
    (``.chat.completions.create``, ``.models.list``); when None, a real
    AsyncOpenAI is created lazily from OPENAI_API_KEY.
    """

    def __init__(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        cache_dir: str | Path = "cache",
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = 60.0,
        base_delay: float = 1.0,
        api: Any | None = None,
    ) -> None:
        load_dotenv()
        self._api = api
        self._apis: dict[str, Any] = {}
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._sem = asyncio.Semaphore(concurrency)
        self.cache = DiskCache(cache_dir)
        self.cost = CostMeter()
        # counters
        self.n_api_calls = 0  # fresh calls that succeeded
        self.n_cache_hits = 0
        self.n_429 = 0
        self.n_retryable_errors = 0  # 5xx / timeouts / connection errors
        self.n_failures = 0  # gave up after max retries

    def _get_api(self, model: str) -> Any:
        if self._api is None:
            provider = _provider_for_model(model)
            existing = self._apis.get(provider)
            if existing is not None:
                return existing
            if provider == "deepseek":
                key = os.environ.get("DEEPSEEK_API_KEY")
                if not key:
                    raise RuntimeError(
                        "DEEPSEEK_API_KEY is not set (expected in the environment or .env)"
                    )
                api = openai.AsyncOpenAI(
                    api_key=key,
                    base_url="https://api.deepseek.com",
                    timeout=self._timeout,
                    max_retries=0,
                )
            else:
                key = os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise RuntimeError(
                        "OPENAI_API_KEY is not set (expected in the environment or .env)"
                    )
                api = openai.AsyncOpenAI(api_key=key, timeout=self._timeout, max_retries=0)
            self._apis[provider] = api
            return api
        return self._api

    def stats(self) -> dict[str, Any]:
        attempts = self.n_api_calls + self.n_429 + self.n_retryable_errors
        return {
            "api_calls": self.n_api_calls,
            "cache_hits": self.n_cache_hits,
            "n_429": self.n_429,
            "retryable_errors": self.n_retryable_errors,
            "failures": self.n_failures,
            "rate_429": self.n_429 / attempts if attempts else 0.0,
            "error_rate": (self.n_429 + self.n_retryable_errors) / attempts if attempts else 0.0,
        }

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """One chat completion; returns a serializable record (never the API key).

        Record: {cache_key, cached, request, response} where response is the
        full API response dict (choices, logprobs, usage, ...).
        """
        if not logprobs:
            # top_logprobs is omitted from the wire request when logprobs is
            # off — normalize it out of the cache key too, so identical wire
            # requests never get distinct keys (no spurious paid duplicates).
            top_logprobs = None
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "seed": seed,
        }
        key = cache_key(model, messages, temperature, max_tokens, logprobs, top_logprobs, seed)
        cached = self.cache.get(key)
        if cached is not None:
            self.n_cache_hits += 1
            return {"cache_key": key, "cached": True, "request": request, "response": cached}

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logprobs:
            kwargs["logprobs"] = True
            if top_logprobs is not None:
                kwargs["top_logprobs"] = top_logprobs
        if seed is not None:
            kwargs["seed"] = seed
        if _provider_for_model(model) == "deepseek":
            kwargs.pop("seed", None)
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        api = self._get_api(model)
        async with self._sem:
            for attempt in range(self._max_retries + 1):
                try:
                    response = await api.chat.completions.create(**kwargs)
                    break
                except Exception as exc:
                    if isinstance(exc, openai.RateLimitError):
                        self.n_429 += 1
                    elif _is_retryable(exc):
                        self.n_retryable_errors += 1
                    else:
                        raise
                    if attempt >= self._max_retries:
                        self.n_failures += 1
                        raise
                    delay = self._base_delay * (2**attempt)
                    sleep_s = min(delay + random.uniform(0, delay), 60.0)
                    # one line per retry so a nohup log captures live 429/5xx
                    # activity (review M2) — the watchdog's error-marker scan
                    # keys on the literal '429' / error class name below
                    marker = "429 " if isinstance(exc, openai.RateLimitError) else ""
                    print(
                        f"retry: {marker}{type(exc).__name__} attempt "
                        f"{attempt + 1}/{self._max_retries} sleeping {sleep_s:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    await asyncio.sleep(sleep_s)

        data = response.model_dump()
        usage = data.get("usage") or {}
        self.cost.add(
            model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            prompt_cache_hit_tokens=usage.get("prompt_cache_hit_tokens"),
            prompt_cache_miss_tokens=usage.get("prompt_cache_miss_tokens"),
        )
        self.n_api_calls += 1
        self.cache.put(key, data)
        return {"cache_key": key, "cached": False, "request": request, "response": data}

    async def list_models(self) -> list[str]:
        api = self._get_api("gpt-4.1")
        page = await api.models.list()
        return sorted(m.id for m in page.data)

    async def aclose(self) -> None:
        self.cache.close()
        apis = list(self._apis.values())
        if self._api is not None:
            apis.append(self._api)
        for api in apis:
            close = getattr(api, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result


def response_text(record: dict[str, Any]) -> str:
    """Message content of the first choice."""
    return record["response"]["choices"][0]["message"]["content"] or ""


def first_token_logprobs(record: dict[str, Any]) -> dict[str, Any] | None:
    """Logprob entry at the first completion token (None if logprobs absent).

    Entry has keys: token, logprob, top_logprobs (list of {token, logprob}).
    """
    lp = record["response"]["choices"][0].get("logprobs")
    if not lp or not lp.get("content"):
        return None
    return lp["content"][0]
