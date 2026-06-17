"""Client tests: cache key stability, cache hit/miss, cost meter, retries.

No network — the API surface is faked (see conftest.FakeAPI).
"""

from __future__ import annotations

import asyncio
import json

import httpx
import openai
import pytest

from icl_articulation.client import OpenAIClient, cache_key
from tests.conftest import FakeAPI, fake_response_data

MESSAGES = [
    {"role": "system", "content": "You are a precise classifier."},
    {"role": "user", "content": "Input: hello\nLabel:"},
]


def test_cache_key_stable() -> None:
    k1 = cache_key("gpt-4.1", MESSAGES, 0.0, 2, True, 5, 0)
    k2 = cache_key("gpt-4.1", list(MESSAGES), 0.0, 2, True, 5, 0)
    assert k1 == k2
    assert len(k1) == 64
    assert all(c in "0123456789abcdef" for c in k1)


def test_cache_key_sensitive_to_every_param() -> None:
    base = cache_key("gpt-4.1", MESSAGES, 0.0, 2, True, 5, 0)
    assert cache_key("gpt-4o", MESSAGES, 0.0, 2, True, 5, 0) != base
    assert cache_key("gpt-4.1", MESSAGES, 1.0, 2, True, 5, 0) != base
    assert cache_key("gpt-4.1", MESSAGES, 0.0, 16, True, 5, 0) != base
    assert cache_key("gpt-4.1", MESSAGES, 0.0, 2, False, 5, 0) != base
    assert cache_key("gpt-4.1", MESSAGES, 0.0, 2, True, 3, 0) != base
    assert cache_key("gpt-4.1", MESSAGES, 0.0, 2, True, 5, 1) != base
    other = [{"role": "user", "content": "different"}]
    assert cache_key("gpt-4.1", other, 0.0, 2, True, 5, 0) != base


def test_cache_hit_skips_api_call(tmp_path) -> None:
    api = FakeAPI()
    client = OpenAIClient(cache_dir=tmp_path / "cache", api=api)

    async def go():
        r1 = await client.complete("gpt-4.1", MESSAGES, max_tokens=2, logprobs=True,
                                   top_logprobs=5, seed=0)
        r2 = await client.complete("gpt-4.1", MESSAGES, max_tokens=2, logprobs=True,
                                   top_logprobs=5, seed=0)
        return r1, r2

    r1, r2 = asyncio.run(go())
    assert len(api.calls) == 1  # second call served from cache
    assert r1["cached"] is False and r2["cached"] is True
    assert r1["response"] == r2["response"]
    assert client.n_cache_hits == 1 and client.n_api_calls == 1


def test_cache_persists_across_client_instances(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    api1, api2 = FakeAPI(), FakeAPI()
    c1 = OpenAIClient(cache_dir=cache_dir, api=api1)
    asyncio.run(c1.complete("gpt-4.1", MESSAGES))
    c2 = OpenAIClient(cache_dir=cache_dir, api=api2)
    record = asyncio.run(c2.complete("gpt-4.1", MESSAGES))
    assert record["cached"] is True
    assert api2.calls == []  # crash-resume: no API call after restart


def test_cache_hit_when_logprobs_off_ignores_top_logprobs(tmp_path) -> None:
    # top_logprobs is omitted from the wire request when logprobs=False, so it
    # must not bust the cache either: identical wire requests -> one paid call.
    api = FakeAPI()
    client = OpenAIClient(cache_dir=tmp_path / "cache", api=api)

    async def go():
        await client.complete("gpt-4.1", MESSAGES, logprobs=False, top_logprobs=5)
        return await client.complete("gpt-4.1", MESSAGES, logprobs=False, top_logprobs=None)

    r2 = asyncio.run(go())
    assert r2["cached"] is True
    assert len(api.calls) == 1


def test_cache_miss_on_param_change(tmp_path) -> None:
    api = FakeAPI()
    client = OpenAIClient(cache_dir=tmp_path / "cache", api=api)

    async def go():
        await client.complete("gpt-4.1", MESSAGES, temperature=0.0)
        await client.complete("gpt-4.1", MESSAGES, temperature=1.0)

    asyncio.run(go())
    assert len(api.calls) == 2


def test_cost_meter_arithmetic(tmp_path) -> None:
    api = FakeAPI(fake_response_data(prompt_tokens=1000, completion_tokens=50))
    client = OpenAIClient(cache_dir=tmp_path / "cache", api=api)
    asyncio.run(client.complete("gpt-4.1", MESSAGES))
    # gpt-4.1: $2/Mtok prompt, $8/Mtok completion
    expected = (1000 * 2.00 + 50 * 8.00) / 1_000_000
    assert client.cost.total_usd == pytest.approx(expected)
    summary = client.cost.summary()
    assert summary["prompt_tokens"] == {"gpt-4.1": 1000}
    assert summary["completion_tokens"] == {"gpt-4.1": 50}
    # cached repeat adds no cost
    asyncio.run(client.complete("gpt-4.1", MESSAGES))
    assert client.cost.total_usd == pytest.approx(expected)


def _rate_limit_error() -> openai.RateLimitError:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    return openai.RateLimitError("rate limited", response=resp, body=None)


def test_retry_on_429_then_success(tmp_path) -> None:
    api = FakeAPI()
    api.errors_to_raise = [_rate_limit_error(), _rate_limit_error()]
    client = OpenAIClient(cache_dir=tmp_path / "cache", api=api, base_delay=0.001)
    record = asyncio.run(client.complete("gpt-4.1", MESSAGES))
    assert record["cached"] is False
    assert client.n_429 == 2
    assert client.n_api_calls == 1
    assert len(api.calls) == 3
    assert client.stats()["rate_429"] == pytest.approx(2 / 3)


def test_retry_prints_visible_stderr_line(tmp_path, capsys) -> None:
    # review M2: a nohup log (and the watchdog's error-marker scan) must see
    # live 429 activity — one stderr line per retry with the literal '429',
    # the error class, the attempt number, and the sleep
    api = FakeAPI()
    api.errors_to_raise = [_rate_limit_error()]
    client = OpenAIClient(cache_dir=tmp_path / "cache", api=api, base_delay=0.001)
    asyncio.run(client.complete("gpt-4.1", MESSAGES))
    err = capsys.readouterr().err
    lines = [l for l in err.splitlines() if l.startswith("retry:")]
    assert len(lines) == 1
    assert "429" in lines[0]
    assert "RateLimitError" in lines[0]
    assert "attempt 1/" in lines[0]
    assert "sleeping" in lines[0]


def test_gives_up_after_max_retries(tmp_path) -> None:
    api = FakeAPI()
    api.errors_to_raise = [_rate_limit_error() for _ in range(10)]
    client = OpenAIClient(cache_dir=tmp_path / "cache", api=api,
                          max_retries=2, base_delay=0.001)
    with pytest.raises(openai.RateLimitError):
        asyncio.run(client.complete("gpt-4.1", MESSAGES))
    assert client.n_failures == 1
    assert len(api.calls) == 3  # initial + 2 retries


def test_non_retryable_error_raises_immediately(tmp_path) -> None:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    api = FakeAPI()
    api.errors_to_raise = [openai.BadRequestError("bad", response=resp, body=None)]
    client = OpenAIClient(cache_dir=tmp_path / "cache", api=api, base_delay=0.001)
    with pytest.raises(openai.BadRequestError):
        asyncio.run(client.complete("gpt-4.1", MESSAGES))
    assert len(api.calls) == 1


def test_logprob_params_passed_through(tmp_path) -> None:
    api = FakeAPI()
    client = OpenAIClient(cache_dir=tmp_path / "cache", api=api)
    asyncio.run(client.complete("gpt-4.1", MESSAGES, max_tokens=2,
                                logprobs=True, top_logprobs=5, seed=7))
    call = api.calls[0]
    assert call["logprobs"] is True and call["top_logprobs"] == 5
    assert call["seed"] == 7 and call["max_tokens"] == 2 and call["temperature"] == 0.0
    # and omitted when logprobs is off
    asyncio.run(client.complete("gpt-4.1", MESSAGES, logprobs=False))
    assert "logprobs" not in api.calls[1] and "top_logprobs" not in api.calls[1]


def test_deepseek_wire_params_disable_thinking_and_omit_seed(tmp_path) -> None:
    api = FakeAPI()
    client = OpenAIClient(cache_dir=tmp_path / "cache", api=api)
    asyncio.run(
        client.complete(
            "deepseek-v4-flash",
            MESSAGES,
            max_tokens=2,
            logprobs=True,
            top_logprobs=5,
            seed=7,
        )
    )
    call = api.calls[0]
    assert call["model"] == "deepseek-v4-flash"
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "seed" not in call


def test_record_is_serializable_and_key_free(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE-NEVER-LOGGED")
    api = FakeAPI()
    client = OpenAIClient(cache_dir=tmp_path / "cache", api=api)
    record = asyncio.run(client.complete("gpt-4.1", MESSAGES))
    blob = json.dumps(record)
    assert "sk-FAKE-NEVER-LOGGED" not in blob
