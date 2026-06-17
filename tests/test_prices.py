"""Cost meter arithmetic and price lookup tests."""

from __future__ import annotations

import pytest

from icl_articulation.client import CostMeter
from icl_articulation.prices import cost_usd, price_for


def test_cost_usd_arithmetic() -> None:
    assert cost_usd("gpt-4.1", 1_000_000, 0) == pytest.approx(2.00)
    assert cost_usd("gpt-4.1", 0, 1_000_000) == pytest.approx(8.00)
    assert cost_usd("gpt-4.1-mini", 500_000, 250_000) == pytest.approx(0.20 + 0.40)
    assert cost_usd("gpt-4o", 100, 10) == pytest.approx((100 * 2.5 + 10 * 10.0) / 1e6)
    assert cost_usd("gpt-4.1", 0, 0) == 0.0
    assert cost_usd("deepseek-v4-flash", 1_000_000, 1_000_000) == pytest.approx(0.14 + 0.28)
    assert cost_usd(
        "deepseek-v4-flash",
        1_000_000,
        1_000_000,
        prompt_cache_hit_tokens=250_000,
        prompt_cache_miss_tokens=750_000,
    ) == pytest.approx(0.0028 * 0.25 + 0.14 * 0.75 + 0.28)


def test_price_for_prefix_matches_dated_snapshots() -> None:
    assert price_for("gpt-4.1-2025-04-14") == price_for("gpt-4.1")
    # longest prefix wins: -mini snapshot must NOT match bare gpt-4.1
    assert price_for("gpt-4.1-mini-2025-04-14") == price_for("gpt-4.1-mini")
    assert price_for("gpt-4o-2024-08-06") == price_for("gpt-4o")
    assert price_for("deepseek-v4-flash") == price_for("deepseek-v4-flash")


def test_price_for_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        price_for("gpt-99-ultra")


def test_cost_meter_accumulates_per_model() -> None:
    meter = CostMeter()
    meter.add("gpt-4.1", 1000, 100)
    meter.add("gpt-4.1", 2000, 200)
    meter.add("gpt-4.1-mini", 10_000, 1000)
    expected = (
        (3000 * 2.00 + 300 * 8.00) / 1e6
        + (10_000 * 0.40 + 1000 * 1.60) / 1e6
    )
    assert meter.total_usd == pytest.approx(expected)
    summary = meter.summary()
    assert summary["prompt_tokens"] == {"gpt-4.1": 3000, "gpt-4.1-mini": 10_000}
    assert summary["completion_tokens"] == {"gpt-4.1": 300, "gpt-4.1-mini": 1000}
    assert summary["total_usd"] == pytest.approx(expected)


def test_cost_meter_tracks_deepseek_cache_usage() -> None:
    meter = CostMeter()
    meter.add(
        "deepseek-v4-flash",
        1_000_000,
        1_000_000,
        prompt_cache_hit_tokens=250_000,
        prompt_cache_miss_tokens=750_000,
    )
    expected = (250_000 * 0.0028 + 750_000 * 0.14 + 1_000_000 * 0.28) / 1e6
    assert meter.total_usd == pytest.approx(expected)
    assert meter.summary()["prompt_cache_hit_tokens"] == {"deepseek-v4-flash": 250_000}
    assert meter.summary()["prompt_cache_miss_tokens"] == {"deepseek-v4-flash": 750_000}
