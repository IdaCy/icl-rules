"""Unit tests for wilson_ci and cluster_bootstrap_ci (deterministic seeds)."""

from __future__ import annotations

import numpy as np
import pytest

from icl_articulation.stats import (
    binom_pmf,
    binom_test_two_sided,
    cluster_bootstrap_ci,
    wilson_ci,
)


def test_binom_test_known_values() -> None:
    # symmetric at p=0.5: every outcome is equally-or-less likely than the center
    assert binom_test_two_sided(5, 10) == pytest.approx(1.0)
    # 0/10: only the two extremes (0 and 10) are <= P(0); 2 * (0.5**10)
    assert binom_test_two_sided(0, 10) == pytest.approx(2 * 0.5 ** 10)
    # 6/7 (the original physically_impossible empirical gap): exact two-sided
    # p = P(X in {0,1,6,7}) = (1+7+7+1)/128 = 16/128 = 0.125
    assert binom_test_two_sided(6, 7) == pytest.approx(0.125)
    # 23/23 (the corrected discriminating result): vanishingly small
    assert binom_test_two_sided(23, 23) == pytest.approx(2 * 0.5 ** 23)


def test_binom_test_matches_pmf_sum() -> None:
    n, k = 8, 2
    p = binom_test_two_sided(k, n)
    obs = binom_pmf(k, n, 0.5)
    brute = sum(binom_pmf(j, n, 0.5) for j in range(n + 1)
                if binom_pmf(j, n, 0.5) <= obs * (1 + 1e-9))
    assert p == pytest.approx(brute)
    assert p == pytest.approx(2 * (binom_pmf(0, 8, .5) + binom_pmf(1, 8, .5) + binom_pmf(2, 8, .5)))


def test_binom_test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        binom_test_two_sided(5, 0)
    with pytest.raises(ValueError):
        binom_test_two_sided(11, 10)


def test_wilson_known_value() -> None:
    # 50/100 at 95%: classic Wilson interval ~ (0.4038, 0.5962)
    low, high = wilson_ci(50, 100)
    assert low == pytest.approx(0.4038, abs=2e-4)
    assert high == pytest.approx(0.5962, abs=2e-4)


def test_wilson_known_value_asymmetric() -> None:
    # 95/100 at 95%: asymmetric pin catches wrong-formula variants
    # (continuity-corrected, swapped half-width) that pass at p=0.5.
    low, high = wilson_ci(95, 100)
    assert low == pytest.approx(0.8882495307680808, abs=1e-9)
    assert high == pytest.approx(0.9784563208456319, abs=1e-9)


def test_wilson_symmetry_at_half() -> None:
    low, high = wilson_ci(60, 120)
    assert low + high == pytest.approx(1.0)


def test_wilson_edges() -> None:
    low0, high0 = wilson_ci(0, 20)
    assert low0 == 0.0 and 0 < high0 < 0.25
    low1, high1 = wilson_ci(20, 20)
    assert high1 == 1.0 and 0.75 < low1 < 1.0


def test_wilson_narrows_with_n() -> None:
    w_small = wilson_ci(9, 10)
    w_big = wilson_ci(90, 100)
    assert (w_big[1] - w_big[0]) < (w_small[1] - w_small[0])


def test_wilson_validates_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_ci(1, 0)
    with pytest.raises(ValueError):
        wilson_ci(5, 4)


def test_bootstrap_deterministic_with_seed() -> None:
    rng = np.random.default_rng(123)
    data = (rng.random((3, 120)) < 0.9).astype(float)
    r1 = cluster_bootstrap_ci(data, n_boot=2000, seed=42)
    r2 = cluster_bootstrap_ci(data, n_boot=2000, seed=42)
    assert r1 == r2
    r3 = cluster_bootstrap_ci(data, n_boot=2000, seed=43)
    assert r3 != r1  # different seed -> different draws


def test_bootstrap_degenerate_all_correct() -> None:
    data = np.ones((3, 120))
    point, low, high = cluster_bootstrap_ci(data, n_boot=1000, seed=0)
    assert point == low == high == 1.0


def test_bootstrap_antithetic_pins_item_level_resampling() -> None:
    # Context 0 is correct exactly on items 0..59, context 1 exactly on
    # items 60..119. Under JOINT item (column) resampling, every bootstrap
    # draw has per-context accuracies summing to 1, so the mean over contexts
    # is exactly 0.5 in every draw -> degenerate CI at the point estimate.
    # An incorrect call-level bootstrap (resampling per context independently)
    # would produce a wide CI here. Pins the pre-specified statistic.
    arr = np.zeros((2, 120))
    arr[0, :60] = 1.0
    arr[1, 60:] = 1.0
    point, low, high = cluster_bootstrap_ci(arr, n_boot=2000, seed=0)
    assert point == low == high == 0.5


def test_bootstrap_brackets_point_estimate() -> None:
    rng = np.random.default_rng(7)
    data = (rng.random((3, 120)) < 0.85).astype(float)
    point, low, high = cluster_bootstrap_ci(data, n_boot=10_000, seed=0)
    assert low <= point <= high
    assert 0.0 < low < high < 1.0
    assert point == pytest.approx(data.mean(axis=1).mean())
    # CI width sane for n=120, p~0.85 (binomial SE ~0.033 -> ~±2 SE)
    assert 0.03 < high - low < 0.20


def test_bootstrap_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(np.ones(10))
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(np.ones((3, 0)))
