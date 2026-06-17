"""Pre-registered statistics helpers (PLAN step-1 config).

- Wilson 95% CI at the independent-unit level (per-context accuracy, n items).
- Item-level cluster bootstrap for the pooled summary across contexts:
  resample ITEMS with replacement (items are the cluster unit shared across
  contexts), recompute the mean of per-context accuracies, 10k draws,
  percentile CI. NO Wilson on pooled counts (non-independent).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


def binom_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) for X ~ Binomial(n, p)."""
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def binom_test_two_sided(successes: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (sum-of-small-probabilities method).

    The two-sided p-value is the total probability of every outcome at least as
    UNLIKELY as the observed one under Binomial(n, p) — the convention SciPy's
    binomtest uses. For p=0.5 the distribution is symmetric so this equals
    2*min(P(X<=k), P(X>=k)) capped at 1; the general form is computed here so it
    is correct for any p. Used to put an honest p on the small step-3 gaps."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    observed = binom_pmf(successes, n, p)
    # include every k whose pmf does not exceed the observed pmf (with a tiny
    # tolerance so ties at the symmetric point are not dropped by float error)
    tol = observed * (1.0 + 1e-9)
    total = sum(binom_pmf(k, n, p) for k in range(n + 1) if binom_pmf(k, n, p) <= tol)
    return min(1.0, total)


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))


class BootstrapResult(NamedTuple):
    point: float  # mean of per-context accuracies on the real data
    low: float
    high: float


def cluster_bootstrap_ci(
    correct: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Item-level cluster bootstrap CI for pooled accuracy across contexts.

    ``correct``: array of shape (n_contexts, n_items), entries 0/1 (correct
    or not), with the SAME items (columns) under each context. Items are
    resampled with replacement; the statistic is the mean over contexts of
    per-context accuracy. Percentile CI from ``n_boot`` draws.
    """
    arr = np.asarray(correct, dtype=float)
    if arr.ndim != 2:
        raise ValueError("correct must be 2-D: (n_contexts, n_items)")
    n_items = arr.shape[1]
    if n_items == 0:
        raise ValueError("need at least one item")
    point = float(arr.mean(axis=1).mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_items, size=(n_boot, n_items))
    # (n_contexts, n_boot, n_items) -> per-context accuracy -> mean over contexts
    draws = arr[:, idx].mean(axis=2).mean(axis=0)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(point, float(low), float(high))
