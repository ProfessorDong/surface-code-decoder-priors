"""Statistics for paired decoder comparisons.

All logical-error comparisons in the rework are *paired*: every prior decodes
the identical detector-event shots. The primitives here therefore operate on
boolean per-shot error vectors and their discordant counts.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps


# ----------------------------------------------------------------------------
# Confidence intervals for a single proportion
# ----------------------------------------------------------------------------

def clopper_pearson(num_errors: int, num_trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided interval for a binomial proportion."""
    k, n = int(num_errors), int(num_trials)
    if n <= 0:
        raise ValueError("num_trials must be positive")
    alpha = 1.0 - confidence
    lo = 0.0 if k == 0 else float(sps.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# ----------------------------------------------------------------------------
# Paired comparisons
# ----------------------------------------------------------------------------

def discordant_counts(errors_a: np.ndarray, errors_b: np.ndarray) -> Tuple[int, int]:
    """Return (b, c): shots where only A fails, and shots where only B fails."""
    a = np.asarray(errors_a, dtype=bool)
    b = np.asarray(errors_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired error vectors must have identical shape")
    only_a = int(np.count_nonzero(a & ~b))
    only_b = int(np.count_nonzero(~a & b))
    return only_a, only_b


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return 1.0
    return float(sps.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def paired_delta_ci(errors_a: np.ndarray, errors_b: np.ndarray,
                    confidence: float = 0.95) -> Tuple[float, float, float]:
    """Difference of paired proportions P(A fails) - P(B fails) with a Wald CI.

    Uses the paired variance ((b + c)/n - (b - c)^2/n^2)/n, which accounts for
    the shared shots. Returns (delta, lo, hi).
    """
    a = np.asarray(errors_a, dtype=bool)
    n = a.size
    b, c = discordant_counts(errors_a, errors_b)
    delta = (b - c) / n
    var = ((b + c) / n - delta ** 2) / n
    z = float(sps.norm.ppf(0.5 + confidence / 2))
    half = z * float(np.sqrt(max(var, 0.0)))
    return float(delta), float(delta - half), float(delta + half)


def block_bootstrap_delta(errors_a: np.ndarray, errors_b: np.ndarray, block_len: int,
                          n_boot: int = 2000, seed: int = 0,
                          confidence: float = 0.95) -> Tuple[float, float, float]:
    """Moving-block bootstrap CI for a paired difference on a serially dependent stream."""
    a = np.asarray(errors_a, dtype=np.int8)
    b = np.asarray(errors_b, dtype=np.int8)
    diff = a - b
    n = diff.size
    if block_len < 1 or block_len > n:
        raise ValueError("block_len must be in [1, n]")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    starts_max = n - block_len + 1
    csum = np.concatenate([[0], np.cumsum(diff)])
    block_sums = csum[block_len:] - csum[:-block_len]          # sum over each window start
    boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        boot[i] = block_sums[starts].sum() / (n_blocks * block_len)
    alpha = 1.0 - confidence
    return float(diff.mean()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def cluster_bootstrap_delta(pairs: Sequence[Tuple[np.ndarray, np.ndarray]], n_boot: int = 2000,
                            seed: int = 0, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap over clusters (e.g. qubit placements) of paired error vectors.

    The statistic is the pooled paired difference (A minus B) over all shots of
    the resampled clusters. Returns (delta, lo, hi).
    """
    if len(pairs) == 0:
        raise ValueError("need at least one cluster")
    sums = np.array([(np.asarray(a, dtype=np.int64).sum() - np.asarray(b, dtype=np.int64).sum())
                     for a, b in pairs], dtype=float)
    sizes = np.array([np.asarray(a).size for a, _ in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    k = len(pairs)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        boot[i] = sums[idx].sum() / sizes[idx].sum()
    alpha = 1.0 - confidence
    return float(sums.sum() / sizes.sum()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def holm(pvalues: Iterable[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(list(pvalues), dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


def mcnemar_shots_required(discordance_rate: float, delta: float, alpha: float = 0.01,
                           power: float = 0.9) -> int:
    """Approximate paired-test shot count to detect a paired difference ``delta``
    in failure probability when a fraction ``discordance_rate`` of shots are discordant."""
    z_a = float(sps.norm.ppf(1 - alpha / 2))
    z_b = float(sps.norm.ppf(power))
    return int(np.ceil((z_a + z_b) ** 2 * discordance_rate / delta ** 2))


# ----------------------------------------------------------------------------
# Per-round logical error and Lambda
# ----------------------------------------------------------------------------

def per_round_ler_google(p_shot: float, rounds: int) -> float:
    """Google's convention: epsilon = (1 - (1 - 2 P)^(1/T)) / 2."""
    base = 1.0 - 2.0 * float(p_shot)
    if base <= 0.0:
        return 0.5
    return 0.5 * (1.0 - base ** (1.0 / rounds))


def per_round_ler_naive(p_shot: float, rounds: int) -> float:
    """1 - (1 - P)^(1/T); the naive per-round rate, kept only for comparison."""
    return 1.0 - (1.0 - float(p_shot)) ** (1.0 / rounds)


def fit_eps_per_round(rounds: Sequence[int], p_shot: Sequence[float], n_shots: Sequence[int] | None = None,
                      min_signal: float = 0.05, with_intercept: bool = True) -> Tuple[float, float, int]:
    """Weighted least squares of ln(1 - 2P) against rounds.

    Points with 1 - 2P <= ``min_signal`` (saturated experiments) are excluded.
    Returns (eps, slope_std_err, n_points_used). eps = (1 - exp(slope)) / 2.
    """
    r = np.asarray(rounds, dtype=float)
    p = np.asarray(p_shot, dtype=float)
    sig = 1.0 - 2.0 * p
    mask = sig > min_signal
    if mask.sum() < (2 if with_intercept else 1):
        return float("nan"), float("nan"), int(mask.sum())
    y = np.log(sig[mask])
    x = r[mask]
    if n_shots is None:
        w = np.ones_like(x)
    else:
        n = np.asarray(n_shots, dtype=float)[mask]
        # delta-method variance of ln(1-2P): 4 P(1-P) / (n (1-2P)^2)
        var = 4.0 * p[mask] * (1 - p[mask]) / (n * sig[mask] ** 2)
        w = 1.0 / np.maximum(var, 1e-18)
    if with_intercept:
        X = np.column_stack([np.ones_like(x), x])
    else:
        X = x[:, None]
    W = np.diag(w)
    XtW = X.T @ W
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ XtW @ y
    slope = beta[-1]
    resid = y - X @ beta
    dof = max(int(mask.sum()) - X.shape[1], 1)
    scale = float((resid ** 2 * w).sum() / dof)
    se = float(np.sqrt(cov[-1, -1] * max(scale, 1.0)))
    return float(0.5 * (1.0 - np.exp(slope))), se, int(mask.sum())


def lambda_factor(eps_small_d: float, eps_large_d: float) -> float:
    """Error suppression factor per distance step of two."""
    return float(eps_small_d / eps_large_d)
