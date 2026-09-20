"""Per-round logical error and suppression factor Lambda, Google's convention.

epsilon_d is fitted per (code, basis) by weighted least squares of ln(1 - 2 p_L)
against rounds (two parameters), then averaged over bases and placements.
Lambda comes from a weighted regression of ln(eps_d) on (d + 1) / 2:
Lambda = exp(-slope). Saturated points (1 - 2 p_L <= min_signal) are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .stats import fit_eps_per_round


@dataclass
class EpsFit:
    eps: float
    se: float
    n_points: int


def fit_eps_groups(rows: Iterable[Tuple[str, int, int, int, int]], min_signal: float = 0.05) -> Dict[Tuple[str, int], EpsFit]:
    """rows: (group, distance, rounds, n_errors, n_shots) -> {(group, distance): EpsFit}."""
    acc: Dict[Tuple[str, int], List[Tuple[int, int, int]]] = {}
    for g, d, r, k, n in rows:
        acc.setdefault((g, d), []).append((r, k, n))
    out = {}
    for key, pts in acc.items():
        r = [p[0] for p in pts]
        p_l = [p[1] / p[2] for p in pts]
        n = [p[2] for p in pts]
        eps, se, used = fit_eps_per_round(r, p_l, n, min_signal=min_signal, with_intercept=True)
        out[key] = EpsFit(eps, se, used)
    return out


def average_eps(fits: Dict[Tuple[str, int], EpsFit]) -> Dict[int, Tuple[float, float, int]]:
    """Average eps over groups per distance: (mean, standard error of the mean, n_groups)."""
    by_d: Dict[int, List[float]] = {}
    for (g, d), f in fits.items():
        if np.isfinite(f.eps):
            by_d.setdefault(d, []).append(f.eps)
    return {d: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"), len(v))
            for d, v in by_d.items()}


def lambda_regression(eps_by_d: Dict[int, Tuple[float, float, int]]) -> Tuple[float, float]:
    """Weighted regression of ln eps on (d+1)/2; returns (Lambda, delta Lambda)."""
    ds = sorted(d for d, v in eps_by_d.items() if np.isfinite(v[0]) and v[0] > 0)
    if len(ds) < 2:
        return float("nan"), float("nan")
    x = np.array([(d + 1) / 2 for d in ds], dtype=float)
    y = np.array([np.log(eps_by_d[d][0]) for d in ds])
    se = np.array([eps_by_d[d][1] / eps_by_d[d][0] if np.isfinite(eps_by_d[d][1]) else 1.0 for d in ds])
    w = 1.0 / np.maximum(se, 1e-6) ** 2
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    beta = cov @ X.T @ (w * y)
    slope = beta[1]
    return float(np.exp(-slope)), float(np.exp(-slope) * np.sqrt(cov[1, 1]))


def lambda_pairwise(eps_by_d: Dict[int, Tuple[float, float, int]]) -> Dict[str, float]:
    ds = sorted(eps_by_d)
    return {f"{a}->{b}": eps_by_d[a][0] / eps_by_d[b][0] for a, b in zip(ds[:-1], ds[1:])}
