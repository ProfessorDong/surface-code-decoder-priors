"""Detector-correlation (p_ij) estimation of matching-graph edge probabilities.

Bulk edges (Google 2021 SI Eq. 11 = 2023 SI Eq. 21)::

    p_ij = 1/2 - 1/2 * sqrt( 1 - 4 (<x_i x_j> - <x_i><x_j>) / (1 - 2<x_i> - 2<x_j> + 4<x_i x_j>) )

Boundary edges (2021 SI Eqs. 14-16): with p_iS the probability that an odd
number of the bulk edges at detector i fired, combined with g(p, q) = p + q - 2pq,

    p_iB = (<x_i> - p_iS) / (1 - 2 p_iS).

Invalid estimates (negative discriminant, negative probability, or too few
events) fall back to the template's probability and are counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..config import P_MAX, P_MIN
from ..dem.edges import EdgeClass, EdgeTable
from .base import Estimator, FitResult


def unpack_chunk(dets_packed: np.ndarray, num_detectors: int, start: int, stop: int) -> np.ndarray:
    """Unpack rows [start, stop) to bool[shots, D] (little-endian bit order, as Stim writes)."""
    chunk = dets_packed[start:stop]
    bits = np.unpackbits(chunk, axis=1, bitorder="little")
    return bits[:, :num_detectors].astype(bool)


def pairwise_moments(dets_packed: np.ndarray, num_detectors: int, pairs: np.ndarray,
                     chunk: int = 2000) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (<x_i> for all detectors, <x_i x_j> for each pair, shots)."""
    n = dets_packed.shape[0]
    xi = np.zeros(num_detectors, dtype=np.int64)
    xij = np.zeros(len(pairs), dtype=np.int64)
    u = pairs[:, 0]
    v = pairs[:, 1]
    for s in range(0, n, chunk):
        X = unpack_chunk(dets_packed, num_detectors, s, min(n, s + chunk))
        xi += X.sum(axis=0)
        if len(pairs):
            xij += (X[:, u] & X[:, v]).sum(axis=0)
    return xi / n, xij / n, n


def pij_from_moments(xi: np.ndarray, xj: np.ndarray, xij: np.ndarray) -> np.ndarray:
    num = 4.0 * (xij - xi * xj)
    den = 1.0 - 2.0 * xi - 2.0 * xj + 4.0 * xij
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = 1.0 - num / den
        p = 0.5 - 0.5 * np.sqrt(arg)
    p = np.where((den > 0) & (arg >= 0) & np.isfinite(p), p, np.nan)
    return p


def combine_odd(ps: np.ndarray) -> float:
    """Probability that an odd number of independent events with probabilities ps occur."""
    return 0.5 * (1.0 - np.prod(1.0 - 2.0 * np.asarray(ps, dtype=float)))


@dataclass
class PijEstimator(Estimator):
    """Per-edge p_ij fit on the template's edge set."""

    min_events: int = 1          # minimum coincidence count for a bulk edge to be trusted
    p_floor: float = P_MIN
    name: str = "pij"
    needs_labels: bool = False

    def fit(self, template: EdgeTable, dets_packed: np.ndarray,
            obs_packed: Optional[np.ndarray] = None) -> FitResult:
        D = template.num_detectors
        bulk = np.flatnonzero(template.v >= 0)
        bnd = np.flatnonzero(template.v < 0)
        pairs = np.stack([template.u[bulk], template.v[bulk]], axis=1) if bulk.size else np.zeros((0, 2), int)
        xi, xij, n = pairwise_moments(dets_packed, D, pairs)
        return self.fit_moments(template, xi, xij, n)

    def fit_moments(self, template: EdgeTable, xi: np.ndarray, xij: np.ndarray, n: float) -> FitResult:
        """The estimator applied to moments <x_i> and <x_u x_v> (bulk edges in template order).

        ``n`` is the number of shots behind the moments; ``np.inf`` gives the
        population limit (no minimum-count fallback)."""
        bulk = np.flatnonzero(template.v >= 0)
        bnd = np.flatnonzero(template.v < 0)
        pairs = np.stack([template.u[bulk], template.v[bulk]], axis=1) if bulk.size else np.zeros((0, 2), int)
        p_new = template.p.copy()
        n_fallback_bulk = 0
        if bulk.size:
            est = pij_from_moments(xi[pairs[:, 0]], xi[pairs[:, 1]], xij)
            counts = xij * n
            bad = ~np.isfinite(est) | (est <= 0) | (counts < self.min_events)
            n_fallback_bulk = int(bad.sum())
            est = np.where(bad, template.p[bulk], est)
            p_new[bulk] = np.clip(est, self.p_floor, P_MAX)
        # boundary edges from the residual marginal
        n_fallback_bnd = 0
        if bnd.size:
            # bulk probabilities incident to each detector
            incident: Dict[int, list] = {}
            for k in bulk:
                incident.setdefault(int(template.u[k]), []).append(p_new[k])
                incident.setdefault(int(template.v[k]), []).append(p_new[k])
            # group boundary edges by detector (a detector may carry several: with/without logical)
            groups: Dict[int, list] = {}
            for k in bnd:
                groups.setdefault(int(template.u[k]), []).append(k)
            for det, ks in groups.items():
                p_sum = combine_odd(incident.get(det, []))
                den = 1.0 - 2.0 * p_sum
                est = (xi[det] - p_sum) / den if den > 0 else np.nan
                tmpl_total = float(template.p[ks].sum())
                if not np.isfinite(est) or est <= 0:
                    n_fallback_bnd += len(ks)
                    continue
                # split the detector's boundary mass across its boundary edges as in the template
                for k in ks:
                    share = template.p[k] / tmpl_total if tmpl_total > 0 else 1.0 / len(ks)
                    p_new[k] = np.clip(est * share, self.p_floor, P_MAX)
        table = template.with_probabilities(p_new)
        params = {"n_cal_shots": int(n) if np.isfinite(n) else -1, "n_fallback_bulk": n_fallback_bulk,
                  "n_fallback_boundary": n_fallback_bnd,
                  "n_bulk": int(bulk.size), "n_boundary": int(bnd.size), "min_events": self.min_events}
        return FitResult(table=table, params=params, p_hat=float(np.mean(xi)))
