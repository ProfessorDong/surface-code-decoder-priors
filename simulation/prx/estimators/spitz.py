"""Spitz et al. (2018) streaming estimator: boxcar window over chronological shots.

Moments are accumulated per block of shots; the estimate used for block b is
built from the ``window_blocks`` blocks that precede it (strictly causal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

from ..config import P_MAX, P_MIN
from ..dem.edges import EdgeTable
from .base import FitResult
from .pij import combine_odd, pairwise_moments, pij_from_moments


@dataclass
class SpitzStream:
    template: EdgeTable
    block_shots: int = 100
    window_blocks: int = 20

    def _table_from(self, xi: np.ndarray, xij: np.ndarray, n: int) -> Tuple[EdgeTable, int]:
        t = self.template
        bulk = np.flatnonzero(t.v >= 0)
        pairs = np.stack([t.u[bulk], t.v[bulk]], axis=1)
        p_new = t.p.copy()
        est = pij_from_moments(xi[pairs[:, 0]], xi[pairs[:, 1]], xij)
        bad = ~np.isfinite(est) | (est <= 0)
        p_new[bulk] = np.where(bad, t.p[bulk], est)
        incident = {}
        for k in bulk:
            incident.setdefault(int(t.u[k]), []).append(p_new[k])
            incident.setdefault(int(t.v[k]), []).append(p_new[k])
        n_fb = int(bad.sum())
        for k in np.flatnonzero(t.v < 0):
            det = int(t.u[k])
            p_sum = combine_odd(incident.get(det, []))
            den = 1 - 2 * p_sum
            e = (xi[det] - p_sum) / den if den > 0 else np.nan
            if np.isfinite(e) and e > 0:
                same = (t.u == det) & (t.v < 0)
                p_new[k] = e * t.p[k] / max(t.p[same].sum(), P_MIN)
            else:
                n_fb += 1
        return t.with_probabilities(np.clip(p_new, P_MIN, P_MAX)), n_fb

    def run(self, dets_packed: np.ndarray) -> Iterator[Tuple[int, FitResult]]:
        """Yield (block_index, FitResult) for every block that has a full causal window."""
        t = self.template
        bulk = np.flatnonzero(t.v >= 0)
        pairs = np.stack([t.u[bulk], t.v[bulk]], axis=1)
        n = dets_packed.shape[0]
        nb = n // self.block_shots
        xi_b = np.zeros((nb, t.num_detectors))
        xij_b = np.zeros((nb, len(bulk)))
        for b in range(nb):
            s = b * self.block_shots
            xi, xij, _ = pairwise_moments(dets_packed[s:s + self.block_shots], t.num_detectors, pairs, chunk=self.block_shots)
            xi_b[b] = xi
            xij_b[b] = xij
        cxi = np.vstack([np.zeros((1, t.num_detectors)), np.cumsum(xi_b, axis=0)])
        cxij = np.vstack([np.zeros((1, len(bulk))), np.cumsum(xij_b, axis=0)])
        W = self.window_blocks
        for b in range(W, nb):
            xi = (cxi[b] - cxi[b - W]) / W
            xij = (cxij[b] - cxij[b - W]) / W
            table, n_fb = self._table_from(xi, xij, W * self.block_shots)
            yield b, FitResult(table=table, params={"block": b, "window_shots": W * self.block_shots, "n_fallback": n_fb},
                               p_hat=float(xi.mean()))
