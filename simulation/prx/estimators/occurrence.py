"""Occurrence-based edge estimation: the alignment re-weighting of DGR.

Wang et al. (arXiv:2311.16214, Sec. IV A) decode past syndromes with the current
matching graph, count how often each edge appears in the returned matchings, and
use that frequency as the edge probability. It is label-free, like the pairwise
estimators, but it infers edges through the decoder instead of from detector
moments, so its accuracy is bounded by the decoder's (their Sec. IV A).

Implementation choices, which the paper leaves open:
  * one pass, decoding with the reference prior (the paper's alignment step
    decodes with the current weights; iterating is optional there);
  * an edge that never appears keeps the reference probability, because a
    frequency of zero would give it infinite weight;
  * counts use every calibration shot; no sliding window, since the noise here
    is static.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import P_MAX, P_MIN
from ..dem.edges import EdgeTable
from .base import Estimator, FitResult


@dataclass
class OccurrenceEstimator(Estimator):
    passes: int = 1
    name: str = "dgr"
    needs_labels: bool = False

    def fit(self, template: EdgeTable, dets_packed: np.ndarray,
            obs_packed: Optional[np.ndarray] = None) -> FitResult:
        n = int(dets_packed.shape[0])
        # canonical keys: (min, max) for bulk edges, (detector, -1) for boundary edges
        index = {((min(int(u), int(v)), max(int(u), int(v))) if v >= 0 else (int(u), -1)): k
                 for k, (u, v) in enumerate(zip(template.u, template.v))}
        assert len(index) == template.num_edges, "parallel edges are not supported"
        table = template
        counts = np.zeros(template.num_edges, dtype=np.int64)
        unmatched = 0
        for _ in range(self.passes):
            m = table.to_matching()
            counts[:] = 0
            bits = np.unpackbits(dets_packed, axis=1, bitorder="little")[:, :template.num_detectors].astype(np.uint8)
            for s in range(n):
                if not bits[s].any():
                    continue
                for a, b in m.decode_to_edges_array(bits[s]):
                    a, b = int(a), int(b)
                    key = (min(a, b), max(a, b)) if a >= 0 and b >= 0 else (max(a, b), -1)
                    k = index.get(key)
                    if k is None:
                        unmatched += 1
                    else:
                        counts[k] += 1
            p_new = np.where(counts > 0, counts / n, template.p)
            table = template.with_probabilities(np.clip(p_new, P_MIN, P_MAX))
        params = {"n_cal_shots": n, "passes": self.passes, "n_edges_seen": int((counts > 0).sum()),
                  "n_edges": int(template.num_edges), "n_unmatched_edge_ids": unmatched}
        return FitResult(table=table, params=params, p_hat=None)
