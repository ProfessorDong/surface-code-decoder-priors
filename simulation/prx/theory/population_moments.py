"""Exact population moments of detector events under a detector error model.

For independent mechanisms m with probabilities p_m and detector sets D_m, the
parity s_S = prod_{i in S} (1 - 2 x_i) of any detector set S has expectation

    E[s_S] = prod over m with |D_m intersect S| odd of (1 - 2 p_m).

Hence <x_i> = (1 - E[s_i]) / 2 and <x_i x_j> = (1 - E[s_i] - E[s_j] + E[s_i s_j]) / 4.
Feeding these moments to an estimator gives its population limit: the prior it
would return from infinitely many calibration shots. Comparing the decoding risk
of that prior with the oracle separates the estimator's bias (the limit) from its
finite-data error (the distance of the finite-N fit from the limit).

The DEM must not be decomposed: the components of a decomposed error fire
together and are one mechanism.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import stim


def mechanisms(dem: stim.DetectorErrorModel):
    """(probabilities, list of detector-index arrays) of every error in the flattened DEM."""
    probs, dets = [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ds = sorted({t.val for t in inst.targets_copy() if t.is_relative_detector_id()})
        # a detector listed twice (possible across "^" components) cancels
        cnt = {}
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                cnt[t.val] = cnt.get(t.val, 0) + 1
        ds = sorted(k for k, v in cnt.items() if v % 2 == 1)
        probs.append(float(inst.args_copy()[0]))
        dets.append(np.asarray(ds, dtype=np.int64))
    return np.asarray(probs), dets


def population_moments(dem: stim.DetectorErrorModel, pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact <x_i> for every detector and <x_u x_v> for each row (u, v) of ``pairs``."""
    D = dem.num_detectors
    probs, dets = mechanisms(dem)
    lg = np.log1p(-2.0 * probs)                     # log(1 - 2 p_m)
    log_si = np.zeros(D)
    both: dict = {}                                 # (u, v) -> sum of lg over mechanisms containing both
    want = {(int(min(u, v)), int(max(u, v))) for u, v in pairs}
    for l, ds in zip(lg, dets):
        log_si[ds] += l
        if len(ds) >= 2:
            for a in range(len(ds)):
                for b in range(a + 1, len(ds)):
                    key = (int(ds[a]), int(ds[b]))
                    if key in want:
                        both[key] = both.get(key, 0.0) + l
    e_si = np.exp(log_si)
    xi = (1.0 - e_si) / 2.0
    xij = np.empty(len(pairs))
    for k, (u, v) in enumerate(pairs):
        key = (int(min(u, v)), int(max(u, v)))
        e_sij = np.exp(log_si[u] + log_si[v] - 2.0 * both.get(key, 0.0))
        xij[k] = (1.0 - e_si[u] - e_si[v] + e_sij) / 4.0
    return xi, xij
