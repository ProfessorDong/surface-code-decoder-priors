"""Exact logical error rates by complete enumeration of a detector error model.

For a DEM with N independent error mechanisms we enumerate all 2^N subsets,
so every quantity below is exact to floating-point, with no Monte Carlo error:

* P(syndrome, logical) for every syndrome;
* the maximum-likelihood (most-likely-coset) failure probability, which is the
  optimal decoder for this noise;
* the failure probability of minimum-weight perfect matching with an arbitrary
  prior, so the risk surface R(theta) can be mapped exactly and its minimiser
  compared with the physical prior.

The distribution is built by XOR convolution over mechanisms, which costs
O(M 2^D) in the number of mechanisms M and detectors D. It is therefore exact
for circuits with arbitrarily many error mechanisms, and is limited only by the
detector count (D <= 26 fits in memory), covering repetition-code memories and
rotated surface-code memories at d = 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pymatching
import stim

from ..dem.edges import EdgeTable


@dataclass
class ExactModel:
    """Enumerated syndrome distribution of a detector error model."""

    num_detectors: int
    syndromes: np.ndarray        # integer syndrome bitmask per distinct syndrome
    p_logical0: np.ndarray       # P(syndrome, logical = 0)
    p_logical1: np.ndarray       # P(syndrome, logical = 1)
    mechanisms: int

    @property
    def total_probability(self) -> float:
        return float(self.p_logical0.sum() + self.p_logical1.sum())

    def ml_failure(self) -> float:
        """Most-likely-coset decoder: pick the more probable logical class."""
        return float(np.minimum(self.p_logical0, self.p_logical1).sum())

    def decoder_failure(self, predictions: np.ndarray) -> float:
        """Failure probability of a decoder that predicts ``predictions[i]`` for syndrome i."""
        pred = np.asarray(predictions).astype(bool)
        return float(np.where(pred, self.p_logical0, self.p_logical1).sum())


def _mechanisms(dem: stim.DetectorErrorModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(probabilities, detector bitmask as an integer, logical parity) per error instruction."""
    probs: List[float] = []
    masks: List[int] = []
    logicals: List[int] = []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        m = 0
        log = 0
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                m ^= 1 << t.val
            elif t.is_logical_observable_id():
                log ^= 1
        probs.append(ins.args_copy()[0])
        masks.append(m)
        logicals.append(log)
    return np.asarray(probs, dtype=float), np.asarray(masks, dtype=np.int64), np.asarray(logicals, dtype=np.int8)


def enumerate_dem(dem: stim.DetectorErrorModel, max_detectors: int = 26) -> ExactModel:
    """Exact joint distribution of (syndrome, logical parity) by XOR convolution.

    Each independent mechanism acts on the distribution as
        new[s, l] = (1 - p) old[s, l] + p old[s ^ mask, l ^ flip],
    so the full distribution costs O(M 2^D) rather than O(2^M): it is exact for
    any number of mechanisms, and limited only by the detector count D.
    """
    dem = dem.flattened()
    D = dem.num_detectors
    if D > max_detectors:
        raise ValueError(f"{D} detectors exceeds the exact-convolution limit {max_detectors}")
    probs, masks, logicals = _mechanisms(dem)
    size = 1 << D
    state = np.zeros((size, 2), dtype=np.float64)
    state[0, 0] = 1.0
    ar = np.arange(size, dtype=np.int64)
    for p, m, l in zip(probs, masks, logicals):
        src = state[ar ^ m]                       # permuted copy
        if l:
            src = src[:, ::-1]
        state *= (1.0 - p)
        state += p * src
    keep = state.sum(axis=1) > 0
    idx = np.flatnonzero(keep)
    return ExactModel(D, idx.astype(np.int64), state[idx, 0].copy(), state[idx, 1].copy(), int(len(probs)))


def syndrome_bits(model: "ExactModel") -> np.ndarray:
    """Unpack the stored syndrome integers into a bool array [n_syndromes, D]."""
    s = model.syndromes[:, None]
    return ((s >> np.arange(model.num_detectors, dtype=np.int64)[None, :]) & 1).astype(np.uint8)


def mwpm_predictions(model: ExactModel, matching: pymatching.Matching) -> np.ndarray:
    """Decode every distinct syndrome once; returns the predicted logical bit."""
    preds = matching.decode_batch(syndrome_bits(model))
    return preds[:, 0].astype(bool)


def risk(model: ExactModel, table: EdgeTable) -> float:
    return model.decoder_failure(mwpm_predictions(model, table.to_matching()))


@dataclass
class TruncatedModel:
    """The highest-probability syndromes of an ExactModel, with a bound on the rest.

    Decoding every syndrome of a large model is expensive, but almost all of the
    probability sits on a few low-weight syndromes. Keeping the syndromes that
    carry 1 - ``neglected`` of the mass gives
        R_kept <= R <= R_kept + neglected,
    an exact two-sided bound. Comparisons between priors are valid whenever the
    differences of interest exceed ``neglected``.
    """

    bits: np.ndarray
    p_logical0: np.ndarray
    p_logical1: np.ndarray
    neglected: float
    ml_failure_kept: float

    def decoder_failure(self, predictions: np.ndarray) -> float:
        pred = np.asarray(predictions).astype(bool)
        return float(np.where(pred, self.p_logical0, self.p_logical1).sum())

    def risk(self, table: EdgeTable) -> float:
        preds = table.to_matching().decode_batch(self.bits)[:, 0].astype(bool)
        return self.decoder_failure(preds)


def truncate(model: ExactModel, tol: float = 1e-9) -> TruncatedModel:
    """Keep the most probable syndromes until the neglected mass is below ``tol``.

    ``tol <= 0`` keeps every enumerated syndrome. (Searching the cumulative sum for
    1.0 would not: in floating point it reaches 1.0 before the last syndromes, which
    would then be dropped silently.)
    """
    total = model.p_logical0 + model.p_logical1
    order = np.argsort(total)[::-1]
    cumulative = np.cumsum(total[order])
    if tol <= 0:
        keep_n = order.size
    else:
        keep_n = int(np.searchsorted(cumulative, 1.0 - tol) + 1)
    keep_n = min(keep_n, order.size)
    idx = order[:keep_n]
    neglected = float(max(0.0, 1.0 - cumulative[keep_n - 1]))
    sub = ExactModel(model.num_detectors, model.syndromes[idx], model.p_logical0[idx], model.p_logical1[idx],
                     model.mechanisms)
    return TruncatedModel(syndrome_bits(sub), sub.p_logical0, sub.p_logical1, neglected,
                          float(np.minimum(sub.p_logical0, sub.p_logical1).sum()))


def repetition_memory(distance: int, rounds: int, p_data: float, p_meas: float) -> stim.Circuit:
    """Phenomenological repetition-code memory: data flips between rounds, noisy measurements."""
    return stim.Circuit.generated(
        "repetition_code:memory", distance=distance, rounds=rounds,
        before_round_data_depolarization=p_data,
        before_measure_flip_probability=p_meas,
    )
