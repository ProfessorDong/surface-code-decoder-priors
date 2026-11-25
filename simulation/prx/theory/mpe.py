"""Most-probable-error decoding by max-product enumeration, independent of matching.

Lemma 1 of the manuscript states that minimum-weight matching with log-odds
weights returns the most probable single fault set of the graphlike model it is
given. A numerical check of that statement is only meaningful if the reference
does not itself run a matching algorithm. This module computes the reference by
the same XOR recursion used for the exact joint law (Proposition 1), with the
semiring changed from (+, x) to (max, x):

    M_k[s, l] = max( (1 - p_k) M_{k-1}[s, l],  p_k M_{k-1}[s ^ m_k, l ^ l_k] ).

After all edges, M[s, l] is the largest probability of any single fault
configuration of the decomposed model producing syndrome s and logical parity l.
The most probable error for s therefore has logical parity argmax_l M[s, l]. No
graph, shortest path or blossom step is involved, so agreement with PyMatching
is an independent test, not a comparison of an algorithm with itself.

Ties are handled explicitly: where the two logical classes attain the same
maximum to within a relative tolerance, either answer is a most probable error,
and matching is counted as agreeing whichever it returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dem.edges import EdgeTable


@dataclass
class MPEResult:
    syndromes: np.ndarray      # integer syndrome masks, one per syndrome with nonzero mass
    logical: np.ndarray        # most-probable-error logical parity (bool)
    tied: np.ndarray           # both parities attain the maximum
    log_max: np.ndarray        # log of the maximising configuration probability


def max_product(table: EdgeTable, num_detectors: int) -> MPEResult:
    """Exact most-probable-error parity for every syndrome of the decomposed model."""
    if num_detectors > 26:
        raise ValueError("max-product enumeration limited to 26 detectors")
    size = 1 << num_detectors
    neg = -np.inf
    L = np.full((size, 2), neg)               # log domain avoids underflow
    L[0, 0] = 0.0
    ar = np.arange(size, dtype=np.int64)
    for u, v, p, fm in zip(table.u, table.v, table.p, table.fault_mask):
        m = (1 << int(u)) | ((1 << int(v)) if int(v) >= 0 else 0)
        flip = int(fm) & 1
        stay = L + np.log1p(-p)
        src = L[ar ^ m]
        if flip:
            src = src[:, ::-1]
        move = src + np.log(p)
        np.maximum(stay, move, out=L)
    reach = np.isfinite(L).any(axis=1)
    idx = np.flatnonzero(reach)
    a, b = L[idx, 0], L[idx, 1]
    hi = np.maximum(a, b)
    tied = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= 1e-9 * np.maximum(1.0, np.abs(hi)))
    return MPEResult(idx.astype(np.int64), b > a, tied, hi)


def check_matching_is_mpe(table: EdgeTable, num_detectors: int, syndromes: np.ndarray,
                          predictions: np.ndarray) -> dict:
    """Compare matching predictions on ``syndromes`` with the max-product reference.

    Returns counts of agreements, disagreements outside ties, and syndromes the
    decomposed model cannot produce at all (which matching may still decode).
    """
    ref = max_product(table, num_detectors)
    pos = np.searchsorted(ref.syndromes, syndromes)
    pos_c = np.clip(pos, 0, len(ref.syndromes) - 1)
    found = ref.syndromes[pos_c] == syndromes
    pred = np.asarray(predictions).astype(bool)
    agree = found & ((ref.logical[pos_c] == pred) | ref.tied[pos_c])
    disagree = found & ~agree
    return {"n_checked": int(found.sum()), "n_agree": int(agree.sum()),
            "n_disagree": int(disagree.sum()), "n_ties": int((found & ref.tied[pos_c]).sum()),
            "n_unreachable": int((~found).sum())}
