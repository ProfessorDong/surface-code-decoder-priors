"""Exact three-way split of the matching-to-maximum-likelihood gap.

The gap between a matching decoder and optimal decoding has two named causes:
error degeneracy (matching optimises over errors, not over stabiliser cosets)
and the graphlike approximation (hyperedges are decomposed into graphlike
components which the decoder then treats as independent). Ablating the
hyperedges confounds them, because deleting mechanisms also lowers the total
fault probability and so moves the model deeper into the ordered phase, where
degeneracy is small for an unrelated reason.

This module separates them on a single detector error model, with one error
budget and no ablation. Write pi_true for the exact distribution of
(syndrome, logical) under the full model and pi_dec for the same distribution
under the decomposed model, that is, the graphlike components treated as
independent mechanisms with the probabilities the decoder is given. Every
failure probability below is evaluated under pi_true; only the decision rule
changes:

    R_match   matching, i.e. the most-probable-error rule of pi_dec
    R_MPdec   most probable single error under pi_dec          (= R_match)
    R_MLdec   most probable coset under pi_dec
    R_MLfull  most probable coset under pi_true                 (optimal)

    R_match - R_MLfull  =  (R_MPdec - R_MLdec)   [degeneracy of the decomposed model]
                        +  (R_MLdec - R_MLfull)  [cost of the graphlike approximation]

The identity R_match = R_MPdec holds because minimum-weight matching on a
graphlike model returns the most probable single error of that model
(Higgott and Gidney, Quantum 9, 1600 (2025), Sec. 2.2-2.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import stim

from ..dem.edges import EdgeTable
from .exact import ExactModel, enumerate_dem, syndrome_bits


def decomposed_dem(table: EdgeTable) -> stim.DetectorErrorModel:
    """The model the matching decoder actually uses: one independent mechanism
    per graphlike edge, with the probability PyMatching assigned to it."""
    dem = stim.DetectorErrorModel()
    for k in range(table.num_edges):
        targets = [stim.target_relative_detector_id(int(table.u[k]))]
        if table.v[k] >= 0:
            targets.append(stim.target_relative_detector_id(int(table.v[k])))
        for f in sorted(table.fault_ids(k)):
            targets.append(stim.target_logical_observable_id(int(f)))
        dem.append(stim.DemInstruction("error", [float(table.p[k])], targets))
    return dem


def _align(full: ExactModel, dec: ExactModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put both distributions on the union of their syndromes.

    Returns (syndromes, pi_true[:, 2], pi_dec[:, 2]).
    """
    keys = np.union1d(full.syndromes, dec.syndromes)
    pt = np.zeros((keys.size, 2))
    pd = np.zeros((keys.size, 2))
    ia = np.searchsorted(keys, full.syndromes)
    pt[ia, 0] = full.p_logical0
    pt[ia, 1] = full.p_logical1
    ib = np.searchsorted(keys, dec.syndromes)
    pd[ib, 0] = dec.p_logical0
    pd[ib, 1] = dec.p_logical1
    return keys, pt, pd


@dataclass
class ThreeWaySplit:
    R_match: float
    R_MLdec: float
    R_MLfull: float
    degeneracy: float        # R_match - R_MLdec
    decomposition: float     # R_MLdec - R_MLfull
    total_gap: float         # R_match - R_MLfull
    match_equals_MPdec: bool
    n_syndromes: int
    mpe_checked: int = 0
    mpe_disagree: int = 0
    mpe_ties: int = 0
    mpe_unreachable: int = 0

    def as_row(self) -> Dict[str, float]:
        return {"R_match": self.R_match, "R_MLdec": self.R_MLdec, "R_MLfull": self.R_MLfull,
                "gap_total_%": 100 * self.total_gap / self.R_match,
                "gap_degeneracy_%": 100 * self.degeneracy / self.R_match,
                "gap_decomposition_%": 100 * self.decomposition / self.R_match,
                "degeneracy_share_%": 100 * self.degeneracy / self.total_gap if self.total_gap > 0 else float("nan"),
                "match_equals_MPdec": self.match_equals_MPdec, "n_syndromes": self.n_syndromes,
                "mpe_checked": self.mpe_checked, "mpe_disagree": self.mpe_disagree,
                "mpe_ties": self.mpe_ties, "mpe_unreachable": self.mpe_unreachable}


def split(circuit: stim.Circuit, max_detectors: int = 26) -> ThreeWaySplit:
    dem_full = circuit.detector_error_model(decompose_errors=True)
    table = EdgeTable.from_dem(dem_full)
    dem_dec = decomposed_dem(table)

    full = enumerate_dem(dem_full, max_detectors=max_detectors)
    dec = enumerate_dem(dem_dec, max_detectors=max_detectors)
    keys, pt, pd = _align(full, dec)

    # decision rules
    ml_full = pt[:, 1] > pt[:, 0]                 # optimal under the true model
    ml_dec = pd[:, 1] > pd[:, 0]                  # optimal coset under the decomposed model
    bits = ((keys[:, None] >> np.arange(full.num_detectors, dtype=np.int64)[None, :]) & 1).astype(np.uint8)
    match = table.to_matching().decode_batch(bits)[:, 0].astype(bool)

    # Independent check of Lemma 1: the most probable single error of the decomposed
    # model, computed by max-product enumeration with no matching algorithm involved.
    # (An earlier version called the same matching decoder here, so the check could
    # not fail; see research/prx_quantum/audit_20260918/VERIFICATION_OF_AUDIT.md.)
    from .mpe import check_matching_is_mpe
    mpe = check_matching_is_mpe(table, full.num_detectors, keys, match)

    def risk(dec_rule: np.ndarray) -> float:
        return float(np.where(dec_rule, pt[:, 0], pt[:, 1]).sum())

    R_match, R_MLdec, R_MLfull = risk(match), risk(ml_dec), risk(ml_full)
    return ThreeWaySplit(
        R_match=R_match, R_MLdec=R_MLdec, R_MLfull=R_MLfull,
        degeneracy=R_match - R_MLdec, decomposition=R_MLdec - R_MLfull,
        total_gap=R_match - R_MLfull,
        match_equals_MPdec=bool(mpe["n_disagree"] == 0),
        n_syndromes=int(keys.size),
        mpe_checked=mpe["n_checked"], mpe_disagree=mpe["n_disagree"],
        mpe_ties=mpe["n_ties"], mpe_unreachable=mpe["n_unreachable"])


def main():
    import argparse
    import pandas as pd
    from ..dem.priors import build_circuit
    from .exact import repetition_memory

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", default="surface", choices=["surface", "repetition"])
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--p-grid", nargs="+", type=float, default=[0.0005, 0.001, 0.002, 0.004])
    ap.add_argument("--out", default="research/prx_quantum/results/theory")
    a = ap.parse_args()
    rows = []
    for p in a.p_grid:
        c = (build_circuit("si1000", a.distance, a.rounds, p, variant="google2025")
             if a.code == "surface" else repetition_memory(a.distance, a.rounds, p, p))
        s = split(c)
        row = {"code": a.code, "d": a.distance, "T": a.rounds, "p": p, **s.as_row()}
        rows.append(row)
        print(f"p={p:<8g} total gap={row['gap_total_%']:7.3f}%  "
              f"degeneracy={row['gap_degeneracy_%']:7.4f}%  decomposition={row['gap_decomposition_%']:7.3f}%  "
              f"(degeneracy is {row['degeneracy_share_%']:.2f}% of the gap)  "
              f"match==MP_dec: {row['match_equals_MPdec']}", flush=True)
    df = pd.DataFrame(rows)
    from pathlib import Path
    Path(a.out).mkdir(parents=True, exist_ok=True)
    f = Path(a.out) / f"three_way_{a.code}_d{a.distance}_T{a.rounds}.csv"
    df.to_csv(f, index=False)
    print("\nwrote", f)


if __name__ == "__main__":
    main()
