"""Population limit of the occurrence (DGR alignment) estimator, and its exact risk.

With unlimited calibration data the occurrence estimator of ``estimators.occurrence``
returns, for every edge, the probability that the edge appears in the matching the
reference decoder returns:

    p_e = sum_s pi(s) 1[e in M(s)],

M(s) being matching's correction for syndrome s under the reference prior (here the
physical prior, so the limit measures the estimator's bias, not a misspecification).
Edges that never appear keep the reference probability. We evaluate p_e over every
syndrome of the enumerated model (T = 2) or over the syndromes carrying all but
``tol`` of the mass (T = 3), which bounds the error of every p_e by ``tol``, and then
compute the exact risk of the resulting prior with no truncation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import P_MAX, P_MIN, RESULTS_ROOT
from ..dem.edges import EdgeTable
from ..dem.priors import build_circuit
from .exact import enumerate_dem, risk, syndrome_bits, truncate


def occurrence_limit(model, phys: EdgeTable, tol: float) -> tuple:
    kept = truncate(model, tol) if tol > 0 else truncate(model, 0.0)
    w = kept.p_logical0 + kept.p_logical1
    m = phys.to_matching()
    index = {((min(int(u), int(v)), max(int(u), int(v))) if v >= 0 else (int(u), -1)): k
             for k, (u, v) in enumerate(zip(phys.u, phys.v))}
    mass = np.zeros(phys.num_edges)
    for s in range(kept.bits.shape[0]):
        if w[s] == 0 or not kept.bits[s].any():
            continue
        for a, b in m.decode_to_edges_array(kept.bits[s]):
            a, b = int(a), int(b)
            key = (min(a, b), max(a, b)) if a >= 0 and b >= 0 else (max(a, b), -1)
            mass[index[key]] += w[s]
    p = np.where(mass > 0, mass, phys.p)
    return phys.with_probabilities(np.clip(p, P_MIN, P_MAX)), kept.neglected


def main() -> None:
    th = RESULTS_ROOT / "theory"
    lim = pd.read_csv(th / "moment_limit_exact.csv").set_index(["T", "p"])
    rows = []
    for (T, p), r in lim.iterrows():
        dem = build_circuit("si1000", 3, int(T), float(p), variant="google2025").detector_error_model(decompose_errors=True)
        model = enumerate_dem(dem)
        phys = EdgeTable.from_dem(dem)
        table, neglected = occurrence_limit(model, phys, 0.0 if T == 2 else 1e-10)
        r_occ = risk(model, table)
        rows.append({"T": int(T), "p": float(p), "R_phys": float(r.R_phys), "R_occurrence_limit": r_occ,
                     "gain_occurrence_limit": 1 - r_occ / float(r.R_phys), "gain_correlation_limit": float(r.gain_limit),
                     "occurrence_mass_neglected": neglected})
        print(f"T={int(T)} p={p:g}: occurrence limit gain {100*rows[-1]['gain_occurrence_limit']:+.3f}%  "
              f"(correlation limit {100*r.gain_limit:+.3f}%)", flush=True)
    out = th / "dgr_limit_exact.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
