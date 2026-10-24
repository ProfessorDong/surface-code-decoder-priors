"""Exact risk of the pairwise estimator's population limit under homogeneous noise.

For the distance-3 memories of Table II, the pairwise estimator (with or without
pooling; they coincide in the limit) is applied to the exact population moments of
the true detector error model, and the resulting graph prior is decoded on every
syndrome (no truncation). This is what unlimited syndrome data would give the
correlation fit, compared with the physical graph prior, with the best per-class
prior found by search (Table II), and with maximum-likelihood decoding.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from ..config import RESULTS_ROOT
from ..dem.edges import EdgeTable
from ..dem.priors import build_circuit
from ..estimators.pij import PijEstimator
from ..estimators.pooled import PooledPijEstimator
from .exact import enumerate_dem, risk
from .population_moments import population_moments


CERT = {2: "certified_class_family_T2_wspan.csv", 3: "certified_class_family_T3_gpu.csv"}


def main() -> None:
    th = RESULTS_ROOT / "theory"
    cand = pd.read_csv(th / "exact_candidates_untruncated.csv")
    best = cand.loc[cand.groupby(["T", "p"]).gain.idxmax()].set_index(["T", "p"])
    # the branch and bound reached rates the coordinate scan never visited; the exact
    # distribution is the same object, so report those operating points too
    points = set(best.index)
    for T, fname in CERT.items():
        f = th / fname
        if f.exists():
            points |= {(T, float(x)) for x in pd.read_csv(f).p}
    rows = []
    for (T, p) in sorted(points):
        circ = build_circuit("si1000", 3, int(T), float(p), variant="google2025")
        dem = circ.detector_error_model(decompose_errors=True)
        model = enumerate_dem(dem)
        phys = EdgeTable.from_dem(dem)
        bulk = np.flatnonzero(phys.v >= 0)
        pairs = np.stack([phys.u[bulk], phys.v[bulk]], axis=1)
        xi, xij = population_moments(circ.detector_error_model(decompose_errors=False), pairs)
        lim = PijEstimator().fit_moments(phys, xi, xij, np.inf).table
        lim_pooled = PooledPijEstimator(kappa=0.0).fit_moments(phys, xi, xij, np.inf).table
        r_phys, r_lim, r_limp, r_ml = risk(model, phys), risk(model, lim), risk(model, lim_pooled), model.ml_failure()
        r_best = float(best.R_cand[(T, p)]) if (T, p) in best.index else float("nan")
        rows.append({"T": int(T), "p": float(p), "R_phys": r_phys, "R_limit": r_lim, "R_limit_pooled": r_limp,
                     "R_best_class": r_best, "R_ml": r_ml,
                     "gain_limit": 1 - r_lim / r_phys, "gain_limit_pooled": 1 - r_limp / r_phys,
                     "gain_best_class": 1 - r_best / r_phys, "gap_to_ml": 1 - r_ml / r_phys,
                     "mass_deficit": 1 - model.total_probability})
        r = rows[-1]
        print(f"T={int(T)} p={p:g}: limit gain {100*r['gain_limit']:.3f}% (pooled {100*r['gain_limit_pooled']:.3f}%)"
              f"  best class {100*r['gain_best_class']:.3f}%  ML gap {100*r['gap_to_ml']:.2f}%", flush=True)
    out = th / "moment_limit_exact.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
