"""Do the correlation and occurrence population limits make the same decisions?

At d = 3, T = 3, p = 5e-4 (SI1000), compute both limit priors (exact moments; exact
occurrence frequencies over the syndromes carrying all but 1e-10 of the mass), decode
every syndrome carrying all but 1e-12 of the mass with each, and report the
probability mass of syndromes on which their logical decisions differ.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from ..config import RESULTS_ROOT
from ..dem.edges import EdgeTable
from ..dem.priors import build_circuit
from ..estimators.pij import PijEstimator
from .dgr_limit_exact import occurrence_limit
from .exact import enumerate_dem, truncate
from .population_moments import population_moments


def main() -> None:
    p = 0.0005
    circ = build_circuit("si1000", 3, 3, p, variant="google2025"); dem = circ.detector_error_model(decompose_errors=True)
    model = enumerate_dem(dem); phys = EdgeTable.from_dem(dem)
    pairs = np.stack([phys.u[phys.v >= 0], phys.v[phys.v >= 0]], 1)
    xi, xij = population_moments(circ.detector_error_model(decompose_errors=False), pairs)
    corr = PijEstimator().fit_moments(phys, xi, xij, np.inf).table
    occ, _ = occurrence_limit(model, phys, 1e-10)
    kept = truncate(model, 1e-12); w = kept.p_logical0 + kept.p_logical1
    dc = corr.to_matching().decode_batch(kept.bits)[:, 0]; do = occ.to_matching().decode_batch(kept.bits)[:, 0]
    lr = np.log(occ.p / corr.p)
    out = pd.DataFrame([{"T": 3, "p": p, "mass_decisions_differ": float(w[dc != do].sum()), "mass_neglected": kept.neglected,
                         "median_abs_ln_ratio": float(np.median(np.abs(lr))), "max_abs_ln_ratio": float(np.abs(lr).max())}])
    out.to_csv(RESULTS_ROOT / "theory" / "limit_agreement.csv", index=False); print(out.to_string(index=False))


if __name__ == "__main__":
    main()
