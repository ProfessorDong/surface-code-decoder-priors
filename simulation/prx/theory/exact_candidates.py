"""Untruncated re-evaluation of the priors reported in Table I.

The risk-surface scans ran with a truncation tolerance, so each reported risk
carries a two-sided bound R_kept <= R <= R_kept + eps (Theorem 2). Propagated
into a gain, that bound is not negligible at the printed precision: at d = 3,
T = 3, p = 5e-4 the gain interval is 2.52% to 2.83%. For d = 3 the detector count
is small enough (D = 16 at T = 2, D = 24 at T = 3) to enumerate every syndrome,
so this module re-evaluates the physical prior, the maximum-likelihood decoder
and each reported candidate prior with no truncation at all. The only remaining
error is floating-point summation.

The candidates are read from the stored scans, never typed in.
"""
from __future__ import annotations

import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESULTS_ROOT
from ..dem.edges import EdgeTable
from ..dem.priors import build_circuit
from .exact import enumerate_dem, risk

CLASSES = ("space", "time", "diag", "boundary")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    a = ap.parse_args()
    th = a.results_root / "theory"
    jobs = []
    for T in (2, 3):
        surf = pd.read_csv(th / f"risk_surface_surf_d3_T{T}.csv")
        for r in surf.itertuples():
            jobs.append((T, float(r.p), "scan optimum", {c: float(getattr(r, f"s_{c}")) for c in CLASSES}))
    rb = th / "optimum_robustness_d3_T3_p0.004.csv"
    if rb.exists():
        d = pd.read_csv(rb)
        best = d[d.candidate == "refined"].iloc[0]
        jobs.append((3, 0.004, "robustness re-test", {c: float(best[f"s_{c}"]) for c in CLASSES}))

    rows, cache = [], {}
    for T, p, name, scales in jobs:
        key = (T, p)
        if key not in cache:
            t0 = time.time()
            dem = build_circuit("si1000", 3, T, p, variant="google2025").detector_error_model(decompose_errors=True)
            model = enumerate_dem(dem)                       # full, untruncated
            phys = EdgeTable.from_dem(dem)
            cache[key] = (model, phys, risk(model, phys), model.ml_failure(), time.time() - t0)
            print(f"T={T} p={p:g}: {len(model.syndromes)} syndromes, total mass "
                  f"{model.total_probability:.15f}", flush=True)
        model, phys, r_phys, r_ml, secs = cache[key]
        r_c = risk(model, phys.scaled_by_class(scales))
        rows.append({"T": T, "p": p, "candidate": name, **{f"s_{c}": scales[c] for c in CLASSES},
                     "R_phys": r_phys, "R_cand": r_c, "R_ml": r_ml, "gain": 1 - r_c / r_phys,
                     "gap_to_ml": 1 - r_ml / r_phys,
                     "recovered": (r_phys - r_c) / (r_phys - r_ml) if r_phys > r_ml else np.nan,
                     "n_syndromes": len(model.syndromes), "mass_deficit": 1 - model.total_probability,
                     "truncation": "none"})
        print(f"   {name:20s} gain={100 * rows[-1]['gain']:.4f}%  gap={100 * rows[-1]['gap_to_ml']:.3f}%",
              flush=True)
    out = th / "exact_candidates_untruncated.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
