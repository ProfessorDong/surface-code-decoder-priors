"""Exact risk surface of a matching decoder as a function of its prior.

Everything is computed by complete enumeration of the detector error model, so
there is no Monte Carlo error: for each prior on the grid we decode every
distinct syndrome once and sum the exact probability of a logical failure.

Outputs (CSV, one row per (model, p, prior)):
  R_phys   exact failure probability with the physical prior
  R_opt    minimum over the scanned prior family
  R_ml     most-likely-coset (optimal decoder) failure probability
  scales   the minimising per-class scales
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..dem.edges import CLASS_NAMES, EdgeClass, EdgeTable
from ..dem.priors import build_circuit
from .exact import ExactModel, TruncatedModel, enumerate_dem, repetition_memory, risk, truncate


def scan_exhaustive(model: ExactModel, phys: EdgeTable, classes: Sequence[str],
                    grid: Sequence[float]) -> Tuple[Dict[str, float], float, List[dict]]:
    """Exhaustive scan over the product grid (only affordable for small syndrome spaces)."""
    best_scales = {c: 1.0 for c in classes}
    best = model.risk(phys)
    log = [{"scales": dict(best_scales), "risk": best}]
    for combo in itertools.product(grid, repeat=len(classes)):
        scales = {c: float(s) for c, s in zip(classes, combo)}
        r = model.risk(phys.scaled_by_class(scales))
        log.append({"scales": scales, "risk": r})
        if r < best:
            best, best_scales = r, scales
    return best_scales, best, log


def scan_coordinate(model: ExactModel, phys: EdgeTable, classes: Sequence[str], grid: Sequence[float],
                    sweeps: int = 4, refine: int = 2) -> Tuple[Dict[str, float], float, List[dict]]:
    """Coordinate descent with local refinement; exact risk at every evaluation.

    After the coarse sweeps converge, each class is refined on a geometric
    sub-grid bracketing its current value, ``refine`` times.
    """
    scales = {c: 1.0 for c in classes}
    best = model.risk(phys)
    log = [{"scales": dict(scales), "risk": best}]

    def try_grid(c: str, values: Sequence[float]) -> bool:
        nonlocal best
        improved = False
        cur = scales[c]
        for s in values:
            if abs(s - scales[c]) < 1e-12:
                continue
            trial = dict(scales)
            trial[c] = float(s)
            r = model.risk(phys.scaled_by_class(trial))
            log.append({"scales": trial, "risk": r})
            if r < best - 1e-18:
                best, cur, improved = r, float(s), True
        scales[c] = cur
        return improved

    for _ in range(sweeps):
        # Materialise the list: any() would short-circuit and skip the remaining
        # classes as soon as one of them improved.
        improved = [try_grid(c, grid) for c in classes]
        if not any(improved):
            break
    for _ in range(refine):
        for c in classes:
            lo, hi = scales[c] / 1.6, scales[c] * 1.6
            try_grid(c, list(np.geomspace(lo, hi, 9)))
    return scales, best, log


def scan(model: ExactModel, phys: EdgeTable, classes: Sequence[str], grid: Sequence[float],
         method: str = "coordinate") -> Tuple[Dict[str, float], float, List[dict]]:
    if method == "exhaustive":
        return scan_exhaustive(model, phys, classes, grid)
    return scan_coordinate(model, phys, classes, grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("research/prx_quantum/results/theory"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--code", default="repetition", choices=["repetition", "surface"])
    ap.add_argument("--p-grid", nargs="+", type=float,
                    default=[0.002, 0.005, 0.01, 0.02, 0.04, 0.08])
    ap.add_argument("--scale-grid", nargs="+", type=float,
                    default=[0.15, 0.2, 0.28, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75, 2.1, 2.6, 3.2, 4.0, 5.0])
    ap.add_argument("--classes", nargs="+", default=["space", "time", "diag", "boundary"])
    ap.add_argument("--method", default="coordinate", choices=["coordinate", "exhaustive"])
    ap.add_argument("--tol", type=float, default=1e-10, help="neglected probability mass")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = []
    for p in args.p_grid:
        if args.code == "repetition":
            circuit = repetition_memory(args.distance, args.rounds, p, p)
            dem = circuit.detector_error_model()
        else:
            circuit = build_circuit("si1000", args.distance, args.rounds, p, variant="google2025")
            dem = circuit.detector_error_model(decompose_errors=True)
        t0 = time.time()
        full = enumerate_dem(dem)
        model = truncate(full, args.tol)
        phys = EdgeTable.from_dem(dem)
        counts = phys.class_counts()
        classes = [c for c in args.classes if counts.get(c, 0) > 0]
        r_phys = model.risk(phys)
        r_ml = model.ml_failure_kept
        build_seconds = time.time() - t0
        scales, r_opt, log = scan(model, phys, classes, args.scale_grid, args.method)
        rows.append({"distance": args.distance, "rounds": args.rounds, "p": p, "mechanisms": full.mechanisms,
                     "edges": phys.num_edges, **{f"n_{k}": v for k, v in counts.items()},
                     "R_phys": r_phys, "R_opt": r_opt, "R_ml": r_ml, "bound_width": model.neglected,
                     "syndromes_kept": len(model.bits), "build_seconds": build_seconds,
                     "gain_vs_phys": 1 - r_opt / r_phys, "gap_phys_to_ml": 1 - r_ml / r_phys,
                     "frac_of_ml_gap_recovered": (r_phys - r_opt) / (r_phys - r_ml) if r_phys > r_ml else np.nan,
                     **{f"s_{c}": scales.get(c, 1.0) for c in args.classes}})
        for entry in log:
            curves.append({"p": p, "risk": entry["risk"], **{f"s_{k}": v for k, v in entry["scales"].items()}})
        print(f"p={p:<7g} R_phys={r_phys:.6e} R_opt={r_opt:.6e} R_ml={r_ml:.6e} "
              f"gain={100*(1-r_opt/r_phys):.3f}% mlgap={100*(1-r_ml/r_phys):.2f}% "
              f"bound={model.neglected:.1e} scales={ {k: round(v,3) for k,v in scales.items()} }", flush=True)
    df = pd.DataFrame(rows)
    tag = f"{'rep' if args.code == 'repetition' else 'surf'}_d{args.distance}_T{args.rounds}"
    df.to_csv(args.out / f"risk_surface_{tag}.csv", index=False)
    pd.DataFrame(curves).to_csv(args.out / f"risk_scan_{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.5g}"))


if __name__ == "__main__":
    main()
