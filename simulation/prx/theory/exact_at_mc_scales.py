"""Exact risk evaluated at the per-class scales that coordinate search found on
sampled data.

This is the exact-versus-Monte-Carlo validation quoted in Sec. III. The point is
to hold the *prior* fixed and change only the evaluation method: the scales are
read from the sampled gate-0 run rather than typed in, and the risk is then
recomputed by complete enumeration of the detector error model. Any number the
manuscript quotes for this comparison is produced here and read back by
manuscript/verify_numbers.py.
"""
from __future__ import annotations

import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESULTS_ROOT
from ..dem.edges import EdgeTable
from ..dem.priors import build_circuit
from .exact import enumerate_dem, truncate

CLASSES = ("space", "time", "diag", "boundary")


def latest_gate0(root: Path) -> Path:
    cands = sorted((root / "prior_sweep").glob("*_gate0"))
    if not cands:
        raise FileNotFoundError("no prior_sweep/*_gate0 run under %s" % root)
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.001)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    gate0 = latest_gate0(a.results_root)
    df = pd.read_csv(gate0 / "rows.csv")
    sel = df[(df.noise_family == "si1000:google2025") & (df.distance == a.distance)
             & (df.rounds == a.rounds) & (np.isclose(df.p_true, a.p))]
    cs = sel[sel.prior_id == "class_search"]
    ph = sel[sel.prior_id == "oracle_physical"]
    if cs.empty or ph.empty:
        raise SystemExit("gate0 run %s lacks the d%d T%d p=%g cell" % (gate0.name, a.distance, a.rounds, a.p))
    scales = {c: float(cs["scale_%s" % c].iloc[0]) for c in CLASSES}
    mc_phys, mc_cls = float(ph.ler.iloc[0]), float(cs.ler.iloc[0])
    mc_shots = int(ph.n_test_shots.iloc[0])
    mc_gain = 1.0 - mc_cls / mc_phys
    print("gate0 run      :", gate0.name)
    print("MC scales      :", {k: round(v, 6) for k, v in scales.items()})
    print("MC gain        : %.4f%%  (%d paired shots)" % (100 * mc_gain, mc_shots))

    t0 = time.time()
    dem = build_circuit("si1000", a.distance, a.rounds, a.p,
                        variant="google2025").detector_error_model(decompose_errors=True)
    model = truncate(enumerate_dem(dem), a.tol)
    phys = EdgeTable.from_dem(dem)
    r_phys = model.risk(phys)
    r_cls = model.risk(phys.scaled_by_class(scales))
    exact_gain = 1.0 - r_cls / r_phys
    build = time.time() - t0
    print("exact R_phys   : %.6e" % r_phys)
    print("exact R_class  : %.6e" % r_cls)
    print("exact gain     : %.4f%%   (neglected mass %.1e, %d syndromes, %.0f s)"
          % (100 * exact_gain, model.neglected, len(model.bits), build))
    print("MC - exact     : %.4f percentage points" % (100 * abs(exact_gain - mc_gain)))

    out = a.out or (a.results_root / "theory" / "exact_at_mc_scales.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "distance": a.distance, "rounds": a.rounds, "p": a.p, "gate0_run": gate0.name,
        **{"s_%s" % c: scales[c] for c in CLASSES},
        "mc_ler_phys": mc_phys, "mc_ler_class": mc_cls, "mc_shots": mc_shots,
        "mc_gain": mc_gain, "exact_R_phys": r_phys, "exact_R_class": r_cls,
        "exact_gain": exact_gain, "abs_gain_difference": abs(exact_gain - mc_gain),
        "neglected_mass": model.neglected, "syndromes_kept": len(model.bits),
        "tol": a.tol, "build_seconds": build,
    }]).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
