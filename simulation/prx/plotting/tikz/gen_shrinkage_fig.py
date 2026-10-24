"""Data for the shrinkage figure (fig_shrinkage).

Panel (a): the shrinkage strength the empirical-Bayes rule chooses on each
calibration budget, against the strength tuned once on the development devices.
Panel (b): the tuning criterion against a fixed shrinkage strength, with the
empirical-Bayes and no-shrinkage arms as reference levels.

Run from the repository root:
    python -m simulation.prx.plotting.tikz.gen_shrinkage_fig
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DEV = ROOT / "research" / "prx_quantum" / "results" / "kappa_dev" / "20260920T022238Z_d8a61a_dev"
OUT = ROOT / "manuscript" / "figdata"

KMIN, KMAX = 0.35, 1400.0          # x range of panel (b), used for the level lines


def write(name, rows):
    path = OUT / f"fig_shrinkage_{name}.dat"
    with path.open("w") as fh:
        fh.write("x y\n")
        for x, y in rows:
            fh.write(f"{x:g} {y:.6f}\n")
    print(f"wrote {path.name} ({len(rows)} points)")


def main():
    OUT.mkdir(exist_ok=True)

    # --- panel (a): kappa chosen by empirical Bayes, per budget ---
    k = pd.read_csv(DEV / "kappa_eb_values.csv")
    k = k[k.estimator == "pooled_eb_global"]
    med = k[np.isfinite(k.kappa_hat)].groupby("n_cal").kappa_hat.median()
    write("kappahat", list(med.items()))
    nlo, nhi = float(med.index.min()), float(med.index.max())
    write("tuned_level", [(nlo, 20.0), (nhi, 20.0)])

    # --- panel (b): the tuning criterion against fixed kappa ---
    c = pd.read_csv(DEV / "kappa_eb_criterion.csv").set_index("arm")
    fixed = {}
    for arm in c.index:
        if arm.startswith("kappa="):
            kap = float(arm.split("=")[1])
            if kap > 0:
                fixed[kap] = float(c.loc[arm, "criterion_mean_rel_excess"])
    write("criterion", sorted(fixed.items()))
    best_k = min(fixed, key=fixed.get)
    write("criterion_best", [(best_k, fixed[best_k])])
    for name, arm in (("eb_level", "pooled_eb_global"), ("none_level", "kappa=0.0")):
        v = float(c.loc[arm, "criterion_mean_rel_excess"])
        write(name, [(KMIN, v), (KMAX, v)])


if __name__ == "__main__":
    main()
