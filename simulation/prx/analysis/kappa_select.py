"""Select kappa on the development devices, exactly as the protocol fixes it
(research/prx_quantum/preregistration/kappa_tuning_protocol.md).

Primary criterion: for each kappa, the mean over the 120 development conditions and
the eight budgets of R_fit / R_oracle - 1, with R_fit pooled over calibration
replicates. kappa* = argmin; candidates within 1e-4 of the minimum resolve to the
smallest kappa. Secondary summaries are reported but cannot change kappa*.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    a = ap.parse_args()
    d = pd.read_csv(a.run_dir / "rows.csv")
    man = json.loads((a.run_dir / "manifest.json").read_text())
    assert d.cond_id.nunique() == len(man["conditions"]) == 120, "incomplete development run"
    orc = d[d.estimator == "oracle"].set_index("cond_id")
    st = d[d.estimator == "static"].set_index("cond_id")
    f = d[d.estimator == "pooled_shrink"]
    g = f.groupby(["cond_id", "kappa", "n_cal"]).agg(k=("n_errors", "sum"), n=("n_test", "sum")).reset_index()
    g["ler"] = g.k / g.n
    g["excess"] = g.ler / (orc.loc[g.cond_id, "n_errors"].values / orc.loc[g.cond_id, "n_test"].values) - 1
    g["harm"] = g.ler > st.loc[g.cond_id, "n_errors"].values / st.loc[g.cond_id, "n_test"].values
    assert g.groupby("kappa").size().nunique() == 1, "unequal cells per kappa"
    crit = g.groupby("kappa").excess.mean().sort_index()
    best = crit.min()
    kstar = float(min(k for k, v in crit.items() if v <= best + 1e-4))
    summ = pd.DataFrame({"criterion_mean_rel_excess": crit,
                         "harm_frac_N<=1e3": g[g.n_cal <= 1000].groupby("kappa").harm.mean(),
                         "median_excess_N=100": g[g.n_cal == 100].groupby("kappa").excess.median(),
                         "median_excess_N=1e3": g[g.n_cal == 1000].groupby("kappa").excess.median()})
    summ.to_csv(a.run_dir / "kappa_criterion.csv")
    (a.run_dir / "kappa_selected.json").write_text(json.dumps({"kappa_star": kstar, "criterion": float(crit[kstar]),
                                                                "criterion_kappa5": float(crit[5.0])}, indent=1))
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string())
    print(f"\nkappa* = {kstar:g}  (criterion {crit[kstar]:.5f}; kappa = 5 gives {crit[5.0]:.5f})")


if __name__ == "__main__":
    main()
