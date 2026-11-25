"""Post hoc: how an empirical-Bayes shrinkage weight compares with the tuned constant.

The weight n/(n+kappa) is the posterior-mean weight for a Gaussian prior of variance
1/kappa on the edge log-odds. The empirical-Bayes choice estimates that variance from
the same syndromes (moment estimate, truncated at zero), giving kappa_hat = 1/tau_hat^2
per fit. This script evaluates the EB variants on the DEVELOPMENT conditions only, with
the criterion of the archived protocol, alongside the kappa grid. It cannot change the
frozen kappa*: it is reported as a post hoc comparison.
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
    orc = d[d.estimator == "oracle"].set_index("cond_id")
    st = d[d.estimator == "static"].set_index("cond_id")
    f = d[d.estimator.isin(["pooled_shrink"]) | d.estimator.str.startswith("pooled_eb")].copy()
    f["arm"] = np.where(f.estimator == "pooled_shrink", "kappa=" + f.kappa.astype(str), f.estimator)
    g = f.groupby(["cond_id", "arm", "n_cal"]).agg(k=("n_errors", "sum"), n=("n_test", "sum")).reset_index()
    g["ler"] = g.k / g.n
    g["excess"] = g.ler / (orc.loc[g.cond_id, "n_errors"].values / orc.loc[g.cond_id, "n_test"].values) - 1
    g["harm"] = g.ler > st.loc[g.cond_id, "n_errors"].values / st.loc[g.cond_id, "n_test"].values
    summ = pd.DataFrame({"criterion_mean_rel_excess": g.groupby("arm").excess.mean(),
                         "harm_frac_N<=1e3": g[g.n_cal <= 1000].groupby("arm").harm.mean(),
                         "median_excess_N=100": g[g.n_cal == 100].groupby("arm").excess.median(),
                         "median_excess_N=1e3": g[g.n_cal == 1000].groupby("arm").excess.median(),
                         "median_excess_N=3e5": g[g.n_cal == 300000].groupby("arm").excess.median()})
    summ = summ.sort_values("criterion_mean_rel_excess")
    summ.to_csv(a.run_dir / "kappa_eb_criterion.csv")
    # what the EB rule actually chose
    eb = f[f.estimator.str.startswith("pooled_eb") & f.kappa_eb_json.notna()].copy()
    ks = []
    for est, n_cal, js in zip(eb.estimator, eb.n_cal, eb.kappa_eb_json):
        for key, v in json.loads(js).items():
            ks.append({"estimator": est, "n_cal": n_cal, "component": key, "kappa_hat": float(v)})
    kh = pd.DataFrame(ks)
    kh.to_csv(a.run_dir / "kappa_eb_values.csv", index=False)
    med = kh[np.isfinite(kh.kappa_hat)].groupby(["estimator", "n_cal"]).kappa_hat.median().unstack()
    pd.set_option("display.width", 220)
    print(summ.round(4).to_string())
    print("\nmedian kappa_hat chosen by the EB rule (rows: variant, columns: calibration shots)")
    print(med.round(2).to_string())
    print(f"\ninfinite kappa_hat (shrink fully to the reference): {100*(~np.isfinite(kh.kappa_hat)).mean():.1f}% of fits")


if __name__ == "__main__":
    main()
