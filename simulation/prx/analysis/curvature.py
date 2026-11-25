"""Curvature of the risk surface from prior_sweep search rows.

For each condition and edge class, fit ln(n_errors) as a quadratic in ln(scale)
along the coordinate-search path (holding the other classes at their final
values), giving the relative curvature H_kk = d^2 ln R / d(ln s_k)^2 and the
per-class optimum s_k*. These feed the bias-variance law of the frontier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fit_curvatures(search_rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    df = search_rows.copy()
    df["cond"] = df["noise_family"] + "|d" + df["distance"].astype(str) + "|T" + df["rounds"].astype(str) + "|p" + df["p_true"].map(lambda x: f"{x:g}")
    df["cls"] = df["prior_id"].str.extract(r"search:(\w+)=")[0]
    df["s"] = df["prior_id"].str.extract(r"=([0-9.eE+-]+)$")[0].astype(float)
    for cond, g in df.groupby("cond"):
        last_sweep = g["estimator_params_json"].str.extract(r'"sweep": (\d+)')[0].astype(int)
        g = g.assign(sweep=last_sweep)
        n_search = int(g["n_test_shots"].iloc[0])
        for cls, h in g[g.cls.notna()].groupby("cls"):
            hh = h[h.sweep == h.sweep.max()]
            if len(hh) < 4 or (hh.n_errors <= 0).any():
                continue
            x = np.log(hh.s.values)
            y = np.log(hh.n_errors.values)
            X = np.column_stack([np.ones_like(x), x, x ** 2])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            curv = 2 * beta[2]
            s_opt = float(np.exp(-beta[1] / (2 * beta[2]))) if beta[2] > 0 else np.nan
            resid = y - X @ beta
            out.append({"cond": cond, "class": cls, "n_search": n_search, "min_errors": int(hh.n_errors.min()),
                        "curvature": float(curv), "s_opt": s_opt, "rms_resid": float(np.sqrt(np.mean(resid ** 2))),
                        "poisson_sd": float(1 / np.sqrt(hh.n_errors.mean()))})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    a = ap.parse_args()
    sr = pd.read_csv(a.run_dir / "search_rows.csv")
    res = fit_curvatures(sr)
    res.to_csv(a.run_dir / "curvatures.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
