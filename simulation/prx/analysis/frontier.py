"""Frontier analysis: fit the excess-risk law and locate saturation / crossover.

Model (Proposition 4):  Delta_N = A / N + B,  Delta_N = LER(estimator at N) - LER(best).
A is the variance term (~ tr(H Sigma_1)/2), B the decoder-mismatch bias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..stats import clopper_pearson


def load(run_dir: Path) -> pd.DataFrame:
    """Rows of a frontier run, plus the estimators of any companion run listed in
    ``companions.json``. A companion is a run on the same conditions with the same base
    seed (so identical calibration and test shots); its oracle and static rows must
    equal this run's exactly, which is asserted before its estimator rows are appended."""
    df = pd.read_csv(run_dir / "rows.csv")
    comp = run_dir / "companions.json"
    if comp.exists():
        for rel, ests in json.loads(comp.read_text()).items():
            other = pd.read_csv((run_dir / rel).resolve() / "rows.csv")
            key = ["distance", "rounds", "p_true", "noise_extra_json", "estimator"]
            ref_a = df[df.estimator.isin(["oracle_physical", "static"])].set_index(key).n_errors.sort_index()
            ref_b = other[other.estimator.isin(["oracle_physical", "static"])].set_index(key).n_errors.sort_index()
            assert ref_a.equals(ref_b), f"companion {rel}: reference rows differ; not the same shots"
            df = pd.concat([df, other[other.estimator.isin(ests)]], ignore_index=True)
    ex = df.noise_extra_json.fillna("{}").map(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    df["sigma_h"] = [e.get("sigma_h", 0.0) for e in ex]
    df["het_seed"] = [e.get("het_seed", 0) for e in ex]
    df["p_nominal"] = [e.get("p_nominal", np.nan) for e in ex]
    df["cond"] = ("d" + df.distance.astype(str) + "|T" + df.rounds.astype(str) + "|p" + df.p_true.map(lambda x: f"{x:g}")
                  + "|sh" + df.sigma_h.map(lambda x: f"{x:g}") + "|s" + df.het_seed.astype(str))
    return df


def fit_law(n: np.ndarray, delta: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """Weighted least squares of delta = A/N + B."""
    ok = np.isfinite(delta) & (n > 0)
    if ok.sum() < 3:
        return {"A": np.nan, "B": np.nan, "N_sat": np.nan, "r2": np.nan}
    x = 1.0 / n[ok]
    y = delta[ok]
    ww = w[ok]
    X = np.column_stack([x, np.ones_like(x)])
    W = np.diag(ww)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    A, B = float(beta[0]), float(beta[1])
    pred = X @ beta
    ss_res = float(((y - pred) ** 2 * ww).sum())
    ss_tot = float(((y - np.average(y, weights=ww)) ** 2 * ww).sum())
    n_sat = A / (0.1 * abs(B)) if B != 0 and np.isfinite(B) else np.nan
    return {"A": A, "B": B, "N_sat": n_sat, "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cond, g in df.groupby("cond"):
        oracle = g[g.estimator == "oracle_physical"]
        static = g[g.estimator == "static"]
        if oracle.empty:
            continue
        ler_oracle = float(oracle.ler.iloc[0])
        ler_static = float(static[static.prior_id == "static_nominal"].ler.iloc[0]) if not static.empty else np.nan
        for est, h in g.groupby("estimator"):
            if est in ("oracle_physical", "static"):
                continue
            agg = h.groupby("n_cal_shots").agg(ler=("ler", "mean"), k=("n_errors", "sum"), n=("n_test_shots", "sum"),
                                               reps=("ler", "size")).reset_index()
            if agg.empty:
                continue
            delta = agg.ler.values - ler_oracle
            law = fit_law(agg.n_cal_shots.values.astype(float), delta, agg.n.values.astype(float))
            best_row = agg.loc[agg.ler.idxmin()]
            out.append({"cond": cond, "distance": int(g.distance.iloc[0]), "rounds": int(g.rounds.iloc[0]),
                        "p": float(g.p_true.iloc[0]), "sigma_h": float(g.sigma_h.iloc[0]),
                        "estimator": est, "needs_labels": bool(h.needs_labels.iloc[0]),
                        "ler_oracle": ler_oracle, "ler_static": ler_static,
                        "ler_best": float(best_row.ler), "N_best": int(best_row.n_cal_shots),
                        "excess_best_rel": float(best_row.ler / ler_oracle - 1),
                        "A": law["A"], "B": law["B"], "N_sat": law["N_sat"], "r2": law["r2"]})
    return pd.DataFrame(out)


def frontier_curves(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, est), g in df.groupby(["cond", "estimator"]):
        if est in ("oracle_physical", "static"):
            continue
        oracle = df[(df.cond == cond) & (df.estimator == "oracle_physical")]
        if oracle.empty:
            continue
        lo = float(oracle.ler.iloc[0])
        for n_cal, h in g.groupby("n_cal_shots"):
            k, n = h.n_errors.sum(), h.n_test_shots.sum()
            rows.append({"cond": cond, "distance": int(h.distance.iloc[0]), "rounds": int(h.rounds.iloc[0]),
                         "p": float(h.p_true.iloc[0]), "sigma_h": float(h.sigma_h.iloc[0]), "estimator": est,
                         "n_cal": int(n_cal), "reps": len(h), "ler": k / n, "rel_excess": (k / n) / lo - 1,
                         "ler_oracle": lo})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    a = ap.parse_args()
    df = load(a.run_dir)
    s = summarize(df)
    c = frontier_curves(df)
    s.to_csv(a.run_dir / "frontier_fits.csv", index=False)
    c.to_csv(a.run_dir / "frontier_curves.csv", index=False)
    pd.set_option("display.width", 250)
    print("=== fits of Delta_N = A/N + B ===")
    print(s.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
