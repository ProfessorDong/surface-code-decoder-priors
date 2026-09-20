"""Calibration budgets, stated so that failures and non-events are visible.

A budget reported as "the first sampled N at which the mean excess over the
oracle falls below 10%" hides three things: conditions that never reach the
target, conditions where the uncalibrated prior already meets it, and curves that
cross the threshold and cross back. This module reports all three, together with
the two quantities that answer the practical question directly:

    beats(N)     whether the fitted prior has lower risk than leaving the prior alone
    recovery(N)  (R_static - R_est(N)) / (R_static - R_oracle), the fraction of the
                 available gap recovered, reported only where the denominator is
                 resolved by at least three paired standard errors.

Each condition is one independent noise draw at one geometry (d, T, p); the
calibration replicates within a condition are averaged, since they do not create
new devices. Budgets are given in shots and in syndrome rounds N T.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .frontier import load

TARGET = 0.10


def per_condition(df: pd.DataFrame, estimator: str) -> pd.DataFrame:
    out = []
    for cond, g in df.groupby("cond"):
        orc = g[g.estimator == "oracle_physical"]; st = g[g.estimator == "static"]
        est = g[(g.estimator == estimator) & (g.n_cal_shots > 0)]
        if orc.empty or st.empty or est.empty:
            continue
        n_test = float(orc.n_test_shots.iloc[0])
        r_o, r_s = float(orc.ler.iloc[0]), float(st.ler.iloc[0])
        # paired SE of R_static - R_oracle: both decode the same test shots, and the static
        # row stores its discordances against the oracle (b: static only, c: oracle only)
        b, c = float(st.mcnemar_b.iloc[0]), float(st.mcnemar_c.iloc[0])
        se = np.sqrt(max((b + c) / n_test - ((b - c) / n_test) ** 2, 0.0) / n_test)
        denom_ok = (r_s - r_o) > 3 * se
        curve = est.groupby("n_cal_shots").apply(lambda h: h.n_errors.sum() / h.n_test_shots.sum())
        curve = curve.sort_index()
        ex = curve / r_o - 1
        ok = ex <= TARGET
        first = int(ok[ok].index[0]) if ok.any() else None
        recross = bool(first is not None and (~ok[ok.index > first]).any())
        T = int(g.rounds.iloc[0])
        for n_cal, r_e in curve.items():
            out.append({"cond": cond, "distance": int(g.distance.iloc[0]), "rounds": T,
                        "p": float(g.p_true.iloc[0]), "sigma_h": float(g.sigma_h.iloc[0]),
                        "draw": int(g.het_seed.iloc[0]), "n_cal": int(n_cal), "NT": int(n_cal) * T,
                        "excess": float(r_e / r_o - 1), "static_excess": r_s / r_o - 1,
                        "beats_static": bool(r_e < r_s),
                        "recovery": float((r_s - r_e) / (r_s - r_o)) if denom_ok else np.nan,
                        "budget": first, "budget_NT": (first * T) if first else None,
                        "censored": first is None, "recross": recross,
                        "static_meets_target": bool(r_s / r_o - 1 <= TARGET)})
    return pd.DataFrame(out)


def summarize(pc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sg, d), g in pc.groupby(["sigma_h", "distance"]):
        c = g.drop_duplicates("cond")
        b = c.budget.astype(float).fillna(np.inf)
        rows.append({"sigma_h": sg, "distance": d, "conditions": len(c), "draws": c.draw.nunique(),
                     "censored": int(c.censored.sum()), "static_meets_target": int(c.static_meets_target.sum()),
                     "recrossing": int(c.recross.sum()),
                     "budget_median": float(np.quantile(b, 0.5, method="inverted_cdf")),
                     "budget_q25": float(np.quantile(b, 0.25, method="inverted_cdf")),
                     "budget_q75": float(np.quantile(b, 0.75, method="inverted_cdf")),
                     "budget_NT_median": float(np.quantile(c.budget_NT.astype(float).fillna(np.inf), 0.5, method="inverted_cdf"))})
    return pd.DataFrame(rows)


def by_budget(pc: pd.DataFrame) -> pd.DataFrame:
    return (pc.groupby(["sigma_h", "distance", "n_cal"])
              .agg(conditions=("cond", "nunique"), median_excess=("excess", "median"),
                   frac_beats_static=("beats_static", "mean"), median_recovery=("recovery", "median"),
                   q25_recovery=("recovery", lambda x: x.quantile(0.25)),
                   q75_recovery=("recovery", lambda x: x.quantile(0.75)),
                   resolved=("recovery", lambda x: int(x.notna().sum())))
              .reset_index())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--estimators", nargs="+", default=["pooled_shrink", "pooled", "pij",
                                                        "class_search", "uniform_search", "density", "dgr"])
    a = ap.parse_args()
    df = load(a.run_dir)
    allpc, allsum, allbud = [], [], []
    for est in a.estimators:
        pc = per_condition(df, est)
        if pc.empty:
            continue
        allpc.append(pc.assign(estimator=est))
        allsum.append(summarize(pc).assign(estimator=est))
        allbud.append(by_budget(pc).assign(estimator=est))
    pc, sm, bd = pd.concat(allpc), pd.concat(allsum), pd.concat(allbud)
    pc.to_csv(a.run_dir / "budget_per_condition.csv", index=False)
    sm.to_csv(a.run_dir / "budget_summary.csv", index=False)
    bd.to_csv(a.run_dir / "budget_by_n.csv", index=False)
    pd.set_option("display.width", 220)
    f = lambda v: f"{v:.3g}"
    print(sm.to_string(index=False, float_format=f))
    print()
    print(bd[bd.estimator == "pooled_shrink"].to_string(index=False, float_format=f))


if __name__ == "__main__":
    main()
