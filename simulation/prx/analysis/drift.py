"""Analysis of the drift experiments.

within-circuit: for each schedule, compare priors against the oracle built from
the true drifting circuit, and report the window length that minimises risk.
stream: compare causal windowed estimators against the per-block oracle, with
moving-block bootstrap intervals, and report excess failures during recovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "rows.csv")
    df["cond"] = ("d" + df.distance.astype(str) + "|T" + df.rounds.astype(str)
                  + "|p" + df.p_true.map(lambda x: f"{x:g}") + "|" + df.drift_id.astype(str))
    return df


def within_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cond, g in df.groupby("cond"):
        orc = g[g.prior_id == "oracle_drifting"]
        if orc.empty:
            continue
        base = float(orc.ler.iloc[0])
        for est, h in g.groupby("estimator"):
            if est == "oracle":
                continue
            agg = h.groupby("n_cal_shots").agg(k=("n_errors", "sum"), n=("n_test_shots", "sum")).reset_index()
            for _, r in agg.iterrows():
                ler = r.k / r.n
                out.append({"cond": cond, "drift": g.drift_id.iloc[0], "distance": int(g.distance.iloc[0]),
                            "rounds": int(g.rounds.iloc[0]), "p": float(g.p_true.iloc[0]),
                            "estimator": est, "n_cal": int(r.n_cal_shots),
                            "ler": ler, "excess_vs_oracle": ler / base - 1, "ler_oracle": base})
    return pd.DataFrame(out)


def stream_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cond, g in df.groupby("cond"):
        orc = g[g.prior_id == "oracle_block"]
        if orc.empty:
            continue
        base = float(orc.ler.iloc[0])
        for _, r in g.iterrows():
            if r.prior_id == "oracle_block":
                continue
            out.append({"cond": cond, "drift": r.drift_id, "distance": int(r.distance),
                        "estimator": r.estimator, "prior_id": r.prior_id,
                        "window_shots": r.n_cal_shots, "ler": r.ler,
                        "excess_vs_oracle": r.ler / base - 1,
                        "delta": r.delta_ler_vs_ref, "lo": r.delta_ci_lo, "hi": r.delta_ci_hi,
                        "recovery_excess_failures": r.p_hat})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--mode", choices=["within", "stream"], default=None)
    a = ap.parse_args()
    df = load(a.run_dir)
    mode = a.mode or ("within" if "within" in str(a.run_dir) else "stream")
    pd.set_option("display.width", 250)
    if mode == "within":
        t = within_table(df)
        t.to_csv(a.run_dir / "within_summary.csv", index=False)
        print("=== excess over the drifting oracle (%), by schedule and estimator ===")
        piv = t.pivot_table(index=["distance", "drift"], columns=["estimator", "n_cal"],
                            values="excess_vs_oracle") * 100
        print(piv.to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n=== best window half-width per schedule (largest budget) ===")
        w = t[t.estimator.str.startswith("window_hw=")]
        if not w.empty:
            top = w.loc[w.groupby(["distance", "drift", "n_cal"]).excess_vs_oracle.idxmin()]
            print(top[["distance", "drift", "n_cal", "estimator", "excess_vs_oracle"]].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        t = stream_table(df)
        t.to_csv(a.run_dir / "stream_summary.csv", index=False)
        print("=== excess over the per-block oracle (%), chronological streams ===")
        print(t.pivot_table(index=["distance", "drift"], columns="prior_id",
                            values="excess_vs_oracle").to_string(float_format=lambda x: f"{100*x:.2f}"))
        print("\n=== recovery excess failures after a step ===")
        st = t[t.drift == "step"]
        if not st.empty:
            print(st.pivot_table(index="distance", columns="prior_id",
                                 values="recovery_excess_failures").to_string(float_format=lambda x: f"{x:.0f}"))


if __name__ == "__main__":
    main()
