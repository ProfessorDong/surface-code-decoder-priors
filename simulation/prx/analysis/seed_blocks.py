"""Uncertainty over noise realizations for the heterogeneity frontier.

The draws profile reuses six seeded multiplier maps per heterogeneity level across
all operating points of a distance, so the conditions of a (sigma, d) stratum are
four (two at d = 7) operating points on the same six devices. Uncertainty across
devices is therefore assessed by resampling whole seeds, each seed carrying all of
its operating points; each bootstrap statistic is recomputed exactly as for the
point estimate.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .willow import stable_seed

BOOT = 4000


def first_from(h, col, thr):
    h = h.sort_values("n_cal"); good = (h[col] >= thr).values
    for i in range(len(good)):
        if good[i:].all():
            return float(h.n_cal.values[i])
    return np.inf


def stats(g: pd.DataFrame) -> dict:
    med = lambda v: float(np.quantile(np.asarray(v, float), 0.5, method="inverted_cdf"))
    c = list(g.groupby("cond"))
    b100 = g[g.n_cal == 100].beats_static.mean()
    return {"beats_at_100": 100 * b100,
            "median_beats_budget": med([first_from(h.assign(b=h.beats_static.astype(float)), "b", 1.0) for _, h in c]),
            "median_rec50": med([first_from(h, "recovery", 0.5) for _, h in c]),
            "median_rec80": med([first_from(h, "recovery", 0.8) for _, h in c]),
            "median_excess_top": 100 * float(np.median(g[g.n_cal == g.n_cal.max()].excess))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--estimator", default="pooled_shrink")
    a = ap.parse_args()
    pc = pd.read_csv(a.run_dir / "budget_per_condition.csv")
    pc = pc[pc.estimator == a.estimator]
    out = []
    for (sg, d), g in pc.groupby(["sigma_h", "distance"]):
        seeds = sorted(g.draw.unique())
        pt = stats(g)
        rng = np.random.default_rng(stable_seed("seed_blocks", a.estimator, sg, int(d)))
        draws = []
        for _ in range(BOOT):
            pick = rng.choice(seeds, size=len(seeds), replace=True)
            gb = pd.concat([g[g.draw == s].assign(cond=g[g.draw == s].cond + f"#{i}") for i, s in enumerate(pick)])
            draws.append(stats(gb))
        D = pd.DataFrame(draws)
        for k, v in pt.items():
            lo, hi = np.quantile(D[k].replace(np.inf, np.nan).dropna(), [0.025, 0.975]) if D[k].notna().any() else (np.nan, np.nan)
            out.append({"sigma_h": sg, "distance": int(d), "seeds": len(seeds), "statistic": k, "point": v,
                        "lo95": float(np.quantile(D[k], 0.025)), "hi95": float(np.quantile(D[k], 0.975))})
    df = pd.DataFrame(out)
    df.to_csv(a.run_dir / f"seed_blocks_{a.estimator}.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.pivot_table(index=["sigma_h", "distance"], columns="statistic", values=["point", "lo95", "hi95"]).round(2).to_string())


if __name__ == "__main__":
    main()
