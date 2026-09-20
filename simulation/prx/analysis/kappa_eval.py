"""Evaluate the frozen kappa* on the published (test) devices against kappa = 5
(protocol: research/prx_quantum/preregistration/kappa_tuning_protocol.md).

Both runs use the published seeds, so calibration and test shots are identical; the
oracle and uncalibrated rows are asserted equal before any comparison. Reported: the
primary criterion (mean relative excess over conditions and budgets) for each kappa and
its paired difference with a 95% interval from resampling whole disorder seeds within
each sigma; the probability of harm per budget; and the Table V budgets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .willow import stable_seed

BOOT = 10000


def per_cell(run_dir: Path) -> pd.DataFrame:
    pc = pd.read_csv(run_dir / "budget_per_condition.csv")
    pc = pc[pc.estimator == "pooled_shrink"]
    return pc[["distance", "rounds", "p", "sigma_h", "draw", "n_cal", "excess", "beats_static", "static_excess"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("published", type=Path)
    ap.add_argument("tuned", type=Path)
    a = ap.parse_args()
    key = ["distance", "rounds", "p_true", "noise_extra_json", "estimator"]
    ra = pd.read_csv(a.published / "rows.csv", usecols=key + ["n_errors"])
    rb = pd.read_csv(a.tuned / "rows.csv", usecols=key + ["n_errors"])
    refa = ra[ra.estimator.isin(["oracle_physical", "static"])].set_index(key).n_errors.sort_index()
    refb = rb[rb.estimator.isin(["oracle_physical", "static"])].set_index(key).n_errors.sort_index()
    assert refa.equals(refb), "reference rows differ: the runs did not use the same shots"
    k = ["distance", "rounds", "p", "sigma_h", "draw", "n_cal"]
    m = per_cell(a.published).merge(per_cell(a.tuned), on=k, suffixes=("_k5", "_kt"))
    assert len(m) == 120 * 8, len(m)
    crit = lambda d, s: float(d["excess" + s].mean())
    c5, ct = crit(m, "_k5"), crit(m, "_kt")
    rng = np.random.default_rng(stable_seed("kappa_eval"))
    diffs = np.empty(BOOT)
    for b in range(BOOT):
        parts = []
        for sg, g in m.groupby("sigma_h"):
            seeds = np.array(sorted(g.draw.unique()))
            for s in rng.choice(seeds, len(seeds), replace=True):
                parts.append(g[g.draw == s])
        gb = pd.concat(parts)
        diffs[b] = crit(gb, "_kt") - crit(gb, "_k5")
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    harm = m.groupby("n_cal").agg(harm_k5=("beats_static_k5", lambda x: 1 - x.mean()),
                                  harm_kt=("beats_static_kt", lambda x: 1 - x.mean()),
                                  median_excess_k5=("excess_k5", "median"), median_excess_kt=("excess_kt", "median"))
    harm.to_csv(a.tuned / "kappa_eval_by_budget.csv")
    res = {"criterion_kappa5": c5, "criterion_kappa_star": ct, "difference": ct - c5, "diff_lo95": float(lo),
           "diff_hi95": float(hi), "conditions": int(m[k[:5]].drop_duplicates().shape[0]),
           "cells_better": int((m.excess_kt < m.excess_k5).sum()), "cells_worse": int((m.excess_kt > m.excess_k5).sum())}
    (a.tuned / "kappa_eval.json").write_text(json.dumps(res, indent=1))
    pd.set_option("display.width", 200)
    print(json.dumps(res, indent=1))
    print((harm * 100).round(1).to_string())


if __name__ == "__main__":
    main()
