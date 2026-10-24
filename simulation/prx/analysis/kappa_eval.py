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


def individual_harm(run_dir: Path) -> pd.DataFrame:
    """Fraction of individual fits whose held-out risk exceeds the uncalibrated prior's.

    This is E_C 1{R(theta_C) > R_0}, not 1{E_C R(theta_C) > R_0}: the two differ, and the
    second is what ``beats_static`` in budget_per_condition.csv reports.
    """
    key = ["distance", "rounds", "p_true", "noise_extra_json"]
    d = pd.read_csv(run_dir / "rows.csv",
                    usecols=key + ["estimator", "n_cal_shots", "replicate", "n_errors", "n_test_shots"])
    ref = (d[d.estimator == "static"].groupby(key)
           .apply(lambda g: g.n_errors.sum() / g.n_test_shots.sum(), include_groups=False))
    f = d[d.estimator == "pooled_shrink"].copy()
    f["ref"] = list(ref.loc[list(map(tuple, f[key].values))])
    f["worse"] = f.n_errors / f.n_test_shots > f.ref
    return (f.groupby("n_cal_shots")
            .agg(harm_ind=("worse", "mean"), n_fits=("worse", "size"))
            .reset_index().rename(columns={"n_cal_shots": "n_cal"}))


def individual_harm_seed_ci(published: Path, tuned: Path, boot: int = 20000) -> pd.DataFrame:
    """Seed-resampled interval for the change in individual-fit harm, per budget.

    The two runs decode identical shots, so the comparison is paired fit by fit. The
    resampling unit is the disorder seed, within each heterogeneity stratum, because
    seeds are what the conditions share; the observed held-out signs are held fixed, so
    this propagates variation across devices and not test-shot noise, and it is not an
    interval for the probability that the true risk increases.
    """
    key = ["distance", "rounds", "p_true", "noise_extra_json"]

    def fits(run: Path) -> pd.DataFrame:
        d = pd.read_csv(run / "rows.csv",
                        usecols=key + ["estimator", "replicate", "n_cal_shots", "n_errors"])
        ref = (d[d.estimator == "static"][key + ["n_errors"]]
               .rename(columns={"n_errors": "ref_errors"}))
        v = d[d.estimator == "pooled_shrink"].merge(ref, on=key, validate="many_to_one")
        v["harm"] = v.n_errors > v.ref_errors
        return v[key + ["replicate", "n_cal_shots", "harm"]]

    m = fits(published).merge(fits(tuned), on=key + ["replicate", "n_cal_shots"],
                              suffixes=("_k5", "_kt"), validate="one_to_one")
    extra = m.noise_extra_json.map(json.loads)
    m["sigma_h"] = extra.map(lambda x: x["sigma_h"])
    m["draw"] = extra.map(lambda x: x["het_seed"])
    rng = np.random.default_rng(stable_seed("kappa_eval_harm_seeds"))
    rows = []
    for n, d in m.groupby("n_cal_shots"):
        per_seed = d.groupby(["sigma_h", "draw"])[["harm_k5", "harm_kt"]].mean()
        acc = np.zeros((boot, 2))
        strata = list(per_seed.groupby("sigma_h"))
        for _, g in strata:
            v = g.values
            acc += v[rng.integers(0, len(v), size=(boot, len(v)))].mean(axis=1) / len(strata)
        dif = acc[:, 1] - acc[:, 0]
        rows.append({"n_cal": int(n),
                     "harm_ind_delta": float(d.harm_kt.mean() - d.harm_k5.mean()),
                     "harm_ind_delta_lo95": float(np.quantile(dif, 0.025)),
                     "harm_ind_delta_hi95": float(np.quantile(dif, 0.975)),
                     "harm_seeds": int(per_seed.shape[0])})
    return pd.DataFrame(rows).set_index("n_cal")


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
    # The columns above average the calibration replicates within a condition and then ask
    # whether that mean is worse than the uncalibrated prior. A reader deploys one fit, so we
    # also count how often a single fit is worse, from the primitive rows of both runs.
    ind = individual_harm(a.published).merge(individual_harm(a.tuned), on="n_cal",
                                             suffixes=("_k5", "_kt")).set_index("n_cal")
    harm["harm_ind_k5"] = ind.harm_ind_k5
    harm["harm_ind_kt"] = ind.harm_ind_kt
    harm["n_fits"] = ind.n_fits_k5
    assert (ind.n_fits_k5 == ind.n_fits_kt).all(), "the two runs hold different numbers of fits"
    ci = individual_harm_seed_ci(a.published, a.tuned)
    for col in ci.columns:
        harm[col] = ci[col]
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
