"""Is the empirical-Bayes shrinkage weight small because its variance model is wrong?

The empirical-Bayes weight is tau^2 / (tau^2 + D_g) with D_g the sampling variance of
signature g's fitted log-odds. We model D_g = 1/n_g from its event count. If the pooled
p_ij estimate is noisier than that, tau^2 is inflated, kappa = 1/tau^2 comes out too
small, and the estimator under-shrinks for a reason that has nothing to do with the
decoder's loss.

This measures D_g directly: R independent calibration sets per condition and budget,
the same fit on each, then the variance of the fitted log-odds across replicates. It
reports the ratio of measured to modelled variance, and the kappa that empirical Bayes
would have chosen had it known the measured variance ("oracle-variance kappa").
"""
from __future__ import annotations

import argparse
import csv
from typing import Any, Dict, List

import numpy as np

from ..config import SeedPlan
from ..dem.edges import EdgeTable
from ..estimators.pij import pairwise_moments
from ..estimators.pooled import PooledPijEstimator, edge_signatures, eb_kappa_mle
from ..experiments.common import Condition, build_condition_circuit, run_pool, sample_shots
from ..experiments.frontier import nominal_template
from ..results_io import new_run_dir, write_manifest

FIELDS = ["cond_id", "distance", "rounds", "p", "sigma_h", "het_seed", "n_cal", "n_signatures",
          "var_ratio_median", "var_ratio_p90", "kappa_modelled", "kappa_oracle_variance"]


def _logit(p):
    return np.log(p / (1 - p))


def task(t: Dict[str, Any]) -> Dict[str, Any]:
    cond: Condition = t["cond"]
    plan = SeedPlan(t["base_seed"])
    circ = build_condition_circuit(cond)
    template = nominal_template(cond)
    bulk = np.flatnonzero(template.v >= 0)
    pairs = np.stack([template.u[bulk], template.v[bulk]], axis=1)
    n_cal, R = t["n_cal"], t["replicates"]
    dets, _ = sample_shots(circ, n_cal * R, plan.for_("eb_var", cond.cond_id, n_cal))
    sigs = edge_signatures(template)
    # one representative edge per signature, and the fit on each replicate
    reps: Dict[tuple, int] = {}
    for k, s in enumerate(sigs):
        reps.setdefault(s, k)
    idx = np.array(sorted(reps.values()))
    z = np.empty((R, idx.size))
    ne = np.empty((R, idx.size))
    for r in range(R):
        xi, xij, n = pairwise_moments(dets[r * n_cal:(r + 1) * n_cal], template.num_detectors, pairs)
        fit = PooledPijEstimator(kappa=0.0).fit_moments(template, xi, xij, n)
        z[r] = _logit(np.clip(fit.table.p[idx], 1e-12, 0.5 - 1e-12))
        ne[r] = np.array([max(v, 1e-9) for v in _n_eff(template, xi, xij, n)[idx]])
    measured = z.var(axis=0, ddof=1)
    modelled = 1.0 / ne.mean(axis=0)
    ok = np.isfinite(measured) & np.isfinite(modelled) & (measured > 0)
    ratio = measured[ok] / modelled[ok]
    # kappa from a single replicate, with the modelled and with the measured variance
    delta = z[0] - _logit(np.clip(template.p[idx], 1e-12, 0.5 - 1e-12))
    k_model = eb_kappa_mle(delta[ok], ne.mean(axis=0)[ok])
    k_oracle = eb_kappa_mle(delta[ok], 1.0 / measured[ok])
    return {"cond_id": cond.cond_id, "distance": cond.distance, "rounds": cond.rounds, "p": cond.p,
            "sigma_h": cond.extra["sigma_h"], "het_seed": cond.extra["het_seed"], "n_cal": n_cal,
            "n_signatures": int(ok.sum()), "var_ratio_median": float(np.median(ratio)),
            "var_ratio_p90": float(np.quantile(ratio, 0.9)), "kappa_modelled": float(k_model),
            "kappa_oracle_variance": float(k_oracle)}


def _n_eff(template: EdgeTable, xi, xij, n) -> np.ndarray:
    """The counts the estimator itself uses as the weight of each edge."""
    fit = PooledPijEstimator(kappa=1.0).fit_moments(template, xi, xij, n)
    # recover n_eff from the applied weight: w = n_eff / (n_eff + 1)
    w = (_logit(fit.table.p) - _logit(template.p))
    raw = PooledPijEstimator(kappa=0.0).fit_moments(template, xi, xij, n)
    d = _logit(raw.table.p) - _logit(template.p)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.abs(d) > 1e-12, w / d, 0.0)
        ne = np.where(frac < 1, frac / np.maximum(1 - frac, 1e-12), 0.0)
    return ne


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=20260920)
    a = ap.parse_args()
    conds = [Condition("si1000", d, T, 0.0015, extra={"variant": "google2025", "p_nominal": 0.001,
                                                      "sigma_h": s, "het_seed": seed})
             for d, T in ((3, 3), (5, 5)) for s, seed in ((0.6, 301), (1.0, 401))]
    tasks = [{"cond": c, "n_cal": n, "replicates": a.replicates, "base_seed": a.base_seed}
             for c in conds for n in (100, 1000, 10000)]
    run_dir = new_run_dir("eb_variance", label="check")
    write_manifest(run_dir, {"experiment": "eb_variance", "base_seed": a.base_seed,
                             "replicates": a.replicates, "conditions": [c.cond_id for c in conds]})
    with open(run_dir / "rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for i, row in enumerate(run_pool(tasks, task, workers=a.workers), 1):
            w.writerow(row); fh.flush()
            print(f"[{i}/{len(tasks)}] d={row['distance']} sigma={row['sigma_h']} N={row['n_cal']}: "
                  f"variance ratio {row['var_ratio_median']:.1f}x, kappa {row['kappa_modelled']:.2f} "
                  f"-> {row['kappa_oracle_variance']:.2f} with the measured variance", flush=True)
    print("run_dir:", run_dir)


if __name__ == "__main__":
    main()
