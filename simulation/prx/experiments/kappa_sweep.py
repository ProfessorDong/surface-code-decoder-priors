"""Development sweep of the shrinkage strength kappa (protocol:
research/prx_quantum/preregistration/kappa_tuning_protocol.md).

Development devices only: disorder seeds 301-306 (sigma = 0.6) and 401-406
(sigma = 1.0), disjoint from every seed used in the published results. For each
condition, budget and calibration replicate the pooled moments are computed once and
every kappa on the grid is applied to the same moments, so differences between kappa
values are paired; all decode the same development test shots.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..config import SeedPlan
from ..dem.edges import EdgeTable
from ..estimators.pij import pairwise_moments
from ..estimators.pooled import PooledPijEstimator
from ..results_io import new_run_dir, write_manifest
from .common import Condition, build_condition_circuit, decode_errors, run_pool, sample_shots
from .frontier import N_GRID, nominal_template

KAPPAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
EB_VARIANTS = ["global", "per_class"]
FIELDS = ["cond_id", "distance", "rounds", "p", "sigma_h", "het_seed", "estimator", "kappa", "n_cal",
          "replicate", "n_test", "n_errors", "kappa_eb_json"]


def dev_conditions() -> List[Condition]:
    conds = []
    for d in (3, 5, 7):
        for T in (d, 3 * d):
            if d == 7 and T > d:
                continue
            for p in (0.0015, 0.003):
                for sigma, seeds in ((0.6, range(301, 307)), (1.0, range(401, 407))):
                    for s in seeds:
                        conds.append(Condition("si1000", d, T, p, extra={
                            "variant": "google2025", "p_nominal": p / 1.5, "sigma_h": sigma, "het_seed": s}))
    return conds


def task(t: Dict[str, Any]) -> List[Dict[str, Any]]:
    cond: Condition = t["cond"]
    plan = SeedPlan(t["base_seed"])
    circ = build_condition_circuit(cond)
    tdets, tobs = sample_shots(circ, t["n_test"], plan.for_("dev_test", cond.cond_id))
    pdets, _ = sample_shots(circ, t["n_pool"], plan.for_("dev_pool", cond.cond_id))
    template = nominal_template(cond)
    phys = EdgeTable.from_circuit(circ)
    base = {"cond_id": cond.cond_id, "distance": cond.distance, "rounds": cond.rounds, "p": cond.p,
            "sigma_h": cond.extra["sigma_h"], "het_seed": cond.extra["het_seed"], "n_test": t["n_test"]}
    rows = [dict(base, estimator="oracle", kappa="", n_cal=0, replicate=0,
                 n_errors=int(decode_errors(phys.to_matching(), tdets, tobs).sum())),
            dict(base, estimator="static", kappa="", n_cal=0, replicate=0,
                 n_errors=int(decode_errors(template.to_matching(), tdets, tobs).sum()))]
    bulk = np.flatnonzero(template.v >= 0)
    pairs = np.stack([template.u[bulk], template.v[bulk]], axis=1)
    for n_cal in [n for n in N_GRID if n <= t["n_pool"]]:
        reps = min(t["replicates"], t["n_pool"] // n_cal)
        for rep in range(reps):
            chunk = pdets[rep * n_cal:(rep + 1) * n_cal]
            xi, xij, n = pairwise_moments(chunk, template.num_detectors, pairs)
            for kappa in KAPPAS:
                fit = PooledPijEstimator(kappa=kappa).fit_moments(template, xi, xij, n)
                e = decode_errors(fit.table.to_matching(), tdets, tobs)
                rows.append(dict(base, estimator="pooled_shrink", kappa=kappa, n_cal=n_cal, replicate=rep,
                                 n_errors=int(e.sum())))
            for eb in EB_VARIANTS:
                fit = PooledPijEstimator(eb=eb).fit_moments(template, xi, xij, n)
                e = decode_errors(fit.table.to_matching(), tdets, tobs)
                rows.append(dict(base, estimator="pooled_eb_" + eb, kappa="", n_cal=n_cal, replicate=rep,
                                 n_errors=int(e.sum()), kappa_eb_json=fit.params["kappa_eb_json"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--base-seed", type=int, default=20260919)
    ap.add_argument("--n-test", type=int, default=200_000)
    ap.add_argument("--n-pool", type=int, default=300_000)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    conds = dev_conditions()
    if a.limit:
        conds = conds[:a.limit]
    run_dir = new_run_dir("kappa_dev", label="dev" if not a.limit else "smoke")
    write_manifest(run_dir, {"experiment": "kappa_dev", "protocol": "preregistration/kappa_tuning_protocol.md",
                             "protocol_sha256": "721e00d921bdc76d3d6b06b5ce5b29db0aa67e40274f82a7b2257874d0643b2b",
                             "base_seed": a.base_seed, "kappas": KAPPAS, "eb_variants": EB_VARIANTS, "n_grid": N_GRID,
                             "cfg": {"n_test": a.n_test, "n_pool": a.n_pool, "replicates": a.replicates},
                             "conditions": [c.cond_id for c in conds]})
    tasks = [{"cond": c, "base_seed": a.base_seed, "n_test": a.n_test, "n_pool": a.n_pool,
              "replicates": a.replicates} for c in conds]
    out = run_dir / "rows.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for i, rows in enumerate(run_pool(tasks, task, workers=a.workers), 1):
            w.writerows(rows); fh.flush()
            print(f"[{i}/{len(tasks)}] {rows[0]['cond_id']}", flush=True)
    print("run_dir:", run_dir)


if __name__ == "__main__":
    main()
