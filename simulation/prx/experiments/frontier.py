"""E2: logical error versus calibration budget N for every estimator (the frontier).

For each condition a reference (nominal) prior is deliberately misspecified
relative to the true circuit (different p, and/or per-qubit heterogeneity).
Estimators see N calibration shots (label-free ones only the detector events)
and are scored on one common seeded test set, paired against the physical
prior of the true circuit and against the nominal prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import SeedPlan, versions
from ..dem.edges import EdgeTable, join_probabilities
from ..dem.priors import physical_prior
from ..estimators.density import DensityCalibrationCurve, DensityEstimator
from ..estimators.oracle import StaticPrior
from ..estimators.pij import PijEstimator
from ..estimators.occurrence import OccurrenceEstimator
from ..estimators.pooled import PooledPijEstimator
from ..estimators.supervised import ClassScaleSearch, UniformScaleSearch
from ..results_io import RowWriter, SCHEMA, new_run_dir, save_error_vector, utc_now, write_manifest
from ..stats import per_round_ler_google, per_round_ler_naive
from .common import (Condition, Timer, build_condition_circuit, decode_errors, paired_summary, run_pool,
                     sample_shots)

N_GRID = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000]


def make_estimator(name: str, cond: Condition, cfg: Dict[str, Any]):
    if name == "static":
        return StaticPrior()
    if name == "density":
        curve = DensityCalibrationCurve(cond.family, cond.distance, cond.rounds, cond.basis,
                                        variant=cond.extra.get("variant"), shots_per_point=cfg["density_shots"]).build()
        return DensityEstimator(curve)
    if name == "pij":
        return PijEstimator()
    if name == "pooled":
        return PooledPijEstimator(kappa=0.0)
    if name == "pooled_shrink":
        return PooledPijEstimator(kappa=cfg["kappa"])
    if name == "class_search":
        return ClassScaleSearch(grid=cfg["class_grid"], sweeps=cfg["sweeps"])
    if name == "dgr":
        return OccurrenceEstimator()
    if name == "uniform_search":
        return UniformScaleSearch()
    raise KeyError(name)


def nominal_template(cond: Condition) -> EdgeTable:
    """The misspecified reference prior: homogeneous circuit at p_nominal."""
    p_nom = cond.extra.get("p_nominal", cond.p)
    kw = {"variant": cond.extra["variant"]} if "variant" in cond.extra else {}
    return physical_prior(cond.family, cond.distance, cond.rounds, p_nom, cond.basis, **kw)


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    cond: Condition = task["cond"]
    est_name: str = task["estimator"]
    cfg = task["cfg"]
    run_dir = Path(task["run_dir"])
    plan = SeedPlan(task["base_seed"])
    run_id = run_dir.name

    circuit = build_condition_circuit(cond)
    seed_test = plan.for_("test", cond.cond_id)
    seed_pool = plan.for_("pool", cond.cond_id)
    tdets, tobs = sample_shots(circuit, cfg["n_test"], seed_test)
    pdets, pobs = sample_shots(circuit, cfg["n_pool"], seed_pool)
    template = nominal_template(cond)
    phys = EdgeTable.from_circuit(circuit)
    # references
    ref_phys = decode_errors(phys.to_matching(), tdets, tobs)
    ref_static = decode_errors(template.to_matching(), tdets, tobs)
    est = make_estimator(est_name, cond, cfg)
    tag = cond.cond_id.replace("|", "_").replace(" ", "")
    rows: List[Dict[str, Any]] = []

    def base_row(prior_id: str, n_cal: int, rep: int, seed_cal: int) -> Dict[str, Any]:
        row = {k: "" for k in SCHEMA}
        row.update(cond.row_fields())
        row.update({"experiment": "frontier", "run_id": run_id, "created_utc": utc_now(), "seed_test": seed_test,
                    "seed_cal": seed_cal, "replicate": rep, "estimator": est_name, "needs_labels": est.needs_labels,
                    "n_cal_shots": n_cal, "n_cal_rounds": cond.rounds, "prior_id": prior_id, "decoder": "pymatching_mwpm",
                    "decoder_params_json": "{}", "ref_prior_id": "oracle_physical",
                    "stim_version": versions()["stim"], "pymatching_version": versions()["pymatching"]})
        return row

    def finish(row: Dict[str, Any], errs: np.ndarray, save: bool):
        row.update(paired_summary(errs, ref_phys))
        b2, c2 = int(np.count_nonzero(errs & ~ref_static)), int(np.count_nonzero(~errs & ref_static))
        row["estimator_params_json"] = json.dumps(dict(json.loads(row["estimator_params_json"] or "{}"),
                                                       vs_static_b=b2, vs_static_c=c2), sort_keys=True)
        row["ler_per_round_google"] = per_round_ler_google(row["ler"], cond.rounds)
        row["ler_per_round_naive"] = per_round_ler_naive(row["ler"], cond.rounds)
        if save:
            row["error_vector_path"] = save_error_vector(run_dir, f"{tag}__{row['prior_id']}", errs)
        rows.append(row)

    # reference rows (once per task, tagged by estimator for joins)
    if est_name == "static":
        r = base_row("oracle_physical", 0, 0, 0); r["estimator"] = "oracle_physical"; r["prior_id"] = "oracle_physical"
        r["estimator_params_json"] = "{}"; r["decode_seconds"] = 0.0
        finish(r, ref_phys, True)
        r = base_row("static_nominal", 0, 0, 0); r["estimator_params_json"] = "{}"; r["decode_seconds"] = 0.0
        finish(r, ref_static, True)
        return {"cond_id": cond.cond_id, "estimator": est_name, "rows": rows}

    n_grid = [n for n in cfg["n_grid"] if n <= cfg["n_pool"]]
    for n_cal in n_grid:
        reps = min(cfg["replicates"], cfg["n_pool"] // n_cal)
        for rep in range(reps):
            s = rep * n_cal
            cd, co = pdets[s:s + n_cal], pobs[s:s + n_cal]
            with Timer() as tf:
                res = est.fit(template, cd, co if est.needs_labels else None)
            with Timer() as td:
                errs = decode_errors(res.table.to_matching(), tdets, tobs)
            row = base_row(f"{est_name}_N={n_cal}_rep={rep}", n_cal, rep, seed_pool)
            row["estimator_params_json"] = json.dumps(res.params, sort_keys=True, default=str)
            row["p_hat"] = "" if res.p_hat is None else res.p_hat
            row["fit_seconds"] = tf.seconds
            row["decode_seconds"] = td.seconds
            if "scales" in res.params:
                for k, v in res.params["scales"].items():
                    row[f"scale_{k}"] = v
            finish(row, errs, save=(rep == 0))
    return {"cond_id": cond.cond_id, "estimator": est_name, "rows": rows}


def default_conditions(profile: str) -> List[Condition]:
    conds: List[Condition] = []
    if profile == "smoke":
        conds.append(Condition("si1000", 3, 3, 0.003, extra={"variant": "google2025", "p_nominal": 0.002}))
        return conds
    if profile == "main":
        for d in (3, 5, 7):
            for T in (d, 3 * d):
                if d == 7 and T > d:
                    continue
                for p in (0.0015, 0.003):
                    for sigma_h, het_seed in ((0.0, 0), (0.6, 1), (0.6, 2), (1.0, 3)):
                        extra = {"variant": "google2025", "p_nominal": p / 1.5}
                        if sigma_h > 0:
                            extra.update({"sigma_h": sigma_h, "het_seed": het_seed})
                        conds.append(Condition("si1000", d, T, p, extra=extra))
        return conds
    if profile == "draws":
        # Independent noise draws at each heterogeneity level. The main profile has
        # two draws at sigma = 0.6 and one at sigma = 1, so its per-distance budgets
        # can be driven by a single realization; calibration replicates do not create
        # new devices. Seeds are disjoint from the main profile (1, 2, 3).
        for d in (3, 5, 7):
            for T in (d, 3 * d):
                if d == 7 and T > d:
                    continue
                for p in (0.0015, 0.003):
                    for sigma_h, seeds in ((0.6, range(101, 107)), (1.0, range(201, 207))):
                        for het_seed in seeds:
                            conds.append(Condition("si1000", d, T, p, extra={
                                "variant": "google2025", "p_nominal": p / 1.5,
                                "sigma_h": sigma_h, "het_seed": het_seed}))
        return conds
    if profile == "device_like":
        # The only misspecification is per-qubit measurement heterogeneity, with the
        # reference prior homogeneous at the true p: the hardware situation, where
        # Google's SI1000 DEM has the right geometry but uniform measurement rates.
        # sigma_h = 0.65 reproduces the time-like spread fitted on Willow (sd log p ~ 0.33).
        for d in (3, 5, 7):
            for T in (d, 3 * d):
                if d == 7 and T > d:
                    continue
                for sigma in (0.35, 0.65, 1.0):
                    for seed in (1, 2):
                        conds.append(Condition("si1000", d, T, 0.002, extra={
                            "variant": "google2025", "p_nominal": 0.002,
                            "sigma_h": sigma, "het_seed": seed, "het_channels": ["meas_flip"]}))
        return conds
    raise KeyError(profile)


def main(argv: Optional[Sequence[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="main", choices=["smoke", "main", "device_like", "draws"])
    ap.add_argument("--n-test", type=int, default=1_000_000)
    ap.add_argument("--n-pool", type=int, default=300_000)
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--base-seed", type=int, default=20260918)
    ap.add_argument("--estimators", nargs="+",
                    default=["static", "density", "pij", "pooled", "pooled_shrink", "uniform_search", "class_search"])
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)
    cfg = {"n_test": args.n_test, "n_pool": args.n_pool, "replicates": args.replicates, "n_grid": N_GRID,
           "kappa": args.kappa, "class_grid": [float(x) for x in np.geomspace(0.35, 2.85, 9)], "sweeps": 2,
           "density_shots": 20_000}
    conds = default_conditions(args.profile)
    run_dir = new_run_dir("frontier", label=args.label or args.profile)
    write_manifest(run_dir, {"experiment": "frontier", "profile": args.profile, "cfg": cfg, "base_seed": args.base_seed,
                             "estimators": args.estimators, "conditions": [c.cond_id for c in conds]})
    tasks = [{"cond": c, "estimator": e, "run_dir": str(run_dir), "base_seed": args.base_seed, "cfg": cfg}
             for c in conds for e in args.estimators]
    tasks.sort(key=lambda t: -(t["cond"].distance ** 2 * t["cond"].rounds))
    writer = RowWriter(run_dir)
    for i, res in enumerate(run_pool(tasks, run_task, workers=args.workers), 1):
        writer.write_many(res["rows"])
        print(f"[{i}/{len(tasks)}] {res['estimator']:14s} {res['cond_id']} rows={len(res['rows'])}", flush=True)
    print("run_dir:", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
