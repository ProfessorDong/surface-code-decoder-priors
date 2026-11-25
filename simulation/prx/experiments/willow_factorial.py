"""E5: prior x decoder factorial on Willow hardware.

Question: does a better prior still help once the decoder uses the hyperedge
correlations that the matching graph discards? For every test leaf we decode
the same 50,000 shots with

    priors   = {si1000_shipped, rl_shipped, pooled_shrink(N)}
    decoders = {plain MWPM, correlated matching (PyMatching enable_correlations)}

The fitted prior is transferred into the DEM by `dem.reweight.reweight_dem`, so
both decoders see the same prior information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pymatching

from ..config import WILLOW_ZIP, versions
from ..dem.edges import EdgeTable
from ..dem.reweight import reweight_dem
from ..estimators.pooled import PooledPijEstimator, apply_signature_map, signature_map
from ..results_io import RowWriter, SCHEMA, new_run_dir, save_error_vector, utc_now, write_manifest
from ..stats import per_round_ler_google
from ..willow.reader import LeafKey, WillowArchive, shot_errors
from .common import Timer, paired_summary, run_pool
from .willow import RL_PATHWAY, SI_PATHWAY, TEST_ROUNDS


def _decode(dem, dets, actual, correlated: bool) -> np.ndarray:
    m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=correlated)
    preds = m.decode_batch(dets, bit_packed_shots=True, bit_packed_predictions=True,
                           enable_correlations=correlated)
    return shot_errors(preds, actual)


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    ar = WillowArchive(task["zip"])
    key: LeafKey = task["test_key"]
    cal: LeafKey = task["cal_key"]
    n_cal = task["n_cal"]
    run_dir = Path(task["run_dir"]); run_id = run_dir.name
    dets = ar.detection_events(key)
    actual = ar.obs_actual(key)
    si_dem = ar.dem(key, SI_PATHWAY)
    rl_dem = ar.dem(key, RL_PATHWAY)
    template = EdgeTable.from_dem(si_dem)

    cal_dets = ar.detection_events(cal)[:n_cal]
    cal_template = EdgeTable.from_dem(ar.dem(cal, SI_PATHWAY))
    with Timer() as tf:
        fit = PooledPijEstimator(kappa=task["kappa"]).fit(cal_template, cal_dets)
        fitted, missing = apply_signature_map(template, signature_map(fit.table))
        fit_dem = reweight_dem(si_dem, template, fitted)

    priors = {"si1000_shipped": si_dem, "rl_shipped": rl_dem, f"pooled_shrink_N={n_cal}": fit_dem}
    rows: List[Dict[str, Any]] = []
    errs: Dict[tuple, np.ndarray] = {}
    for prior_id, dem in priors.items():
        for correlated in (False, True):
            with Timer() as td:
                e = _decode(dem, dets, actual, correlated)
            errs[(prior_id, correlated)] = e
            row = {k: "" for k in SCHEMA}
            row.update({"experiment": "willow_factorial", "run_id": run_id, "created_utc": utc_now(),
                        "code_family": "willow_xzzx", "distance": key.distance, "rounds": key.rounds,
                        "basis": key.basis, "noise_family": "hardware", "patch": key.patch,
                        "leaf_id": key.leaf_id, "replicate": 0,
                        "estimator": "pooled_shrink" if prior_id.startswith("pooled") else "fixed_prior",
                        "needs_labels": prior_id == "rl_shipped",
                        "n_cal_shots": n_cal if prior_id.startswith("pooled") else (50000 if prior_id == "rl_shipped" else 0),
                        "n_cal_rounds": 13, "prior_id": prior_id,
                        "decoder": "pymatching_correlated" if correlated else "pymatching_mwpm",
                        "decoder_params_json": "{}", "ref_prior_id": "si1000_shipped",
                        "stim_version": versions()["stim"], "pymatching_version": versions()["pymatching"],
                        "decode_seconds": td.seconds, "fit_seconds": tf.seconds if prior_id.startswith("pooled") else 0.0,
                        "estimator_params_json": json.dumps(dict(fit.params, n_unmatched=missing), sort_keys=True, default=str)
                        if prior_id.startswith("pooled") else "{}"})
            ref = errs.get(("si1000_shipped", correlated))
            row.update(paired_summary(e, None if prior_id == "si1000_shipped" else ref))
            row["ler_per_round_google"] = per_round_ler_google(row["ler"], key.rounds)
            row["error_vector_path"] = save_error_vector(run_dir, f"{key.leaf_id.replace('/', '_')}__{prior_id}__{'corr' if correlated else 'mwpm'}", e)
            rows.append(row)
    return {"leaf": key.leaf_id, "rows": rows}


def main(argv: Optional[Sequence[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-cal", type=int, default=10000)
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)
    ar = WillowArchive(WILLOW_ZIP)
    run_dir = new_run_dir("willow_factorial", label=args.label or "primary")
    tasks = []
    for d in args.distances:
        for patch in ar.patches(d):
            for basis in ("X", "Z"):
                for r in TEST_ROUNDS[d]:
                    tasks.append({"test_key": LeafKey(d, patch, basis, r), "cal_key": LeafKey(d, patch, basis, 13),
                                  "n_cal": args.n_cal, "kappa": args.kappa, "run_dir": str(run_dir),
                                  "zip": str(WILLOW_ZIP)})
    tasks.sort(key=lambda t: -(t["test_key"].distance ** 2 * t["test_key"].rounds))
    if args.limit:
        tasks = tasks[:args.limit]
    write_manifest(run_dir, {"experiment": "willow_factorial", "n_cal": args.n_cal, "kappa": args.kappa,
                             "n_tasks": len(tasks)})
    writer = RowWriter(run_dir)
    for i, res in enumerate(run_pool(tasks, run_task, workers=args.workers), 1):
        writer.write_many(res["rows"])
        print(f"[{i}/{len(tasks)}] {res['leaf']}", flush=True)
    print("run_dir:", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
