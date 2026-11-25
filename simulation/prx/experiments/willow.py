"""E4: hardware replay on the Willow archive.

Primary protocol (Google-like): every estimator is calibrated on the r13 leaf
of a (patch, basis) using the first N shots, transferred by edge signature to
the test leaf's matching graph, and scored on all 50k shots of each test leaf
(rounds with 1 - 2P > min_signal under the SI1000 prior; r01 and r13 excluded).

Priors evaluated on every test leaf (all with plain PyMatching):
  E0  si1000_shipped   Google's shipped SI1000 DEM (the reference template)
  E0' circuit_dem      DEM of circuit_noisy_si1000.stim (decompose_errors)
  E1  rl_shipped       Google's shipped RL-optimized DEM
  E2  uniform_search   E0 scaled uniformly; scale chosen by LER on r13 labels (equivalence control)
  E3  class_search     E0 scaled per edge class; scales chosen by LER on r13 labels
  E5  pooled           time-pooled p_ij fitted on r13 (label-free), transferred by signature
  E6  pooled_shrink    E5 with shrinkage toward E0 (label-free; primary)
  E7  pooled_shrink+class  E6 followed by a per-class scale search on r13 labels
plus Google's shipped decoder predictions as reference rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pymatching

from ..config import SeedPlan, versions
from ..dem.edges import EdgeTable
from ..estimators.pooled import PooledPijEstimator, apply_signature_map, signature_map
from ..estimators.supervised import ClassScaleSearch, UniformScaleSearch
from ..results_io import RowWriter, SCHEMA, new_run_dir, save_error_vector, utc_now, write_manifest
from ..stats import clopper_pearson, per_round_ler_google, per_round_ler_naive
from ..willow.reader import PATHWAYS, LeafKey, WillowArchive, prior_name, shot_errors
from .common import Timer, decode_errors, paired_summary, run_pool

SI_PATHWAY = "correlated_matching_decoder_with_si1000_prior"
RL_PATHWAY = "correlated_matching_decoder_with_rl_optimized_prior"
N_GRID = [100, 300, 1_000, 3_000, 10_000, 30_000, 50_000]
TEST_ROUNDS = {3: [10, 30, 50, 70], 5: [10, 30, 50, 70, 90, 110], 7: [10, 30, 50, 70, 90, 110, 130, 150]}


def _row(key: LeafKey, run_id: str, prior_id: str, estimator: str, n_cal: int, rep: int, needs_labels: bool,
         decoder: str = "pymatching_mwpm") -> Dict[str, Any]:
    row = {k: "" for k in SCHEMA}
    row.update({"experiment": "willow", "run_id": run_id, "created_utc": utc_now(), "code_family": "willow_xzzx",
                "distance": key.distance, "rounds": key.rounds, "basis": key.basis, "noise_family": "hardware",
                "p_true": "", "patch": key.patch, "leaf_id": key.leaf_id, "replicate": rep, "estimator": estimator,
                "needs_labels": needs_labels, "n_cal_shots": n_cal, "n_cal_rounds": 13, "prior_id": prior_id,
                "decoder": decoder, "decoder_params_json": "{}", "ref_prior_id": "si1000_shipped",
                "stim_version": versions()["stim"], "pymatching_version": versions()["pymatching"]})
    return row


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    ar = WillowArchive(task["zip"])
    test_key: LeafKey = task["test_key"]
    cal_key: LeafKey = task["cal_key"]
    cfg = task["cfg"]
    run_dir = Path(task["run_dir"])
    run_id = run_dir.name
    rows: List[Dict[str, Any]] = []
    tag = test_key.leaf_id.replace("/", "_")

    dets = ar.detection_events(test_key)
    actual = ar.obs_actual(test_key)
    si_dem = ar.dem(test_key, SI_PATHWAY)
    template = EdgeTable.from_dem(si_dem)
    ref_errs = decode_errors(template.to_matching(), dets, actual)

    def emit(prior_id: str, estimator: str, errs: np.ndarray, n_cal: int = 0, rep: int = 0, needs_labels=False,
             params=None, decoder="pymatching_mwpm", secs=0.0, fit_secs=0.0, scales=None, save=True):
        row = _row(test_key, run_id, prior_id, estimator, n_cal, rep, needs_labels, decoder)
        row.update(paired_summary(errs, None if prior_id == "si1000_shipped" else ref_errs))
        row["ler_per_round_google"] = per_round_ler_google(row["ler"], test_key.rounds)
        row["ler_per_round_naive"] = per_round_ler_naive(row["ler"], test_key.rounds)
        row["estimator_params_json"] = json.dumps(params or {}, sort_keys=True, default=str)
        row["decode_seconds"] = secs
        row["fit_seconds"] = fit_secs
        if scales:
            for k, v in scales.items():
                row[f"scale_{k}"] = v
        if save:
            row["error_vector_path"] = save_error_vector(run_dir, f"{tag}__{prior_id}", errs)
        rows.append(row)

    # ---- fixed references
    emit("si1000_shipped", "fixed_prior", ref_errs)
    with Timer() as t:
        errs = decode_errors(EdgeTable.from_circuit(ar.circuit(test_key, "noisy_si1000")).to_matching(), dets, actual)
    emit("circuit_dem", "fixed_prior", errs, secs=t.seconds)
    rl_table = EdgeTable.from_dem(ar.dem(test_key, RL_PATHWAY))
    with Timer() as t:
        errs = decode_errors(rl_table.to_matching(), dets, actual)
    emit("rl_shipped", "fixed_prior", errs, secs=t.seconds, needs_labels=True, n_cal=50_000)
    if cfg.get("correlations"):
        m = pymatching.Matching.from_detector_error_model(si_dem, enable_correlations=True)
        preds = m.decode_batch(dets, bit_packed_shots=True, bit_packed_predictions=True, enable_correlations=True)
        emit("si1000_shipped", "fixed_prior", shot_errors(preds, actual), decoder="pymatching_correlated")
    for pw in PATHWAYS:
        if ar.has_pathway(test_key, pw):
            emit(f"google:{prior_name(pw)}", "google_shipped", shot_errors(ar.obs_predicted(test_key, pw), actual),
                 decoder=pw, needs_labels="rl" in pw, save=False)

    # ---- calibration on r13
    cal_dets = ar.detection_events(cal_key)
    cal_obs = ar.obs_actual(cal_key)
    cal_template = EdgeTable.from_dem(ar.dem(cal_key, SI_PATHWAY))
    for n_cal in [n for n in cfg["n_grid"] if n <= cal_dets.shape[0]]:
        reps = min(cfg["replicates"], cal_dets.shape[0] // n_cal)
        for rep in range(reps):
            s = rep * n_cal
            cd, co = cal_dets[s:s + n_cal], cal_obs[s:s + n_cal]
            fitted: Dict[str, Tuple[EdgeTable, Dict[str, Any], bool, float, Optional[Dict[str, float]]]] = {}
            # label-free
            for name, kappa in (("pooled", 0.0), ("pooled_shrink", cfg["kappa"])):
                with Timer() as tf:
                    res = PooledPijEstimator(kappa=kappa).fit(cal_template, cd)
                    smap = signature_map(res.table)
                    table, missing = apply_signature_map(template, smap)
                params = dict(res.params, n_unmatched=missing)
                fitted[name] = (table, params, False, tf.seconds, None)
            # label-requiring (r13 labels), applied to the test template
            if cfg.get("labeled", True):
                with Timer() as tf:
                    res_u = UniformScaleSearch().fit(cal_template, cd, co)
                fitted["uniform_search"] = (template.scaled(res_u.params["s"]), res_u.params, True, tf.seconds, None)
                with Timer() as tf:
                    res_c = ClassScaleSearch(grid=cfg["class_grid"], sweeps=cfg["sweeps"]).fit(cal_template, cd, co)
                fitted["class_search"] = (template.scaled_by_class(res_c.params["scales"]), res_c.params, True, tf.seconds,
                                          res_c.params["scales"])
                # E7: shrinkage fit then class refinement on the calibration leaf
                base_cal = PooledPijEstimator(kappa=cfg["kappa"]).fit(cal_template, cd).table
                with Timer() as tf:
                    res_7 = ClassScaleSearch(grid=cfg["class_grid"], sweeps=cfg["sweeps"]).fit(base_cal, cd, co)
                t6, _ = apply_signature_map(template, signature_map(base_cal))
                fitted["pooled_shrink+class"] = (t6.scaled_by_class(res_7.params["scales"]), res_7.params, True, tf.seconds,
                                                  res_7.params["scales"])
            for name, (table, params, needs_labels, fit_secs, scales) in fitted.items():
                with Timer() as td:
                    errs = decode_errors(table.to_matching(), dets, actual)
                emit(f"{name}_N={n_cal}_rep={rep}", name, errs, n_cal=n_cal, rep=rep, needs_labels=needs_labels,
                     params=params, secs=td.seconds, fit_secs=fit_secs, scales=scales,
                     save=(rep == 0 or cfg.get("save_all_replicates", False)))
    return {"leaf": test_key.leaf_id, "rows": rows}


def build_tasks(ar: WillowArchive, cfg: Dict[str, Any], run_dir: Path, zip_path: str,
                distances: Sequence[int]) -> List[Dict[str, Any]]:
    tasks = []
    for d in distances:
        for patch in ar.patches(d):
            for basis in ("X", "Z"):
                cal = LeafKey(d, patch, basis, 13)
                for r in TEST_ROUNDS[d]:
                    tk = LeafKey(d, patch, basis, r)
                    tasks.append({"test_key": tk, "cal_key": cal, "cfg": cfg, "run_dir": str(run_dir), "zip": zip_path})
    tasks.sort(key=lambda t: -(t["test_key"].distance ** 2 * t["test_key"].rounds))
    return tasks


def main(argv: Optional[Sequence[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--no-labeled", action="store_true")
    ap.add_argument("--correlations", action="store_true")
    ap.add_argument("--n-grid", nargs="+", type=int, default=N_GRID)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first k tasks")
    ap.add_argument("--label", default="")
    ap.add_argument("--save-all-replicates", action="store_true",
                    help="store per-shot vectors for every calibration replicate, not only the first")
    args = ap.parse_args(argv)
    from ..config import WILLOW_ZIP
    ar = WillowArchive(WILLOW_ZIP)
    cfg = {"n_grid": args.n_grid, "replicates": args.replicates, "kappa": args.kappa, "labeled": not args.no_labeled,
           "correlations": args.correlations, "class_grid": [float(x) for x in np.geomspace(0.25, 5.0, 13)], "sweeps": 2}
    if args.save_all_replicates:          # only then, so the original manifest's cfg is reproduced exactly otherwise
        cfg["save_all_replicates"] = True
    run_dir = new_run_dir("willow", label=args.label or "primary")
    tasks = build_tasks(ar, cfg, run_dir, str(WILLOW_ZIP), args.distances)
    if args.limit:
        tasks = tasks[-args.limit:]
    write_manifest(run_dir, {"experiment": "willow", "cfg": cfg, "n_tasks": len(tasks),
                             "test_rounds": TEST_ROUNDS, "leaves": [t["test_key"].leaf_id for t in tasks]})
    writer = RowWriter(run_dir)
    for i, res in enumerate(run_pool(tasks, run_task, workers=args.workers), 1):
        writer.write_many(res["rows"])
        print(f"[{i}/{len(tasks)}] {res['leaf']} rows={len(res['rows'])}", flush=True)
    print("run_dir:", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
