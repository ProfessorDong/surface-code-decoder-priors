"""E3: drifting noise, two regimes.

``--mode within``: the physical rate changes from round to round inside one
memory circuit (leakage-like buildup, bursts, steps). Estimators calibrate on
N shots of the same drifting circuit and are scored on a common test set,
paired against the oracle prior built from the true drifting circuit.

``--mode stream``: chronological blocks of shots come from circuits whose rate
p(t) drifts across blocks. Every block t is decoded with a prior estimated
causally from the W preceding blocks (Spitz-style boxcar window; density
scalar; static). Metrics: total failures with moving-block bootstrap CIs,
and excess failures during recovery after a step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import stim

from ..config import SeedPlan, versions
from ..dem.edges import EdgeTable, join_probabilities
from ..dem.priors import build_circuit, physical_prior
from ..estimators.density import DensityCalibrationCurve, detection_fraction
from ..estimators.pij import PijEstimator
from ..estimators.pooled import PooledPijEstimator, TimeWindowPijEstimator
from ..estimators.spitz import SpitzStream
from ..estimators.supervised import ClassScaleSearch
from ..noise.drift import RoundSchedule, drifting_circuit
from ..results_io import RowWriter, SCHEMA, new_run_dir, save_error_vector, utc_now, write_manifest
from ..stats import block_bootstrap_delta, per_round_ler_google
from .common import Timer, decode_errors, paired_summary, run_pool, sample_shots

VARIANT = "google2025"


def within_schedules(T: int, p0: float) -> Dict[str, RoundSchedule]:
    return {
        "constant": RoundSchedule.constant(T, p0),
        "sinusoid": RoundSchedule.sinusoid(T, p0, 0.5 * p0, period=T / 4, onset=T // 10),
        "step": RoundSchedule.step(T, p0, 2 * p0, T // 2),
        "burst": RoundSchedule.burst(T, p0, 4 * p0, T // 2, max(3, T // 20)),
        "ou": RoundSchedule.ornstein_uhlenbeck(T, p0, 0.3, T / 10, seed=11),
    }


# ----------------------------------------------------------------------------
# within-circuit drift
# ----------------------------------------------------------------------------

def run_within(task: Dict[str, Any]) -> Dict[str, Any]:
    d, T, p0, sched_name = task["d"], task["T"], task["p0"], task["schedule"]
    cfg = task["cfg"]
    run_dir = Path(task["run_dir"]); run_id = run_dir.name
    plan = SeedPlan(task["base_seed"])
    sched = within_schedules(T, p0)[sched_name]
    circuit = drifting_circuit(d, T, sched, VARIANT)
    cond_id = f"within|d{d}|T{T}|p{p0:g}|{sched_name}"
    tdets, tobs = sample_shots(circuit, cfg["n_test"], plan.for_("test", cond_id))
    pdets, pobs = sample_shots(circuit, cfg["n_pool"], plan.for_("pool", cond_id))
    oracle = EdgeTable.from_circuit(circuit)
    static = physical_prior("si1000", d, T, p0, variant=VARIANT)
    mean_prior = physical_prior("si1000", d, T, float(sched.values.mean()), variant=VARIANT)
    ref = decode_errors(oracle.to_matching(), tdets, tobs)
    rows: List[Dict[str, Any]] = []
    tag = cond_id.replace("|", "_")

    def emit(prior_id, estimator, errs, n_cal=0, rep=0, needs_labels=False, params=None, secs=0.0, fit_secs=0.0, save=True, scales=None):
        row = {k: "" for k in SCHEMA}
        row.update({"experiment": "drift_within", "run_id": run_id, "created_utc": utc_now(),
                    "code_family": "rotated_surface_memory_Z", "distance": d, "rounds": T, "basis": "Z",
                    "noise_family": f"si1000:{VARIANT}", "p_true": p0, "drift_id": sched_name,
                    "noise_extra_json": json.dumps({"schedule": sched_name, "values": sched.values.round(6).tolist()}),
                    "replicate": rep, "estimator": estimator, "needs_labels": needs_labels, "n_cal_shots": n_cal,
                    "n_cal_rounds": T, "prior_id": prior_id, "decoder": "pymatching_mwpm", "decoder_params_json": "{}",
                    "ref_prior_id": "oracle_drifting", "stim_version": versions()["stim"],
                    "pymatching_version": versions()["pymatching"], "fit_seconds": fit_secs, "decode_seconds": secs,
                    "estimator_params_json": json.dumps(params or {}, sort_keys=True, default=str)})
        row.update(paired_summary(errs, None if prior_id == "oracle_drifting" else ref))
        row["ler_per_round_google"] = per_round_ler_google(row["ler"], T)
        if scales:
            for k, v in scales.items():
                row[f"scale_{k}"] = v
        if save:
            row["error_vector_path"] = save_error_vector(run_dir, f"{tag}__{prior_id}", errs)
        rows.append(row)

    emit("oracle_drifting", "oracle", ref)
    emit("static_p0", "static", decode_errors(static.to_matching(), tdets, tobs))
    emit("static_mean", "static_mean", decode_errors(mean_prior.to_matching(), tdets, tobs))
    for n_cal in [n for n in cfg["n_grid"] if n <= cfg["n_pool"]]:
        reps = min(cfg["replicates"], cfg["n_pool"] // n_cal)
        for rep in range(reps):
            s = rep * n_cal
            cd, co = pdets[s:s + n_cal], pobs[s:s + n_cal]
            ests = [("pij", PijEstimator(), False), ("pooled_shrink", PooledPijEstimator(kappa=cfg["kappa"]), False)]
            for hw in cfg["half_widths"]:
                ests.append((f"window_hw={hw}", TimeWindowPijEstimator(half_width=hw), False))
            if cfg.get("labeled", True):
                ests.append(("class_search", ClassScaleSearch(grid=cfg["class_grid"], sweeps=cfg["sweeps"]), True))
            for name, est, needs in ests:
                with Timer() as tf:
                    res = est.fit(static, cd, co if needs else None)
                with Timer() as td:
                    errs = decode_errors(res.table.to_matching(), tdets, tobs)
                emit(f"{name}_N={n_cal}_rep={rep}", name, errs, n_cal, rep, needs, res.params, td.seconds, tf.seconds,
                     save=(rep == 0), scales=res.params.get("scales"))
    return {"cond_id": cond_id, "rows": rows}


# ----------------------------------------------------------------------------
# chronological shot streams
# ----------------------------------------------------------------------------

def stream_schedule(name: str, n_blocks: int, p0: float, seed: int) -> np.ndarray:
    t = np.arange(n_blocks)
    if name == "constant":
        return np.full(n_blocks, p0)
    if name == "sinusoid":
        return p0 * (1 + 0.5 * np.sin(2 * np.pi * t / (n_blocks / 3)))
    if name == "step":
        v = np.full(n_blocks, p0); v[n_blocks // 2:] = 2 * p0; return v
    if name == "ou":
        return RoundSchedule.ornstein_uhlenbeck(n_blocks, p0, 0.3, n_blocks / 10, seed).values
    raise KeyError(name)


def run_stream(task: Dict[str, Any]) -> Dict[str, Any]:
    d, T, p0, sched_name = task["d"], task["T"], task["p0"], task["schedule"]
    cfg = task["cfg"]
    run_dir = Path(task["run_dir"]); run_id = run_dir.name
    plan = SeedPlan(task["base_seed"])
    rep = task["rep"]
    n_blocks, B = cfg["n_blocks"], cfg["block_shots"]
    cond_id = f"stream|d{d}|T{T}|p{p0:g}|{sched_name}|rep{rep}"
    p_t = stream_schedule(sched_name, n_blocks, p0, plan.for_("sched", cond_id))
    # circuits cached by rounded p
    circ_cache: Dict[float, Tuple[stim.Circuit, EdgeTable]] = {}

    def circ_for(p: float):
        key = float(f"{p:.4g}")
        if key not in circ_cache:
            c = build_circuit("si1000", d, T, key, variant=VARIANT)
            circ_cache[key] = (c, EdgeTable.from_circuit(c))
        return circ_cache[key]

    dets_blocks, obs_blocks, oracle_errs = [], [], []
    for b in range(n_blocks):
        c, tab = circ_for(p_t[b])
        db, ob = sample_shots(c, B, plan.for_("block", cond_id, b))
        dets_blocks.append(db); obs_blocks.append(ob)
        oracle_errs.append(decode_errors(tab.to_matching(), db, ob))
    dets = np.vstack(dets_blocks); obs = np.vstack(obs_blocks)
    oracle = np.concatenate(oracle_errs)
    template = physical_prior("si1000", d, T, p0, variant=VARIANT)
    static_errs = decode_errors(template.to_matching(), dets, obs)
    curve = DensityCalibrationCurve("si1000", d, T, variant=VARIANT, shots_per_point=cfg["density_shots"]).build()
    rows: List[Dict[str, Any]] = []
    tag = cond_id.replace("|", "_")

    def emit(prior_id, estimator, errs, params=None, save=True):
        row = {k: "" for k in SCHEMA}
        row.update({"experiment": "drift_stream", "run_id": run_id, "created_utc": utc_now(),
                    "code_family": "rotated_surface_memory_Z", "distance": d, "rounds": T, "basis": "Z",
                    "noise_family": f"si1000:{VARIANT}", "p_true": p0, "drift_id": sched_name,
                    "noise_extra_json": json.dumps({"schedule": sched_name, "n_blocks": n_blocks, "block_shots": B}),
                    "replicate": rep, "estimator": estimator, "needs_labels": False, "n_cal_rounds": T,
                    "prior_id": prior_id, "decoder": "pymatching_mwpm", "decoder_params_json": "{}",
                    "ref_prior_id": "oracle_block", "stim_version": versions()["stim"],
                    "pymatching_version": versions()["pymatching"],
                    "estimator_params_json": json.dumps(params or {}, sort_keys=True, default=str)})
        # score only blocks every estimator can decode causally (after the longest warm-up)
        w0 = cfg["warmup_blocks"] * B
        e = errs[w0:]; o = oracle[w0:]
        row.update(paired_summary(e, None if prior_id == "oracle_block" else o))
        dlt, lo, hi = block_bootstrap_delta(e, o, block_len=B, n_boot=500, seed=plan.for_("boot", cond_id, prior_id))
        row["delta_ler_vs_ref"], row["delta_ci_lo"], row["delta_ci_hi"] = dlt, lo, hi
        row["n_cal_shots"] = params.get("window_shots", 0) if params else 0
        # recovery excess after a step: failures in the K blocks after the step minus oracle
        if sched_name == "step":
            s0 = (n_blocks // 2) * B; s1 = s0 + cfg["recovery_blocks"] * B
            row["p_hat"] = int(errs[s0:s1].sum() - oracle[s0:s1].sum())
        if save:
            row["error_vector_path"] = save_error_vector(run_dir, f"{tag}__{prior_id}", errs)
        rows.append(row)

    emit("oracle_block", "oracle", oracle)
    emit("static_p0", "static", static_errs)
    for W in cfg["windows"]:
        # Spitz boxcar (per-edge p_ij from the preceding W blocks)
        errs = static_errs.copy()
        for b, res in SpitzStream(template, block_shots=B, window_blocks=W).run(dets):
            s = b * B
            errs[s:s + B] = decode_errors(res.table.to_matching(), dets[s:s + B], obs[s:s + B])
        emit(f"spitz_W={W}", "spitz", errs, {"window_shots": W * B})
        # scalar density over the same window
        errs = static_errs.copy()
        for b in range(W, n_blocks):
            f = detection_fraction(dets[(b - W) * B:b * B], template.num_detectors)
            p_hat = curve.invert(f)
            tab = template.with_probabilities(join_probabilities(template, circ_for(p_hat)[1]))
            s = b * B
            errs[s:s + B] = decode_errors(tab.to_matching(), dets[s:s + B], obs[s:s + B])
        emit(f"density_W={W}", "density", errs, {"window_shots": W * B})
    return {"cond_id": cond_id, "rows": rows}


def main(argv: Optional[Sequence[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["within", "stream"], default="within")
    ap.add_argument("--profile", choices=["smoke", "main"], default="main")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--base-seed", type=int, default=20260919)
    ap.add_argument("--label", default="")
    ap.add_argument("--n-test", type=int, default=1_000_000)
    ap.add_argument("--n-pool", type=int, default=100_000)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--n-blocks", type=int, default=600)
    ap.add_argument("--n-grid", nargs="+", type=int, default=[300, 1_000, 3_000, 10_000, 30_000, 100_000])
    ap.add_argument("--half-widths", nargs="+", type=int, default=[1, 2, 5, 10])
    ap.add_argument("--windows", nargs="+", type=int, default=[5, 20, 50, 200])
    ap.add_argument("--block-shots", type=int, default=100)
    args = ap.parse_args(argv)
    cfg = {"n_test": args.n_test, "n_pool": args.n_pool, "replicates": args.replicates,
           "n_grid": args.n_grid, "kappa": 5.0, "half_widths": args.half_widths,
           "class_grid": [float(x) for x in np.geomspace(0.25, 5.0, 13)], "sweeps": 2, "labeled": True,
           "n_blocks": args.n_blocks, "block_shots": args.block_shots, "windows": args.windows,
           "warmup_blocks": 200, "recovery_blocks": 50, "density_shots": 20_000}
    tasks: List[Dict[str, Any]] = []
    if args.mode == "within":
        grid = [(3, 12, 0.002, "step")] if args.profile == "smoke" else \
               [(d, T, p0, s) for d, T in ((3, 100), (5, 100), (7, 50)) for p0 in (0.0015, 0.003)
                for s in ("constant", "sinusoid", "step", "burst", "ou")]
        for d, T, p0, s in grid:
            tasks.append({"d": d, "T": T, "p0": p0, "schedule": s, "cfg": cfg, "base_seed": args.base_seed})
        fn = run_within
    else:
        grid = [(3, 3, 0.002, "step", 0)] if args.profile == "smoke" else \
               [(d, d, p0, s, r) for d in (3, 5, 7) for p0 in (0.0015, 0.003) for s in ("constant", "sinusoid", "step", "ou")
                for r in range(args.replicates)]
        for d, T, p0, s, r in grid:
            tasks.append({"d": d, "T": T, "p0": p0, "schedule": s, "rep": r, "cfg": cfg, "base_seed": args.base_seed})
        fn = run_stream
    run_dir = new_run_dir(f"drift_{args.mode}", label=args.label or args.profile)
    for t in tasks:
        t["run_dir"] = str(run_dir)
    write_manifest(run_dir, {"experiment": f"drift_{args.mode}", "cfg": cfg, "base_seed": args.base_seed,
                             "tasks": [{k: v for k, v in t.items() if k not in ("cfg", "run_dir")} for t in tasks]})
    tasks.sort(key=lambda t: -(t["d"] ** 2 * t["T"]))
    writer = RowWriter(run_dir)
    for i, res in enumerate(run_pool(tasks, fn, workers=args.workers), 1):
        writer.write_many(res["rows"])
        print(f"[{i}/{len(tasks)}] {res['cond_id']} rows={len(res['rows'])}", flush=True)
    print("run_dir:", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
