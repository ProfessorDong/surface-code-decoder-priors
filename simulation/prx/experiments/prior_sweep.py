"""E1: how logical error depends on the prior for a fixed decoder (Gate 0).

For every condition (family, d, T, p) this runner decodes one seeded test set
with a family of priors and stores paired results against the physical prior:

* ``oracle_physical``: matching graph of the circuit itself;
* ``stale_pcal=<p_cal>``: matching graph of the same family calibrated at p_cal (Table V);
* ``uniform_s=<s>``: physical prior with every probability scaled by s (null control);
* ``d1_offset_only`` / ``d1_structure_only``: the Table V decomposition diagnostics;
* ``class_search``: per-edge-class scales found by coordinate search on a
  separate seeded search set, then evaluated on the test set (the risk-optimal
  prior within the per-class family). Search rows go to ``search_rows.csv``.
* optional ``ensemble_k<M>_sigma<s>``: majority vote over M perturbed-weight
  matchings (proxy for the MWPM-to-ML gap; recipe aligned with Harmony later).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import RESULTS_ROOT, SeedPlan, versions
from ..dem.edges import CLASS_NAMES, EdgeClass, EdgeTable
from ..dem.priors import physical_prior, stale_prior
from ..results_io import RowWriter, SCHEMA, new_run_dir, save_error_vector, utc_now, write_manifest
from ..stats import per_round_ler_google, per_round_ler_naive
from .common import (Condition, Timer, build_condition_circuit, decode_errors, geometric_mean_ratio,
                     paired_summary, run_pool, sample_shots)

CLASSES = ["space", "time", "diag", "boundary"]


def _blank_row() -> Dict[str, Any]:
    return {k: "" for k in SCHEMA}


def _base_row(cond: Condition, run_id: str, seed_test: int, prior_id: str, ref: str) -> Dict[str, Any]:
    row = _blank_row()
    row.update(cond.row_fields())
    row.update({"experiment": "prior_sweep", "run_id": run_id, "created_utc": utc_now(),
                "seed_test": seed_test, "replicate": 0, "estimator": "fixed_prior", "needs_labels": False,
                "n_cal_shots": 0, "n_cal_rounds": 0, "prior_id": prior_id, "decoder": "pymatching_mwpm",
                "decoder_params_json": "{}", "ref_prior_id": ref, "stim_version": versions()["stim"],
                "pymatching_version": versions()["pymatching"]})
    return row


def _fill_scales(row: Dict[str, Any], scales: Optional[Dict[str, float]]) -> None:
    if scales is None:
        return
    for k in CLASSES:
        row[f"scale_{k}"] = float(scales.get(k, 1.0))


def evaluate_prior(table: EdgeTable, dets: np.ndarray, obs: np.ndarray) -> Tuple[np.ndarray, float]:
    with Timer() as t:
        m = table.to_matching()
        errs = decode_errors(m, dets, obs)
    return errs, t.seconds


def coordinate_search(base: EdgeTable, dets: np.ndarray, obs: np.ndarray, grid: Sequence[float],
                      sweeps: int, classes: Sequence[str] = CLASSES,
                      log: Optional[List[Dict[str, Any]]] = None) -> Tuple[Dict[str, float], int]:
    """Coordinate descent over per-class multiplicative scales minimising failures on (dets, obs)."""
    scales = {c: 1.0 for c in classes}
    best_fail = int(evaluate_prior(base.scaled_by_class(scales), dets, obs)[0].sum())
    if log is not None:
        log.append({"sweep": 0, "class": "init", "scale": 1.0, "n_errors": best_fail, "scales": dict(scales)})
    for sweep in range(1, sweeps + 1):
        improved = False
        for c in classes:
            cand_best = scales[c]
            for s in grid:
                if abs(s - scales[c]) < 1e-12:
                    continue
                trial = dict(scales)
                trial[c] = float(s)
                fail = int(evaluate_prior(base.scaled_by_class(trial), dets, obs)[0].sum())
                if log is not None:
                    log.append({"sweep": sweep, "class": c, "scale": float(s), "n_errors": fail, "scales": trial})
                if fail < best_fail:
                    best_fail, cand_best, improved = fail, float(s), True
            scales[c] = cand_best
        if not improved:
            break
    return scales, best_fail


def perturbed_ensemble_errors(base: EdgeTable, dets: np.ndarray, obs: np.ndarray, members: int,
                              sigma: float, seed: int) -> np.ndarray:
    """Majority vote of ``members`` matchings whose edge probabilities are jittered by exp(sigma*xi)."""
    rng = np.random.default_rng(seed)
    votes = np.zeros(dets.shape[0], dtype=np.int32)
    for _ in range(members):
        jitter = np.exp(sigma * rng.standard_normal(base.num_edges))
        m = base.scaled(jitter).to_matching()
        preds = m.decode_batch(dets, bit_packed_shots=True, bit_packed_predictions=True)
        votes += (preds[:, 0] & 1).astype(np.int32)
    majority = (votes * 2 > members).astype(np.uint8)
    return ((majority ^ (obs[:, 0] & 1)) & 1).astype(bool)


# ----------------------------------------------------------------------------
# One task = one condition, one prior group
# ----------------------------------------------------------------------------

def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    cond: Condition = task["cond"]
    run_dir = Path(task["run_dir"])
    plan = SeedPlan(task["base_seed"])
    group = task["group"]
    cfg = task["cfg"]
    run_id = run_dir.name

    circuit = build_condition_circuit(cond)
    seed_test = plan.for_("test", cond.cond_id)
    with Timer() as ts:
        dets, obs = sample_shots(circuit, cfg["n_test"], seed_test)
    phys = EdgeTable.from_circuit(circuit)
    ref_errs, ref_secs = evaluate_prior(phys, dets, obs)
    tag = cond.cond_id.replace("|", "_").replace(" ", "")
    rows: List[Dict[str, Any]] = []
    search_rows: List[Dict[str, Any]] = []

    def emit(prior_id: str, errs: np.ndarray, secs: float, scales=None, p_hat="", fit_secs=0.0, params=None):
        row = _base_row(cond, run_id, seed_test, prior_id, "oracle_physical")
        row.update(paired_summary(errs, None if prior_id == "oracle_physical" else ref_errs))
        row["ler_per_round_google"] = per_round_ler_google(row["ler"], cond.rounds)
        row["ler_per_round_naive"] = per_round_ler_naive(row["ler"], cond.rounds)
        row["decode_seconds"] = secs
        row["fit_seconds"] = fit_secs
        row["p_hat"] = p_hat
        row["estimator_params_json"] = json.dumps(params or {}, sort_keys=True)
        _fill_scales(row, scales)
        row["error_vector_path"] = save_error_vector(run_dir, f"{tag}__{prior_id}", errs)
        rows.append(row)

    if group == "fixed":
        emit("oracle_physical", ref_errs, ref_secs, p_hat=cond.p)
        for p_cal in cfg["stale_grid"]:
            st = stale_prior(cond.family, cond.distance, cond.rounds, float(p_cal), cond.basis,
                             **({"variant": cond.extra["variant"]} if "variant" in cond.extra else {}))
            errs, secs = evaluate_prior(st, dets, obs)
            emit(f"stale_pcal={p_cal:g}", errs, secs, p_hat=float(p_cal))
            if abs(float(p_cal) - cfg["d1_pcal"]) < 1e-12 and abs(cond.p - cfg["d1_pcal"]) > 1e-12:
                # D1 decomposition: offset-only (physical structure at stale's mean level) and
                # structure-only (stale structure at physical's mean level)
                r = geometric_mean_ratio(st, phys)
                errs_o, secs_o = evaluate_prior(phys.scaled(r), dets, obs)
                emit(f"d1_offset_only_pcal={p_cal:g}", errs_o, secs_o, params={"ratio": r})
                errs_s, secs_s = evaluate_prior(st.scaled(1.0 / r), dets, obs)
                emit(f"d1_structure_only_pcal={p_cal:g}", errs_s, secs_s, params={"ratio": r})
        for s in cfg["uniform_grid"]:
            errs, secs = evaluate_prior(phys.scaled(float(s)), dets, obs)
            emit(f"uniform_s={s:g}", errs, secs, params={"s": float(s)})
        if cfg.get("ensemble_members", 0) > 0:
            with Timer() as te:
                errs = perturbed_ensemble_errors(phys, dets, obs, cfg["ensemble_members"], cfg["ensemble_sigma"],
                                                 plan.for_("ensemble", cond.cond_id))
            emit(f"ensemble_k{cfg['ensemble_members']}_sigma{cfg['ensemble_sigma']:g}", errs, te.seconds,
                 params={"members": cfg["ensemble_members"], "sigma": cfg["ensemble_sigma"]})

    elif group == "search":
        seed_search = plan.for_("search", cond.cond_id)
        sd, so = sample_shots(circuit, cfg["n_search"], seed_search)
        log: List[Dict[str, Any]] = []
        with Timer() as tf:
            scales, best = coordinate_search(phys, sd, so, cfg["class_grid"], cfg["sweeps"], log=log)
        errs, secs = evaluate_prior(phys.scaled_by_class(scales), dets, obs)
        emit("class_search", errs, secs, scales=scales, fit_secs=tf.seconds,
             params={"n_search": cfg["n_search"], "seed_search": seed_search, "grid": list(cfg["class_grid"]),
                     "sweeps": cfg["sweeps"], "best_search_errors": best})
        for entry in log:
            row = _base_row(cond, run_id, seed_search, f"search:{entry['class']}={entry['scale']:g}", "oracle_physical")
            row.update({"experiment": "prior_sweep:search", "n_test_shots": int(cfg["n_search"]),
                        "n_errors": entry["n_errors"], "ler": entry["n_errors"] / cfg["n_search"],
                        "estimator_params_json": json.dumps({"sweep": entry["sweep"]})})
            _fill_scales(row, entry["scales"])
            search_rows.append(row)
    else:
        raise ValueError(group)

    return {"cond_id": cond.cond_id, "group": group, "rows": rows, "search_rows": search_rows,
            "sample_seconds": ts.seconds}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def default_conditions(profile: str) -> List[Condition]:
    conds: List[Condition] = []
    if profile == "smoke":
        conds.append(Condition("legacy_from_physical_rate", 3, 3, 0.01))
        conds.append(Condition("si1000", 3, 3, 0.002, extra={"variant": "google2025"}))
        return conds
    if profile == "gate0":
        for d in (3, 5, 7):
            for p in (0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02):
                conds.append(Condition("legacy_from_physical_rate", d, d, p))
            for p in (0.001, 0.0015, 0.002, 0.003, 0.005):
                conds.append(Condition("si1000", d, d, p, extra={"variant": "google2025"}))
        for d in (3, 5):
            for p in (0.005, 0.01):
                conds.append(Condition("legacy_from_physical_rate", d, 3 * d, p))
            for p in (0.0015, 0.003):
                conds.append(Condition("si1000", d, 3 * d, p, extra={"variant": "google2025"}))
        return conds
    if profile == "gate0_deep":
        # below-threshold SI1000 cells at d>=5 with longer memories; needs --n-test 16000000 --n-search 8000000
        for d, Ts in ((5, (5, 15, 25)), (7, (7, 21))):
            for T in Ts:
                for p in (0.0015, 0.002, 0.003):
                    conds.append(Condition("si1000", d, T, p, extra={"variant": "google2025"}))
        return conds
    raise KeyError(profile)


def main(argv: Optional[Sequence[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="gate0", choices=["smoke", "gate0", "gate0_deep"])
    ap.add_argument("--class-grid", nargs=3, type=float, default=[0.35, 2.85, 9], metavar=("LO", "HI", "N"))
    ap.add_argument("--n-test", type=int, default=4_000_000)
    ap.add_argument("--n-search", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--base-seed", type=int, default=20260917)
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--ensemble-members", type=int, default=0)
    ap.add_argument("--ensemble-sigma", type=float, default=0.5)
    ap.add_argument("--label", default="")
    ap.add_argument("--groups", nargs="+", default=["fixed", "search"])
    args = ap.parse_args(argv)

    cfg = {
        "n_test": args.n_test, "n_search": args.n_search, "sweeps": args.sweeps,
        "stale_grid": [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03],
        "uniform_grid": [0.25, 0.5, 0.7, 1.4, 2.0, 4.0],
        "class_grid": [float(x) for x in np.geomspace(args.class_grid[0], args.class_grid[1], int(args.class_grid[2]))],
        "d1_pcal": 0.005,
        "ensemble_members": args.ensemble_members, "ensemble_sigma": args.ensemble_sigma,
    }
    conds = default_conditions(args.profile)
    run_dir = new_run_dir("prior_sweep", label=args.label or args.profile)
    write_manifest(run_dir, {"experiment": "prior_sweep", "profile": args.profile, "cfg": cfg,
                             "base_seed": args.base_seed, "workers": args.workers,
                             "conditions": [c.cond_id for c in conds]})
    tasks = [{"cond": c, "run_dir": str(run_dir), "base_seed": args.base_seed, "group": g, "cfg": cfg}
             for c in conds for g in args.groups]
    # heaviest first: larger distance and rounds
    tasks.sort(key=lambda t: -(t["cond"].distance ** 2 * t["cond"].rounds))
    writer = RowWriter(run_dir)
    swriter = RowWriter(run_dir, filename="search_rows.csv")
    done = 0
    for res in run_pool(tasks, run_task, workers=args.workers):
        writer.write_many(res["rows"])
        swriter.write_many(res["search_rows"])
        done += 1
        print(f"[{done}/{len(tasks)}] {res['group']:6s} {res['cond_id']} rows={len(res['rows'])} "
              f"sample={res['sample_seconds']:.1f}s", flush=True)
    print("run_dir:", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
