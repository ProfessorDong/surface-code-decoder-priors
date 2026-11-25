"""Heterogeneity penalty across many noise realisations.

The main frontier ran a single per-qubit draw at each heterogeneity level, which
is not enough: the penalty for using a homogeneous prior is a heavy-tailed
statistic over draws, and a single draw containing one very bad qubit is not
representative. This runner repeats the measurement over many seeds and reports
the distribution.

It also separates two effects that the plain parameterisation confounds. With
lambda_q = exp(sigma xi_q) the mean multiplier is exp(sigma^2 / 2), so raising
sigma also raises the average error rate: at sigma = 1 the patch is 1.65x
noisier on average, and part of any "heterogeneity penalty" is really a change
of operating point. The mean-preserving variant lambda_q = exp(sigma xi_q -
sigma^2 / 2) has unit mean, so the two can be told apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import SeedPlan, versions
from ..dem.edges import EdgeTable
from ..dem.priors import physical_prior
from ..noise.si1000 import (SI1000Rates, inject_si1000, noiseless_memory_circuit)
from ..results_io import RowWriter, SCHEMA, new_run_dir, utc_now, write_manifest
from ..stats import clopper_pearson, discordant_counts, exact_mcnemar, paired_delta_ci
from .common import Timer, decode_errors, run_pool, sample_shots

VARIANT = "google2025"


def qubit_scales(num_qubits: int, sigma: float, seed: int, mean_preserving: bool) -> Dict[int, float]:
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal(num_qubits)
    shift = 0.5 * sigma * sigma if mean_preserving else 0.0
    return {q: float(np.exp(sigma * xi[q] - shift)) for q in range(num_qubits)}


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    d, T, p = task["d"], task["T"], task["p"]
    sigma, seed, mp = task["sigma"], task["seed"], task["mean_preserving"]
    channels = task["channels"]
    cfg = task["cfg"]
    run_dir = Path(task["run_dir"])
    plan = SeedPlan(task["base_seed"])

    noiseless = noiseless_memory_circuit(d, T)
    rates = SI1000Rates.from_variant(VARIANT, p)
    scales = qubit_scales(noiseless.num_qubits, sigma, seed, mp) if sigma > 0 else None
    qs = ({c: scales for c in channels} if (scales and channels) else scales)
    circuit = inject_si1000(noiseless, rates, qubit_scale=qs)

    tag = f"d{d}|T{T}|p{p:g}|sigma{sigma:g}|seed{seed}|{'mp' if mp else 'raw'}|{'+'.join(channels) if channels else 'all'}"
    dets, obs = sample_shots(circuit, cfg["n_test"], plan.for_("test", tag))
    oracle = EdgeTable.from_circuit(circuit)
    static = physical_prior("si1000", d, T, p, variant=VARIANT)

    e_or = decode_errors(oracle.to_matching(), dets, obs)
    e_st = decode_errors(static.to_matching(), dets, obs)
    b, c = discordant_counts(e_st, e_or)
    delta, lo, hi = paired_delta_ci(e_st, e_or)
    lam = np.array(list(scales.values())) if scales else np.array([1.0])

    row = {k: "" for k in SCHEMA}
    row.update({
        "experiment": "hetero_seeds", "run_id": run_dir.name, "created_utc": utc_now(),
        "code_family": f"rotated_surface_memory_Z", "distance": d, "rounds": T, "basis": "Z",
        "noise_family": f"si1000:{VARIANT}", "p_true": p, "replicate": seed,
        "noise_extra_json": json.dumps({"sigma": sigma, "seed": seed, "mean_preserving": mp,
                                        "channels": channels, "lambda_mean": float(lam.mean()),
                                        "lambda_max": float(lam.max()), "lambda_median": float(np.median(lam))},
                                       sort_keys=True),
        "estimator": "static_homogeneous", "needs_labels": False, "n_cal_shots": 0,
        "prior_id": "static_homogeneous", "ref_prior_id": "oracle_physical",
        "decoder": "pymatching_mwpm", "decoder_params_json": "{}",
        "n_test_shots": int(e_st.size), "n_errors": int(e_st.sum()), "ler": float(e_st.mean()),
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p_exact": exact_mcnemar(b, c),
        "delta_ler_vs_ref": delta, "delta_ci_lo": lo, "delta_ci_hi": hi,
        "p_hat": float(e_or.mean()),          # the oracle rate, for the penalty ratio
        "stim_version": versions()["stim"], "pymatching_version": versions()["pymatching"],
    })
    lo_ci, hi_ci = clopper_pearson(int(e_st.sum()), int(e_st.size))
    row["ler_cp_lo"], row["ler_cp_hi"] = lo_ci, hi_ci
    penalty = float(e_st.mean() / e_or.mean() - 1) if e_or.mean() > 0 else float("nan")
    row["estimator_params_json"] = json.dumps({"penalty_vs_oracle": penalty}, sort_keys=True)
    return {"tag": tag, "rows": [row], "penalty": penalty}


def main(argv: Optional[Sequence[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.6, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--n-test", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=20260920)
    ap.add_argument("--channels", nargs="*", default=[], help="restrict heterogeneity to these channels")
    ap.add_argument("--both-parameterisations", action="store_true",
                    help="also run the mean-preserving variant")
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    cfg = {"n_test": args.n_test}
    mps = [False, True] if args.both_parameterisations else [False]
    tasks = []
    for d in args.distances:
        for sigma in args.sigmas:
            seeds = args.seeds if sigma > 0 else args.seeds[:1]
            for seed in seeds:
                for mp in (mps if sigma > 0 else [False]):
                    tasks.append({"d": d, "T": d, "p": args.p, "sigma": sigma, "seed": seed,
                                  "mean_preserving": mp, "channels": list(args.channels),
                                  "cfg": cfg, "base_seed": args.base_seed})
    run_dir = new_run_dir("hetero_seeds", label=args.label or "main")
    for t in tasks:
        t["run_dir"] = str(run_dir)
    write_manifest(run_dir, {"experiment": "hetero_seeds", "cfg": cfg, "n_tasks": len(tasks),
                             "sigmas": args.sigmas, "seeds": args.seeds, "p": args.p,
                             "channels": args.channels, "base_seed": args.base_seed})
    tasks.sort(key=lambda t: -(t["d"] ** 2 * t["T"]))
    writer = RowWriter(run_dir)
    for i, res in enumerate(run_pool(tasks, run_task, workers=args.workers), 1):
        writer.write_many(res["rows"])
        print(f"[{i}/{len(tasks)}] {res['tag']} penalty={100*res['penalty']:+.2f}%", flush=True)
    print("run_dir:", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
