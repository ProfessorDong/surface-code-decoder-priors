"""Intervals for the hardware shot-budget frontier (Fig. 4), with pairing preserved.

Every calibration replicate of every estimator decodes the same test shots as the
reference prior. For each (distance, estimator, budget) we load the per-shot error
vectors of all replicates and of the reference, classify each shot of each leaf by
its joint failure pattern, and bootstrap:

  * patch-basis groups, resampled within distance where there are at least
    ``MIN_GROUPS`` of them (distances three and five); distance seven has one
    patch in two bases, so its interval is conditional on that patch;
  * calibration replicates, resampled with replacement;
  * shots, by one multinomial draw per leaf over the joint patterns, so the
    reference and every replicate move together.

The statistic is the frontier's point estimate, mean over replicates of
K_rep / K_ref - 1 with K pooled over the leaves of a distance. Intervals are
pointwise, one budget at a time; they are not simultaneous bands over the frontier.

This needs per-shot vectors for every replicate (run the Willow experiment with
``--save-all-replicates``). An earlier version resampled each replicate's 2x2
table against the reference separately, which discards the covariance between
replicates that share test shots; it is superseded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..results_io import load_error_vector
from .willow import stable_seed

MIN_GROUPS = 5
BOOT = 4000
REF = "si1000_shipped"
ESTIMATORS = ("pooled_shrink", "pooled", "class_search", "uniform_search", "pooled_shrink+class")


def joint_counts(run_dir: Path, leaf: str, names: List[str]) -> np.ndarray:
    tag = leaf.replace("/", "_")
    code = None
    for j, nm in enumerate(names):
        v = load_error_vector(run_dir, f"errors/{tag}__{nm}.npz").astype(np.int64)
        code = v << j if code is None else code | (v << j)
    return np.bincount(code, minlength=1 << len(names))


def interval(run_dir: Path, rows: pd.DataFrame, d: int, names: List[str], seed: int, boot: int = BOOT,
             force_groups: bool = False):
    """names[0] is the reference; names[1:] are the replicates (or a single fixed prior)."""
    leaves = rows[rows.distance == d][["leaf_id", "patch", "basis"]].drop_duplicates().sort_values("leaf_id")
    C = np.stack([joint_counts(run_dir, lf, names) for lf in leaves.leaf_id])
    K = C.shape[1]
    F = np.array([[(k >> j) & 1 for j in range(len(names))] for k in range(K)])
    group = (leaves.patch + "/" + leaves.basis).values
    groups = sorted(set(group))
    members = {g: np.flatnonzero(group == g) for g in groups}
    R = len(names) - 1

    def stat(Cx, pick_r):
        e = (Cx @ F).sum(axis=0)
        return float(np.mean([e[1 + r] / e[0] - 1 for r in pick_r]))

    point = stat(C, range(R))
    rng = np.random.default_rng(seed)
    resample_groups = force_groups or len(groups) >= MIN_GROUPS
    draws = np.empty(boot)
    for b in range(boot):
        pg = [groups[i] for i in rng.integers(0, len(groups), len(groups))] if resample_groups else groups
        ix = np.concatenate([members[g] for g in pg])
        Cb = rng.multinomial(C[ix].sum(axis=1), C[ix] / C[ix].sum(axis=1, keepdims=True))
        draws[b] = stat(Cb, rng.integers(0, R, R))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi), len(groups), R, resample_groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--boot", type=int, default=BOOT)
    a = ap.parse_args()
    rows = pd.read_csv(a.run_dir / "rows.csv", low_memory=False)
    out = []
    for d in (3, 5, 7):
        dd = rows[rows.distance == d]
        for est in ESTIMATORS:
            for n in sorted(dd[dd.estimator == est].n_cal_shots.unique()):
                reps = sorted(dd[(dd.estimator == est) & (dd.n_cal_shots == n)].replicate.unique())
                names = [REF] + [f"{est}_N={int(n)}_rep={int(r)}" for r in reps]
                pt, lo, hi, ng, nr, rg = interval(a.run_dir, rows, d, names, stable_seed("fig4joint", d, est, int(n)), a.boot)
                out.append({"distance": d, "estimator": est, "n_cal": int(n), "rel_ler": pt, "ci_lo": lo, "ci_hi": hi,
                            "n_groups": ng, "n_replicates": nr, "groups_resampled": rg})
        pt, lo, hi, ng, nr, rg = interval(a.run_dir, rows, d, [REF, "rl_shipped"], stable_seed("fig4joint", d, "rl"), a.boot)
        out.append({"distance": d, "estimator": "rl_shipped", "n_cal": 0, "rel_ler": pt, "ci_lo": lo, "ci_hi": hi,
                    "n_groups": ng, "n_replicates": nr, "groups_resampled": rg})
    df = pd.DataFrame(out)
    df.to_csv(a.run_dir / "frontier_joint_ci.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
