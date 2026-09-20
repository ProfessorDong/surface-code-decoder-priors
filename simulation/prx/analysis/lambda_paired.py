"""Paired uncertainty for differences in the error-suppression factor Lambda.

Two priors decoding the same shots produce correlated Lambda fits, so the
uncertainty of their difference is not the quadrature sum of the two marginal
uncertainties:

    Var(ln L_A - ln L_B) = Var(ln L_A) + Var(ln L_B) - 2 Cov(ln L_A, ln L_B).

This module bootstraps the complete fitting procedure (per-group epsilon fits,
averaging over groups, weighted regression on distance) with the pairing
preserved. Within each test leaf the same resampled shots are used for every
prior: each shot is classified by the joint pattern of which priors fail on it,
and a multinomial draw over those patterns resamples all priors at once.

Two resampling schemes are reported, because they answer different questions.
  shots          conditional on the observed patches: shot noise only.
  shots+groups   patch-basis groups are also resampled with replacement.
At distance seven the release has one patch and two bases, so the second scheme
cannot represent a population of distance-seven devices; it is reported for the
fixed device only.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..results_io import load_error_vector
from ..suppression import average_eps, fit_eps_groups, lambda_regression
from .willow import stable_seed


def load_patterns(run_dir: Path, priors: List[str]) -> pd.DataFrame:
    """One row per leaf: group, distance, rounds, and counts of each joint failure pattern."""
    rows = pd.read_csv(run_dir / "rows.csv")
    leaves = rows[["leaf_id", "patch", "basis", "distance", "rounds"]].drop_duplicates().sort_values("leaf_id")
    out = []
    for lf in leaves.itertuples():
        tag = lf.leaf_id.replace("/", "_")
        vecs = [load_error_vector(run_dir, f"errors/{tag}__{p}.npz") for p in priors]
        code = np.zeros(vecs[0].size, dtype=np.int64)
        for j, v in enumerate(vecs):
            code |= v.astype(np.int64) << j
        counts = np.bincount(code, minlength=1 << len(priors))
        out.append({"leaf_id": lf.leaf_id, "group": f"{lf.patch}/{lf.basis}", "distance": int(lf.distance),
                    "rounds": int(lf.rounds), "n": int(vecs[0].size), "counts": counts})
    return pd.DataFrame(out)


def lambdas(leaves: pd.DataFrame, errors: np.ndarray, group_label: np.ndarray) -> np.ndarray:
    """Lambda for each prior, given per-leaf error counts errors[leaf, prior]."""
    res = []
    for j in range(errors.shape[1]):
        rows = [(g, int(d), int(r), int(k), int(n)) for g, d, r, k, n in
                zip(group_label, leaves.distance, leaves.rounds, errors[:, j], leaves.n)]
        lam, _ = lambda_regression(average_eps(fit_eps_groups(rows)))
        res.append(lam)
    return np.array(res)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--priors", nargs="+",
                    default=["si1000_shipped", "rl_shipped"] + [f"pooled_shrink_N=3000_rep={r}" for r in range(5)])
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    P = a.priors
    L = load_patterns(a.run_dir, P)
    K = 1 << len(P)
    fails = np.array([[(pat >> j) & 1 for j in range(len(P))] for pat in range(K)])   # [K, priors]
    C = np.stack(L.counts.values)                                                      # [leaves, K]
    point = lambdas(L, C @ fails, L.group.values)

    groups = L.group.unique()
    by_group = {g: np.flatnonzero(L.group.values == g) for g in groups}
    pairs = list(itertools.combinations(range(len(P)), 2))
    records = []
    for scheme in ("shots", "shots+groups"):
        rng = np.random.default_rng(stable_seed("lambda_paired", scheme, *P))
        draws = np.full((a.boot, len(P)), np.nan)
        for b in range(a.boot):
            if scheme == "shots":
                idx = np.arange(len(L)); labels = L.group.values
            else:
                # resample patch-basis groups within each distance, preserving the distance design
                idx_parts, lab_parts = [], []
                for d in sorted(L.distance.unique()):
                    gs = [g for g in groups if int(L.distance.values[by_group[g][0]]) == d]
                    pick = rng.choice(len(gs), size=len(gs), replace=True)
                    for c, gi in enumerate(pick):
                        members = by_group[gs[gi]]
                        idx_parts.append(members)
                        lab_parts.append(np.array([f"{gs[gi]}#{c}"] * len(members)))
                idx = np.concatenate(idx_parts); labels = np.concatenate(lab_parts)
            sub = L.iloc[idx]
            Cb = np.stack([rng.multinomial(int(n), c / c.sum()) for n, c in zip(sub.n.values, C[idx])])
            draws[b] = lambdas(sub, Cb @ fails, labels)
        for i, j in pairs:
            dl = np.log(draws[:, i]) - np.log(draws[:, j])
            dl = dl[np.isfinite(dl)]
            pt = np.log(point[i]) - np.log(point[j])
            lo, hi = np.quantile(dl, [0.025, 0.975])
            records.append({"scheme": scheme, "prior_A": P[i], "prior_B": P[j],
                            "Lambda_A": point[i], "Lambda_B": point[j], "dlnLambda": pt,
                            "dlnLambda_lo": lo, "dlnLambda_hi": hi,
                            "ratio": float(np.exp(pt)), "ratio_lo": float(np.exp(lo)), "ratio_hi": float(np.exp(hi)),
                            "sd_marginal_A": float(np.nanstd(np.log(draws[:, i]))),
                            "sd_marginal_B": float(np.nanstd(np.log(draws[:, j]))),
                            "sd_difference": float(np.std(dl)), "n_boot": int(dl.size)})
    df = pd.DataFrame(records)
    out = a.run_dir / "lambda_paired.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(df[["scheme", "prior_A", "prior_B", "Lambda_A", "Lambda_B", "ratio", "ratio_lo", "ratio_hi",
              "sd_marginal_A", "sd_difference"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("wrote", out)


if __name__ == "__main__":
    main()
