"""Bootstrap uncertainty of Lambda for every decoder in Fig. 5, from error counts.

The "dLambda" of ``lambda.csv`` is the model-based standard error of a weighted
regression through three points, with weights from the spread between
patch-basis groups; at distance seven that spread rests on two groups. This module
instead bootstraps the whole fitting procedure (per-group epsilon fits, averaging
over groups, regression on distance), with the scheme of ``lambda_paired``:

  shots+groups   patch-basis groups resampled within each distance, and shots
                 resampled within each leaf.

A marginal uncertainty needs no pairing between decoders, so shots are resampled
binomially from each leaf's error count. That also covers Google's shipped
decoders, whose per-shot predictions are not stored. ``lambda_paired`` computed the
same marginal SD for three priors from per-shot vectors; the two are compared.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..suppression import average_eps, fit_eps_groups, lambda_regression
from .willow import REF, stable_seed

BOOT = 10000


def selections(df: pd.DataFrame, n_primary: int = 3000) -> dict:
    """The same selections as ``willow.main`` uses for lambda.csv."""
    sels = {"pymatching+si1000": df[df.prior_id == REF],
            "pymatching+rl": df[df.prior_id == "rl_shipped"]}
    best = df[(df.estimator == "pooled_shrink") & (df.n_cal_shots == n_primary) & (df.replicate == 0)]
    sels[f"pymatching+pooled_shrink_N={n_primary}"] = best
    for pw in df[df.estimator == "google_shipped"].decoder.unique():
        sels["google:" + pw] = df[(df.estimator == "google_shipped") & (df.decoder == pw)]
    return sels


def lam(sub: pd.DataFrame, k: np.ndarray, labels: np.ndarray) -> float:
    rows = [(g, int(d), int(r), int(kk), int(n)) for g, d, r, kk, n in
            zip(labels, sub.distance, sub.rounds, k, sub.n_test_shots)]
    return lambda_regression(average_eps(fit_eps_groups(rows)))[0]


def bootstrap(sel: pd.DataFrame, seed: int, boot: int = BOOT) -> np.ndarray:
    sel = sel.reset_index(drop=True)
    group = (sel.patch + "/" + sel.basis).values
    rng = np.random.default_rng(seed)
    out = np.full(boot, np.nan)
    for b in range(boot):
        idx_parts, lab_parts = [], []
        for d in sorted(sel.distance.unique()):
            gs = sorted(set(group[sel.distance.values == d]))
            for c, gi in enumerate(rng.integers(0, len(gs), len(gs))):
                members = np.flatnonzero(group == gs[gi])
                idx_parts.append(members)
                lab_parts.append(np.array([f"{gs[gi]}#{c}"] * len(members)))
        idx = np.concatenate(idx_parts)
        sub = sel.iloc[idx]
        k = rng.binomial(sub.n_test_shots.values.astype(int), (sub.n_errors / sub.n_test_shots).values)
        out[b] = lam(sub, k, np.concatenate(lab_parts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    a = ap.parse_args()
    df = pd.read_csv(a.run_dir / "rows.csv").sort_values(["leaf_id", "prior_id", "decoder", "replicate"], kind="stable")
    rows = []
    for name, sel in selections(df).items():
        point = lam(sel, sel.n_errors.values, (sel.patch + "/" + sel.basis).values)
        draws = bootstrap(sel, stable_seed("lambda_marginal", name))
        ok = np.isfinite(draws)
        rows.append({"selection": name, "Lambda": point, "sd_Lambda": float(np.std(draws[ok])),
                     "sd_lnLambda": float(np.std(np.log(draws[ok]))),
                     "lo95": float(np.quantile(draws[ok], 0.025)), "hi95": float(np.quantile(draws[ok], 0.975)),
                     "n_boot": int(ok.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(a.run_dir / "lambda_marginal.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
