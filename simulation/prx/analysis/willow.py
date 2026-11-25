"""Analysis of a Willow run: hardware shot-budget frontier, predeclared tests, and Lambda.

Aggregation across placements/bases uses a cluster bootstrap over (patch, basis)
groups; within a leaf the comparison is paired per shot, so we aggregate the
paired differences rather than the raw rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..results_io import load_error_vector
from ..results_io import load_error_vector
from ..stats import discordant_counts, exact_mcnemar, holm
from ..suppression import average_eps, fit_eps_groups, lambda_pairwise, lambda_regression

REF = "si1000_shipped"


def stable_seed(*parts) -> int:
    """Deterministic 31-bit seed from the string form of ``parts``.

    Python's built-in ``hash`` of a string is randomized per process
    (PYTHONHASHSEED), so seeds derived from it made the bootstrap intervals
    irreproducible between runs.
    """
    import hashlib
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def load(run_dir: Path) -> pd.DataFrame:
    # sorted, so that every bootstrap is independent of the order in which rows were written
    df = pd.read_csv(run_dir / "rows.csv").sort_values(["leaf_id", "prior_id", "decoder", "replicate"], kind="stable").reset_index(drop=True)
    df["patch_basis"] = df.patch + "/" + df.basis
    return df


def _paired_rel(df: pd.DataFrame, estimator: str, n_cal: int) -> pd.DataFrame:
    """Per-leaf paired difference vs the reference prior, for one estimator and budget."""
    ref = df[df.prior_id == REF].set_index("leaf_id")
    sub = df[(df.estimator == estimator) & (df.n_cal_shots == n_cal)]
    rows = []
    for leaf, g in sub.groupby("leaf_id"):
        if leaf not in ref.index:
            continue
        r = ref.loc[leaf]
        # average over replicates: sum errors and shots
        k = g.n_errors.sum()
        n = g.n_test_shots.sum()
        reps = len(g)
        rows.append({"leaf_id": leaf, "patch": g.patch.iloc[0], "basis": g.basis.iloc[0],
                     "patch_basis": g.patch_basis.iloc[0], "distance": int(g.distance.iloc[0]),
                     "rounds": int(g.rounds.iloc[0]), "reps": reps,
                     "k_est": k / reps, "n": n / reps, "k_ref": r.n_errors, "n_ref": r.n_test_shots,
                     "ler_est": k / n, "ler_ref": r.ler,
                     "b": g.mcnemar_b.mean(), "c": g.mcnemar_c.mean()})
    return pd.DataFrame(rows)


def cluster_bootstrap_rel(per_leaf: pd.DataFrame, n_boot: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap the pooled relative LER change over (patch, basis) clusters."""
    if per_leaf.empty:
        return np.nan, np.nan, np.nan
    groups = list(per_leaf.groupby("patch_basis"))
    k_est = np.array([g.k_est.sum() for _, g in groups])
    k_ref = np.array([g.k_ref.sum() for _, g in groups])
    rng = np.random.default_rng(seed)
    point = k_est.sum() / k_ref.sum() - 1
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    boot = k_est[idx].sum(axis=1) / k_ref[idx].sum(axis=1) - 1
    return float(point), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def pooled_mcnemar(per_leaf: pd.DataFrame) -> float:
    """Exact McNemar on summed discordances of ONE frozen pair of decoders.

    Valid only when b and c are genuine per-shot discordant counts of a single
    pair. It must not be applied to counts averaged over calibration replicates
    (see ``_paired_rel``); callers that average return NaN instead. Hardware
    inference for the paper is in ``analysis.willow_inference``.
    """
    if per_leaf.empty or per_leaf.b.isna().all():
        return float("nan")
    if "reps" in per_leaf.columns and (per_leaf.reps > 1).any():
        return float("nan")
    b = per_leaf.b.dropna().sum()
    c = per_leaf.c.dropna().sum()
    return exact_mcnemar(int(round(b)), int(round(c)))


def frontier_table(df: pd.DataFrame, estimators: List[str], distances=(3, 5, 7)) -> pd.DataFrame:
    out = []
    budgets = sorted(x for x in df.n_cal_shots.unique() if x and x > 0 and x != 50000 or x == 50000)
    for d in distances:
        dd = df[df.distance == d]
        for est in estimators:
            for n_cal in sorted(dd[dd.estimator == est].n_cal_shots.unique()):
                per_leaf = _paired_rel(dd, est, n_cal)
                if per_leaf.empty:
                    continue
                pt, lo, hi = cluster_bootstrap_rel(per_leaf, seed=stable_seed(d, est, n_cal))
                out.append({"distance": d, "estimator": est, "n_cal": int(n_cal), "n_leaves": len(per_leaf),
                            "rel_ler": pt, "ci_lo": lo, "ci_hi": hi, "mcnemar_p": pooled_mcnemar(per_leaf)})
    return pd.DataFrame(out)


def fixed_prior_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for d in (3, 5, 7):
        dd = df[df.distance == d]
        ref = dd[dd.prior_id == REF].set_index("leaf_id")
        for (prior, dec), g in dd[dd.estimator.isin(["fixed_prior", "google_shipped"])].groupby(["prior_id", "decoder"]):
            if prior == REF and dec == "pymatching_mwpm":
                continue
            per_leaf = []
            for leaf, gg in g.groupby("leaf_id"):
                if leaf not in ref.index:
                    continue
                r = ref.loc[leaf]
                per_leaf.append({"patch_basis": gg.patch_basis.iloc[0], "k_est": gg.n_errors.iloc[0],
                                 "k_ref": r.n_errors, "b": gg.mcnemar_b.iloc[0], "c": gg.mcnemar_c.iloc[0]})
            pl = pd.DataFrame(per_leaf)
            if pl.empty:
                continue
            pt, lo, hi = cluster_bootstrap_rel(pl.assign(patch_basis=pl.patch_basis), seed=stable_seed(d, prior, dec))
            out.append({"distance": d, "prior": prior, "decoder": dec, "n_leaves": len(pl), "rel_ler": pt,
                        "ci_lo": lo, "ci_hi": hi, "mcnemar_p": pooled_mcnemar(pl)})
    return pd.DataFrame(out)


def lambda_table(df: pd.DataFrame, selections: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """selections: name -> rows to use (one row per leaf)."""
    out = []
    for name, sel in selections.items():
        rows = [(f"{r.patch}/{r.basis}", int(r.distance), int(r.rounds), int(r.n_errors), int(r.n_test_shots))
                for r in sel.itertuples()]
        fits = fit_eps_groups(rows)
        eps = average_eps(fits)
        lam, dlam = lambda_regression(eps)
        pair = lambda_pairwise(eps)
        rec = {"selection": name, "Lambda_fit": lam, "dLambda": dlam}
        for d, (e, se, n) in sorted(eps.items()):
            rec[f"eps_{d}"] = e
            rec[f"n_{d}"] = n
        rec.update(pair)
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n-primary", type=int, default=3000)
    a = ap.parse_args()
    run_dir = a.run_dir
    df = load(a.run_dir)
    pd.set_option("display.width", 250)
    ests = ["pooled", "pooled_shrink", "uniform_search", "class_search", "pooled_shrink+class"]

    fx = fixed_prior_table(df)
    fr = frontier_table(df, ests)
    fx.to_csv(a.run_dir / "fixed_priors.csv", index=False)
    fr.to_csv(a.run_dir / "frontier.csv", index=False)

    # ---- SUPERSEDED: the predeclared tests below mixed estimands (a five-replicate
    # mean effect with a replicate-0 p-value). They are kept only to reproduce the
    # earlier numbers; the paper uses analysis.willow_inference, which separates the
    # fitting procedure from a frozen fit.
    # ---- predeclared tests at the primary budget
    tests = []
    # (run_dir is needed to load the stored per-shot vectors for H2)
    for d in (3, 5, 7):
        dd = df[df.distance == d]
        # H1: pooled_shrink vs si1000 (label-free)
        pl = _paired_rel(dd, "pooled_shrink", a.n_primary)
        pt, lo, hi = cluster_bootstrap_rel(pl, seed=1)
        tests.append({"test": "H1 pooled_shrink vs si1000", "distance": d, "rel": pt, "lo": lo, "hi": hi,
                      "p": pooled_mcnemar(pl)})
        # H2: pooled_shrink vs the RL prior (both under plain MWPM)
        rl = dd[dd.prior_id == "rl_shipped"].set_index("leaf_id")
        sub = dd[(dd.estimator == "pooled_shrink") & (dd.n_cal_shots == a.n_primary)]
        rows = []
        for leaf, g in sub.groupby("leaf_id"):
            if leaf not in rl.index:
                continue
            # Paired discordant counts from the stored per-shot outcome vectors:
            # without these the headline comparison has no significance test.
            b = c = np.nan
            ev_a = g.error_vector_path.dropna()
            ev_b = rl.loc[leaf].error_vector_path
            if len(ev_a) and isinstance(ev_b, str) and ev_b:
                try:
                    ea = load_error_vector(run_dir, ev_a.iloc[0])
                    eb = load_error_vector(run_dir, ev_b)
                    b, c = discordant_counts(ea, eb)
                except FileNotFoundError:
                    pass
            rows.append({"patch_basis": g.patch_basis.iloc[0], "k_est": g.n_errors.mean(),
                         "k_ref": rl.loc[leaf].n_errors, "b": b, "c": c})
        pl2 = pd.DataFrame(rows)
        pt, lo, hi = cluster_bootstrap_rel(pl2, seed=2)
        tests.append({"test": "H2 pooled_shrink vs rl_shipped", "distance": d, "rel": pt, "lo": lo, "hi": hi,
                      "p": pooled_mcnemar(pl2) if pl2.b.notna().any() else np.nan})
        # H3: uniform scale equivalence
        pl3 = _paired_rel(dd, "uniform_search", a.n_primary)
        pt, lo, hi = cluster_bootstrap_rel(pl3, seed=3)
        tests.append({"test": "H3 uniform_search vs si1000 (equivalence)", "distance": d, "rel": pt, "lo": lo, "hi": hi,
                      "p": pooled_mcnemar(pl3)})
    tt = pd.DataFrame(tests)
    ps = tt.p.dropna()
    tt.loc[ps.index, "p_holm"] = holm(ps.tolist())
    tt.to_csv(a.run_dir / "predeclared_tests.csv", index=False)

    # ---- Lambda, like-for-like on the common test rounds
    sels = {}
    sels["pymatching+si1000"] = df[(df.prior_id == REF)]
    sels["pymatching+rl"] = df[(df.prior_id == "rl_shipped")]
    best = df[(df.estimator == "pooled_shrink") & (df.n_cal_shots == a.n_primary)].groupby("leaf_id").first().reset_index()
    sels[f"pymatching+pooled_shrink_N={a.n_primary}"] = best
    for pw in df[df.estimator == "google_shipped"].decoder.unique():
        sels["google:" + pw] = df[(df.estimator == "google_shipped") & (df.decoder == pw)]
    lt = lambda_table(df, sels)
    lt.to_csv(a.run_dir / "lambda.csv", index=False)

    print("=== fixed priors (relative to shipped SI1000 DEM under plain PyMatching) ===")
    print(fx.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== hardware frontier: relative LER vs calibration shots ===")
    print(fr.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== predeclared tests ===")
    print(tt.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== Lambda (Google convention fits on the common test rounds) ===")
    print(lt.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
