"""Hardware inference with the estimand fixed before the numbers are computed.

The earlier analysis attached p-values to the wrong quantities: discordant counts
were averaged over calibration replicates and rounded before an exact McNemar
test, and the headline comparison against the RL prior paired a five-replicate
mean effect with a replicate-0 p-value. This module reports two estimands and
never mixes them.

A. A frozen calibration fit. Each calibration replicate is one fitted prior,
   evaluated on fixed test shots. For a frozen fit the exact McNemar test on the
   summed per-shot discordances is a valid conditional test: shots are
   independent within and across leaves, and under the null each discordant shot
   is equally likely to favour either prior. Replicate 0 is the predeclared
   primary fit (the first contiguous calibration prefix); the other four are
   reported alongside it.

B. The fitting procedure. The quantity of interest is the expected effect over
   calibration draws. Its uncertainty is estimated by a bootstrap that resamples
   patch-basis groups within each distance, the five calibration replicates, and
   the test shots jointly across every prior (so shared test shots are never
   counted as independent). No p-value is attached to B.

Predeclared hypotheses (plan of 17 Sep 2026, before the hardware run):
   H1  pooled+shrinkage vs the shipped SI1000 prior, N = 3000
   H2  pooled+shrinkage vs the shipped RL prior,    N = 3000
   H3  uniform rescaling equivalent to SI1000, margin 2% (two one-sided tests)
   Holm correction at family-wise alpha = 0.01.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from ..results_io import load_error_vector
from ..stats import exact_mcnemar, holm
from .willow import stable_seed

MARGIN = 0.02
ALPHA = 0.01


def leaf_vectors(run_dir: Path, leaf_id: str, names: List[str]) -> List[np.ndarray]:
    tag = leaf_id.replace("/", "_")
    return [load_error_vector(run_dir, f"errors/{tag}__{n}.npz") for n in names]


def joint_counts(vecs: List[np.ndarray]) -> np.ndarray:
    code = np.zeros(vecs[0].size, dtype=np.int64)
    for j, v in enumerate(vecs):
        code |= v.astype(np.int64) << j
    return np.bincount(code, minlength=1 << len(vecs))


def build(run_dir: Path, n_cal: int, reps: int) -> Dict[str, object]:
    rows = pd.read_csv(run_dir / "rows.csv")
    leaves = (rows[["leaf_id", "patch", "basis", "distance", "rounds"]].drop_duplicates()
              .sort_values("leaf_id").reset_index(drop=True))          # order-independent
    names = [f"pooled_shrink_N={n_cal}_rep={r}" for r in range(reps)] + \
            ["si1000_shipped", "rl_shipped", f"uniform_search_N={n_cal}_rep=0"]
    C = np.stack([joint_counts(leaf_vectors(run_dir, lf, names)) for lf in leaves.leaf_id])
    K = 1 << len(names)
    fails = np.array([[(k >> j) & 1 for j in range(len(names))] for k in range(K)])
    return {"leaves": leaves, "names": names, "C": C, "fails": fails}


def frozen_tests(B: Dict[str, object], reps: int) -> pd.DataFrame:
    L, C, F, names = B["leaves"], B["C"], B["fails"], B["names"]
    idx = {n: i for i, n in enumerate(names)}
    err = C @ F                                              # [leaves, priors] error counts
    out = []

    def disc(a: int, b: int, mask):
        only_a = (F[:, a] == 1) & (F[:, b] == 0)
        only_b = (F[:, a] == 0) & (F[:, b] == 1)
        return int(C[mask][:, only_a].sum()), int(C[mask][:, only_b].sum())

    for d in (3, 5, 7):
        m = (L.distance == d).values
        si, rl = idx["si1000_shipped"], idx["rl_shipped"]
        for r in range(reps):
            a = idx[f"pooled_shrink_N=3000_rep={r}"]
            for ref, lab in ((si, "SI1000"), (rl, "RL")):
                b_, c_ = disc(a, ref, m)
                out.append({"estimand": "A frozen fit", "distance": d, "replicate": r,
                            "comparison": f"pooled+shrinkage vs {lab}",
                            "rel": err[m, a].sum() / err[m, ref].sum() - 1,
                            "b": b_, "c": c_, "p_mcnemar": exact_mcnemar(b_, c_)})
        # d = 7: one physical patch, two bases, reported separately
        if d == 7:
            for basis in ("X", "Z"):
                mb = m & (L.basis == basis).values
                a = idx["pooled_shrink_N=3000_rep=0"]
                for ref, lab in ((si, "SI1000"), (rl, "RL")):
                    b_, c_ = disc(a, ref, mb)
                    out.append({"estimand": f"A frozen fit, basis {basis}", "distance": d, "replicate": 0,
                                "comparison": f"pooled+shrinkage vs {lab}",
                                "rel": err[mb, a].sum() / err[mb, ref].sum() - 1,
                                "b": b_, "c": c_, "p_mcnemar": exact_mcnemar(b_, c_)})
    return pd.DataFrame(out)


def paired_ratio_se(Cd: np.ndarray, fa: np.ndarray, fb: np.ndarray) -> Tuple[float, float]:
    """Delta-method SE of R = K_a / K_b - 1 for pooled failure counts on shared shots.

    Cd holds, per leaf, the counts of every joint failure pattern; fa and fb are
    the failure indicators of decoders a and b per pattern. Shots are independent
    within a leaf and leaves are independent (stratified), so
        Var K_a = sum_l n_l q_a (1 - q_a),  Cov(K_a, K_b) = sum_l n_l (q_ab - q_a q_b),
    and Var(K_a / K_b) follows by the delta method. Returns (R, SE).
    """
    n = Cd.sum(axis=1).astype(float)
    ka_l, kb_l, kab_l = Cd @ fa, Cd @ fb, Cd @ (fa * fb)
    qa, qb, qab = ka_l / n, kb_l / n, kab_l / n
    va, vb = (n * qa * (1 - qa)).sum(), (n * qb * (1 - qb)).sum()
    cov = (n * (qab - qa * qb)).sum()
    Ka, Kb = ka_l.sum(), kb_l.sum()
    ratio = Ka / Kb
    var = ratio ** 2 * (va / Ka ** 2 + vb / Kb ** 2 - 2 * cov / (Ka * Kb))
    return float(ratio - 1), float(np.sqrt(max(var, 0.0)))


def h3_tost(B: Dict[str, object], boot: int) -> pd.DataFrame:
    """Equivalence of uniform rescaling to SI1000 for the frozen fit: paired TOST.

    Two one-sided z-tests of H0: rel <= -margin and H0: rel >= +margin, with the
    paired, leaf-stratified delta-method SE of ``paired_ratio_se``. With ~1e5 to
    ~1e6 test shots per distance the normal approximation is adequate; a percentile
    bootstrap over shots (joint failure patterns) gives the 98% interval as a check,
    and equivalence at alpha = 0.01 per test is equivalent to that interval lying
    inside the margin. (An earlier version reported bootstrap tail fractions floored
    at 1/(B+1); those are not calibrated p-values and are no longer used.)
    """
    from scipy.stats import norm
    L, C, F, names = B["leaves"], B["C"], B["fails"], B["names"]
    u, si = names.index("uniform_search_N=3000_rep=0"), names.index("si1000_shipped")
    out = []
    for d in (3, 5, 7):
        m = (L.distance == d).values
        Cd = C[m]
        rel, se = paired_ratio_se(Cd, F[:, u].astype(float), F[:, si].astype(float))
        z_lo, z_hi = (rel + MARGIN) / se, (MARGIN - rel) / se
        p_lo, p_hi = float(norm.sf(z_lo)), float(norm.sf(z_hi))
        rng = np.random.default_rng(stable_seed("h3", d))
        draws = np.empty(boot)
        for b in range(boot):
            Cb = rng.multinomial(Cd.sum(axis=1), Cd / Cd.sum(axis=1, keepdims=True))
            draws[b] = (Cb @ F[:, u]).sum() / (Cb @ F[:, si]).sum() - 1
        lo, hi = np.quantile(draws, [ALPHA, 1 - ALPHA])
        zq = norm.ppf(1 - ALPHA)
        out.append({"distance": d, "rel": rel, "se": se, "wald98_lo": rel - zq * se, "wald98_hi": rel + zq * se,
                    "ci98_lo": lo, "ci98_hi": hi, "boot_sd": float(draws.std()),
                    "z_min": float(min(z_lo, z_hi)), "p_tost": max(p_lo, p_hi), "margin": MARGIN})
    return pd.DataFrame(out)


def procedure_effect(B: Dict[str, object], reps: int, boot: int) -> pd.DataFrame:
    """Expected effect over calibration draws: bootstrap groups, replicates and shots jointly."""
    L, C, F, names = B["leaves"], B["C"], B["fails"], B["names"]
    R = [names.index(f"pooled_shrink_N=3000_rep={r}") for r in range(reps)]
    refs = {"SI1000": names.index("si1000_shipped"), "RL": names.index("rl_shipped")}
    out = []
    for d in (3, 5, 7):
        md = (L.distance == d).values
        groups = sorted(set(L.patch[md] + "/" + L.basis[md]))
        gidx = {g: np.flatnonzero(md & ((L.patch + "/" + L.basis) == g).values) for g in groups}
        err = C @ F
        for lab, ref in refs.items():
            per_rep = [err[md, a].sum() / err[md, ref].sum() - 1 for a in R]
            rng = np.random.default_rng(stable_seed("procedure", d, lab))
            draws = np.empty(boot)
            for b in range(boot):
                pick_g = rng.choice(len(groups), size=len(groups), replace=True)
                ix = np.concatenate([gidx[groups[g]] for g in pick_g])
                Cb = np.stack([rng.multinomial(int(c.sum()), c / c.sum()) for c in C[ix]])
                eb = Cb @ F
                pick_r = rng.choice(reps, size=reps, replace=True)
                draws[b] = np.mean([eb[:, R[r]].sum() / eb[:, ref].sum() - 1 for r in pick_r])
            lo, hi = np.quantile(draws, [0.025, 0.975])
            out.append({"distance": d, "comparison": f"pooled+shrinkage vs {lab}",
                        "mean_over_replicates": float(np.mean(per_rep)),
                        "min_replicate": float(np.min(per_rep)), "max_replicate": float(np.max(per_rep)),
                        "ci95_lo": lo, "ci95_hi": hi, "n_groups": len(groups), "n_replicates": reps})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--boot", type=int, default=20000)
    a = ap.parse_args()
    B = build(a.run_dir, a.n_cal, a.replicates)
    A = frozen_tests(B, a.replicates)
    H3 = h3_tost(B, a.boot)
    P = procedure_effect(B, a.replicates, a.boot)

    # predeclared family: H1, H2 on the primary frozen fit (replicate 0), H3 by TOST
    fam = []
    for d in (3, 5, 7):
        for lab, hname in (("SI1000", "H1"), ("RL", "H2")):
            r = A[(A.estimand == "A frozen fit") & (A.distance == d) & (A.replicate == 0)
                  & (A.comparison == f"pooled+shrinkage vs {lab}")].iloc[0]
            fam.append({"test": hname, "distance": d, "rel": r.rel, "p": r.p_mcnemar})
        h = H3[H3.distance == d].iloc[0]
        fam.append({"test": "H3 (TOST, margin 2%)", "distance": d, "rel": h.rel, "p": h.p_tost})
    fam = pd.DataFrame(fam)
    fam["p_holm"] = holm(fam.p.tolist())
    fam["reject_at_0.01"] = fam.p_holm < ALPHA

    A.to_csv(a.run_dir / "inference_frozen_fits.csv", index=False)
    H3.to_csv(a.run_dir / "inference_h3_tost.csv", index=False)
    P.to_csv(a.run_dir / "inference_procedure.csv", index=False)
    fam.to_csv(a.run_dir / "inference_predeclared.csv", index=False)
    pd.set_option("display.width", 220)
    f = lambda v: f"{v:.4g}"
    print("=== predeclared family (primary frozen fit, Holm at 0.01) ===")
    print(fam.to_string(index=False, float_format=f))
    print("\n=== A: every frozen fit ===")
    print(A.to_string(index=False, float_format=f))
    print("\n=== H3 equivalence detail ===")
    print(H3.to_string(index=False, float_format=f))
    print("\n=== B: fitting procedure (no p-values) ===")
    print(P.to_string(index=False, float_format=f))


if __name__ == "__main__":
    main()
