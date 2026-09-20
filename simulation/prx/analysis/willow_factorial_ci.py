"""Intervals for the prior-by-decoder factorial on hardware (Fig. 6).

All six cells (three priors x two decoders) decode identical shots, and every
cell's per-shot error vector is stored. Each shot of each leaf is classified by its
joint failure pattern over the six cells; a bootstrap draw resamples shots by a
multinomial over those patterns, so all cells move together and every comparison
stays paired. Patch-basis groups are resampled within distance where there are at
least ``MIN_GROUPS`` of them (distances three and five); distance seven, one patch
in two bases, is conditional on that patch, as in Fig. 4.

The statistic for a cell is its error count pooled over the leaves of a distance,
relative to a reference cell, minus one: the same quantity the figure plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..results_io import load_error_vector
from .willow import stable_seed

MIN_GROUPS = 5
BOOT = 4000
PRIORS = ["si1000_shipped", "rl_shipped", "pooled_shrink_N=10000"]
DECODERS = [("pymatching_mwpm", "mwpm"), ("pymatching_correlated", "corr")]
CELLS = [(p, d) for d, _ in DECODERS for p in PRIORS]


def load(run_dir: Path):
    rows = pd.read_csv(run_dir / "rows.csv")
    leaves = rows[["leaf_id", "patch", "basis", "distance"]].drop_duplicates().reset_index(drop=True)
    counts = []
    for lf in leaves.itertuples():
        tag = lf.leaf_id.replace("/", "_")
        code = None
        for j, (p, d) in enumerate(CELLS):
            v = load_error_vector(run_dir, f"errors/{tag}__{p}__{dict(DECODERS)[d]}.npz").astype(np.int64)
            code = v << j if code is None else code | (v << j)
        counts.append(np.bincount(code, minlength=1 << len(CELLS)))
    fails = np.array([[(pat >> j) & 1 for j in range(len(CELLS))] for pat in range(1 << len(CELLS))])
    # cross-check against the stored counts
    err = np.stack(counts) @ fails
    for j, (p, d) in enumerate(CELLS):
        stored = rows[(rows.prior_id == p) & (rows.decoder == d)].set_index("leaf_id").n_errors
        assert (stored.reindex(leaves.leaf_id).values == err[:, j]).all(), (p, d)
    return leaves, np.stack(counts), fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    a = ap.parse_args()
    leaves, C, F = load(a.run_dir)
    group = (leaves.patch + "/" + leaves.basis).values
    idx = {c: j for j, c in enumerate(CELLS)}
    # comparisons: every cell against SI1000 + MWPM (the bars), plus the caption's contrasts
    comps = [(c, ("si1000_shipped", "pymatching_mwpm")) for c in CELLS if c != ("si1000_shipped", "pymatching_mwpm")]
    comps += [(("rl_shipped", d), ("si1000_shipped", d)) for d, _ in DECODERS]
    comps += [(("pooled_shrink_N=10000", d), ("si1000_shipped", d)) for d, _ in DECODERS]
    comps += [(("pooled_shrink_N=10000", d), ("rl_shipped", d)) for d, _ in DECODERS]
    out = []
    for dist in sorted(leaves.distance.unique()):
        md = np.flatnonzero(leaves.distance.values == dist)
        gs = sorted(set(group[md]))
        members = {g: md[group[md] == g] for g in gs}
        err = C[md] @ F
        point = {c: err[:, idx[c[0]]].sum() / err[:, idx[c[1]]].sum() - 1 for c in comps}
        rng = np.random.default_rng(stable_seed("factorial", int(dist)))
        draws = {c: np.empty(BOOT) for c in comps}
        for b in range(BOOT):
            pick = [gs[i] for i in rng.integers(0, len(gs), len(gs))] if len(gs) >= MIN_GROUPS else gs
            ix = np.concatenate([members[g] for g in pick])
            Cb = rng.multinomial(C[ix].sum(axis=1), C[ix] / C[ix].sum(axis=1, keepdims=True))
            eb = (Cb @ F).sum(axis=0)
            for c in comps:
                draws[c][b] = eb[idx[c[0]]] / eb[idx[c[1]]] - 1
        for c in comps:
            lo, hi = np.quantile(draws[c], [0.025, 0.975])
            out.append({"distance": int(dist), "prior": c[0][0], "decoder": c[0][1], "ref_prior": c[1][0],
                        "ref_decoder": c[1][1], "rel": float(point[c]), "ci_lo": float(lo), "ci_hi": float(hi),
                        "n_groups": len(gs), "groups_resampled": len(gs) >= MIN_GROUPS})
    df = pd.DataFrame(out)
    df.to_csv(a.run_dir / "factorial_ci.csv", index=False)
    pd.set_option("display.width", 220)
    f = lambda v: f"{v:.4f}"
    print(df.to_string(index=False, float_format=f))


if __name__ == "__main__":
    main()
