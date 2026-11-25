"""Hardware sensitivity to kappa (post hoc, labelled): the fitting-procedure estimand at
the primary budget with the development-selected kappa, against the shipped SI1000 and
RL priors, with the joint bootstrap of ``willow_frontier_ci`` (groups at d = 3, 5;
calibration draws; shots jointly). The kappa = 5 result is recomputed alongside with
the same routine, so the two differ only in kappa.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .willow import stable_seed
from .willow_frontier_ci import interval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--boot", type=int, default=20000)
    a = ap.parse_args()
    rows = pd.read_csv(a.run_dir / "rows.csv", low_memory=False)
    out = []
    for label, tmpl in ((f"kappa={a.kappa:g}", f"pooled_shrink_kappa{a.kappa:g}_N={a.n_cal}_rep={{r}}"),
                        ("kappa=5", f"pooled_shrink_N={a.n_cal}_rep={{r}}")):
        fits = [tmpl.format(r=r) for r in range(5)]
        for ref in ("si1000_shipped", "rl_shipped"):
            for d in (3, 5, 7):
                pt, lo, hi, ng, nr, rg = interval(a.run_dir, rows, d, [ref] + fits,
                                                  stable_seed("willow_kappa", label, ref, d), a.boot, force_groups=True)
                out.append({"setting": label, "reference": ref, "distance": d, "rel": pt, "ci_lo": lo, "ci_hi": hi})
    df = pd.DataFrame(out)
    df.to_csv(a.run_dir / f"kappa{a.kappa:g}_sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.assign(rel=100 * df.rel, ci_lo=100 * df.ci_lo, ci_hi=100 * df.ci_hi).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
