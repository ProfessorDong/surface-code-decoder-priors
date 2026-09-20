"""Gate 0 analysis of a prior_sweep run: headroom of prior deformation over the physical prior.

Usage: python -m simulation.prx.analysis.gate0 <run_dir> [--out GATE0.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..stats import holm


def load(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "rows.csv")
    df["cond"] = df["noise_family"] + "|d" + df["distance"].astype(str) + "|T" + df["rounds"].astype(str) + "|p" + df["p_true"].map(lambda x: f"{x:g}")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cond, g in df.groupby("cond", sort=False):
        phys = g[g.prior_id == "oracle_physical"]
        if phys.empty:
            continue
        phys = phys.iloc[0]
        rec = {"cond": cond, "family": phys.noise_family, "d": int(phys.distance), "T": int(phys.rounds),
               "p": float(phys.p_true), "n": int(phys.n_test_shots), "ler_phys": float(phys.ler),
               "errors_phys": int(phys.n_errors)}
        def best(prefix):
            h = g[g.prior_id.str.startswith(prefix)]
            if h.empty:
                return None
            row = h.loc[h.n_errors.idxmin()]
            return row
        for name, prefix in (("stale", "stale_pcal="), ("uniform", "uniform_s="), ("class", "class_search")):
            row = best(prefix)
            if row is None:
                continue
            rec[f"{name}_id"] = row.prior_id
            rec[f"{name}_gain"] = 1.0 - row.ler / phys.ler if phys.ler > 0 else np.nan   # relative LER reduction
            rec[f"{name}_p"] = float(row.mcnemar_p_exact) if row.mcnemar_p_exact == row.mcnemar_p_exact else np.nan
            rec[f"{name}_b"] = row.mcnemar_b
            rec[f"{name}_c"] = row.mcnemar_c
            if name == "class":
                rec["scales"] = (row.scale_space, row.scale_time, row.scale_diag, row.scale_boundary)
        d1o = g[g.prior_id.str.startswith("d1_offset_only")]
        d1s = g[g.prior_id.str.startswith("d1_structure_only")]
        st5 = g[g.prior_id == "stale_pcal=0.005"]
        if not d1o.empty and not st5.empty:
            rec["d1_stale005_gain"] = 1 - st5.iloc[0].ler / phys.ler
            rec["d1_offset_gain"] = 1 - d1o.iloc[0].ler / phys.ler
            rec["d1_structure_gain"] = 1 - d1s.iloc[0].ler / phys.ler
        out.append(rec)
    return pd.DataFrame(out)


def threshold_note(summary: pd.DataFrame) -> str:
    lines = []
    for fam, g in summary.groupby("family"):
        for T_mode in ("T=d", "T=3d"):
            sub = g[(g["T"] == g["d"]) if T_mode == "T=d" else (g["T"] == 3 * g["d"])]
            if sub.empty:
                continue
            piv = sub.pivot_table(index="p", columns="d", values="ler_phys")
            lines.append(f"\n{fam} ({T_mode}): physical-prior LER by p (rows) and d (cols)\n" + piv.to_string(float_format=lambda x: f"{x:.3e}"))
    return "\n".join(lines)


def report(run_dir: Path, out: Path) -> None:
    df = load(run_dir)
    s = summarize(df)
    # Holm over the class-search tests (the predeclared headroom test)
    mask = s["class_p"].notna()
    s.loc[mask, "class_p_holm"] = holm(s.loc[mask, "class_p"].tolist())
    cols = ["cond", "n", "ler_phys", "stale_gain", "stale_p", "uniform_gain", "uniform_p", "class_gain", "class_p", "class_p_holm", "scales"]
    cols = [c for c in cols if c in s]
    txt = ["# Gate 0 report: prior_sweep run " + run_dir.name, "",
           "Gains are relative LER reductions versus the physical prior on the same 4e6-shot test set "
           "(positive = the deformed prior is better). p-values: exact McNemar; Holm over the class-search family.", "",
           s[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"), "",
           "## Threshold location (physical prior)", threshold_note(s), ""]
    d1 = s.dropna(subset=["d1_stale005_gain"]) if "d1_stale005_gain" in s else pd.DataFrame()
    if not d1.empty:
        txt += ["## D1 decomposition (stale p_cal=0.005 vs physical): offset-only vs structure-only gains", "",
                d1[["cond", "d1_stale005_gain", "d1_offset_gain", "d1_structure_gain"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"), ""]
    # Gate decision helper: below-threshold cells at d>=5 with class gain
    txt += ["## Gate 0 decision inputs", ""]
    for _, r in s.iterrows():
        if r.d >= 5 and "class_gain" in s and r.class_gain == r.class_gain:
            txt.append(f"- {r.cond}: class-search gain {100*r.class_gain:.2f}% (Holm p={r.get('class_p_holm', float('nan')):.2g}), "
                       f"best stale gain {100*r.stale_gain:.2f}%, uniform gain {100*r.uniform_gain:.2f}%")
    out.write_text("\n".join(txt))
    print("\n".join(txt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    report(a.run_dir, a.out or (a.run_dir / "GATE0.md"))


if __name__ == "__main__":
    main()
