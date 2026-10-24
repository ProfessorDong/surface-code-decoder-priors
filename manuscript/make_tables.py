"""Generate the table bodies of the manuscript from the result files.

Numbers in these tables are never typed by hand. Each table body is written to
manuscript/tables/*.tex and \\input by prior_value.tex; verify_numbers.py checks
that the files are current.

    python manuscript/make_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TH = ROOT / "research" / "prx_quantum" / "results" / "theory"
OUT = Path(__file__).resolve().parent / "tables"


def sci(x: float) -> str:
    m, e = f"{x:.3e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def pct(x: float, nd: int = 2, sign: bool = False) -> str:
    s = f"{100 * x:+.{nd}f}" if sign else f"{100 * x:.{nd}f}"
    return rf"${s}\%$"


def ptex(p: float) -> str:
    m, e = f"{p:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


CERT = {2: "certified_class_family_T2_wspan.csv", 3: "certified_class_family_T3_gpu.csv"}


def best_found() -> pd.Series:
    """Canonical best per-class gain at each (T, p), over every candidate we have evaluated.

    Two searches produced candidates: the coordinate scan with its robustness re-test
    (exact_candidates_untruncated.csv) and the incumbents of the branch-and-bound runs,
    whose upper value is an exact risk at a point of the same family. The branch and bound
    improves on the scan at every error rate, so the table and the figures must take the
    maximum rather than the scan alone.
    """
    d = pd.read_csv(TH / "exact_candidates_untruncated.csv")
    gains = d.groupby(["T", "p"]).gain.max()
    for T, fname in CERT.items():
        c = pd.read_csv(TH / fname).set_index("p")
        for p in c.index:
            g = float(c.gain_UB_certified[p])
            gains.loc[(T, p)] = max(float(gains.get((T, p), -np.inf)), g)
    return gains.sort_index()


def tab_exact() -> str:
    """Table I: untruncated risks of the best-found per-class priors, T = 2 and T = 3.

    The row set is the operating points at which the exact distribution was enumerated,
    which is moment_limit_exact.csv: the coordinate scan's rates, plus the rates the
    branch-and-bound search reached that the scan never visited.
    """
    d = pd.read_csv(TH / "exact_candidates_untruncated.csv")
    bf = best_found()
    # exact risks of the physical prior, the correlation limit and the ML decoder
    lim = pd.read_csv(TH / "moment_limit_exact.csv").set_index(["T", "p"]).sort_index()
    lines = []
    for T in (2, 3):
        first = True
        for (_, p), lr in lim.loc[T:T].iterrows():
            scan = d[(d["T"] == T) & (d.p == p)]
            if len(scan):                                # cross-check the two enumerations
                r = scan.loc[scan.gain.idxmax()]
                assert abs(lr.R_phys - r.R_phys) < 1e-12 * max(1.0, r.R_phys) + 1e-15, "R_phys mismatch"
                assert abs(lr.gap_to_ml - r.gap_to_ml) < 1e-9, "gap-to-ML mismatch"
            gain = float(bf.loc[(T, p)])                 # scan or branch-and-bound incumbent
            lines.append(" & ".join([f"${T}$" if first else "", ptex(p), sci(lr.R_phys), pct(gain, 3),
                                     pct(lr.gain_limit, 3), pct(lr.gap_to_ml, 1),
                                     pct(gain / lr.gap_to_ml, 1)]) + r" \\")
            first = False
        if T == 2:
            lines.append(r"\hline")
    return "\n".join(lines) + "\n"


def tab_mechanism() -> str:
    """Table II: operational split of the matching-to-ML gap."""
    parts = [("surface, $T{=}2$", "three_way_surface_d3_T2.csv"),
             ("surface, $T{=}3$", "three_way_surface_d3_T3.csv"),
             ("repetition", "three_way_repetition_d5_T3.csv")]
    lines = []
    for k, (label, f) in enumerate(parts):
        d = pd.read_csv(TH / f).sort_values("p")
        for i, (_, r) in enumerate(d.iterrows()):
            tot, rule, model = r["gap_total_%"], r["gap_degeneracy_%"], r["gap_decomposition_%"]
            gl = "0.000" if abs(model) < 5e-4 else f"{model:.2f}"
            lines.append(" & ".join([label if i == 0 else "", ptex(r["p"]), rf"${tot:.2f}\%$",
                                     rf"${rule:+.2f}\%$", rf"${gl}\%$"]) + r" \\")
        if k < len(parts) - 1:
            lines.append(r"\hline")
    return "\n".join(lines) + "\n"


def tab_hetero() -> str:
    """Table III: homogeneous-prior penalty, median [IQR] over ten noise draws."""
    import json
    runs = sorted((ROOT / "research" / "prx_quantum" / "results" / "hetero_seeds").glob("*_main"))
    d = pd.read_csv(runs[-1] / "rows.csv")
    x = d.noise_extra_json.map(json.loads)
    d["sig"] = x.map(lambda e: e["sigma"]); d["mp"] = x.map(lambda e: e["mean_preserving"])
    d["pen"] = 100 * d.estimator_params_json.map(lambda e: json.loads(e)["penalty_vs_oracle"])
    d = d[~d.mp]
    lines = []
    for sg in (0.6, 1.0):
        cells = []
        for dist in (3, 5, 7):
            g = d[np.isclose(d.sig, sg) & (d.distance == dist)].pen
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            nd = 1 if sg < 0.9 else 0
            cells.append(rf"${med:.1f}\%$ $[{q1:.{nd}f}, {q3:.{nd}f}]$")
        lines.append(" & ".join([f"${sg:.1f}$"] + cells) + r" \\")
    return "\n".join(lines) + "\n"


def _tex_n(v: float) -> str:
    if not np.isfinite(v):
        return r"$>3\times10^{5}$"
    e = int(np.floor(np.log10(v))); m = v / 10 ** e
    return rf"$10^{{{e}}}$" if abs(m - 1) < 1e-9 else rf"${m:g}\times10^{{{e}}}$"


def budget_rows(run_dir=None, estimator="pooled_shrink"):
    """Per (sigma, d) budget summary of one estimator (default: pooled with shrinkage)."""
    runs = sorted((ROOT / "research" / "prx_quantum" / "results" / "frontier").glob("*_draws"))
    pc = pd.read_csv((run_dir or runs[-1]) / "budget_per_condition.csv")
    pc = pc[pc.estimator == estimator]
    med = lambda v: float(np.quantile(np.array(v, dtype=float), 0.5, method="inverted_cdf")) if len(v) else np.nan

    def first(h, col, thr):
        """Smallest sampled N from which on h[col] >= thr at every larger budget."""
        h = h.sort_values("n_cal")
        good = (h[col] >= thr).values
        for i in range(len(good)):
            if good[i:].all():
                return float(h.n_cal.values[i])
        return np.inf
    out = []
    for (sg, d), g in pc.groupby(["sigma_h", "distance"]):
        c = list(g.groupby("cond"))
        res = [h for _, h in c if h.recovery.notna().all()]
        out.append({"sigma": sg, "d": int(d), "conds": len(c), "draws": int(g.draw.nunique()),
                    "static": 100 * float(g.groupby("cond").static_excess.first().median()),
                    "beats": med([first(h.assign(b=h.beats_static.astype(float)), "b", 1.0) for _, h in c]),
                    "rec50": med([first(h, "recovery", 0.5) for h in res]),
                    "rec80": med([first(h, "recovery", 0.8) for h in res]),
                    "resolved": len(res),
                    "tenpct": med([first(h.assign(t=-h.excess), "t", -0.10) for _, h in c]),
                    "already": int(g.groupby("cond").static_meets_target.first().sum())})
    return out


def tab_budget() -> str:
    """Table V: when fitting pays, for the pooled estimator with shrinkage and for the
    occurrence (DGR) estimator on the same calibration and test shots."""
    lines = []
    fr = ROOT / "research" / "prx_quantum" / "results" / "frontier"
    tuned = sorted(fr.glob("*_draws_kappa20"))
    blocks = [(None, "pooled_shrink", "pooled + shrinkage, $\\kappa = 5$ (as run)"),
              (None, "dgr", "occurrence (DGR)")]
    if tuned:
        blocks.insert(1, (tuned[-1], "pooled_shrink", "pooled + shrinkage, $\\kappa = 20$ (tuned on separate devices)"))
    for bi, (run, est, label) in enumerate(blocks):
        lines.append(r"\multicolumn{8}{l}{\emph{%s}} \\" % label)
        rows = budget_rows(run, estimator=est)
        for k, r in enumerate(rows):
            lines.append(" & ".join([f"${r['sigma']:.1f}$" if r["d"] == 3 else "", f"${r['d']}$",
                                     f"${r['conds']}$", rf"${r['static']:.0f}\%$", _tex_n(r["beats"]),
                                     _tex_n(r["rec50"]), _tex_n(r["rec80"]),
                                     _tex_n(r["tenpct"]) + f" ({r['already']})"]) + r" \\")
            if r["d"] == 7 and k < len(rows) - 1:
                lines.append(r"\cline{1-8}")
        if bi < len(blocks) - 1:
            lines.append(r"\hline")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, fn in (("tab_exact.tex", tab_exact), ("tab_mechanism.tex", tab_mechanism),
                     ("tab_hetero.tex", tab_hetero), ("tab_budget.tex", tab_budget)):
        try:
            (OUT / name).write_text(fn())
        except FileNotFoundError as e:
            print(f"[skip] {name}: missing input {e.filename}")
    for f in sorted(OUT.glob("*.tex")):
        print(f"--- {f.name}\n{f.read_text()}")


if __name__ == "__main__":
    main()
