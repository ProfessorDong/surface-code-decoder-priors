"""Publication-quality figures for the PRX Quantum manuscript.

Reads the result CSVs written by the PRX experiment drivers under
``research/prx_quantum/results`` and renders one PDF per figure into
``research/prx_quantum/figures``.

Nothing is recomputed here beyond trivial presentation arithmetic
(ratios of numbers that are already in the CSVs, and aggregation across
replicate conditions).  If an input CSV is missing the corresponding
figure is skipped with a warning.

Usage
-----
    python -m simulation.prx.plotting.figures --out research/prx_quantum/figures

Individual run directories can be pinned with the ``--prior-sweep-deep``,
``--prior-sweep-shallow``, ``--frontier``, ``--willow``,
``--willow-factorial`` and ``--theory`` options; otherwise the most recent
matching run directory is discovered by glob.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COL_SINGLE = 3.4        # inches, single column
COL_DOUBLE = 7.0        # inches, double column

FS_TICK = 8             # labels / ticks
FS_AXIS = 9             # axis titles
FS_LEGEND = 7

# Colourblind-safe (Okabe-Ito ordering requested by the style guide).
PALETTE: List[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
]

GREY = "#999999"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": FS_TICK,
            "axes.labelsize": FS_AXIS,
            "axes.titlesize": FS_AXIS,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "errorbar.capsize": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def panel_label(ax: plt.Axes, letter: str, dx: float = -0.03, dy: float = 1.04) -> None:
    """Lowercase bold panel letter, just outside the top-left corner of the axes."""
    ax.text(
        dx,
        dy,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=FS_AXIS,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "research" / "prx_quantum" / "results"


def latest_run(subdir: str, pattern: str, root: Path = RESULTS_ROOT) -> Optional[Path]:
    """Most recent run directory (names start with an ISO-8601 stamp)."""
    base = root / subdir
    if not base.is_dir():
        return None
    matches = sorted(p for p in base.glob(pattern) if p.is_dir())
    return matches[-1] if matches else None


def read_csv(path: Optional[Path], what: str) -> Optional[pd.DataFrame]:
    if path is None or not Path(path).is_file():
        print(f"  [skip] missing {what}: {path}")
        return None
    return pd.read_csv(path)


def save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 1 -- prior geometry
# ---------------------------------------------------------------------------

_S_RE = re.compile(r"uniform_s=([0-9.]+)")
CLASSES = ["space", "time", "diag", "boundary"]
CLASS_LABELS = {
    "space": "space",
    "time": "time",
    "diag": "diagonal",
    "boundary": "boundary",
}


def _uniform_scale_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for one (distance, rounds, p) cell -> table of (s, pct, lo, hi)."""
    base = df.loc[df.prior_id == "oracle_physical", "ler"]
    if base.empty:
        return pd.DataFrame()
    ler0 = float(base.iloc[0])
    recs = [dict(s=1.0, pct=0.0, lo=0.0, hi=0.0)]
    for _, row in df[df.prior_id.str.startswith("uniform_s=")].iterrows():
        m = _S_RE.match(row.prior_id)
        if not m:
            continue
        delta = float(row.delta_ler_vs_ref)
        recs.append(
            dict(
                s=float(m.group(1)),
                pct=100.0 * delta / ler0,
                lo=100.0 * (delta - float(row.delta_ci_lo)) / ler0,
                hi=100.0 * (float(row.delta_ci_hi) - delta) / ler0,
            )
        )
    return pd.DataFrame(recs).sort_values("s").reset_index(drop=True)


def figure1(deep: Optional[Path], shallow: Optional[Path], out_dir: Path,
            p_star: float = 0.002) -> Optional[str]:
    print("Figure 1: prior geometry")
    deep_rows = read_csv(deep, "prior_sweep (deep) rows.csv")
    if deep_rows is None:
        return None
    shallow_rows = read_csv(shallow, "prior_sweep (shallow) rows.csv")

    fam = "si1000:google2025"
    deep_sel = deep_rows[(deep_rows.noise_family == fam) & (deep_rows.p_true == p_star)]
    if deep_sel.empty:
        print(f"  [skip] no si1000 rows at p={p_star} in the deep run")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.7))
    ax = axes[0]

    curves = []
    for i, ((d, T), grp) in enumerate(
        sorted(deep_sel.groupby(["distance", "rounds"]), key=lambda kv: kv[0])
    ):
        tab = _uniform_scale_curve(grp)
        if tab.empty:
            continue
        shots = int(grp.n_test_shots.iloc[0])
        curves.append((f"$d$={d}, $T$={T}", tab, PALETTE[i % len(PALETTE)], "-", shots))

    if shallow_rows is not None:
        sh = shallow_rows[
            (shallow_rows.noise_family == fam)
            & (shallow_rows.p_true == p_star)
            & (shallow_rows.distance == 3)
        ]
        for j, ((d, T), grp) in enumerate(
            sorted(sh.groupby(["distance", "rounds"]), key=lambda kv: kv[0])
        ):
            tab = _uniform_scale_curve(grp)
            if tab.empty:
                continue
            shots = int(grp.n_test_shots.iloc[0])
            colour = PALETTE[(len(curves) + j) % len(PALETTE)]
            curves.append((f"$d$={d}, $T$={T}", tab, colour, "--", shots))

    curves.sort(key=lambda c: tuple(int(v) for v in re.findall(r"=(\d+)", c[0])))
    for label, tab, colour, ls, _shots in curves:
        ax.errorbar(
            tab.s,
            tab.pct,
            yerr=np.vstack([tab.lo, tab.hi]),
            color=colour,
            linestyle=ls,
            marker="o",
            markersize=3,
            linewidth=1.0,
            elinewidth=0.7,
            label=label,
        )
    ax.axhline(0.0, color=GREY, linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel("Uniform prior scale $f$ (dimensionless)")
    ax.set_ylabel("Change in logical error rate (%)")
    ax.legend(frameon=True, loc="upper left", ncol=2, handlelength=1.6, borderpad=0.3,
              columnspacing=0.8)
    panel_label(ax, "a")

    # --- panel (b): per-class optimal scales -------------------------------
    axb = axes[1]
    frames = [("deep", deep_rows, True)]
    if shallow_rows is not None:
        frames.append(("shallow", shallow_rows[shallow_rows.distance == 3], False))

    rec = []
    for _tag, frame, filled in frames:
        sel = frame[(frame.noise_family == fam) & (frame.prior_id == "class_search")]
        for _, row in sel.iterrows():
            if not np.isfinite(row.get("scale_space", np.nan)):
                continue
            rec.append(
                dict(
                    distance=int(row.distance),
                    rounds=int(row.rounds),
                    p=float(row.p_true),
                    filled=filled,
                    **{c: float(row[f"scale_{c}"]) for c in CLASSES},
                )
            )
    cs = pd.DataFrame(rec)
    p_keep = sorted(deep_sel.p_true.unique().tolist() + [0.0015, 0.002, 0.003])
    cs = cs[cs.p.isin(sorted(set(p_keep)))]
    configs = sorted(set(zip(cs.distance, cs.rounds)))
    xpos = {cfg: i for i, cfg in enumerate(configs)}
    markers = {p: m for p, m in zip(sorted(cs.p.unique()), ["o", "s", "^", "D", "v"])}
    offs = np.linspace(-0.28, 0.28, len(CLASSES))
    # small per-p offset so that equal grid values at different p do not overlap
    p_off = dict(zip(sorted(markers), np.linspace(-0.05, 0.05, len(markers))))

    for ci, cls in enumerate(CLASSES):
        for _, row in cs.iterrows():
            x = xpos[(row.distance, row.rounds)] + offs[ci] + p_off[row.p]
            axb.plot(
                x,
                row[cls],
                marker=markers[row.p],
                markersize=3,
                color=PALETTE[ci],
                markerfacecolor=PALETTE[ci] if row.filled else "none",
                markeredgecolor=PALETTE[ci],
                markeredgewidth=0.7,
                linestyle="none",
            )
    axb.axhline(1.0, color=GREY, linewidth=0.8, zorder=0)
    axb.set_xticks(range(len(configs)))
    axb.set_xticklabels([f"$d$={d}\n$T$={T}" for d, T in configs])
    axb.set_xlim(-0.6, len(configs) - 0.4)
    axb.set_ylabel("Best class scale found\n(dimensionless)")
    axb.set_xlabel("Code distance and number of rounds")
    axb.set_ylim(0.4, 4.3)
    axb.set_yticks([1, 2, 3])
    class_handles = [
        Line2D([], [], color=PALETTE[i], marker="s", linestyle="none",
               markersize=3, label=CLASS_LABELS[c])
        for i, c in enumerate(CLASSES)
    ]
    p_handles = [
        Line2D([], [], color="0.35", marker=markers[p], linestyle="none",
               markersize=3, label=f"$p$={p:g}")
        for p in sorted(markers)
    ]
    shot_handles = [
        Line2D([], [], color="0.35", marker="o", linestyle="none", markersize=3,
               label="16M shots"),
        Line2D([], [], color="0.35", marker="o", linestyle="none", markersize=3,
               markerfacecolor="none", label="4M shots"),
    ]
    # one boxed legend, top right; in pgfplots (row-wise) the rows are classes,
    # boundary with the two shot counts, and the three noise strengths
    axb.legend(handles=class_handles[:3] + class_handles[3:] + shot_handles + p_handles,
               loc="upper right", ncol=3, frameon=True, handletextpad=0.3,
               columnspacing=0.8)
    panel_label(axb, "b")

    path = save(fig, out_dir, "fig1_prior_geometry.pdf")

    lines = [f"fig1_prior_geometry.pdf -> {path}"]
    lines.append(
        f"  (a) si1000:google2025 at p={p_star:g}; relative LER change vs the physical "
        f"prior under a uniform rescaling of every edge prior by s."
    )
    for label, tab, _c, ls, shots in curves:
        row4 = tab[np.isclose(tab.s, 4.0)]
        row025 = tab[np.isclose(tab.s, 0.25)]
        best = tab.loc[tab.pct.idxmin()]
        lines.append(
            f"      {label} ({shots/1e6:g}M shots): s=0.25 {float(row025.pct.iloc[0]):+.2f}%, "
            f"s=4 {float(row4.pct.iloc[0]):+.2f}%, best s={best.s:g} at {best.pct:+.2f}%"
            + ("  [shallower run]" if ls == "--" else "")
        )
    lines.append(
        "  (b) class_search optima: "
        + ", ".join(
            f"{CLASS_LABELS[c]} median {cs[c].median():.2f} (range {cs[c].min():.2f}-{cs[c].max():.2f})"
            for c in CLASSES
        )
    )
    lines.append(
        f"      {int((cs.boundary < 1).sum())}/{len(cs)} configurations have s*_boundary<1; "
        f"{int((cs['diag'] > 1).sum())}/{len(cs)} have s*_diag>1; "
        f"{int((cs.time > 1).sum())}/{len(cs)} have s*_time>1."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure 2 -- exact theory
# ---------------------------------------------------------------------------


def _rep_tag(path: Path, df: pd.DataFrame) -> str:
    """Readable label for an auxiliary risk-surface file."""
    family = "repetition" if "_rep_" in path.name else "surface"
    if {"distance", "rounds"} <= set(df.columns):
        return f"{family} $d$={int(df.distance.iloc[0])}, $T$={int(df.rounds.iloc[0])}"
    return family


def figure2(theory_dir: Path, out_dir: Path) -> Optional[str]:
    """(a) best reweighting found and ML decoding, relative to the physical prior,
    for the d = 3 surface-code memory at T = 2 and 3; (b) share of the
    matching-to-ML gap that the best reweighting found recovers, per memory.

    Every value is exact and untruncated: the surface code from
    exact_candidates_untruncated.csv (all 2^16 or 2^24 syndromes, best candidate
    per point), the repetition code from rep_untruncated/ (tol = 0).
    """
    print("Figure 2: exact theory")
    cand = read_csv(theory_dir / "exact_candidates_untruncated.csv", "untruncated exact candidates")
    if cand is None:
        return None
    surf = cand.loc[cand.groupby(["T", "p"]).gain.idxmax()].sort_values(["T", "p"]).reset_index(drop=True)
    reps = sorted((theory_dir / "rep_untruncated").glob("risk_surface_rep_d3_T*.csv"),
                  key=lambda f: int(re.search(r"_T(\d+)", f.name).group(1)))

    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.7))
    ax = axes[0]
    ax.axhline(100.0, color=GREY, linewidth=0.8, zorder=0)
    styles = {2: "-", 3: "--"}
    for T, g in surf.groupby("T"):
        ax.plot(g.p, 100 * g.R_cand / g.R_phys, color=PALETTE[1], marker="s", linestyle=styles[T])
        ax.plot(g.p, 100 * g.R_ml / g.R_phys, color=PALETTE[2], marker="^", linestyle=styles[T])
    ax.set_xscale("log")
    ax.set_xlabel("Physical error rate $p$ (dimensionless)")
    ax.set_ylabel("Logical error relative to\nthe physical prior (%)")
    ax.set_ylim(70, 102)
    handles = [
        Line2D([], [], color=PALETTE[1], marker="s", linestyle="none", markersize=3,
               label="best reweighting found"),
        Line2D([], [], color="0.2", linestyle="-", label="$T$=2"),
        Line2D([], [], color=PALETTE[2], marker="^", linestyle="none", markersize=3,
               label="optimal (ML) decoder"),
        Line2D([], [], color="0.2", linestyle="--", label="$T$=3"),
    ]
    ax.legend(handles=handles, loc="lower right", ncol=2, frameon=True)
    panel_label(ax, "a")

    axb = axes[1]
    for T, g in surf.groupby("T"):
        axb.plot(g.p, 100 * g.recovered, color=PALETTE[1], marker="s", linestyle=styles[T],
                 label=f"surface, $T$={T}")
    rep_colours = [PALETTE[0], PALETTE[3], PALETTE[5]]
    for k, f in enumerate(reps):
        rd = pd.read_csv(f).sort_values("p")
        T = int(rd.rounds.iloc[0])
        axb.plot(rd.p, 100 * rd.frac_of_ml_gap_recovered, color=rep_colours[k % 3], marker="o",
                 linestyle=":", label=f"repetition, $T$={T}")
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_ylim(0.1, 300)
    axb.set_xlabel("Physical error rate $p$ (dimensionless)")
    axb.set_ylabel("Share of the gap to ML\ndecoding recovered (%)")
    axb.legend(loc="upper left", ncol=2, frameon=True)
    panel_label(axb, "b")

    path = save(fig, out_dir, "fig2_exact_theory.pdf")

    lines = [f"fig2_exact_theory.pdf -> {path}"]
    for _, r in surf.iterrows():
        lines.append(f"  surface d=3 T={int(r['T'])} p={r.p:g}: R_opt/R_phys={100*r.R_cand/r.R_phys:.2f}%, "
                     f"R_ml/R_phys={100*r.R_ml/r.R_phys:.2f}%, recovered {100*r.recovered:.2f}% "
                     f"({r.candidate}, truncation {r.truncation})")
    for f in reps:
        rd = pd.read_csv(f).sort_values("p")
        lines.append(f"  {f.name}: recovered {100*rd.frac_of_ml_gap_recovered.min():.2f}%-"
                     f"{100*rd.frac_of_ml_gap_recovered.max():.2f}%, max neglected mass "
                     f"{rd.bound_width.max():.1e}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure 3 -- heterogeneity frontier
# ---------------------------------------------------------------------------

EST_LABELS = {
    "dgr": "occurrence (DGR)",
    "pooled_shrink": "pooled + shrinkage",
    "pooled": "pooled",
    "pij": r"$p_{ij}$ correlations",
    "class_search": "class search",
    "uniform_search": "uniform search",
    "density": "detector density",
    "pooled_shrink+class": "pooled + shrinkage + class",
}
EST_ORDER = ["pooled_shrink", "pooled", "pij", "class_search", "uniform_search", "density", "dgr"]


def figure3(frontier_dir: Optional[Path], out_dir: Path, sigma_star: float = 1.0,
            hetero_dir: Optional[Path] = None) -> Optional[str]:
    """(a-c) calibration frontier at sigma_star; (d) homogeneous-prior penalty over noise draws.

    Panels (a-c) show, for each estimator, the median over conditions (each an
    independent noise draw at one geometry) of the excess over the oracle prior,
    with the interquartile range for the pooled estimator with shrinkage, and the
    median excess of the uncalibrated prior as a dashed reference: an estimator
    helps only where it lies below that line. Panel (d) shows the median and
    interquartile range over ten independent noise draws from the separate
    heterogeneity study, the same data as the heterogeneity table.
    """
    import json as _json
    print("Figure 3: heterogeneity frontier")
    if frontier_dir is None:
        print("  [skip] no frontier run directory found")
        return None
    curves = read_csv(frontier_dir / "frontier_curves.csv", "frontier_curves.csv")
    fits = read_csv(frontier_dir / "frontier_fits.csv", "frontier_fits.csv")
    if curves is None:
        return None
    sel = curves[np.isclose(curves.sigma_h, sigma_star)]
    if sel.empty:
        print(f"  [skip] no frontier curves at sigma_h={sigma_star}")
        return None
    distances = sorted(sel.distance.unique())
    fig, axes = plt.subplots(2, 2, figsize=(COL_DOUBLE, 5.0))
    flat = axes.ravel()
    lines: List[str] = []
    ests = [e for e in EST_ORDER if e in set(sel.estimator) and e != "static"]
    for k, d in enumerate(distances[:3]):
        ax = flat[k]
        sub = sel[sel.distance == d]
        n_cond = sub.cond.nunique()
        n_draw = sub.cond.map(lambda c: c.split("|")[-1]).nunique()
        for i, est in enumerate(ests):
            g = sub[sub.estimator == est].groupby("n_cal")["rel_excess"]
            med = g.median() * 100
            ax.plot(med.index, med.values, color=PALETTE[i % len(PALETTE)], marker="o",
                    markersize=3, label=EST_LABELS.get(est, est))
            if est == "pooled_shrink":
                ax.fill_between(med.index, g.quantile(0.25).values * 100, g.quantile(0.75).values * 100,
                                color=PALETTE[i % len(PALETTE)], alpha=0.18, linewidth=0)
            lines.append(f"      d={d} {EST_LABELS.get(est, est)}: median excess "
                         f"{med.iloc[0]:.1f}% at N={int(med.index[0])} -> {med.iloc[-1]:.1f}% "
                         f"at N={int(med.index[-1])}")
        if fits is not None and {"ler_static", "ler_oracle"} <= set(fits.columns):
            fu = fits.drop_duplicates(subset=["cond"])
            fu = fu[(fu.distance == d) & np.isclose(fu.sigma_h, sigma_star)]
            if len(fu):
                st_med = float(100 * (fu.ler_static / fu.ler_oracle - 1).median())
                ax.axhline(st_med, color="0.25", linestyle="--", linewidth=0.9, label="uncalibrated prior")
                lines.append(f"      d={d} uncalibrated prior: median excess {st_med:.1f}% over {len(fu)} conditions")
        ax.axhline(0.0, color=GREY, linewidth=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("Calibration shots $N_{\\mathrm{cal}}$")
        ax.set_ylabel("Excess over oracle prior (%)")
        ax.set_title(f"$d={d}$, $\\sigma={sigma_star:g}$, {n_cond} conditions", fontsize=FS_TICK)
        panel_label(ax, "abc"[k])
        if k == 0:
            ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.4,
                      columnspacing=0.8, fontsize=FS_LEGEND - 1)

    # (d) probability of harm: fraction of conditions in which the pooled fit with
    # shrinkage decodes worse than the uncalibrated prior, by distance and sigma
    axd = flat[3]
    bpc = frontier_dir / "budget_per_condition.csv"
    if bpc.exists():
        pc = pd.read_csv(bpc)
        pc = pc[pc.estimator == "pooled_shrink"]
        for i, d in enumerate(sorted(pc.distance.unique())):
            for sg, ls in ((0.6, "--"), (1.0, "-")):
                g = pc[(pc.distance == d) & np.isclose(pc.sigma_h, sg)]
                harm = 100 * (1 - g.groupby("n_cal").beats_static.mean())
                axd.plot(harm.index, harm.values, color=PALETTE[i % len(PALETTE)], marker="o", markersize=3,
                         linestyle=ls, label=f"$d={d}$, $\\sigma={sg:g}$")
                lines.append(f"      P(harm) pooled+shrinkage d={d} sigma={sg:g}: "
                             + ", ".join(f"N={int(n)}: {v:.0f}%" for n, v in harm.items()))
        axd.set_xscale("log")
        axd.set_ylim(-5, 105)
        axd.set_xlabel("Calibration shots $N_{\\mathrm{cal}}$")
        axd.set_ylabel("Conditions harmed (%)")
        axd.set_title("pooled + shrinkage", fontsize=FS_TICK)
        axd.legend(frameon=True, loc="upper right", ncol=2)
    else:
        axd.set_axis_off()
    panel_label(axd, "d")

    path = save(fig, out_dir, "fig3_heterogeneity.pdf")
    header = [f"fig3_heterogeneity.pdf -> {path}",
              f"  (a-c) median excess over the oracle prior versus calibration shots, sigma_h={sigma_star:g};"
              " dashed: uncalibrated prior.",
              "  (d) probability of harm: fraction of conditions where pooled+shrinkage is worse than uncalibrated."]
    return "\n".join(header + lines)


# ---------------------------------------------------------------------------
# Figure 4 -- Willow calibration frontier
# ---------------------------------------------------------------------------

WILLOW_ESTS = ["pooled_shrink", "pooled", "class_search", "uniform_search", "pooled_shrink+class"]


def figure4(willow_dir: Optional[Path], out_dir: Path) -> Optional[str]:
    print("Figure 4: Willow calibration frontier")
    if willow_dir is None:
        print("  [skip] no willow run directory found")
        return None
    # intervals resample shots, calibration draws and (at d = 3, 5) patch-basis groups
    fr = read_csv(willow_dir / "frontier_joint_ci.csv", "willow frontier_joint_ci.csv")
    if fr is None:
        return None
    rl_all = fr[fr.estimator == "rl_shipped"]
    fr = fr[fr.estimator != "rl_shipped"]
    leaves = read_csv(willow_dir / "frontier.csv", "willow frontier.csv")

    distances = sorted(fr.distance.unique())
    fig, axes = plt.subplots(1, len(distances), figsize=(COL_DOUBLE, 2.5), sharey=False)
    axes = np.atleast_1d(axes)
    lines: List[str] = []

    for k, d in enumerate(distances):
        ax = axes[k]
        sub = fr[fr.distance == d]
        ests = [e for e in WILLOW_ESTS if e in set(sub.estimator)]
        for i, est in enumerate(ests):
            g = sub[sub.estimator == est].sort_values("n_cal")
            ax.plot(g.n_cal, 100 * g.rel_ler, color=PALETTE[i % len(PALETTE)],
                    marker="o", markersize=3, label=EST_LABELS.get(est, est))
            ax.fill_between(g.n_cal, 100 * g.ci_lo, 100 * g.ci_hi,
                            color=PALETTE[i % len(PALETTE)], alpha=0.16, linewidth=0)
            last = g.iloc[-1]
            lines.append(
                f"      d={d} {EST_LABELS.get(est, est)}: {100*g.rel_ler.iloc[0]:+.2f}% at "
                f"N={int(g.n_cal.iloc[0])} -> {100*last.rel_ler:+.2f}% "
                f"[{100*last.ci_lo:+.2f}, {100*last.ci_hi:+.2f}] at N={int(last.n_cal)}"
            )
        rl = rl_all[rl_all.distance == d]
        if not rl.empty:
            r = rl.iloc[0]
            xs = [sub.n_cal.min() / 1.3, sub.n_cal.max() * 1.3]
            ax.axhline(100 * r.rel_ler, color=PALETTE[6], linestyle="--", linewidth=0.9)
            ax.fill_between(xs, [100 * r.ci_lo] * 2, [100 * r.ci_hi] * 2, color=PALETTE[6], alpha=0.10,
                            linewidth=0)
            lines.append(f"      d={d} RL prior (shipped, PyMatching): {100*r.rel_ler:+.2f}% "
                         f"[{100*r.ci_lo:+.2f}, {100*r.ci_hi:+.2f}]")
        ax.axhline(0.0, color=GREY, linewidth=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("Calibration shots $N_{\\mathrm{cal}}$")
        if k == 0:
            ax.set_ylabel("Relative logical-error change (%)")
        n_leaves = int(leaves[leaves.distance == d].n_leaves.iloc[0]) if leaves is not None else 0
        ax.set_title(f"$d={d}$", fontsize=FS_TICK)
        lines.append(f"      d={d}: {n_leaves} experiments")
        panel_label(ax, "abcdef"[k])
        if k == 0:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([], [], color=PALETTE[6], linestyle="--",
                                  linewidth=0.9, label="RL prior (shipped)"))
            ax.legend(handles=handles, frameon=False, loc="upper right",
                      handlelength=1.4, fontsize=FS_LEGEND - 1)

    path = save(fig, out_dir, "fig4_willow_frontier.pdf")
    header = [
        f"fig4_willow_frontier.pdf -> {path}",
        "  Google Willow memory data; logical error rate of a re-estimated prior relative to the "
        "shipped SI1000 prior (negative is better), PyMatching MWPM throughout. Bands are the "
        "paired bootstrap CIs in the CSV; the dashed black line is the shipped RL prior with its CI.",
    ]
    return "\n".join(header + lines)


# ---------------------------------------------------------------------------
# Figure 5 -- prior versus decoder
# ---------------------------------------------------------------------------

PRIOR_LABELS = {
    "si1000_shipped": "SI1000 prior",
    "rl_shipped": "RL prior",
    "pooled_shrink_N=10000": "pooled + shrinkage",
}
DEC_LABELS = {"pymatching_mwpm": "MWPM", "pymatching_correlated": "correlated matching"}


def figure5(fact_dir: Optional[Path], out_dir: Path) -> Optional[str]:
    print("Figure 5: prior versus decoder")
    if fact_dir is None:
        print("  [skip] no willow_factorial run directory found")
        return None
    rows = read_csv(fact_dir / "rows.csv", "willow_factorial rows.csv")
    if rows is None:
        return None

    agg = (
        rows.groupby(["distance", "prior_id", "decoder"])
        .agg(n_errors=("n_errors", "sum"), n_shots=("n_test_shots", "sum"),
             leaves=("leaf_id", "nunique"))
        .reset_index()
    )
    agg["ler"] = agg.n_errors / agg.n_shots
    # 95% intervals from the joint bootstrap (analysis.willow_factorial_ci)
    ci = read_csv(fact_dir / "factorial_ci.csv", "willow_factorial factorial_ci.csv")
    if ci is not None:
        ci = ci[(ci.ref_prior == "si1000_shipped") & (ci.ref_decoder == "pymatching_mwpm")].drop_duplicates(
            ["distance", "prior", "decoder"]).set_index(["distance", "prior", "decoder"])

    priors = [p for p in PRIOR_LABELS if p in set(agg.prior_id)]
    decoders = [d for d in DEC_LABELS if d in set(agg.decoder)]
    distances = sorted(agg.distance.unique())

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.6))
    width = 0.8 / (len(priors) * len(decoders))
    lines: List[str] = []

    for k, d in enumerate(distances):
        sub = agg[agg.distance == d]
        base = sub[(sub.prior_id == "si1000_shipped") & (sub.decoder == "pymatching_mwpm")]
        if base.empty:
            continue
        ler0 = float(base.ler.iloc[0])
        b = 0
        for di, dec in enumerate(decoders):
            for pi, pr in enumerate(priors):
                r = sub[(sub.prior_id == pr) & (sub.decoder == dec)]
                if r.empty:
                    b += 1
                    continue
                rel = 100.0 * (float(r.ler.iloc[0]) / ler0 - 1.0)
                x = k + (b - (len(priors) * len(decoders) - 1) / 2) * width
                err = None
                if ci is not None and (d, pr, dec) in ci.index:
                    c = ci.loc[(d, pr, dec)]
                    err = [[rel - 100 * c.ci_lo], [100 * c.ci_hi - rel]]
                ax.bar(
                    x, rel, width=width * 0.92, yerr=err,
                    color=PALETTE[pi] if dec == "pymatching_mwpm" else "white",
                    edgecolor=PALETTE[pi],
                    linewidth=0.6,
                    hatch="" if dec == "pymatching_mwpm" else "///",
                )
                lines.append(
                    f"      d={d} {PRIOR_LABELS[pr]} + {DEC_LABELS[dec]}: LER={float(r.ler.iloc[0]):.4f} "
                    f"({rel:+.2f}% vs SI1000+MWPM), {int(r.n_errors.iloc[0])} errors / "
                    f"{int(r.n_shots.iloc[0])} shots, {int(r.leaves.iloc[0])} leaves"
                )
                b += 1
        lines.append(f"      d={d} baseline SI1000+MWPM LER = {ler0:.4f}")

    ax.axhline(0.0, color=GREY, linewidth=0.8, zorder=0)
    ax.set_xticks(range(len(distances)))
    ax.set_xticklabels([f"$d$={d}" for d in distances])
    ax.set_xlabel("Code distance (dimensionless)")
    ax.set_ylabel("Logical error rate relative to\nSI1000 prior with MWPM (%)")
    handles = [
        Patch(facecolor=PALETTE[i], edgecolor=PALETTE[i],
              label=PRIOR_LABELS[p] + (" (reference)" if p == "si1000_shipped" else ""))
        for i, p in enumerate(priors)
    ]
    handles += [
        Patch(facecolor="0.85", edgecolor="0.3", label="MWPM"),
        Patch(facecolor="white", edgecolor="0.3", hatch="///", label="correlated matching"),
    ]
    lo = ax.get_ylim()[0]
    ax.set_ylim(lo - 0.50 * (ax.get_ylim()[1] - lo), ax.get_ylim()[1])   # room for the legend below the bars
    ax.legend(handles=handles, frameon=True, loc="lower center", ncol=2,
              handlelength=1.2, columnspacing=0.8, fontsize=FS_LEGEND - 1)

    path = save(fig, out_dir, "fig5_prior_vs_decoder.pdf")
    header = [
        f"fig5_prior_vs_decoder.pdf -> {path}",
        "  Willow factorial: 3 priors x 2 decoders, pooled over leaves (errors and shots summed). "
        "Reference cell is the shipped SI1000 prior with plain MWPM.",
    ]
    return "\n".join(header + lines)


# ---------------------------------------------------------------------------
# Figure 6 -- error suppression factor
# ---------------------------------------------------------------------------

SEL_LABELS = {
    "pymatching+si1000": "MWPM + SI1000 prior",
    "pymatching+rl": "MWPM + RL prior",
    "google:correlated_matching_decoder_with_si1000_prior": "Google corr. matching + SI1000",
    "google:correlated_matching_decoder_with_rl_optimized_prior": "Google corr. matching + RL",
    "google:harmony_decoder_with_si1000_prior": "Google Harmony + SI1000",
    "google:harmony_decoder_with_rl_optimized_prior": "Google Harmony + RL",
    "google:libra_decoder_with_rl_optimized_prior": "Google Libra + RL",
}


def _sel_label(sel: str) -> str:
    if sel in SEL_LABELS:
        return SEL_LABELS[sel]
    if sel.startswith("pymatching+pooled_shrink"):
        tail = sel.split("pooled_shrink", 1)[1].lstrip("_")
        tail = tail.replace("N=", "$N$=")
        return f"MWPM + pooled + shrinkage ({tail})" if tail else "MWPM + pooled + shrinkage"
    return sel.replace("google:", "Google ").replace("_", " ")


def figure6(willow_dir: Optional[Path], out_dir: Path) -> Optional[str]:
    print("Figure 6: error suppression factor")
    if willow_dir is None:
        print("  [skip] no willow run directory found")
        return None
    lam = read_csv(willow_dir / "lambda.csv", "willow lambda.csv")
    if lam is None:
        return None
    # uncertainty: bootstrap SD over shots and patch-basis groups (analysis.lambda_marginal),
    # not the model-based standard error of the three-point regression
    sd = read_csv(willow_dir / "lambda_marginal.csv", "willow lambda_marginal.csv")
    sd = sd.set_index("selection").sd_Lambda if sd is not None else None

    eps_cols = [(int(c.split("_")[1]), c) for c in lam.columns if re.fullmatch(r"eps_\d+", c)]
    eps_cols.sort()
    distances = [d for d, _ in eps_cols]

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(COL_DOUBLE, 3.0))
    lines: List[str] = []
    # eight lines in a fixed order, so colours are assigned explicitly and stably:
    # our three priors, then Google's decoders by decoder and prior, Libra last (black)
    order = ["pymatching+si1000", "pymatching+rl", "pymatching+pooled_shrink_N=3000",
             "google:correlated_matching_decoder_with_si1000_prior",
             "google:correlated_matching_decoder_with_rl_optimized_prior",
             "google:harmony_decoder_with_si1000_prior", "google:harmony_decoder_with_rl_optimized_prior",
             "google:libra_decoder_with_rl_optimized_prior"]
    lam = lam.set_index("selection").loc[[o for o in order if o in set(lam.selection)]].reset_index()
    colours = PALETTE[:6] + ["#7F7F7F", PALETTE[6]]
    for i, (_, row) in enumerate(lam.iterrows()):
        ours = row.selection.startswith("pymatching")
        ys = [float(row[c]) for _, c in eps_cols]
        ax.plot(
            distances, ys,
            color=colours[i],
            linestyle="-" if ours else "--",
            marker="o" if ours else "s",
            markersize=3,
            label=f"{_sel_label(row.selection)} ($\\Lambda$={row.Lambda_fit:.2f}$\\pm${(sd[row.selection] if sd is not None else row.dLambda):.2f})",
        )
        lines.append(
            f"      {_sel_label(row.selection)}: Lambda={row.Lambda_fit:.2f}+/-"
            f"{(sd[row.selection] if sd is not None else row.dLambda):.2f} (bootstrap SD); "
            + ", ".join(f"eps_{d}={float(row[c]):.5f} (n={int(row[f'n_{d}'])})" for d, c in eps_cols)
        )
    ax.set_yscale("log")
    ax.set_yticks([0.002, 0.003, 0.005, 0.01])
    ax.set_yticklabels(["0.002", "0.003", "0.005", "0.01"])
    ax.set_xticks(distances)
    ax.set_xlabel("Code distance $d$ (dimensionless)")
    ax.set_ylabel("Logical error probability\nper round")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22),
              ncol=1, handlelength=1.8, fontsize=FS_LEGEND - 1)
    panel_label(ax, "a")

    # (b) Lambda of each prior relative to the shipped SI1000 prior, paired intervals
    lp = read_csv(willow_dir / "lambda_paired.csv", "willow lambda_paired.csv")
    if lp is not None:
        ref = "si1000_shipped"
        others = ["rl_shipped"] + [f"pooled_shrink_N=3000_rep={r}" for r in range(5)]
        labels = ["RL"] + [f"fit {r + 1}" for r in range(5)]
        for j, (scheme, mk, off, lab) in enumerate((("shots", "o", -0.12, "shots only"),
                                                    ("shots+groups", "s", 0.12, "shots and groups"))):
            g = lp[(lp.scheme == scheme) & (lp.prior_A == ref)].set_index("prior_B")
            xs, ys, lo, hi = [], [], [], []
            for k, o in enumerate(others):
                r = g.loc[o]
                y = 100 * (1 / r.ratio - 1)
                xs.append(k + off); ys.append(y)
                lo.append(y - 100 * (1 / r.ratio_hi - 1)); hi.append(100 * (1 / r.ratio_lo - 1) - y)
                lines.append(f"      (b) {scheme} {labels[k]}: {y:+.2f}% [{100*(1/r.ratio_hi-1):+.2f}, {100*(1/r.ratio_lo-1):+.2f}]")
            axb.errorbar(xs, ys, yerr=[lo, hi], color=PALETTE[j], marker=mk, linestyle="none", markersize=3,
                         label=lab)
        axb.axhline(0.0, color=GREY, linewidth=0.8, zorder=0)
        axb.set_xticks(range(len(others)))
        axb.set_xticklabels(labels)
        axb.set_xlim(-0.6, len(others) - 0.4)
        axb.set_ylim(-4.6, 3.0)          # room for the legend below the lowest interval
        axb.set_ylabel("Change in $\\Lambda$ vs SI1000 (%)")
        axb.set_xlabel("Prior under plain matching")
        axb.legend(frameon=True, loc="lower right")
        panel_label(axb, "b")

    path = save(fig, out_dir, "fig6_lambda.pdf")
    header = [
        f"fig6_lambda.pdf -> {path}",
        "  Willow memory data: logical error per round versus distance, log scale, with the fitted "
        "error-suppression factor Lambda from lambda.csv in the legend. Solid: our priors under "
        "PyMatching MWPM; dashed: Google's shipped decoders.",
    ]
    return "\n".join(header + lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "research" / "prx_quantum" / "figures")
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--prior-sweep-deep", type=Path, default=None)
    ap.add_argument("--prior-sweep-shallow", type=Path, default=None)
    ap.add_argument("--frontier", type=Path, default=None)
    ap.add_argument("--willow", type=Path, default=None)
    ap.add_argument("--willow-factorial", type=Path, default=None)
    ap.add_argument("--theory", type=Path, default=None)
    ap.add_argument("--hetero-seeds", type=Path, default=None)
    ap.add_argument("--frontier-fig3", type=Path, default=None,
                    help="frontier run for Figure 3 (default: the independent-draws run if present)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of figure numbers to render, e.g. --only 1 4")
    args = ap.parse_args(argv)

    root = args.results_root
    deep_dir = args.prior_sweep_deep or latest_run("prior_sweep", "*gate0_deep", root)
    shallow_dir = args.prior_sweep_shallow or latest_run("prior_sweep", "*_gate0", root)
    frontier_dir = args.frontier or latest_run("frontier", "*main", root)
    willow_dir = args.willow or latest_run("willow", "*primary_allvec", root)
    fact_dir = args.willow_factorial or latest_run("willow_factorial", "*primary", root)
    theory_dir = args.theory or (root / "theory")

    print("Run directories")
    for name, p in [
        ("prior_sweep (deep)", deep_dir),
        ("prior_sweep (shallow)", shallow_dir),
        ("frontier", frontier_dir),
        ("willow", willow_dir),
        ("willow_factorial", fact_dir),
        ("theory", theory_dir),
    ]:
        print(f"  {name:24s} {p}")
    print()

    setup_style()
    out_dir = Path(args.out)
    only = set(args.only) if args.only else None

    jobs = {
        "1": lambda: figure1(deep_dir / "rows.csv" if deep_dir else None,
                             shallow_dir / "rows.csv" if shallow_dir else None, out_dir),
        "2": lambda: figure2(Path(theory_dir), out_dir),
        "3": lambda: figure3(args.frontier_fig3 or latest_run("frontier", "*draws", root) or frontier_dir,
                             out_dir, hetero_dir=args.hetero_seeds or latest_run("hetero_seeds", "*main", root)),
        "4": lambda: figure4(willow_dir, out_dir),
        "5": lambda: figure5(fact_dir, out_dir),
        "6": lambda: figure6(willow_dir, out_dir),
    }

    summaries: List[str] = []
    for key, job in jobs.items():
        if only is not None and key not in only:
            continue
        try:
            s = job()
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")
            s = None
        if s:
            summaries.append(s)

    print("\n" + "=" * 78)
    print("FIGURE SUMMARY")
    print("=" * 78)
    for s in summaries:
        print(s)
        print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
