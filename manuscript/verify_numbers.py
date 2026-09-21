#!/usr/bin/env python
"""Re-derive every number quoted in the manuscript from the result files.

Each check states the manuscript's claim and the value recomputed from the CSVs,
and fails loudly if they disagree beyond the stated tolerance. Run before every
submission build:  python manuscript/verify_numbers.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "research" / "prx_quantum" / "results"
FAILURES: list[str] = []


def latest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(RES / pattern)))
    return Path(hits[-1]) if hits else None


import re as _re

MANUSCRIPT_PRESENT = (Path(__file__).resolve().parent / "prior_value.tex").exists()


def _manuscript_text() -> str:
    """The manuscript with its generated tables inlined and whitespace collapsed.

    The manuscript source is not distributed with the code. Without it the numeric
    checks still run against the result files; the checks that also require a quoted
    phrase to appear in the text are reported as skipped.
    """
    here = Path(__file__).resolve().parent
    if not MANUSCRIPT_PRESENT:
        return ""
    t = (here / "prior_value.tex").read_text()
    t = _re.sub(r"\\(?:input|tablebody)\{(tables/[^}]+)\}",
                lambda m: (here / (m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))).read_text()
                if (here / (m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))).exists() else m.group(0), t)
    return _re.sub(r"\s+", " ", t)

TEXT = _manuscript_text()


def check(name: str, claimed, actual, tol=0.05, unit="", text=None):
    """Relative-tolerance check of the data against the manuscript's claim.

    ``tol`` is relative unless claimed == 0. If ``text`` is given it is a regex
    that must occur in the manuscript: the number being checked has to be the
    number the paper actually prints, not only a value written in this script.
    """
    if actual is None or (isinstance(actual, float) and not np.isfinite(actual)):
        FAILURES.append(f"{name}: no data")
        print(f"  MISSING  {name}")
        return
    if claimed == 0:
        ok = abs(actual) <= tol
    else:
        ok = abs(actual - claimed) <= tol * abs(claimed)
    in_text = True
    if text is not None and MANUSCRIPT_PRESENT:
        in_text = _re.search(text, TEXT) is not None
        if not in_text:
            FAILURES.append(f"{name}: claimed value not found in the manuscript ({text})")
    flag = "ok      " if (ok and in_text) else ("MISMATCH" if not ok else "NOT-IN-TEXT")
    if not ok:
        FAILURES.append(f"{name}: manuscript {claimed}{unit}, data {actual:.4g}{unit}")
    print(f"  {flag} {name}: manuscript {claimed}{unit}, data {actual:.4g}{unit}"
          + ("" if text is None else ("  [in text]" if in_text else "  [NOT IN TEXT]")))


def check_headlines():
    """Numbers stated in the abstract and introduction must appear there verbatim."""
    print("\n[abstract and introduction] headline statements")
    if not MANUSCRIPT_PRESENT:
        print("  skipped: the manuscript source is not distributed with this code")
        return
    for tex in (r"per-class priors found improve logical error rates by at most $0.5\%$ at two rounds and up to $2.7\%$ at three",
                r"converge to that limit, $1\%$ to $2\%$ below the true-rate decoder",
                r"break-even budget of $10^2$ to $10^3$ shots",
                r"reduces held-out logical error rates by $1.3\%$ to $2.2\%$",
                r"by at most $1.1\%$, whereas changing the decoder raises it by $23\%$ to $33\%$",
                r"a few tenths of a percent less than a released prior"):
        ok = _re.search(L(tex), TEXT) is not None
        print(f"  {'ok      ' if ok else 'MISSING '} abstract/intro: {tex[:70]}")
        if not ok: FAILURES.append(f"abstract statement missing: {tex[:60]}")
    # the values behind them are checked in their own sections; here we assert consistency
    f = RES / "theory" / "exact_candidates_untruncated.csv"
    if f.exists():
        d = pd.read_csv(f); best = d.groupby(["T", "p"]).gain.max()
        ok = best.loc[2].max() <= 0.005 and 0.0265 <= best.loc[3].max() <= 0.0275
        print(f"  {'ok      ' if ok else 'MISMATCH'} abstract T=2 max {100*best.loc[2].max():.3f}%, T=3 max {100*best.loc[3].max():.3f}%")
        if not ok: FAILURES.append("abstract reweighting numbers disagree with Table I")


def check_tables_current():
    """Generated tables must match what make_tables.py produces from the data now."""
    print("\n[tables] generated table bodies are current")
    import importlib.util
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("mt", here / "make_tables.py")
    mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
    for name, fn in (("tab_exact.tex", mt.tab_exact), ("tab_mechanism.tex", mt.tab_mechanism),
                     ("tab_hetero.tex", mt.tab_hetero), ("tab_budget.tex", mt.tab_budget)):
        f = here / "tables" / name
        if not MANUSCRIPT_PRESENT:
            fn()                                  # the table still has to build from the data
            print(f"  built    {name} (not distributed with the code; nothing to compare against)")
            continue
        cur = f.read_text() if f.exists() else ""
        ok = cur == fn()
        print(f"  {'ok      ' if ok else 'STALE   '} {name}")
        if not ok:
            FAILURES.append(f"{name} is stale; rerun manuscript/make_tables.py")


def check_exact_theory():
    """Table I: untruncated best-found gains, and the prose that quotes them."""
    print("\n[Sec III] best reweighting found, untruncated (Table I)")
    f = RES / "theory" / "exact_candidates_untruncated.csv"
    if not f.exists():
        FAILURES.append("exact_candidates_untruncated.csv missing"); print("  MISSING"); return
    d = pd.read_csv(f)
    if (d.mass_deficit.abs() > 1e-12).any():
        FAILURES.append("an untruncated model does not sum to one")
    best = d.loc[d.groupby(["T", "p"]).gain.idxmax()].set_index(["T", "p"])
    g = lambda T, p: 100 * float(best.loc[(T, p), "gain"])
    check("T=2 gain range, low end", 0.03, g(2, 0.0005), tol=0.02, unit="%", text=L(r"it is $0.03\%$ to $0.48\%$"))
    check("T=2 gain range, high end", 0.48, g(2, 0.008), tol=0.02, unit="%")
    check("T=3 gain, top of range", 2.7, g(3, 0.0005), tol=0.02, unit="%", text=L(r"$2.5\%$--$2.7\%$ at the three lowest error rates"))
    check("T=3 gain, bottom of range", 2.5, g(3, 0.002), tol=0.02, unit="%")
    check("duration factor at p=5e-4", 90, g(3, 0.0005) / g(2, 0.0005), tol=0.01, text=L(r"a factor of $90$ at"))
    check("duration factor at p=2e-3", 14, g(3, 0.002) / g(2, 0.002), tol=0.04, text=L(r"falling to $14$ at"))
    t2 = [g(2, p) for p in (0.0005, 0.001, 0.002, 0.004, 0.008)]
    check("T=2 span over the p sweep", 16, max(t2) / min(t2), tol=0.03, text=L(r"the factor of $16$ that sweeping"))
    check("drop at p=4e-3, T=3", 0.42, g(3, 0.004), tol=0.01, unit="%", text=L(r"to $0.42\%$ at"))
    scan = 100 * float(d[(d["T"] == 3) & (np.isclose(d.p, 0.004)) & (d.candidate == "scan optimum")].gain.iloc[0])
    check("original scan value at p=4e-3", 0.41, scan, tol=0.01, unit="%", text=L(r"against $0.41\%$ for the original"))
    rb = RES / "theory" / "optimum_robustness_d3_T3_p0.004.csv"
    if rb.exists():
        r = pd.read_csv(rb)
        if str(r.verdict.iloc[0]) != "confirmed" or float(r.improvement.iloc[0]) > 5e-4:
            FAILURES.append("robustness re-test no longer confirms the drop, or improved it by >0.05 points")
        print(f"  ok       robustness re-test: {r.verdict.iloc[0]}, improvement {100*float(r.improvement.iloc[0]):.4f} points")


def check_mechanism():
    """Table II and the independent check of Lemma 1."""
    print("\n[Sec III] operational split and the independent check of Lemma 1")
    rows = []
    for f in ("three_way_surface_d3_T2.csv", "three_way_surface_d3_T3.csv", "three_way_repetition_d5_T3.csv"):
        path = RES / "theory" / f
        if not path.exists():
            FAILURES.append(f"{f} missing"); print(f"  MISSING {f}"); continue
        d = pd.read_csv(path)
        if "mpe_disagree" not in d.columns:
            FAILURES.append(f"{f} predates the independent max-product check"); continue
        n_dis, n_chk = int(d.mpe_disagree.sum()), int(d.mpe_checked.sum())
        ok = n_dis == 0 and n_chk == int(d.n_syndromes.sum())
        print(f"  {'ok      ' if ok else 'PROBLEM '} {f}: matching vs max-product MPE, {n_dis} disagreements "
              f"over {n_chk} syndromes")
        if not ok: FAILURES.append(f"{f}: matching disagrees with the independent MPE reference")
        rows.append(d)
    if len(rows) < 3:
        return
    t2, t3, rep = rows
    check("T=2 decision-rule term, low end", -0.28, float(t2["gap_degeneracy_%"].min()), tol=0.03, unit="%",
          text=L(r"lies between $-0.28\%$ and $+0.48\%$"))
    check("T=2 decision-rule term, high end", 0.48, float(t2["gap_degeneracy_%"].max()), tol=0.02, unit="%")
    check("T=2 model term, low end", 3.5, float(t2["gap_decomposition_%"].min()), tol=0.02, unit="%",
          text=L(r"between $3.5\%$ and $14.3\%$"))
    check("T=2 model term, high end", 14.3, float(t2["gap_decomposition_%"].max()), tol=0.01, unit="%")
    share = 100 * t3["gap_decomposition_%"] / t3["gap_total_%"]
    check("T=3 model share, low end", 86, float(share.min()), tol=0.01, unit="%", text=L(r"accounts for $86\%$ to $96\%$"))
    check("T=3 model share, high end", 96, float(share.max()), tol=0.01, unit="%")
    check("T=3 total gap, low end", 11, float(t3["gap_total_%"].min()), tol=0.03, unit="%",
          text=L(r"grown to between $11\%$ and $22\%$"))
    check("T=3 total gap, high end", 22, float(t3["gap_total_%"].max()), tol=0.02, unit="%")
    rule3 = t3[t3.p < 0.003]["gap_degeneracy_%"]
    check("T=3 decision-rule term at low p", 2.4, float(rule3.mean()), tol=0.03, unit="%",
          text=L(r"$2.4\%$ at the three lower error rates"))
    if rep["gap_decomposition_%"].abs().max() > 5e-4:
        FAILURES.append("repetition-code model term is no longer zero")
    else:
        print("  ok       repetition-code model term is zero at every p")


def check_reweight_mechanism():
    """Does the best reweighting move matching toward coset decisions? (Sec. III)"""
    print("\n[Sec III] reweighting versus the decision-rule term")
    f = RES / "theory" / "reweight_mechanism.csv"
    if not f.exists():
        FAILURES.append("reweight_mechanism.csv missing"); print("  MISSING"); return
    d = pd.read_csv(f)
    d["cut"] = 100 * (1 - d.dis_best_vs_mldec / d.dis_phys_vs_mldec)
    t2, t3 = d[d["T"] == 2], d[d["T"] == 3]
    gap2 = float((100 * (t2.gain_best - np.maximum(t2.rule_term, 0))).abs().max())
    check("T=2 |gain - positive part of rule term|, max", 0.2, gap2, tol=0.15, unit=" pts",
          text=L(r"within $0.2$ percentage points of the positive part"))
    # exact risk attribution: moves toward vs away from the decomposed model's coset decisions
    err = float((d.gain_toward + d.gain_away - d.gain_best).abs().max())
    ok = err < 1e-12
    print(f"  {'ok      ' if ok else 'MISMATCH'} attribution sums to the gain (max error {err:.1e})")
    if not ok: FAILURES.append("risk attribution does not sum to the gain")
    sh = 100 * d.gain_toward / d.gain_best
    s3 = sh[(d["T"] == 3) & (d.p < 0.003)]; s34 = float(sh[(d["T"] == 3) & np.isclose(d.p, 0.004)].iloc[0]); s2 = sh[d["T"] == 2]
    check("share toward, T=3 p<=2e-3, min", 97.7, float(s3.min()), tol=0.05 / 97.7, unit="%")
    check("share toward, T=3 p<=2e-3, max", 99.9, float(s3.max()), tol=0.05 / 99.9, unit="%",
          text=L(r"account for $97.7\%$ to $99.9\%$ of the gain at three rounds"))
    check("share toward, T=3 p=4e-3", 83, s34, tol=0.5 / 83, unit="%", text=L(r"for $83\%$ at $p = 4\times10^{-3}$"))
    check("share toward, T=2 min", 75, float(s2.min()), tol=0.5 / 75, unit="%")
    check("share toward, T=2 max", 99.7, float(s2.max()), tol=0.05 / 99.7, unit="%",
          text=L(r"and for $75\%$ to $99.7\%$ at two rounds"))
    neg_pts = d[d.rule_term < 0]
    ok = len(neg_pts) > 0 and bool((neg_pts.gain_toward > 0).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} moves toward help even where the rule term is negative ({len(neg_pts)} points)")
    if not ok: FAILURES.append("toward-moves claim at negative rule term fails")
    ok = bool((d.gain_away > 0).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} moves away also reduce the risk at every point")
    if not ok: FAILURES.append("away-moves claim fails")
    over = (t3[t3.p < 0.003].gain_best > t3[t3.p < 0.003].rule_term).all()
    under = float(t3[np.isclose(t3.p, 0.004)].gain_best.iloc[0]) < float(t3[np.isclose(t3.p, 0.004)].rule_term.iloc[0])
    ok = bool(over and under)
    print(f"  {'ok      ' if ok else 'MISMATCH'} gain exceeds the rule term at the three lower T=3 rates, not at p=4e-3")
    if not ok: FAILURES.append("gain-vs-rule-term ordering claim no longer holds")


def check_hetero_seeds():
    """Ten-draw heterogeneity study: every sentence-level claim about it."""
    import json
    from scipy.stats import spearmanr
    print("\n[Sec IV] homogeneous-prior penalty over ten noise draws")
    h = latest("hetero_seeds/*_main")
    if h is None:
        FAILURES.append("hetero_seeds run missing"); print("  MISSING"); return
    d = pd.read_csv(h / "rows.csv"); x = d.noise_extra_json.map(json.loads)
    d["sig"] = x.map(lambda e: e["sigma"]); d["mp"] = x.map(lambda e: e["mean_preserving"])
    d["lmax"] = x.map(lambda e: e["lambda_max"])
    d["pen"] = 100 * d.estimator_params_json.map(lambda e: json.loads(e)["penalty_vs_oracle"])
    raw, mp = d[~d.mp], d[d.mp]
    med = lambda f, sg, dist: float(f[np.isclose(f.sig, sg) & (f.distance == dist)].pen.median())
    for sg in (0.6, 1.0):
        m = [med(raw, sg, dist) for dist in (3, 5, 7)]
        ok = m[0] < m[1] < m[2]
        print(f"  {'ok      ' if ok else 'MISMATCH'} median penalty grows with distance at sigma={sg}: {m}")
        if not ok: FAILURES.append(f"median penalty not increasing with d at sigma={sg}")
    r3 = raw[np.isclose(raw.sig, 1.0) & (raw.distance == 3)]
    # worst qubit over the qubits the circuit uses (analysis.worst_qubit), exact permutation p-values
    wq = pd.read_csv(latest("hetero_seeds/*_main") / "worst_qubit_active.csv")
    wq = wq[np.isclose(wq.sigma, 1.0)].set_index("distance")
    for dist, rho, pv, pat in ((3, 0.89, 0.001, r"is $0.89$ (exact permutation $p = 0.001$)"),
                               (5, 0.36, 0.31, r"$0.36$ ($p = 0.31$) at $d = 5$"),
                               (7, 0.62, 0.06, r"$0.62$ ($p = 0.06$) at $d = 7$")):
        check(f"worst active qubit rank correlation, d={dist}", rho, float(wq.rho_active[dist]), tol=0.005 / rho, text=L(pat))
        check(f"worst active qubit exact p, d={dist}", pv, float(wq.p_exact_two_sided[dist]), tol=0.2)   # one significant figure
    check("penalty range at d=3, low", -1.5, float(r3.pen.min()), tol=0.05, unit="%", text=L(r"ranges from $-1.5\%$ to $+100\%$"))
    check("penalty range at d=3, high", 100, float(r3.pen.max()), tol=0.01, unit="%")
    wm3 = max(float(raw[np.isclose(raw.sig, s_) & (raw.distance == 3)].pen.max() / med(raw, s_, 3)) for s_ in (0.6, 1.0))
    check("worst/median at d=3", 6, wm3, tol=0.02, text=L(r"costs up to six times the median"))
    wm57 = [float(raw[np.isclose(raw.sig, s_) & (raw.distance == dist)].pen.max() / med(raw, s_, dist))
            for s_ in (0.6, 1.0) for dist in (5, 7)]
    ok = all(1.5 <= v <= 2.1 for v in wm57)
    print(f"  {'ok      ' if ok else 'MISMATCH'} worst/median at d=5,7 about twice: {[round(v, 2) for v in wm57]}")
    if not ok: FAILURES.append("worst/median at d=5,7 is not about twice")
    diff06 = max(abs(med(raw, 0.6, dd) - med(mp, 0.6, dd)) for dd in (3, 5, 7))
    diff10 = max(abs(med(raw, 1.0, dd) - med(mp, 1.0, dd)) for dd in (3, 5, 7))
    ok = diff06 < 1.0 and diff10 <= 2.5
    print(f"  {'ok      ' if ok else 'MISMATCH'} mean-preserving shift: {diff06:.2f} pts at 0.6, {diff10:.2f} pts at 1.0")
    if not ok: FAILURES.append("mean-preserving shift claim no longer holds")
    if MANUSCRIPT_PRESENT and _re.search(L(r"by at most $2.5$ points at $\sigma = 1$"), TEXT) is None:
        FAILURES.append("mean-preserving sentence not found in text")


def check_device_like():
    """Measurement-only heterogeneity: the robustness frontier quoted in Sec. IV."""
    import json
    print("\n[Sec IV] measurement-only heterogeneity frontier")
    d = latest("frontier/*device_like")
    if d is None:
        FAILURES.append("device_like frontier missing"); print("  MISSING"); return
    df = pd.read_csv(d / "rows.csv")
    df["sigma"] = df.noise_extra_json.map(lambda x: json.loads(x).get("sigma_h"))
    df["seed"] = df.noise_extra_json.map(lambda x: json.loads(x).get("het_seed"))
    key = ["distance", "rounds", "sigma", "seed"]
    orc = df[df.estimator == "oracle_physical"].groupby(key).ler.median().rename("o")
    m = df.merge(orc, on=key, how="left")
    m["excess"] = 100 * (m.ler / m.o - 1)

    def med(est, sig, n=None):
        q = m[(m.estimator == est) & (np.isclose(m.sigma, sig))]
        if n is not None: q = q[q.n_cal_shots == n]
        return q.groupby("distance").excess.median()

    st65, st10 = med("static", 0.65), med("static", 1.0)
    for dd, v in ((3, 4.6), (5, 3.4), (7, 5.1)):
        check("static excess, sigma=0.65, d=%d" % dd, v, float(st65[dd]), tol=0.05, unit="%")
    for dd, v in ((3, 8.1), (5, 13.5), (7, 19.2)):
        check("static excess, sigma=1, d=%d" % dd, v, float(st10[dd]), tol=0.05, unit="%")
    if not (st10[3] < st10[5] < st10[7]):
        FAILURES.append("penalty no longer grows with distance at sigma=1")
    else:
        print("  ok       penalty grows with distance at sigma=1")

    ps = med("pooled_shrink", 0.65, 3000); pij = med("pij", 0.65, 3000)
    for dd, v in ((3, 0.7), (5, 2.1), (7, 5.3)):
        check("pooled+shrinkage at N=3e3, d=%d" % dd, v, float(ps[dd]), tol=0.10, unit="%")
    for dd, v in ((5, 10.2), (7, 13.6)):
        check("raw p_ij at N=3e3, d=%d" % dd, v, float(pij[dd]), tol=0.05, unit="%")

    # the claim that raw p_ij is decisively worse than not calibrating
    q = df[(np.isclose(df.sigma, 0.65)) & (df.estimator == "pij") & (df.n_cal_shots == 3000)
           & (df.distance.isin([5, 7]))]
    worst = float(q.mcnemar_p_exact.max()); nrep = len(q)
    if nrep != 48:
        FAILURES.append("expected 48 p_ij replicates at d=5,7, found %d" % nrep)
    if worst >= 1e-14:
        FAILURES.append("raw p_ij not significant on every replicate (worst p=%.2g)" % worst)
    print("  %s raw p_ij worse than static on all %d replicates, worst exact McNemar p=%.1e"
          % ("ok      " if (worst < 1e-14 and nrep == 48) else "PROBLEM ", nrep, worst))

    # the budget at sigma=0.65: within one percent of the oracle
    for dd, n in ((3, 3000), (5, 10000), (7, 30000)):
        e = float(med("pooled_shrink", 0.65, n)[dd])
        ok_ = e <= 1.0
        print("  %s budget d=%d at N=%d: excess %.2f%% (claim: within one percent)"
              % ("ok      " if ok_ else "PROBLEM ", dd, n, e))
        if not ok_:
            FAILURES.append("d=%d at N=%d is %.2f%% above the oracle, not within one percent" % (dd, n, e))


def check_exact_vs_mc():
    """The exact and sampled pipelines must agree at matched conditions."""
    print("\n[Sec IV] exact-versus-Monte-Carlo validation, d=3 T=3 p=1e-3")
    d = latest("prior_sweep/*_gate0")
    if d is None:
        FAILURES.append("shallow gate0 missing"); print("  MISSING"); return
    df = pd.read_csv(d / "rows.csv")
    sel = df[(df.noise_family == "si1000:google2025") & (df.distance == 3) & (df.rounds == 3)
             & (np.isclose(df.p_true, 0.001))]
    phys = sel[sel.prior_id == "oracle_physical"]
    cls = sel[sel.prior_id == "class_search"]
    if phys.empty or cls.empty:
        FAILURES.append("d3 T3 p=1e-3 cell missing"); print("  MISSING"); return
    mc_gain = 100 * (1 - float(cls.ler.iloc[0]) / float(phys.ler.iloc[0]))
    check("Monte Carlo gain at the searched scales", 2.535, mc_gain, tol=0.02, unit="%")

    # The exact counterpart is recomputed by simulation.prx.theory.exact_at_mc_scales,
    # which reads the same searched scales out of this gate0 run, so nothing here is
    # hand-copied. Regenerate with:
    #   python -m simulation.prx.theory.exact_at_mc_scales --tol 1e-9
    ex = RES / "theory" / "exact_at_mc_scales.csv"
    if not ex.exists():
        FAILURES.append("theory/exact_at_mc_scales.csv missing (run the module)")
        print("  MISSING  exact-at-MC-scales validation"); return
    e = pd.read_csv(ex).iloc[0]
    for c in ("space", "time", "diag", "boundary"):
        if not np.isclose(float(e["s_%s" % c]), float(cls["scale_%s" % c].iloc[0]), rtol=1e-9):
            FAILURES.append("exact run used a different %s scale than gate0" % c)
    print("  ok       exact run used the gate0 scales, run %s" % e["gate0_run"])
    exact_gain = 100 * float(e["exact_gain"])
    check("exact gain at the same scales", 2.554, exact_gain, tol=0.02, unit="%")
    # the manuscript states this as a bound ("agree to two hundredths of a
    # percentage point"), so assert the bound rather than an equality
    disagree = 100 * float(e["abs_gain_difference"])
    if disagree <= 0.02:
        print("  ok       exact-vs-MC disagreement %.4f points, within the stated 0.02" % disagree)
    else:
        FAILURES.append("exact-vs-MC disagreement %.4f points exceeds the stated 0.02" % disagree)
        print("  MISMATCH exact-vs-MC disagreement %.4f points exceeds 0.02" % disagree)
    # the truncation bound must be far below the effect being resolved
    slack = 100 * float(e["neglected_mass"]) / float(e["exact_R_phys"])
    if slack > 0.05 * abs(exact_gain):
        FAILURES.append("truncation slack %.3g%% is not negligible against a %.3g%% gain"
                        % (slack, exact_gain))
    rel_slack = float(e["neglected_mass"]) / float(e["exact_R_phys"])
    check("truncation bound as a fraction of the risk", 4e-7, rel_slack, tol=0.15)
    print("  ok       %d syndromes kept at tol %.0e" % (int(e["syndromes_kept"]), float(e["tol"])))

    # independent cross-check: the exact physical risk must agree with the
    # d=3 T=3 risk surface computed separately at a looser tolerance.
    rs = RES / "theory" / "risk_surface_surf_d3_T3.csv"
    if rs.exists():
        r = pd.read_csv(rs)
        row = r[np.isclose(r.p, 0.001)]
        if not row.empty:
            a, b = float(e["exact_R_phys"]), float(row.R_phys.iloc[0])
            rel = abs(a - b) / a
            ok_ = rel < 1e-3
            print("  %s exact R_phys agrees with the risk surface to %.1e"
                  % ("ok      " if ok_ else "PROBLEM ", rel))
            if not ok_:
                FAILURES.append("exact R_phys disagrees with risk_surface_surf_d3_T3 (%.2e)" % rel)


def check_gate0():
    print("\n[Sec IV] Monte Carlo headroom, deep run")
    d = latest("prior_sweep/*gate0_deep")
    if d is None:
        FAILURES.append("gate0_deep missing"); print("  MISSING"); return
    df = pd.read_csv(d / "rows.csv")
    df["cond"] = df.distance.astype(str) + "|" + df.rounds.astype(str) + "|" + df.p_true.astype(str)
    out = {}
    for cond, g in df.groupby("cond"):
        phys = g[g.prior_id == "oracle_physical"]
        cls = g[g.prior_id == "class_search"]
        if phys.empty or cls.empty:
            continue
        out[cond] = 100 * (1 - cls.ler.iloc[0] / phys.ler.iloc[0])
    d5_Td = [v for k, v in out.items() if k.startswith("5|5|")]
    d5_3d = [v for k, v in out.items() if k.startswith("5|15|")]
    d5_5d = [v for k, v in out.items() if k.startswith("5|25|")]
    d7 = [v for k, v in out.items() if k.startswith("7|")]
    check("d=5 T=d headroom, min", 1.6, min(d5_Td), tol=0.15, text=L(r"$1.6$--$2.4\%$ at $T = d$"))
    check("d=5 T=d headroom, max", 2.4, max(d5_Td), tol=0.15)
    check("d=5 T=3d headroom, max", 3.1, max(d5_3d), tol=0.15)
    check("d=5 T=5d headroom, max", 3.7, max(d5_5d), tol=0.15)
    check("d=7 headroom, max", 2.1, max(d7), tol=0.15, text=L(r"at distance seven it is $1.2$--$2.1\%$"))
    # uniform rescaling: one-sided null. Check both the best gain and the worst loss,
    # pooling the deep and shallow runs over below-threshold conditions only.
    gains, losses, at_f4 = [], [], []
    for run in sorted(glob.glob(str(RES / "prior_sweep" / "2026*gate0*"))):
        dd = pd.read_csv(Path(run) / "rows.csv")
        dd["cond"] = dd.noise_family + "|" + dd.distance.astype(str) + "|" + dd.rounds.astype(str) + "|" + dd.p_true.astype(str)
        for cond, g in dd.groupby("cond"):
            phys = g[g.prior_id == "oracle_physical"]
            u = g[g.prior_id.str.startswith("uniform_s=")].copy()
            if phys.empty or u.empty:
                continue
            thr = 0.005 if "si1000" in str(g.noise_family.iloc[0]) else 0.007
            if float(g.p_true.iloc[0]) >= thr:
                continue
            u["s"] = u.prior_id.str.extract(r"=([0-9.]+)")[0].astype(float)
            rel = 100 * (u.ler / phys.ler.iloc[0] - 1)
            gains.append(-rel.min())
            losses.append(rel.max())
            f4 = rel[np.isclose(u.s, 4.0)]
            if len(f4):
                at_f4.append(float(f4.iloc[0]))
    check("best gain from any uniform rescale, max", 1.7, max(gains), tol=0.05, unit="%",
          text=L(r"the best global scale gains a median of $0.5\%$ and at most $1.7\%$"))
    check("best gain from any uniform rescale, median", 0.5, float(np.median(gains)), tol=0.15, unit="%",
          text=L(r"by a median of $0.5\%$ and at most $1.7\%$"))
    check("worst loss, median over conditions", 14.8, float(np.median(losses)), tol=0.05, unit="%")
    check("worst loss from any uniform rescale", 28.9, max(losses), tol=0.05, unit="%",
          text=L(r"as much as $28.9\%$ at $f = 4$"))
    check("worst loss, rounded in the abstract", 29, max(losses), tol=0.02, unit="%",
          text=L(r"overestimating the noise fourfold costs up to $29\%$"))
    check("mean degradation at f=4", 13.7, float(np.mean(at_f4)), tol=0.05, unit="%")


def check_fig1():
    """Figure 1 caption: uniform-rescale curves at p = 0.002 and per-class search scales."""
    print("\n[Fig 1] uniform rescaling at p=0.002 and best per-class scales")
    deep, sh = latest("prior_sweep/*_gate0_deep"), latest("prior_sweep/*_gate0")
    if deep is None or sh is None:
        FAILURES.append("prior_sweep runs missing"); print("  MISSING"); return
    D, S = pd.read_csv(deep / "rows.csv"), pd.read_csv(sh / "rows.csv")
    fam = "si1000:google2025"
    def curves(df):
        out = []
        for (d, T, p), g in df[df.noise_family == fam].groupby(["distance", "rounds", "p_true"]):
            l0 = g[g.prior_id == "oracle_physical"].ler.iloc[0]
            for _, r in g[g.prior_id.str.startswith("uniform_s=")].iterrows():
                out.append(dict(d=d, T=T, p=p, f=float(r.prior_id.split("=")[1]), pct=100 * (r.ler / l0 - 1),
                                n=int(r.n_test_shots)))
        return pd.DataFrame(out)
    a = pd.concat([curves(D), curves(S[S.distance == 3])])
    a = a[np.isclose(a.p, 0.002)]
    big, small = a[a.d >= 5], a[a.d == 3]
    check("Fig1 d>=5 test shots", 1.6e7, float(big.n.unique().item()), tol=0, text=L(r"($1.6\times10^{7}$ test shots)"))
    check("Fig1 d=3 test shots", 4e6, float(small.n.unique().item()), tol=0, text=L(r"(dashed, $4\times10^{6}$ shots)"))
    down = big[big.f < 1].pct.abs().max()
    ok = down < 1.0
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig1 scaling down at d>=5: max |change| {down:.2f}% (caption < 1%)")
    if not ok: FAILURES.append("Fig1 scale-down change not below 1%")
    f4 = big[big.f == 4].pct
    check("Fig1 f=4 min", 13, float(f4.min()), tol=0.5 / 13, unit="%")
    check("Fig1 f=4 max", 19, float(f4.max()), tol=0.5 / 19, unit="%", text=L(r"$f = 4$ raises it by $13\%$ to $19\%$"))
    d3 = small.pct.abs().max()
    ok = d3 < 2.0
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig1 d=3 max |change| {d3:.2f}% (caption below 2%)")
    if not ok: FAILURES.append("Fig1 d=3 change not below 2%")
    C = ["space", "time", "diag", "boundary"]
    cs = pd.concat([D[(D.noise_family == fam) & (D.prior_id == "class_search")],
                    S[(S.noise_family == fam) & (S.prior_id == "class_search") & (S.distance == 3)
                      & S.p_true.isin([0.0015, 0.002, 0.003])]])
    check("Fig1 configurations", 20, float(len(cs)), tol=0, text=L(r"for $20$ configurations"))
    below = cs.scale_boundary < 1 - 1e-6
    check("Fig1 boundary below one", 17, float(below.sum()), tol=0, text=L(r"below one in $17$ of $20$ configurations"))
    check("Fig1 boundary below one at d>=5", 15, float(below[cs.distance >= 5].sum()), tol=0,
          text=L(r"including all $15$ at $d \ge 5$"))
    check("Fig1 boundary d>=5 count", 15, float((cs.distance >= 5).sum()), tol=0)
    lo = float(cs[[f"scale_{c}" for c in C[:3]]].min().min())
    ok = lo >= 1 - 2e-3          # grid point 0.9987 is the grid's representation of 1
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig1 other classes never below one: min {lo:.4f}")
    if not ok: FAILURES.append("Fig1 non-boundary scale below one")
    import json
    grids = {tuple(np.round(json.loads(x)["grid"], 4)) for x in cs.estimator_params_json}
    g = next(iter(grids))
    ok = len(grids) == 1 and len(g) == 9 and np.isclose(g[0], 0.35) and np.isclose(g[-1], 2.85)
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig1 search grid: {len(g)} points in [{g[0]}, {g[-1]}]")
    if not ok: FAILURES.append("Fig1 search grid differs from caption")
    seeds_differ = all(json.loads(r.estimator_params_json)["seed_search"] != r.seed_test for _, r in cs.iterrows())
    print(f"  {'ok      ' if seeds_differ else 'MISMATCH'} Fig1 search shots separate from test shots")
    if not seeds_differ: FAILURES.append("Fig1 search seed equals test seed")
    check("Fig1 grid text", 0, 0, text=L(r"nine-point grid in $[0.35, 2.85]$"))


def check_fig2():
    """Figure 2: exact, untruncated sources and the caption's bound on the share recovered."""
    print("\n[Fig 2] exact reweighting versus ML decoding")
    th = RES / "theory"
    c = pd.read_csv(th / "exact_candidates_untruncated.csv")
    best = c.loc[c.groupby(["T", "p"]).gain.idxmax()]
    ok = (best.truncation == "none").all() and best.mass_deficit.abs().max() < 1e-13
    print(f"  {'ok      ' if ok else 'MISMATCH'} surface points untruncated: {len(best)} points, "
          f"max |mass deficit| {best.mass_deficit.abs().max():.1e}")
    if not ok: FAILURES.append("Fig2 surface points not untruncated")
    shares = list(100 * best.recovered)
    reps = sorted((th / "rep_untruncated").glob("risk_surface_rep_d3_T*.csv"))
    ok = len(reps) == 3
    for f in reps:
        r = pd.read_csv(f)
        full = (r.syndromes_kept == 2 ** (2 * (r.rounds + 1))).all()   # d = 3: 2(T+1) detectors
        ok &= bool(full) and r.bound_width.max() < 1e-13
        shares += list(100 * r.frac_of_ml_gap_recovered)
    print(f"  {'ok      ' if ok else 'MISMATCH'} repetition curves: {len(reps)} files, complete enumeration")
    if not ok: FAILURES.append("Fig2 repetition curves missing or truncated")
    check("Fig2 largest share of the gap recovered", 19, max(shares), tol=0.5 / 19, unit="%",
          text=L(r"It never exceeds $19\%$"))


def check_fig4():
    """Figure 4: exact joint-bootstrap intervals (all replicates' per-shot vectors) and the caption."""
    print("\n[Fig 4] hardware shot budget")
    w = latest("willow/*_primary_allvec")
    j, f = pd.read_csv(w / "frontier_joint_ci.csv"), pd.read_csv(w / "frontier.csv")
    m = j.merge(f, on=["distance", "estimator", "n_cal"], suffixes=("", "_old"))
    diff = float((m.rel_ler - m.rel_ler_old).abs().max())
    ok = len(m) == len(j[j.estimator != "rl_shipped"]) and diff < 1e-9
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig4 points equal the frontier estimates ({len(m)} points, max diff {diff:.1e})")
    if not ok: FAILURES.append("Fig4 points differ from frontier.csv")
    # the joint bootstrap at N = 3000 must agree with the independent procedure bootstrap (same scheme at d = 3, 5)
    P = pd.read_csv(w / "inference_procedure.csv"); P = P[P.comparison == "pooled+shrinkage vs SI1000"].set_index("distance")
    js = j[(j.estimator == "pooled_shrink") & (j.n_cal == 3000)].set_index("distance")
    dev = max(max(abs(js.ci_lo[dd] - P.ci95_lo[dd]), abs(js.ci_hi[dd] - P.ci95_hi[dd])) for dd in (3, 5))
    ok = dev < 0.001
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig4 joint intervals match the procedure bootstrap at N=3000, d=3,5 (max dev {100*dev:.3f} points)")
    if not ok: FAILURES.append("Fig4 joint intervals disagree with the procedure bootstrap")
    reps = j[j.estimator != "rl_shipped"].groupby("n_cal").n_replicates.unique()
    ok = all(set(r) == ({5} if n <= 10000 else {1}) for n, r in reps.items())
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig4 five draws up to 1e4 shots, one above")
    if not ok: FAILURES.append("Fig4 calibration-draw counts differ from caption")
    check("Fig4 draws text", 0, 0, text=L(r"mean over five calibration draws up to $10^4$ shots and one draw above"))
    for dd, n in ((3, 72), (5, 48), (7, 16)):
        check(f"Fig4 experiments d={dd}", n, float(f[f.distance == dd].n_leaves.iloc[0]), tol=0)
    check("Fig4 experiments text", 0, 0, text=L(r"($72$, $48$ and $16$ experiments)"))
    ok = bool((j[j.distance < 7].groups_resampled).all() and not (j[j.distance == 7].groups_resampled).any())
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig4 groups resampled at d=3,5 only")
    if not ok: FAILURES.append("Fig4 group resampling differs from caption")
    u = float(100 * j[j.estimator == "uniform_search"].rel_ler.abs().max())
    ok = u < 0.5
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig4 uniform search max |change| {u:.2f}% (caption within 0.5%)")
    if not ok: FAILURES.append("Fig4 uniform search exceeds 0.5%")
    check("Fig4 uniform text", 0, 0, text=L(r"The uniform search stays within $0.5\%$ of the shipped prior"))
    d7 = js.loc[7]
    check("Fig4 d=7 N=3000 conditional interval, low", -1.58, 100 * float(d7.ci_lo), tol=0.03, unit="%")
    check("Fig4 d=7 N=3000 conditional interval, high", -0.91, 100 * float(d7.ci_hi), tol=0.03, unit="%")


def check_fig5():
    """Figure 5: Lambda uncertainties are bootstrap SDs, validated against the paired analysis."""
    print("\n[Fig 5] error suppression factor")
    w = latest("willow/*_primary_allvec")
    m = pd.read_csv(w / "lambda_marginal.csv").set_index("selection")
    lam = pd.read_csv(w / "lambda.csv").set_index("selection")
    d = float((m.Lambda - lam.Lambda_fit.reindex(m.index)).abs().max())
    ok = d < 1e-9 and len(m) == 8
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig5 bootstrap point values equal lambda.csv ({len(m)} decoders)")
    if not ok: FAILURES.append("Fig5 bootstrap points differ from lambda.csv")
    lp = pd.read_csv(w / "lambda_paired.csv")
    g = lp[lp.scheme == "shots+groups"]
    paired = {**dict(zip(g.prior_A, g.sd_marginal_A)), **dict(zip(g.prior_B, g.sd_marginal_B))}
    pairs = {"pymatching+si1000": "si1000_shipped", "pymatching+rl": "rl_shipped",
             "pymatching+pooled_shrink_N=3000": "pooled_shrink_N=3000_rep=0"}
    dev = max(abs(m.sd_lnLambda[a] / paired[b] - 1) for a, b in pairs.items())
    ok = dev < 0.05
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig5 count-based SD matches per-shot paired SD (max rel dev {100*dev:.1f}%)")
    if not ok: FAILURES.append("Fig5 bootstrap SD not validated")
    lib = "google:libra_decoder_with_rl_optimized_prior"
    check("Fig5 Libra Lambda sd", 0.11, float(m.sd_Lambda[lib]), tol=0.005 / 0.11,
          text=L(r"Libra gives $2.02 \pm 0.11$ against its published $2.04$"))
    if MANUSCRIPT_PRESENT:
        ok = len(_re.findall(r"2\.02 \\pm 0\.11", TEXT)) == 3 and "2.02 \\pm 0.10" not in TEXT
        print(f"  {'ok      ' if ok else 'MISMATCH'} Fig5 Libra uncertainty consistent across the text")
        if not ok: FAILURES.append("Libra Lambda uncertainty inconsistent in text")


def check_fig6():
    """Figure 6: prior-by-decoder factorial, joint-bootstrap intervals and caption claims."""
    print("\n[Fig 6] prior versus decoder")
    ci = pd.read_csv(latest("willow_factorial/*_primary") / "factorial_ci.csv")
    def rel(pr, dec, rp, rd):
        return 100 * ci[(ci.prior == pr) & (ci.decoder == dec) & (ci.ref_prior == rp) & (ci.ref_decoder == rd)].set_index("distance").rel
    M, Cr, S, R, P = "pymatching_mwpm", "pymatching_correlated", "si1000_shipped", "rl_shipped", "pooled_shrink_N=10000"
    prior_mwpm = pd.concat([-rel(R, M, S, M), -rel(P, M, S, M)])
    check("Fig6 prior change under MWPM, min", 1.6, float(prior_mwpm.min()), tol=0.05 / 1.6, unit="%")
    check("Fig6 prior change under MWPM, max", 2.4, float(prior_mwpm.max()), tol=0.05 / 2.4, unit="%",
          text=L(r"Under MWPM either change of prior is worth $1.6\%$ to $2.4\%$"))
    dec = -rel(S, Cr, S, M)
    check("Fig6 decoder change, min", 8.5, float(dec.min()), tol=0.05 / 8.5, unit="%")
    check("Fig6 decoder change, max", 34.7, float(dec.max()), tol=0.05 / 34.7, unit="%",
          text=L(r"correlated matching $8.5\%$ to $34.7\%$"))
    rl_c = -rel(R, Cr, S, Cr)
    check("Fig6 RL under corr, min", 3.8, float(rl_c.min()), tol=0.05 / 3.8, unit="%")
    check("Fig6 RL under corr, max", 5.7, float(rl_c.max()), tol=0.05 / 5.7, unit="%",
          text=L(r"keeps an advantage of $3.8\%$ to $5.7\%$, the fitted prior at most $1.4\%$"))
    fit_c = -rel(P, Cr, S, Cr)
    check("Fig6 fitted under corr, max", 1.4, float(fit_c.max()), tol=0.05 / 1.4, unit="%")
    ok = bool((ci[ci.distance < 7].groups_resampled).all() and not (ci[ci.distance == 7].groups_resampled).any())
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig6 groups resampled at d=3,5 only")
    if not ok: FAILURES.append("Fig6 group resampling differs from caption")


def check_population_limit():
    """Sec. IV: the estimator's infinite-data limit, from exact population moments."""
    print("\n[Sec IV] population limit of the per-edge estimator")
    d = latest("frontier/*_draws")
    L_ = pd.read_csv(d / "population_limit.csv")
    ok = bool(L_.test_set_reproduced.all()) and len(L_) == 120
    print(f"  {'ok      ' if ok else 'MISMATCH'} limit decoded on the frontier's own test sets ({len(L_)} conditions reproduced)")
    if not ok: FAILURES.append("population-limit test sets not reproduced")
    L_["ex"] = 100 * (L_.k_pooled_limit / L_.k_oracle - 1)
    n, b, c = L_.n_test, L_.b_pooled_limit, L_.c_pooled_limit
    z = L_.ex / (np.sqrt(((b + c) / n - ((b - c) / n) ** 2) / n) * n / L_.k_oracle * 100)
    med = L_.groupby(["sigma_h", "distance"]).ex.median()
    check("limit median excess, most negative stratum", -2.0, float(med.min()), tol=0.05, unit="%")
    check("limit median excess, least negative stratum", -1.2, float(med.max()), tol=0.05, unit="%",
          text=L(r"by $1.2\%$ to $2.0\%$ in median across strata"))
    check("limit significantly better", 79, float((z < -3).sum()), tol=0, text=L(r"errors in $79$ of the $120$ conditions"))
    check("limit significantly worse", 0, float((z > 3).sum()), tol=0, text=L(r"and significantly worse in none"))
    ok = bool((L_.k_pooled_limit == L_.k_pij_limit).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} pooled and raw limits identical")
    pc = pd.read_csv(d / "budget_per_condition.csv"); q = pc[(pc.estimator == "pooled_shrink") & (pc.n_cal == pc.n_cal.max())]
    m = L_.rename(columns={"het_seed": "draw"}).merge(q[["distance", "rounds", "p", "sigma_h", "draw", "excess"]],
                                                     on=["distance", "rounds", "p", "sigma_h", "draw"])
    diff = 100 * m.excess - m.ex
    check("finite fit minus limit, median (max over strata)", 0.3, float(diff.groupby([m.sigma_h, m.distance]).median().max()),
          tol=0.1 / 0.3 + 0.01, unit=" pts", text=L(r"lie within a median of $0.3$ percentage points of that limit"))
    check("finite fit minus limit, max", 1.3, float(diff.max()), tol=0.05 / 1.3, unit=" pts", text=L(r"and within $1.3$ in every condition"))
    above = 100 * q.excess
    check("top-budget conditions above the oracle", 7, float((above > 0).sum()), tol=0, text=L(r"$7$ of the $120$ conditions end above"))
    check("largest top-budget excess", 2.4, float(above.max()), tol=0.05 / 2.4, unit="%", text=L(r"by at most $2.4\%$"))
    e = pd.read_csv(d / "limit_edges.csv")
    bd = e[e["class"] == "boundary"]
    ok = bool((bd.frac_changed == 1).all() and (bd.lr_max < 0).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} limit lowers every boundary edge in every condition")
    if not ok: FAILURES.append("boundary-lowering claim fails")
    check("boundary median reduction", 17, 100 * (1 - float(np.exp(bd.lr_median.median()))), tol=0.5 / 17, unit="%",
          text=L(r"lowers every boundary edge, by $17\%$ in median"))
    oth = e[e["class"] != "boundary"]
    ok = bool((oth.lr_min > 0).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} limit changes non-boundary edges only upward")
    if not ok: FAILURES.append("upward-only claim for non-boundary classes fails")
    dg = e[e["class"] == "diag"]
    check("diagonal median factor", 2.0, float(np.exp(dg.lr_median.median())), tol=0.05, text=L(r"raises by a median\s*factor of two") if False else None)
    check("factor text", 0, 0, text=L(r"which it raises by a median factor of two"))
    sb = pd.read_csv(d / "seed_blocks_pooled_shrink.csv")
    b100 = sb[(sb.statistic == "beats_at_100") & np.isclose(sb.sigma_h, 0.6)].set_index("distance")
    for dist, (lo, hi) in {3: (17, 67), 5: (0, 17), 7: (0, 0)}.items():
        check(f"seed-block beats@100 d={dist} lo", lo, float(b100.lo95[dist]), tol=0.5 / max(lo, 1))
        check(f"seed-block beats@100 d={dist} hi", hi, float(b100.hi95[dist]), tol=0.5 / max(hi, 1))
    check("seed-block text", 0, 0, text=L(r"$[17, 67]$, $[0, 17]$ and $[0, 0]$"))
    ts = pd.read_csv(RES / "willow_timelike_spread" / "timelike_spread.csv")
    w = ts[ts.source == "willow"]; mo = ts[ts.source == "model"]
    check("shipped time-like sd, max", 0.05, float(w.sd_shipped.max()), tol=0.1, text=L(r"is $0.02$--$0.05$ in the shipped device model"))
    check("fitted time-like sd, min", 0.13, float(w.sd_fit.min()), tol=0.05, text=L(r"and $0.13$--$0.61$ in a pooled fit"))
    check("fitted time-like sd, max", 0.61, float(w.sd_fit.max()), tol=0.02)
    for dist, c_ in ((3, 0.30), (5, 0.33), (7, 0.32)):
        check(f"fitted time-like sd median d={dist}", c_, float(w[w.distance == dist].sd_fit.median()), tol=0.02)
    s65 = mo[np.isclose(mo.sigma, 0.65)].groupby("distance").sd_true.median()
    s50 = mo[np.isclose(mo.sigma, 0.5)].groupby("distance").sd_true.median()
    ok = 0.345 <= s65.min() and s65.max() <= 0.405 and 0.265 <= s50.min() and s50.max() <= 0.305
    print(f"  {'ok      ' if ok else 'MISMATCH'} model time-like sd: sigma=0.65 {s65.round(3).tolist()}, sigma=0.5 {s50.round(3).tolist()}")
    if not ok: FAILURES.append("model time-like spread differs from App. C")
    tw = pd.read_csv(RES / "theory" / "three_way_repetition_d5_T3.csv")
    ts_ = pd.read_csv(RES / "theory" / "three_way_surface_d3_T3.csv"); t2 = pd.read_csv(RES / "theory" / "three_way_surface_d3_T2.csv")
    ok = (tw.mpe_ties == 5444).all() and (ts_.mpe_ties == 0).all() and (t2.mpe_ties == 0).all()
    print(f"  {'ok      ' if ok else 'MISMATCH'} max-product ties: repetition 5444 per point, surface none")
    if not ok: FAILURES.append("MPE tie counts differ from the text")


def check_moment_limit_exact():
    """Sec. III: exact risk of the correlation fit's infinite-data prior, d = 3."""
    print("\n[Sec III] exact population limit of the correlation fit")
    m = pd.read_csv(RES / "theory" / "moment_limit_exact.csv")
    ok = bool((m.mass_deficit.abs() < 1e-13).all()) and bool((m.gain_limit == m.gain_limit_pooled).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} untruncated, pooled and raw limits identical")
    if not ok: FAILURES.append("moment-limit exact run truncated or pooled/raw differ")
    t2, t3 = m[m["T"] == 2].sort_values("p"), m[m["T"] == 3].sort_values("p")
    check("T=2 limit gain min", 0.09, 100 * float(t2.gain_limit.min()), tol=0.05 / 0.09 * 0.1, unit="%")
    check("T=2 limit gain max", 0.26, 100 * float(t2.gain_limit.max()), tol=0.005 / 0.26, unit="%",
          text=L(r"by $0.09\%$ to $0.26\%$ at two rounds"))
    for got, want in zip(100 * t3.gain_limit, (2.72, 2.57, 2.50, 0.31)):
        check(f"T=3 limit gain {want}", want, float(got), tol=0.005 / want, unit="%")
    check("T=3 text", 0, 0, text=L(r"by $2.72\%$, $2.57\%$, $2.50\%$ and $0.31\%$ at three"))
    diff = 100 * (t3.gain_limit - t3.gain_best_class).abs()
    check("T=3 |limit - best class|, lower three", 0.05, float(diff.iloc[:3].max()), tol=0.0 + 1e-9 + 0.0 if False else 0.01 / 0.05)
    if float(diff.iloc[:3].max()) > 0.05: FAILURES.append("T=3 limit not within 0.05 points of best class")
    check("T=3 shortfall at p=4e-3", 0.10, float(100 * (t3.gain_best_class - t3.gain_limit).iloc[3]), tol=0.05)
    ok = list((t2.gain_limit > t2.gain_best_class).values) == [True, True, True, False, False]
    print(f"  {'ok      ' if ok else 'MISMATCH'} T=2 limit exceeds the search at the three lowest p only")
    if not ok: FAILURES.append("T=2 limit/search ordering differs from the text")
    ok = bool((m.gain_limit > 0).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} limit better than the physical prior at every point")
    if not ok: FAILURES.append("limit not better than physical everywhere")


def check_dgr():
    """Sec. IV: occurrence (DGR) estimator at matched budgets, and its exact limit."""
    print("\n[Sec IV] occurrence (DGR) estimator")
    d = latest("frontier/*_draws")
    import json
    comp = json.loads((d / "companions.json").read_text())
    ok = any("draws_dgr" in k for k in comp)
    print(f"  {'ok      ' if ok else 'MISMATCH'} DGR companion run registered (reference rows asserted identical at load)")
    pc = pd.read_csv(d / "budget_per_condition.csv")
    def med(est, sg, dist, n):
        h = pc[(pc.estimator == est) & np.isclose(pc.sigma_h, sg) & (pc.distance == dist) & (pc.n_cal == n)]
        return 100 * float(np.median(h.excess)), 100 * (1 - float(h.beats_static.mean()))
    for dist, dg, ps_ in ((5, 33, 4), (7, 42, 17)):
        check(f"DGR harm at 100, sigma=1 d={dist}", dg, med("dgr", 1.0, dist, 100)[1], tol=0.5 / dg, unit="%")
        check(f"pooled harm at 100, sigma=1 d={dist}", ps_, med("pooled_shrink", 1.0, dist, 100)[1], tol=0.5 / ps_, unit="%")
    check("harm text", 0, 0, text=L(r"in $33\%$ and $42\%$ of conditions at distances five and seven, against $4\%$ and $17\%$"))
    for dist, dg, ps_ in ((3, 0.6, 3.9), (5, 2.2, 6.1), (7, 4.5, 10.8)):
        check(f"DGR excess at 3e3 sigma=1 d={dist}", dg, med("dgr", 1.0, dist, 3000)[0], tol=0.05 / dg, unit="%")
        check(f"pooled excess at 3e3 sigma=1 d={dist}", ps_, med("pooled_shrink", 1.0, dist, 3000)[0], tol=0.05 / ps_, unit="%")
    check("3e3 text", 0, 0, text=L(r"is $0.6\%$, $2.2\%$ and $4.5\%$ at distances three, five and seven, against $3.9\%$, $6.1\%$ and $10.8\%$"))
    for dist, dg, ps_ in ((3, -0.8, -2.0), (5, 0.1, -1.5), (7, 0.2, -1.0)):
        check(f"DGR top excess sigma=1 d={dist}", dg, med("dgr", 1.0, dist, 300000)[0], tol=0.051 / abs(dg), unit="%")
        check(f"pooled top excess sigma=1 d={dist}", ps_, med("pooled_shrink", 1.0, dist, 300000)[0], tol=0.051 / abs(ps_), unit="%")
    check("top text", 0, 0, text=L(r"ending at $-0.8\%$, $+0.1\%$ and $+0.2\%$ at $\sigma = 1$ against $-2.0\%$, $-1.5\%$ and $-1.0\%$"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("mt", Path(__file__).resolve().parent / "make_tables.py")
    mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
    rd = {(round(r["sigma"], 1), r["d"]): r for r in mt.budget_rows(d, estimator="dgr")}
    rp = {(round(r["sigma"], 1), r["d"]): r for r in mt.budget_rows(d, estimator="pooled_shrink")}
    ok = all(rd[(1.0, k)]["rec80"] == 1000 for k in (3, 5, 7)) and sorted({rp[(1.0, k)]["rec80"] for k in (3, 5, 7)}) == [3000, 10000]
    print(f"  {'ok      ' if ok else 'MISMATCH'} 80% recovery at sigma=1: DGR 1e3 at every d; pooled 3e3 to 1e4")
    if not ok: FAILURES.append("80% recovery comparison differs from the text")
    check("rec80 text", 0, 0, text=L(r"recovers $80\%$ of the gap by $10^3$ shots at every distance, against $3\times10^3$ to $10^4$"))
    ex = pd.read_csv(RES / "theory" / "dgr_limit_exact.csv")
    t2, t3 = ex[ex["T"] == 2], ex[ex["T"] == 3]
    check("DGR exact limit T=2 min", 0.09, 100 * float(t2.gain_occurrence_limit.min()), tol=0.06, unit="%")
    check("DGR exact limit T=2 max", 0.31, 100 * float(t2.gain_occurrence_limit.max()), tol=0.02, unit="%")
    check("DGR exact limit T=3 min", 0.36, 100 * float(t3.gain_occurrence_limit.min()), tol=0.02, unit="%")
    check("DGR exact limit T=3 max", 2.72, 100 * float(t3.gain_occurrence_limit.max()), tol=0.002, unit="%")
    check("DGR limit text", 0, 0, text=L(r"gains $0.09\%$ to $0.31\%$ at two rounds and $0.36\%$ to $2.72\%$ at three"))
    ok = bool((ex.gain_occurrence_limit > 0).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} DGR limit better than the physical prior at every exact point")
    if not ok: FAILURES.append("DGR limit not better than physical everywhere")
    la = RES / "theory" / "limit_agreement.csv"
    if la.exists():
        a = pd.read_csv(la).iloc[0]
        ok = a.mass_decisions_differ < 1e-8
        print(f"  {'ok      ' if ok else 'MISMATCH'} correlation and occurrence limits differ on {a.mass_decisions_differ:.1e} of the mass (text: all but 1e-8)")
        if not ok: FAILURES.append("limit agreement claim fails")
    else:
        FAILURES.append("limit_agreement.csv missing"); print("  MISSING limit_agreement.csv")
    check("agreement text", 0, 0, text=L(r"make the same decision on all but $10^{-8}$ of the syndrome probability"))


def check_kappa():
    """Shrinkage strength: development selection and frozen test-set evaluation."""
    print("\n[Sec IV / App D] tuned shrinkage strength")
    import json, hashlib
    proto = ROOT / "research" / "prx_quantum" / "preregistration" / "kappa_tuning_protocol.md"
    h = hashlib.sha256(proto.read_bytes()).hexdigest()
    ok = h == "721e00d921bdc76d3d6b06b5ce5b29db0aa67e40274f82a7b2257874d0643b2b"
    print(f"  {'ok      ' if ok else 'MISMATCH'} tuning protocol unchanged since archiving (sha256 {h[:12]})")
    if not ok: FAILURES.append("kappa protocol file changed after archiving")
    dev = RES / "kappa_dev" / "20260919T114029Z_c1ecbd_dev"   # the run the selection was frozen from
    sel = json.loads((dev / "kappa_selected.json").read_text())
    crit = pd.read_csv(dev / "kappa_criterion.csv").set_index("kappa").criterion_mean_rel_excess
    kstar = float(min(k for k, x in crit.items() if x <= crit.min() + 1e-4))
    check("kappa* from development criterion", 20, kstar, tol=0, text=L(r"the value minimizing the mean relative excess over conditions and budgets is $\kappa = 20$"))
    check("kappa* recorded", 20, float(sel["kappa_star"]), tol=0)
    man = json.loads((dev / "manifest.json").read_text())
    seeds = sorted({int(c.split('"het_seed": ')[1].split(",")[0]) for c in man["conditions"]})
    ok = seeds == list(range(301, 307)) + list(range(401, 407)) and len(man["conditions"]) == 120
    print(f"  {'ok      ' if ok else 'MISMATCH'} development seeds 301-306, 401-406, 120 conditions")
    if not ok: FAILURES.append("development seeds differ from the text")
    ev = json.loads((latest("frontier/*_draws_kappa20") / "kappa_eval.json").read_text())
    check("test criterion kappa=5", 6.3, 100 * ev["criterion_kappa5"], tol=0.05 / 6.3, unit="%")
    check("test criterion kappa=20", 5.2, 100 * ev["criterion_kappa_star"], tol=0.05 / 5.2, unit="%",
          text=L(r"lowers that criterion from $6.3\%$ to $5.2\%$"))
    check("difference", -1.2, 100 * ev["difference"], tol=0.05 / 1.2, unit=" pts")
    check("difference CI low", -1.3, 100 * ev["diff_lo95"], tol=0.05 / 1.3, unit=" pts")
    check("difference CI high", -1.0, 100 * ev["diff_hi95"], tol=0.05, unit=" pts",
          text=L(r"(difference $-1.2$ points, 95\% interval $[-1.3, -1.0]$"))
    hb = pd.read_csv(latest("frontier/*_draws_kappa20") / "kappa_eval_by_budget.csv").set_index("n_cal")
    for n, a5, a20 in ((100, 53, 28), (300, 35, 14)):
        check(f"harm N={n} kappa=5", a5, 100 * hb.harm_k5[n], tol=0.5 / a5, unit="%")
        check(f"harm N={n} kappa=20", a20, 100 * hb.harm_kt[n], tol=0.5 / a20, unit="%")
    check("harm text", 0, 0, text=L(r"falls from $53\%$ to $28\%$ at $10^2$ shots and from $35\%$ to $14\%$ at $3\times10^2$"))
    top = sorted([100 * hb.median_excess_k5[300000], 100 * hb.median_excess_kt[300000]])
    ok = -1.65 <= top[0] and top[1] <= -1.45
    print(f"  {'ok      ' if ok else 'MISMATCH'} median excess at 3e5 unchanged: {top}")
    if not ok: FAILURES.append("large-budget kappa comparison differs from text")


def check_eb_shrinkage():
    """App. D: the empirical-Bayes shrinkage strength, and why it is not used."""
    print("\n[App D] empirical-Bayes shrinkage")
    dev = RES / "kappa_dev" / "20260920T022238Z_d8a61a_dev"
    c = pd.read_csv(dev / "kappa_eb_criterion.csv").set_index("arm")
    check("EB criterion", 0.104, float(c.loc["pooled_eb_global", "criterion_mean_rel_excess"]), tol=0.001 / 0.104,
          text=L(r"the criterion rises from $0.059$ to $0.104$"))
    check("tuned criterion on the development set", 0.059, float(c.loc["kappa=20.0", "criterion_mean_rel_excess"]),
          tol=0.001 / 0.059)
    check("EB harm fraction", 57, 100 * float(c.loc["pooled_eb_global", "harm_frac_N<=1e3"]), tol=0.5 / 57, unit="%",
          text=L(r"rises from $17\%$ to $57\%$"))
    check("tuned harm fraction", 17, 100 * float(c.loc["kappa=20.0", "harm_frac_N<=1e3"]), tol=0.5 / 17, unit="%")
    check("no-shrinkage harm fraction", 62, 100 * float(c.loc["kappa=0.0", "harm_frac_N<=1e3"]), tol=0.5 / 62,
          unit="%", text=L(r"which gives $62\%$"))
    k = pd.read_csv(dev / "kappa_eb_values.csv")
    k = k[k.estimator == "pooled_eb_global"]
    med = k[np.isfinite(k.kappa_hat)].groupby("n_cal").kappa_hat.median()
    check("EB kappa, smallest budget", 0.46, float(med.min()), tol=0.05 / 0.46,
          text=L(r"chooses $\kappa$ between $0.5$ and $1.4$ in median"))
    check("EB kappa, largest budget", 1.38, float(med.max()), tol=0.05 / 1.38)
    v = pd.read_csv(latest("eb_variance/*_check") / "rows.csv")
    big = v[v.n_cal >= 1000]
    ok = 2.0 <= big.var_ratio_median.min() and big.var_ratio_median.max() <= 5.0
    print(f"  {'ok      ' if ok else 'MISMATCH'} measured/modelled variance at N >= 1e3 in [2, 5]: "
          f"{big.var_ratio_median.min():.1f} to {big.var_ratio_median.max():.1f}")
    if not ok: FAILURES.append("variance-ratio range differs from the text")
    check("variance text", 0, 0, text=L(r"a factor of $2$ to $5$ more than that model"))
    kv = v.kappa_oracle_variance.replace(np.inf, np.nan).dropna()
    check("kappa with the measured variance", 2.0, float(kv.median()), tol=0.25 / 2.0,
          text=L(r"only raises the chosen $\kappa$ to a median of $2$"))



def check_willow_kappa():
    """Sec. V: post hoc hardware sensitivity with the tuned kappa."""
    print("\n[Sec V] hardware sensitivity, kappa = 20")
    w = latest("willow/*_primary_allvec")
    lv = pd.read_csv(w / "kappa20_sensitivity_leaves.csv")
    ok = bool(lv.kappa5_rep0_reproduced.all()) and len(lv) == 136
    print(f"  {'ok      ' if ok else 'MISMATCH'} kappa = 5 replicate 0 reproduced on {int(lv.kappa5_rep0_reproduced.sum())}/136")
    if not ok: FAILURES.append("hardware kappa determinism check failed")
    s = pd.read_csv(w / "kappa20_sensitivity.csv")
    g = s[(s.setting == "kappa=20") & (s.reference == "si1000_shipped")].set_index("distance")
    for d, (m, lo, hi) in {3: (2.3, 1.4, 3.1), 5: (1.6, 1.0, 2.1), 7: (1.5, 0.4, 2.5)}.items():
        check(f"kappa20 vs SI1000 d={d}", -m, 100 * float(g.rel[d]), tol=0.05 / m, unit="%")
        check(f"kappa20 vs SI1000 d={d} lo", -hi, 100 * float(g.ci_lo[d]), tol=0.05 / hi, unit="%")
        check(f"kappa20 vs SI1000 d={d} hi", -lo, 100 * float(g.ci_hi[d]), tol=0.05 / lo + 0.02, unit="%")
    check("kappa20 SI text", 0, 0, text=L(r"$2.3\%$ $[1.4, 3.1]$, $1.6\%$ $[1.0, 2.1]$ and $1.5\%$ $[0.4, 2.5]$"))
    r = s[(s.setting == "kappa=20") & (s.reference == "rl_shipped")].set_index("distance")
    for d, m in {3: 0.15, 5: 0.21, 7: 0.08}.items():
        check(f"kappa20 vs RL d={d}", m, 100 * float(r.rel[d]), tol=0.005 / m, unit="%")
    ok = bool(((r.ci_lo < 0) & (r.ci_hi > 0)).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} every kappa20-vs-RL interval includes zero")
    if not ok: FAILURES.append("kappa20 RL intervals: zero-inclusion claim fails")
    check("kappa20 RL text", 0, 0, text=L(r"trails the reinforcement-learned prior by $0.15\%$, $0.21\%$ and $0.08\%$"))


def check_certificate():
    """Sec III / App B: branch-and-bound certificate over the per-class family at T = 2."""
    print("\n[Sec III / App B] certified bound, T = 2")
    c = pd.read_csv(RES / "theory" / "certified_class_family_T2_wspan.csv").set_index("p").sort_index()
    e = pd.read_csv(RES / "theory" / "exact_candidates_untruncated.csv")
    e = e[e["T"] == 2]
    best = e.groupby("p").gain.max()
    rml = e.groupby("p").R_ml.first()
    ok = len(c) == 5 and bool((c.boxes >= 600000).all()) and not bool(c.converged.any())
    print(f"  {'ok      ' if ok else 'MISMATCH'} five runs, >= 6e5 boxes each, none closed")
    if not ok: FAILURES.append("certificate runs differ from the text (count, boxes or convergence)")
    ok = bool(np.allclose(c.UB, c.R_best_matching, rtol=0, atol=1e-15))
    print(f"  {'ok      ' if ok else 'MISMATCH'} PyMatching reproduces the certified upper value")
    if not ok: FAILURES.append("certificate UB is not reproduced by PyMatching")
    ok = bool((c.LB <= c.UB).all() and (c.UB <= c.R_phys * (1 - best.reindex(c.index).values) + 1e-15).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} LB <= UB <= best searched risk")
    if not ok: FAILURES.append("certificate ordering LB <= UB <= search fails")
    ceil_ = {0.0005: 3.72, 0.001: 3.48, 0.002: 3.09, 0.004: 2.44, 0.008: 1.73}
    for pp, v in ceil_.items():
        check(f"certified ceiling p={pp}", v, 100 * float(1 - c.LB[pp] / c.R_phys[pp]), tol=0.005 / v, unit="%")
    check("ceiling text (App B)", 0, 0, text=L(r"the certified ceilings on the gain are $3.72\%$, $3.48\%$, $3.09\%$, $2.44\%$ and $1.73\%$"))
    check("ceiling text (Sec III)", 0, 0, text=L(r"by more than $3.7\%$, $3.5\%$, $3.1\%$, $2.4\%$ and $1.7\%$ at the five error rates"))
    check("ceiling range text", 0, 0, text=L(r"$1.7\%$ to $3.7\%$"))
    share = 100 * (c.R_phys - c.LB) / (c.R_phys - rml.reindex(c.index).values)
    check("share of ML gap, low", 27, float(share.min()), tol=0.5 / 27, unit="%",
          text=L(r"recover at most $27\%$ to $44\%$ of the gap"))
    check("share of ML gap, high", 44, float(share.max()), tol=0.5 / 44, unit="%")
    imp = 100 * float((1 - c.UB / c.R_phys - best.reindex(c.index).values).max())
    check("certificate improvement over search", 0.014, imp, tol=0.0005 / 0.014, unit=" pts",
          text=L(r"by at most $0.014$ percentage points"))
    ok = bool(np.isclose(c.hi[0.008], 2.8085, atol=1e-3)) and bool((c.hi.drop(0.008) == 5).all()) and bool((c.lo == 0.15).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} boxes [0.15, 5], [0.15, 2.81] at p = 8e-3")
    if not ok: FAILURES.append("certificate box differs from the text")
    check("box text", 0, 0, text=L(r"in $[0.15, 2.81]$ at $p = 8\times10^{-3}$"))
    b = pd.read_csv(RES / "theory" / "branch_rule_comparison.csv").set_index("rule")
    ok = bool((b.boxes.nunique() == 1) and np.isclose(b.p.iloc[0], 1e-3))
    print(f"  {'ok      ' if ok else 'MISMATCH'} branching rules compared at one budget and error rate")
    if not ok: FAILURES.append("branching comparison is not at a common budget")
    check("branching: weight-span rule", 7.2, float(b.loc["wspan", "ceiling_%"]), tol=0.05 / 7.2, unit="%",
          text=L(r"this rule certifies a ceiling of $7.2\%$ against $8.5\%$ for midpoint splitting"))
    check("branching: midpoint rule", 8.5, float(b.loc["longest", "ceiling_%"]), tol=0.05 / 8.5, unit="%")


def check_frontier():
    """Sec. IV replicated frontier (independent noise draws) and Table IV."""
    print("\n[Sec IV] replicated heterogeneity frontier and Table IV")
    d = latest("frontier/*_draws")
    if d is None or not (d / "budget_per_condition.csv").exists():
        FAILURES.append("draws frontier budget missing"); print("  MISSING"); return
    pc = pd.read_csv(d / "budget_per_condition.csv")
    # continuous quantities (excess, penalty): ordinary median, as in make_tables.py;
    # budgets are sampled grid points and are taken from budget_rows (inverted-CDF median)
    med = lambda v: float(np.median(np.asarray(v, float)))

    # completeness: every manifest condition has results for every estimator
    import json
    man = json.loads((d / "manifest.json").read_text())
    conds = man["conditions"]
    conds = eval(conds) if isinstance(conds, str) else conds
    for est in ("pooled_shrink", "pooled", "pij", "density", "uniform_search", "class_search"):
        n = pc[pc.estimator == est].cond.nunique()
        ok = n == len(conds)
        print(f"  {'ok      ' if ok else 'INCOMPLETE'} {est}: {n}/{len(conds)} conditions")
        if not ok: FAILURES.append(f"draws frontier incomplete for {est}: {n}/{len(conds)}")
    check("draws conditions", 120, float(len(conds)), tol=0, text=L(r"$120$ conditions in all"))

    # independent recomputation of the uncalibrated penalty from rows.csv
    rows = pd.read_csv(d / "rows.csv", usecols=["distance", "noise_extra_json", "rounds",
                                                  "p_true", "estimator", "n_errors", "n_test_shots"])
    rows["sigma_h"] = rows.noise_extra_json.map(lambda j: float(json.loads(j)["sigma_h"]))
    rows["cond"] = (rows.distance.astype(str) + "|" + rows.rounds.astype(str) + "|"
                    + rows.p_true.astype(str) + "|" + rows.noise_extra_json)
    ler = rows[rows.estimator.isin(["static", "oracle_physical"])].groupby(["cond", "estimator"]).apply(
        lambda h: h.n_errors.sum() / h.n_test_shots.sum()).unstack()
    meta = rows.drop_duplicates("cond").set_index("cond")[["distance", "sigma_h"]]
    pen = (100 * (ler.static / ler.oracle_physical - 1)).to_frame("pen").join(meta)
    ps = pc[pc.estimator == "pooled_shrink"]
    for sg, claims in ((1.0, (14, 35, 58)),):
        for dist, c in zip((3, 5, 7), claims):
            v = pen[(pen.distance == dist) & np.isclose(pen.sigma_h, sg)].pen
            check(f"uncalibrated penalty (rows.csv) d={dist} sigma={sg}", c, med(v), tol=0.5 / c, unit="%")
    check("uncal. text sigma=1", 0, 0, text=L(r"uncalibrated prior's $14\%$, $35\%$ and $58\%$"))

    # final (largest budget) excess of each estimator, median over conditions
    top = pc.n_cal.max()
    def final(est, sg, dist):
        h = pc[(pc.estimator == est) & (pc.n_cal == top) & np.isclose(pc.sigma_h, sg) & (pc.distance == dist)]
        return 100 * med(h.excess)
    for est in ("density", "uniform_search"):
        for dist, lo, hi in ((3, 12, 13), (5, 34, 34), (7, 54, 54)):
            v = final(est, 1.0, dist)
            ok = lo - 0.5 <= v < hi + 0.5
            print(f"  {'ok      ' if ok else 'MISMATCH'} {est} final excess d={dist} sigma=1: {v:.2f}% (text {lo}-{hi}%)")
            if not ok: FAILURES.append(f"{est} final excess d={dist}: {v:.2f}%")
    check("scalar final text", 0, 0, text=L(r"they end $12\%$--$13\%$, $34\%$ and $54\%$ above the oracle"))
    gaps = [final(e, 1.0, dd) - u for e in ("density", "uniform_search")
            for dd, u in ((3, med(pen[(pen.distance == 3) & np.isclose(pen.sigma_h, 1)].pen)),
                          (5, med(pen[(pen.distance == 5) & np.isclose(pen.sigma_h, 1)].pen)),
                          (7, med(pen[(pen.distance == 7) & np.isclose(pen.sigma_h, 1)].pen)))]
    worst = float(max(abs(g) for g in gaps))
    ok = worst <= 4.0
    print(f"  {'ok      ' if ok else 'MISMATCH'} scalar final vs uncalibrated: max |diff| {worst:.2f} points (text within four)")
    if not ok: FAILURES.append(f"scalar estimators {worst:.2f} points from uncalibrated")
    check("within-four text", 0, 0, text=L(r"within four percentage points"))
    for dist, c in ((3, 4.5), (5, 32), (7, 44)):
        check(f"class search final excess d={dist} sigma=1", c, final("class_search", 1.0, dist),
              tol=0.05 / c if c < 10 else 0.5 / c, unit="%")
    check("class text", 0, 0, text=L(r"ending $4.5\%$, $32\%$ and $44\%$ above the oracle"))
    check("class caption", 0, 0, text=L(r"the per-class search ends $4.5\%$, $32\%$ and $44\%$ above the oracle"))
    edge = [final(e, 1.0, dd) for e in ("pooled_shrink", "pooled", "pij") for dd in (3, 5, 7)]
    ok = all(-2.5 <= x <= 0.0 for x in edge)
    print(f"  {'ok      ' if ok else 'MISMATCH'} per-edge final excess at sigma=1 in [{min(edge):.2f}, {max(edge):.2f}]% (caption: at or slightly below)")
    if not ok: FAILURES.append("per-edge estimators not at or slightly below the oracle")
    # Fig. 3 caption: the scalar and uniform-search medians stay within four points
    # of the uncalibrated prior at every budget, not only the largest
    worst_any = 0.0
    for e in ("density", "uniform_search"):
        for dd in (3, 5, 7):
            h = pc[(pc.estimator == e) & np.isclose(pc.sigma_h, 1.0) & (pc.distance == dd)]
            u = med(pen[(pen.distance == dd) & np.isclose(pen.sigma_h, 1)].pen)
            worst_any = max(worst_any, float((100 * h.groupby("n_cal").excess.median() - u).abs().max()))
    ok = worst_any <= 4.0
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig3 scalar/uniform within four points at every budget: max {worst_any:.2f}")
    if not ok: FAILURES.append(f"Fig3 scalar/uniform {worst_any:.2f} points from uncalibrated")
    strata = ps.groupby(["sigma_h", "distance"]).n_cal.nunique()
    ok = len(strata) == 6 and bool((strata == 8).all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} Fig3(d) harm curves: 6 strata x 8 budgets")
    if not ok: FAILURES.append("Fig3(d) harm-curve data incomplete")
    check("Fig3(d) caption", 0, 0, text=L(r"decodes worse than the uncalibrated prior, against calibration shots"))
    fin = [final("pooled_shrink", sg, dd) for sg in (0.6, 1.0) for dd in (3, 5, 7)]
    ok = all(-2.05 <= v <= -0.95 for v in fin)
    print(f"  {'ok      ' if ok else 'MISMATCH'} pooled_shrink final excess: {', '.join(f'{v:.2f}' for v in fin)}% (text 1-2% below)")
    if not ok: FAILURES.append("pooled_shrink final excess not 1-2% below oracle")
    check("pooled final text", 0, 0, text=L(r"the median pooled fit ends $1.0\%$ to $2.0\%$ below the oracle in every stratum"))

    # break-even at N = 100, sigma = 0.6
    for dist, c in ((3, 38), (5, 8), (7, 0)):
        h = ps[(ps.n_cal == 100) & np.isclose(ps.sigma_h, 0.6) & (ps.distance == dist)]
        check(f"beats uncalibrated at N=100 sigma=0.6 d={dist}", c, 100 * h.beats_static.mean(),
              tol=0.5 / max(c, 1), unit="%")
    check("beats text", 0, 0, text=L(r"$38\%$, $8\%$ and $0\%$ of conditions"))
    for dist in (5, 7):
        h = ps[(ps.n_cal >= 3000) & np.isclose(ps.sigma_h, 0.6) & (ps.distance == dist)]
        check(f"beats in every condition from 3e3, d={dist}", 100, 100 * h.beats_static.mean(), tol=0, unit="%")

    # unshrunk vs shrunk distance from the oracle at N = 100, sigma = 0.6
    def ratio(est):
        out = []
        for dist in (3, 5, 7):
            h = pc[(pc.estimator == est) & (pc.n_cal == 100) & np.isclose(pc.sigma_h, 0.6) & (pc.distance == dist)]
            out.append(1 - float(h.recovery.median()))
        return out
    r_un, r_sh = ratio("pooled"), ratio("pooled_shrink")
    check("unshrunk ratio min", 3.5, min(r_un), tol=0.05 / 3.5)
    check("unshrunk ratio max", 4.3, max(r_un), tol=0.05 / 4.3, text=L(r"$3.5$ to $4.3$ times"))
    ok = max(r_sh) <= 1.75
    print(f"  {'ok      ' if ok else 'MISMATCH'} shrunk ratio max {max(r_sh):.2f} (text at most 1.7)")
    if not ok: FAILURES.append(f"shrunk ratio {max(r_sh):.2f} > 1.7")
    check("shrunk text", 0, 0, text=L(r"against at most $1.7$ times"))

    # Table IV is regenerated elsewhere; here the numbers the prose repeats from it
    import importlib.util
    spec = importlib.util.spec_from_file_location("mt", Path(__file__).resolve().parent / "make_tables.py")
    mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
    br = {(round(r["sigma"], 1), r["d"]): r for r in mt.budget_rows(d)}
    for key, col, c in (((1.0, 3), "rec50", 1e3), ((1.0, 5), "rec50", 3e2), ((1.0, 7), "rec50", 1e3),
                        ((1.0, 3), "rec80", 1e4), ((1.0, 5), "rec80", 3e3), ((1.0, 7), "rec80", 3e3),
                        ((1.0, 3), "tenpct", 1e3), ((1.0, 7), "tenpct", 1e4)):
        check(f"Table IV {col} sigma={key[0]} d={key[1]}", c, br[key][col], tol=0)
    check("rec text", 0, 0, text=L(r"recovered by $10^3$, $3\times10^2$ and $10^3$ shots"))
    check("rec80 text", 0, 0, text=L(r"$80\%$ by $10^4$, $3\times10^3$ and $3\times10^3$"))
    grow = max(br[(0.6, 7)]["rec50"] / br[(0.6, 3)]["rec50"], br[(0.6, 7)]["rec80"] / br[(0.6, 3)]["rec80"])
    check("growth at sigma=0.6 at most threefold", 3.0, grow, tol=0.01, text=L(r"at most threefold"))
    unres = [br[(0.6, k)]["conds"] - br[(0.6, k)]["resolved"] for k in (3, 5, 7)]
    ok = unres == [0, 0, 0] and all(br[(1.0, k)]["conds"] == br[(1.0, k)]["resolved"] for k in (3, 5, 7))
    print(f"  {'ok      ' if ok else 'MISMATCH'} gap resolved by 3 paired SE in every condition (unresolved at sigma=0.6: {unres})")
    if not ok: FAILURES.append(f"unresolved counts {unres}")
    check("resolved text", 0, 0, text=L(r"resolved by at least three paired standard errors, which holds in every condition"))
    # required recovery to reach within 10%: (s - 0.1)/s at the median penalty
    for dist, c in ((3, 27), (7, 83)):
        s_ = ps[np.isclose(ps.sigma_h, 1.0) & (ps.distance == dist)].groupby("cond").static_excess.first()
        check(f"required recovery for 10% at d={dist}", c, 100 * med((s_ - 0.1) / s_), tol=0.5 / c, unit="%")
    check("required text", 0, 0, text=L(r"median condition must recover $27\%$ of its penalty at distance three and $83\%$ at distance seven"))

    # raw p_ij needs three to ten times more shots than pooled-with-shrinkage for half the gap
    pij_rows = {(round(r["sigma"], 1), r["d"]): r for r in mt.budget_rows(d, estimator="pij")}
    fac = [pij_rows[k]["rec50"] / br[k]["rec50"] for k in br]
    ok = 3 - 1e-9 <= min(fac) and max(fac) <= 10 + 1e-9
    print(f"  {'ok      ' if ok else 'MISMATCH'} p_ij / pooled_shrink 50% budget: {', '.join(f'{x:.3g}' for x in fac)} (text three to ten)")
    if not ok: FAILURES.append(f"pij budget factors {fac}")
    check("pij text", 0, 0, text=L(r"three to ten times more shots to recover half the gap"))


def L(tex: str) -> str:
    """Regex for a literal LaTeX fragment, tolerant of whitespace."""
    return r"\s*".join(_re.escape(part) for part in tex.split())


def check_willow():
    """Hardware: two estimands kept apart, every number tied to its sentence."""
    print("\n[Sec V] hardware, fitting procedure (mean over calibration draws)")
    d = latest("willow/*_primary_allvec")
    need = ["inference_procedure.csv", "inference_frozen_fits.csv", "inference_predeclared.csv",
            "inference_h3_tost.csv", "fixed_priors.csv", "lambda.csv", "lambda_paired.csv"]
    if d is None or not all((d / f).exists() for f in need):
        FAILURES.append("willow inference outputs missing (run analysis.willow_inference)"); print("  MISSING"); return
    P = pd.read_csv(d / "inference_procedure.csv")
    A = pd.read_csv(d / "inference_frozen_fits.csv")
    F = pd.read_csv(d / "inference_predeclared.csv")
    H = pd.read_csv(d / "inference_h3_tost.csv")
    fx = pd.read_csv(d / "fixed_priors.csv")
    lam = pd.read_csv(d / "lambda.csv")
    lp = pd.read_csv(d / "lambda_paired.csv")

    def proc(dist, ref):
        r = P[(P.distance == dist) & (P.comparison == f"pooled+shrinkage vs {ref}")].iloc[0]
        return 100 * r.mean_over_replicates, 100 * r.ci95_lo, 100 * r.ci95_hi
    for dist, (m, lo, hi) in {3: ("2.2", "1.4", "3.1"), 5: ("1.4", "0.9", "2.0"), 7: ("1.3", "0.1", "2.3")}.items():
        v, a_, b_ = proc(dist, "SI1000")
        check(f"procedure vs SI1000, d={dist}", -float(m), v, tol=0.05, unit="%",
              text=L(rf"${m}\%$ $[{lo}, {hi}]$"))
        check(f"  95% interval low end, d={dist}", -float(hi), a_, tol=0.05, unit="%")
        check(f"  95% interval high end, d={dist}", -float(lo), b_, tol=0.08 if float(lo) > 0.2 else 0.9, unit="%")
    for dist, (m, lo, hi) in {3: ("0.18", "-0.54", "0.91"), 5: ("0.36", "-0.01", "0.68"),
                              7: ("0.31", "-0.50", "0.99")}.items():
        v, a_, b_ = proc(dist, "RL")
        check(f"procedure vs RL, d={dist}", float(m), v, tol=0.06, unit="%",
              text=L(rf"${m}\%$ $[{lo}, {hi}]$"))
        check(f"  95% interval low end, d={dist}", float(lo), a_, tol=0.06 if abs(float(lo)) > 0.1 else 1.0, unit="%")
        check(f"  95% interval high end, d={dist}", float(hi), b_, tol=0.06, unit="%")
        if a_ > 0:
            FAILURES.append(f"procedure vs RL d={dist}: interval excludes zero; the text says every interval includes it")
    check("RL includes-zero text", 0, 0, text=L(r"every interval includes zero, the one at distance five only barely"))

    print("\n[Sec V] hardware, primary frozen fit and predeclared tests")
    def frozen(dist, ref, est="A frozen fit", rep=0):
        return A[(A.estimand == est) & (A.distance == dist) & (A.replicate == rep)
                 & (A.comparison == f"pooled+shrinkage vs {ref}")].iloc[0]
    r3 = frozen(3, "RL")
    check("frozen fit vs RL, d=3", 0.27, 100 * r3.rel, tol=0.03, unit="%",
          text=L(r"behind by $0.27\%$ at distance three ($b = 128{,}686$, $c = 126{,}324$"))
    if (int(r3.b), int(r3.c)) != (128686, 126324):
        FAILURES.append("frozen fit d=3 discordant counts changed")
    check("  McNemar p, d=3", 3e-6, float(r3.p_mcnemar), tol=0.1, text=L(r"$p = 3\times10^{-6}$"))
    r5 = frozen(5, "RL")
    check("frozen fit vs RL, d=5", 0.29, 100 * r5.rel, tol=0.03, unit="%", text=L(r"$0.29\%$ at distance five"))
    check("  McNemar p, d=5", 4e-5, float(r5.p_mcnemar), tol=0.05, text=L(r"$p = 4\times10^{-5}$"))
    r7 = frozen(7, "RL")
    check("frozen fit vs RL, d=7", 0.05, 100 * r7.rel, tol=0.1, unit="%", text=L(r"by $0.05\%$ at distance seven ($p = 0.74$)"))
    rx = frozen(7, "RL", "A frozen fit, basis X"); rz = frozen(7, "RL", "A frozen fit, basis Z")
    check("  d=7 X basis", 0.74, 100 * rx.rel, tol=0.02, unit="%", text=L(r"$0.74\%$ behind in the $X$ basis ($p = 1\times10^{-4}$)"))
    check("  d=7 Z basis (ahead)", -0.68, 100 * rz.rel, tol=0.02, unit="%", text=L(r"$0.68\%$ ahead in the $Z$ basis ($p = 1\times10^{-3}$)"))
    rng7 = A[(A.estimand == "A frozen fit") & (A.distance == 7) & (A.comparison == "pooled+shrinkage vs RL")].rel * 100
    check("  d=7 range over five frozen fits, low", 0.05, float(rng7.min()), tol=0.1, unit="%",
          text=L(r"ranges from $0.05\%$ to $0.61\%$"))
    check("  d=7 range over five frozen fits, high", 0.61, float(rng7.max()), tol=0.02, unit="%")
    holm_ok = F.set_index(["test", "distance"])
    for dist in (3, 5):
        if not bool(holm_ok.loc[("H2", dist), "reject_at_0.01"]):
            FAILURES.append(f"H2 at d={dist} no longer survives Holm")
    if bool(holm_ok.loc[("H2", 7), "reject_at_0.01"]):
        FAILURES.append("H2 at d=7 now rejects; the text says it does not")
    h3 = F[F.test.str.startswith("H3")]
    ok = bool(h3["reject_at_0.01"].all())
    print(f"  {'ok      ' if ok else 'MISMATCH'} H3 equivalence rejects both one-sided nulls at every distance after Holm")
    if not ok: FAILURES.append("H3 no longer establishes equivalence")
    zmin = float(H.z_min.min())
    check("H3 smallest one-sided z", 60, zmin, tol=0.02, text=L(r"every one-sided statistic exceeds $z = 60$"))
    if zmin <= 60: FAILURES.append(f"H3 smallest z {zmin:.1f} is not above 60")
    dev = float(max((H.wald98_lo - H.ci98_lo).abs().max(), (H.wald98_hi - H.ci98_hi).abs().max()))
    ok = dev < 4e-5
    print(f"  {'ok      ' if ok else 'MISMATCH'} H3 delta-method and bootstrap 98% intervals agree to {dev:.1e} (text 4e-5)")
    if not ok: FAILURES.append("H3 Wald/bootstrap agreement exceeds the stated 4e-5")
    w = max(abs(H.ci98_lo).max(), abs(H.ci98_hi).max()) * 100
    ok = w <= 0.3
    print(f"  {'ok      ' if ok else 'MISMATCH'} H3 98% intervals within +-0.3%: widest {w:.3f}%")
    if not ok: FAILURES.append("H3 intervals exceed the stated +-0.3%")
    for dist, claimed in {3: -2.4, 5: -1.8, 7: -1.6}.items():
        v = fx[(fx.distance == dist) & (fx.prior == "rl_shipped") & (fx.decoder == "pymatching_mwpm")]
        check(f"RL prior gain, d={dist}", claimed, 100 * float(v.rel_ler.iloc[0]), tol=0.12, unit="%")
    check("RL prior gains quoted", 0, 0, text=L(r"achieve $2.4\%$, $1.8\%$ and $1.6\%$"))

    print("\n[Sec V] error suppression factor, paired")
    lf = lam.set_index("selection").Lambda_fit
    check("Lambda SI1000", 1.529, lf["pymatching+si1000"], tol=0.001, text=L(r"$\Lambda = 1.529$"))
    check("Lambda RL", 1.524, lf["pymatching+rl"], tol=0.001)
    check("Lambda Libra", 2.02, lf["google:libra_decoder_with_rl_optimized_prior"], tol=0.005,
          text=L(r"$2.02$ for Libra"))
    g = lp[lp.scheme == "shots+groups"]
    worst = float(np.exp(max(np.abs(np.log(g.ratio_lo)).max(), np.abs(np.log(g.ratio_hi)).max())) - 1)
    ok = worst < 0.03
    print(f"  {'ok      ' if ok else 'MISMATCH'} paired 95% Lambda-ratio intervals within {100*worst:.2f}% of equality, either orientation (text 3%)")
    if not ok: FAILURES.append("paired Lambda-ratio intervals exceed 3%")
    check("ratio text", 0, 0, text=L(r"lies within $3\%$ of equality, in either orientation"))
    check("largest point difference between priors", 1.1, 100 * float((g.ratio - 1).abs().max()), tol=0.05,
          unit="%", text=L(r"differ by at most $1.1\%$"))
    check("sd of a single ln Lambda, max", 0.042, float(max(g.sd_marginal_A.max(), g.sd_marginal_B.max())), tol=0.05)
    check("sd of a paired difference, max", 0.014, float(g.sd_difference.max()), tol=0.05,
          text=L(r"only $0.003$ to $0.014$"))
    base = {"si1000": lf["pymatching+si1000"], "rl": lf["pymatching+rl"]}
    ups = [lf[k] / base["si1000" if "si1000" in k else "rl"] - 1 for k in lf.index if k.startswith("google:")]
    check("decoder effect on Lambda, low", 23, 100 * min(ups), tol=0.03, unit="%", text=L(r"$23\%$ to $33\%$"))
    check("decoder effect on Lambda, high", 33, 100 * max(ups), tol=0.02, unit="%")
    sh = lp[(lp.scheme == "shots") & (lp.prior_A == "si1000_shipped") & lp.prior_B.str.startswith("pooled")]
    resolved = int((sh.ratio_lo > 1).sum())
    print(f"  {'ok      ' if resolved == 4 else 'MISMATCH'} conditional Lambda shifts resolved for {resolved} of 5 draws")
    if resolved != 4: FAILURES.append("conditional Lambda claim (four of five) no longer holds")


def check_factorial():
    print("\n[Sec VI] prior x decoder factorial")
    d = latest("willow_factorial/*primary")
    if d is None:
        FAILURES.append("factorial missing"); print("  MISSING"); return
    df = pd.read_csv(d / "rows.csv")
    df["prior"] = df.prior_id.str.replace(r"_N=\d+", "", regex=True)
    for dist, claimed in {3: -8.5, 5: -22.4, 7: -34.7}.items():
        g = df[df.distance == dist]
        piv = g.pivot_table(index="leaf_id", columns=["prior", "decoder"], values="n_errors")
        v = 100 * (piv[("si1000_shipped", "pymatching_correlated")].sum() /
                   piv[("si1000_shipped", "pymatching_mwpm")].sum() - 1)
        check(f"correlated matching gain, d={dist}", claimed, v, tol=0.08, unit="%")
    for dist, claimed in {3: -1.4, 5: -0.2, 7: 0.0}.items():
        g = df[df.distance == dist]
        piv = g.pivot_table(index="leaf_id", columns=["prior", "decoder"], values="n_errors")
        v = 100 * (piv[("pooled_shrink", "pymatching_correlated")].sum() /
                   piv[("si1000_shipped", "pymatching_correlated")].sum() - 1)
        check(f"our prior under correlated matching, d={dist}", claimed, v, tol=0.5, unit="%")




# ---------------------------------------------------------------------------
# Manuscript-text checks.
#
# The checks above compare the data against numbers written here in Python. That
# cannot detect a stale number in the manuscript, or an edit that silently failed
# to apply. These checks read prior_value.tex and assert that specific claims are
# present and that retired ones are absent.
# ---------------------------------------------------------------------------

TEX = ROOT / "manuscript" / "prior_value.tex"


def check_tex():
    print("\n[text] manuscript claims")
    if not TEX.exists():
        print("  skipped: the manuscript source is not distributed with this code")
        return
    import re as _re
    # collapse newlines: every claim below is a phrase that LaTeX may wrap
    tex = _re.sub(r'\s+', ' ', TEX.read_text())

    must_contain = {
        "scalar result stated as measured (median gain)": r"by a median of \$0\.5\\%\$",
        "scalar result stated as measured (worst loss)": r"28\.9\\%",
        "scalar result scoped to the experiments": r"measured regularity of the models",
        "best-found language for the reweighting search": r"best per-class",
        "lower-bound interpretation of best-found gains": r"lower bound",
        "per-edge family acknowledged": r"per-edge reweighting",
        "operational split named as such": r"operational split",
        "repetition-code check on the split": r"model term is zero",
        "independent max-product check described": r"maximization in place of summation",
        "hardware residual stated, not called a match": r"a few tenths of a percent less than a released prior",
        "hardware estimands kept apart": r"two estimands",
        "hardware McNemar reported": r"McNemar \$p = 3\\times10\^\{-6\}\$",
        "paired Lambda analysis": r"paired analysis",
        "equivalence test with predeclared margin": r"predeclared \$2\\%\$ margin",
        "heterogeneity reported over ten draws": r"ten independent noise draws",
        "exact-vs-MC validation": r"2\.554",
        "truncation bound stated": r"4\\times10\^\{-7\}",
        "Arms cited first-hand on Willow": r"Arms \\emph\{et al\.\}~\\cite\{arms2026willow\}",
        "Arms Table IV numbers quoted": r"3280 \\pm 80",
        "pooling conceded to Arms": r"is not ours either",
        "Spitz credited for the data-cost curve": r"relative excess over an ideal-weight decoder",
        "DGR credited for logical error vs trials": r"logical error against the number of calibration",
        "Wagner credited for regularization": r"Dirichlet prior",
        "Sivak correlation-vs-RL comparison cited": r"\$3\.3\\%\$",
        "Nickerson-Brown cited": r"nickerson2019correlated",
        "comparison table present": r"label\{tab:related\}",
        "Carroll/Tiurev credited for the distance trend": r"carroll2024nonidentical",
        "Hesner caveat answered": r"measured in situ",
        "Stace-Barrett degeneracy original cited": r"stace2010degeneracy",
        "threshold shift not used as a bound": r"not a bound on the finite-distance",
        "deMarti iOlius pair cited": r"demarti2023recursive",
        "reverse distance trend answered": r"coefficient of variation",
        "Hockings d=63 labelled as extrapolation": r"extrapolating the fitted trends",
        "ACES cost stated structurally": r"84d\^2 - 36d - 24",
        "Tiurev total-vs-Pauli distinction": r"\\emph\{total\} error rate",
        "Libra named for the reproduced value": r"Our fit of Libra",
        "repository named in the availability statement": r"github\.com/ProfessorDong/surface-code-decoder-priors",
    }
    must_not_contain = {
        "retired 'untested default' for kappa": r"untested default",
        "retired 'not certified' claims now that T = 2 is certified": r"rather than certified optima|not certified optima|certified bound over all weights",
        "assistive-tool disclosure (the paper states no such thing)": r"artificial.intelligence|large language model|AI-assisted|AI-generated|machine-generated",
        "retired 'recovers all'": r"recovers all",
        "retired independence claim for the shared-seed frontier": r"every condition is an independent noise draw",
        "retired 'confirm it exactly' (repetition code)": r"confirm it exactly",
        "retired section title": r"Limits on prior reweighting",
        "retired 'only at distance five does the interval exclude zero'": r"only at distance five does the",
        "retired TOST p-value": r"Holm-corrected\s*\$p = 0\.002\$",
        "retired floored bootstrap tail fractions": r"tail fractions floored|1/\(B\+1\)",
        "retired 'keeps the small-budget end from falling below'": r"keeps the small-budget end",
        "retired 'fixed before the hardware run, is the first calibration prefix'": r"fixed before the hardware run, is the first",
        "internal [RUN: ...] provenance tag (see PROVENANCE.md)": r"\[RUN",
        "claim that code is included with the submission": r"code included with the submission",
        "retired 1.3% uniform-rescale claim": r"rescal[^.]{0,120}at most \$1\.3\\%\$",
        "retired 88% single-seed penalty": r"88\\%",
        "retired 'matches a reinforcement-learned prior'": r"shots matches a",
        "retired degeneracy section title": r"The degeneracy ceiling",
        "retired 'risk-optimal prior'": r"risk-optimal prior",
        "retired second-hand Arms citation": r"Earlier work reported parity",
        "retired pooling-novelty claim": r"pooling by edge signature is what multiplies",
        "retired title 'Exact Limits'": r"Exact Limits",
        "retired 'best possible reweighting'": r"best possible reweighting",
        "retired 'cannot help'": r"cannot help",
        "retired ceiling claims": r"exact ceiling|derive the ceiling|The ceiling we compute",
        "retired 'global optima'": r"global optim",
        "retired 'no choice of matching prior can reach'": r"no choice of matching prior can reach",
        "retired 'degeneracy and nothing else'": r"nothing else",
        "retired 'reweighting cannot recover'": r"Reweighting cannot recover",
        "retired universal exponent claim": r"change the exponent|cannot change the exponent",
        "retired 'priors pay only'": r"Priors pay only|only expect them to pay",
        "retired shrinkage guarantee": r"This guarantees that the",
        "retired 'two decades'": r"two decades",
        "retired author error Y. Cao": r"Y\.~Cao",
        "retired 'strongest decoder in the release'": r"strongest decoder in the release",
        "retired 'down-weighted' for probability scales": r"down-weighted",
        "retired 'necessarily distance-dependent'": r"necessarily distance-dependent",
        "retired 'device-faithful'": r"device-faithful",
        "retired 10% Lambda floor": r"which is \$10\\%\$",
        "retired sweeping literature claim": r"A large body of adaptive-decoding work",
        "retired imperative 'Do not calibrate a scalar'": r"Do not calibrate a scalar",
        "retired novelty 'no previous work addresses'": r"no previous work addresses",
        "retired 'not a stalled search'": r"not a stalled search",
        "retired 'less than 0.5% everywhere'": r"less than \$0\.5\\%\$ everywhere",
    }
    import re
    for name, pat in must_contain.items():
        ok = re.search(pat, tex) is not None
        print(f"  {'ok      ' if ok else 'MISSING '} {name}")
        if not ok:
            FAILURES.append(f"manuscript does not state: {name}")
    for name, pat in must_not_contain.items():
        bad = re.search(pat, tex) is not None
        print(f"  {'ok      ' if not bad else 'STALE   '} absent: {name}")
        if bad:
            FAILURES.append(f"manuscript still contains: {name}")

    # every \cite key must resolve, and every bibitem must be cited
    defined = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    cited = set()
    for m in re.findall(r"\\cite\{([^}]+)\}", tex.split(r"\begin{thebibliography}")[0]):
        cited.update(x.strip() for x in m.split(","))
    if cited - defined:
        FAILURES.append(f"undefined citations: {sorted(cited - defined)}")
    if defined - cited:
        FAILURES.append(f"uncited references: {sorted(defined - cited)}")
    print(f"  {'ok      ' if not (cited ^ defined) else 'PROBLEM '} bibliography closed"
          f" ({len(defined)} entries, {len(cited)} cited)")

    # entries are numbered in the order they appear, so they must be listed in citation order
    body, bibtext = TEX.read_text().split(r"\begin{thebibliography}")
    order = re.findall(r"\\bibitem\{([^}]+)\}", bibtext)
    first: dict[str, int] = {}
    for m in re.finditer(r"\\cite\{([^}]+)\}", body):
        for k in m.group(1).split(","):
            first.setdefault(k.strip(), m.start())
    seq = sorted(first, key=first.get)
    ok = order == seq
    bad = next((o for o, c in zip(order, seq) if o != c), None)
    print(f"  {'ok      ' if ok else 'PROBLEM '} bibliography in citation order"
          + ("" if ok else f" (first out of place: {bad})"))
    if not ok:
        FAILURES.append(f"bibliography is not in citation order (first out of place: {bad})")


def main():
    print("Verifying manuscript numbers against result files")
    check_headlines()
    check_tables_current()
    check_exact_theory()
    check_certificate()
    check_mechanism()
    check_reweight_mechanism()
    check_hetero_seeds()
    check_device_like()
    check_exact_vs_mc()
    check_gate0()
    check_fig1()
    check_fig2()
    check_fig4()
    check_fig5()
    check_fig6()
    check_moment_limit_exact()
    check_frontier()
    check_population_limit()
    check_dgr()
    check_kappa()
    check_eb_shrinkage()
    check_willow_kappa()
    check_willow()
    check_factorial()
    check_tex()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} PROBLEM(S):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("All manuscript numbers reproduce from the result files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
