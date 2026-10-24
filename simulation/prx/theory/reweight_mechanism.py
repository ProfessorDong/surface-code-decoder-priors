"""What does the best per-class reweighting change, decision by decision?

At d = 3 the best reweighting gain found in Table I is close to the positive part
of the decision-rule term of the operational split (matching versus the most
probable coset of the decomposed model). That suggests a mechanism: reweighting a
graphlike prior moves matching toward the coset decisions of the same graphlike
model, a partial correction for degeneracy, without recovering the model term.

This module tests the mechanism directly on the exact joint law. For each
operating point it compares four decision rules on every syndrome,

    match_phys   matching with the physical prior
    match_best   matching with the best per-class prior found
    ml_dec       most probable coset of the decomposed model
    ml_full      most probable coset of the full model (optimal),

and reports probability-weighted disagreement rates between them under the true
law. The mechanism predicts that match_best agrees with ml_dec more often than
match_phys does. If it does not, the correspondence in Table I is a coincidence.

"The best per-class prior found" must mean the same prior here as it does in the
gain table, so the candidate is taken from the same canonical selection: the
coordinate scan and its robustness re-test in exact_candidates_untruncated.csv,
together with the branch-and-bound incumbent in the certificate files. Each row
records which candidate it used and that candidate's four scales, and the risk
recomputed here is checked against the gain the candidate was recorded with.

The same attribution is applied to a prior nobody searched for: the population limit
of the pairwise correlation fit (Proposition 3), which is what unlimited label-free
syndrome data would give. It is not a member of the per-class family, so whether its
gain also comes from moving toward the decomposed model's coset decisions is a
separate question from the one the searched prior answers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESULTS_ROOT
from ..dem.edges import EdgeTable
from ..dem.priors import build_circuit
from ..estimators.pij import PijEstimator
from .exact import enumerate_dem
from .population_moments import population_moments
from .three_way import _align, decomposed_dem

CLASSES = ("space", "time", "diag", "boundary")
CERT_FILES = {2: "certified_class_family_T2_wspan.csv", 3: "certified_class_family_T3_gpu.csv"}


def canonical_candidates(th: Path) -> dict:
    """Best candidate at each (T, p) over everything evaluated, as Table II takes it.

    Returns (T, p) -> (source, {class: scale}, gain as recorded), where the gain is the
    one stored with the candidate; main() recomputes it and checks the two agree.
    """
    out = {}
    for _, r in pd.read_csv(th / "exact_candidates_untruncated.csv").iterrows():
        key = (int(r["T"]), float(r["p"]))
        if key not in out or r["gain"] > out[key][2]:
            out[key] = ("scan", {c: float(r[f"s_{c}"]) for c in CLASSES}, float(r["gain"]))
    for T, fname in CERT_FILES.items():
        for _, r in pd.read_csv(th / fname).iterrows():
            key = (T, float(r["p"]))
            scales = dict(zip(CLASSES, json.loads(r["best_scales"])))
            if key not in out or r["gain_UB_certified"] > out[key][2]:
                out[key] = ("incumbent", scales, float(r["gain_UB_certified"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--rounds", type=int, nargs="*", default=None,
                    help="restrict to these durations (default: every condition on file)")
    a = ap.parse_args()
    th = a.results_root / "theory"
    cands = canonical_candidates(th)
    rows = []
    for (T, p), (source, scales, gain_recorded) in sorted(cands.items()):
        if a.rounds and T not in a.rounds:
            continue
        dem = build_circuit("si1000", 3, int(T), float(p), variant="google2025").detector_error_model(
            decompose_errors=True)
        table = EdgeTable.from_dem(dem)
        full, dec = enumerate_dem(dem), enumerate_dem(decomposed_dem(table))
        keys, pt, pdd = _align(full, dec)
        w = pt.sum(axis=1)                                   # true probability of each syndrome
        bits = ((keys[:, None] >> np.arange(full.num_detectors, dtype=np.int64)[None, :]) & 1).astype(np.uint8)
        # the pairwise fit's population limit, the prior unlimited label-free data gives
        circ = build_circuit("si1000", 3, int(T), float(p), variant="google2025")
        bulk = np.flatnonzero(table.v >= 0)
        pairs = np.stack([table.u[bulk], table.v[bulk]], axis=1)
        xi, xij = population_moments(circ.detector_error_model(decompose_errors=False), pairs)
        limit = PijEstimator().fit_moments(table, xi, xij, np.inf).table
        rule = {
            "match_phys": table.to_matching().decode_batch(bits)[:, 0].astype(bool),
            "match_best": table.scaled_by_class(scales).to_matching().decode_batch(bits)[:, 0].astype(bool),
            "match_limit": limit.to_matching().decode_batch(bits)[:, 0].astype(bool),
            "ml_dec": pdd[:, 1] > pdd[:, 0],
            "ml_full": pt[:, 1] > pt[:, 0],
        }
        risk = {k: float(np.where(v, pt[:, 0], pt[:, 1]).sum()) for k, v in rule.items()}
        dis = lambda x, y: float(w[rule[x] != rule[y]].sum())
        # exact attribution of the risk change: per syndrome, g(s) = risk of the physical
        # decision minus risk of the best decision; changed decisions either move to the
        # decomposed model's coset decision ("toward") or away from it ("away")
        risk_s = lambda v: np.where(v, pt[:, 0], pt[:, 1])

        def split(which):
            """(toward, away) shares of the risk reduction of rule `which` over match_phys."""
            gg = risk_s(rule["match_phys"]) - risk_s(which)
            ch = rule["match_phys"] != which
            tw = ch & (which == rule["ml_dec"])
            aw = ch & (rule["match_phys"] == rule["ml_dec"])
            assert not (tw & aw).any() and (tw | aw)[ch].all()
            return gg, tw, aw

        g, toward, away = split(rule["match_best"])
        gl, toward_l, away_l = split(rule["match_limit"])
        gain_best = 1 - risk["match_best"] / risk["match_phys"]
        assert abs(gain_best - gain_recorded) < 1e-9, (
            f"T={T} p={p}: recomputed gain {gain_best} does not match the {source} candidate's "
            f"recorded gain {gain_recorded}")
        rows.append({"T": int(T), "p": float(p), "candidate": source,
                     **{f"s_{c}": scales[c] for c in CLASSES},
                     **{f"R_{k}": v for k, v in risk.items()},
                     "gain_best": gain_best,
                     "rule_term": (risk["match_phys"] - risk["ml_dec"]) / risk["match_phys"],
                     "dis_phys_vs_mldec": dis("match_phys", "ml_dec"),
                     "dis_best_vs_mldec": dis("match_best", "ml_dec"),
                     "dis_phys_vs_mlfull": dis("match_phys", "ml_full"),
                     "dis_best_vs_mlfull": dis("match_best", "ml_full"),
                     "dis_phys_vs_best": dis("match_phys", "match_best"),
                     "gain_toward": float(g[toward].sum()) / risk["match_phys"],
                     "gain_away": float(g[away].sum()) / risk["match_phys"],
                     "mass_toward": float(w[toward].sum()), "mass_away": float(w[away].sum()),
                     "gain_limit": 1 - risk["match_limit"] / risk["match_phys"],
                     "gain_limit_toward": float(gl[toward_l].sum()) / risk["match_phys"],
                     "gain_limit_away": float(gl[away_l].sum()) / risk["match_phys"],
                     "dis_limit_vs_mldec": dis("match_limit", "ml_dec"),
                     "dis_best_vs_limit": dis("match_best", "match_limit")})
        r = rows[-1]
        print(f"T={int(T)} p={p:g} [{source}]: gain {100 * r['gain_best']:.3f}%  rule term {100 * r['rule_term']:+.3f}%  "
              f"| P[disagree with ML_dec]: physical {r['dis_phys_vs_mldec']:.3e} -> best "
              f"{r['dis_best_vs_mldec']:.3e} | gain toward {100*r['gain_toward']:.4f} away {100*r['gain_away']:.4f}"
              f" | limit gain {100*r['gain_limit']:.4f} (toward {100*r['gain_limit_toward']:.4f} "
              f"away {100*r['gain_limit_away']:.4f})", flush=True)
    out = th / "reweight_mechanism.csv"
    df = pd.DataFrame(rows)
    if a.rounds and out.exists():                 # keep the durations this run did not touch
        keep = pd.read_csv(out)
        keep = keep[~keep["T"].isin(a.rounds)]
        df = pd.concat([keep, df]).sort_values(["T", "p"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
