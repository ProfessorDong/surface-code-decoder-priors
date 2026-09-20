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
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESULTS_ROOT
from ..dem.edges import EdgeTable
from ..dem.priors import build_circuit
from .exact import enumerate_dem
from .three_way import _align, decomposed_dem

CLASSES = ("space", "time", "diag", "boundary")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    a = ap.parse_args()
    th = a.results_root / "theory"
    cand = pd.read_csv(th / "exact_candidates_untruncated.csv")
    rows = []
    for (T, p), g in cand.groupby(["T", "p"]):
        best = g.loc[g.gain.idxmax()]
        scales = {c: float(best[f"s_{c}"]) for c in CLASSES}
        dem = build_circuit("si1000", 3, int(T), float(p), variant="google2025").detector_error_model(
            decompose_errors=True)
        table = EdgeTable.from_dem(dem)
        full, dec = enumerate_dem(dem), enumerate_dem(decomposed_dem(table))
        keys, pt, pdd = _align(full, dec)
        w = pt.sum(axis=1)                                   # true probability of each syndrome
        bits = ((keys[:, None] >> np.arange(full.num_detectors, dtype=np.int64)[None, :]) & 1).astype(np.uint8)
        rule = {
            "match_phys": table.to_matching().decode_batch(bits)[:, 0].astype(bool),
            "match_best": table.scaled_by_class(scales).to_matching().decode_batch(bits)[:, 0].astype(bool),
            "ml_dec": pdd[:, 1] > pdd[:, 0],
            "ml_full": pt[:, 1] > pt[:, 0],
        }
        risk = {k: float(np.where(v, pt[:, 0], pt[:, 1]).sum()) for k, v in rule.items()}
        dis = lambda x, y: float(w[rule[x] != rule[y]].sum())
        # exact attribution of the risk change: per syndrome, g(s) = risk of the physical
        # decision minus risk of the best decision; changed decisions either move to the
        # decomposed model's coset decision ("toward") or away from it ("away")
        risk_s = lambda v: np.where(v, pt[:, 0], pt[:, 1])
        g = risk_s(rule["match_phys"]) - risk_s(rule["match_best"])
        changed = rule["match_phys"] != rule["match_best"]
        toward = changed & (rule["match_best"] == rule["ml_dec"])
        away = changed & (rule["match_phys"] == rule["ml_dec"])
        assert not (toward & away).any() and (toward | away)[changed].all()
        rows.append({"T": int(T), "p": float(p), **{f"R_{k}": v for k, v in risk.items()},
                     "gain_best": 1 - risk["match_best"] / risk["match_phys"],
                     "rule_term": (risk["match_phys"] - risk["ml_dec"]) / risk["match_phys"],
                     "dis_phys_vs_mldec": dis("match_phys", "ml_dec"),
                     "dis_best_vs_mldec": dis("match_best", "ml_dec"),
                     "dis_phys_vs_mlfull": dis("match_phys", "ml_full"),
                     "dis_best_vs_mlfull": dis("match_best", "ml_full"),
                     "dis_phys_vs_best": dis("match_phys", "match_best"),
                     "gain_toward": float(g[toward].sum()) / risk["match_phys"],
                     "gain_away": float(g[away].sum()) / risk["match_phys"],
                     "mass_toward": float(w[toward].sum()), "mass_away": float(w[away].sum())})
        r = rows[-1]
        print(f"T={int(T)} p={p:g}: gain {100 * r['gain_best']:.3f}%  rule term {100 * r['rule_term']:+.3f}%  "
              f"| P[disagree with ML_dec]: physical {r['dis_phys_vs_mldec']:.3e} -> best "
              f"{r['dis_best_vs_mldec']:.3e} | gain toward {100*r['gain_toward']:.4f} away {100*r['gain_away']:.4f}", flush=True)
    out = th / "reweight_mechanism.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
