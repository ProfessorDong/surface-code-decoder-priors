"""Population limit of the per-edge estimators on the frontier conditions.

For each condition of a frontier run, the estimators are applied to the exact
population moments of the true detector error model (``theory.population_moments``),
which is what they would return from unlimited calibration data. The resulting
priors are decoded on the frontier's own test shots, regenerated from the run's
seed plan; the regenerated oracle errors must equal the stored ones exactly, so
every comparison is paired with the frontier's rows.

The limit separates two sources of excess risk at finite N:
    excess(N) - excess(inf)   finite-data error of the fit,
    excess(inf)               bias of the estimator itself (its formula assumes a
                              graphlike model; circuit noise has hyperedges).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..config import SeedPlan
from ..dem.edges import EdgeTable
from ..estimators.pij import PijEstimator
from ..estimators.pooled import PooledPijEstimator
from ..stats import discordant_counts
from ..theory.population_moments import population_moments
from .common import build_condition_circuit, decode_errors, run_pool, sample_shots
from .frontier import default_conditions, nominal_template


def task(t: Dict[str, Any]) -> Dict[str, Any]:
    cond = t["cond"]
    circuit = build_condition_circuit(cond)
    plan = SeedPlan(t["base_seed"])
    tdets, tobs = sample_shots(circuit, t["n_test"], plan.for_("test", cond.cond_id))
    phys = EdgeTable.from_circuit(circuit)
    template = nominal_template(cond)
    e_or = decode_errors(phys.to_matching(), tdets, tobs)
    e_st = decode_errors(template.to_matching(), tdets, tobs)
    dem = circuit.detector_error_model(decompose_errors=False)
    bulk = np.flatnonzero(template.v >= 0)
    pairs = np.stack([template.u[bulk], template.v[bulk]], axis=1)
    xi, xij = population_moments(dem, pairs)
    out = {"cond_id": cond.cond_id, "distance": cond.distance, "rounds": cond.rounds, "p": cond.p,
           "sigma_h": cond.extra.get("sigma_h", 0.0), "het_seed": cond.extra.get("het_seed", 0),
           "n_test": int(t["n_test"]), "k_oracle": int(e_or.sum()), "k_static": int(e_st.sum())}
    # edge-level distance of each limit from the true graph prior, on edges both tables share
    true_p = {(int(u), int(v), int(m)): p for u, v, m, p in zip(phys.u, phys.v, phys.fault_mask, phys.p)}
    for name, est in (("pij_limit", PijEstimator()), ("pooled_limit", PooledPijEstimator(kappa=0.0))):
        fit = est.fit_moments(template, xi, xij, np.inf)
        e = decode_errors(fit.table.to_matching(), tdets, tobs)
        b, c = discordant_counts(e, e_or)
        tab = fit.table
        lr = [np.log(p / true_p[(int(u), int(v), int(m))]) for u, v, m, p in zip(tab.u, tab.v, tab.fault_mask, tab.p)
              if (int(u), int(v), int(m)) in true_p]
        out.update({f"k_{name}": int(e.sum()), f"b_{name}": int(b), f"c_{name}": int(c),
                    f"median_abs_logratio_{name}": float(np.median(np.abs(lr))),
                    f"fallback_{name}": int(fit.params.get("n_fallback", fit.params.get("n_fallback_bulk", 0)))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first k conditions only")
    a = ap.parse_args()
    man = json.loads((a.run_dir / "manifest.json").read_text())
    ids = man["conditions"]
    ids = eval(ids) if isinstance(ids, str) else ids
    conds = {c.cond_id: c for c in default_conditions(man["profile"])}
    assert set(ids) <= set(conds), "manifest conditions not reproduced by default_conditions"
    tasks = [{"cond": conds[i], "base_seed": man["base_seed"], "n_test": man["cfg"]["n_test"]} for i in ids]
    if a.limit:
        tasks = tasks[:a.limit]
    rows = [r for r in run_pool(tasks, task, workers=a.workers)]
    df = pd.DataFrame(rows)
    # the regenerated test set must be the frontier's: oracle and static error counts must match the stored rows
    fr = pd.read_csv(a.run_dir / "rows.csv", low_memory=False)
    ref = fr[fr.estimator.isin(["oracle_physical", "static"])]
    key = lambda r: (int(r.distance), int(r.rounds), float(r.p_true), r.noise_extra_json)
    sk = {}
    for r in ref.itertuples():
        sk.setdefault(key(r), {})[r.estimator] = int(r.n_errors)
    mism = 0
    for r in df.itertuples():
        c = conds[r.cond_id]
        k = (c.distance, c.rounds, float(c.p), json.dumps(c.extra, sort_keys=True))
        got = sk.get(k)
        if got is None:
            # fall back to matching on the parsed JSON
            cand = [v for kk, v in sk.items() if kk[:3] == k[:3] and json.loads(kk[3]) == c.extra]
            got = cand[0] if len(cand) == 1 else None
        if got is None or got.get("oracle_physical") != r.k_oracle or got.get("static") != r.k_static:
            mism += 1
    df["test_set_reproduced"] = mism == 0
    out = a.run_dir / ("population_limit.csv" if not a.limit else "population_limit_smoke.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}; test sets reproduced for all conditions: {mism == 0} ({mism} mismatches)")


if __name__ == "__main__":
    main()
