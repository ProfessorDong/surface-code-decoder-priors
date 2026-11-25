"""Edge-level form of the pairwise estimator's population limit, per edge class.

For every condition of a frontier run, the limit prior (exact population moments)
is compared with the true graph prior edge by edge; reported per class are the
fraction of edges that differ (|ln ratio| > 1e-3) and the quantiles of ln ratio
over the differing edges.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from ..dem.edges import CLASS_NAMES, EdgeTable
from ..estimators.pij import PijEstimator
from ..experiments.common import build_condition_circuit
from ..experiments.frontier import default_conditions, nominal_template
from ..theory.population_moments import population_moments


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("run_dir", type=Path); a = ap.parse_args()
    man = json.loads((a.run_dir / "manifest.json").read_text())
    ids = eval(man["conditions"]) if isinstance(man["conditions"], str) else man["conditions"]
    conds = {c.cond_id: c for c in default_conditions(man["profile"])}
    rows = []
    for cid in ids:
        c = conds[cid]; circ = build_condition_circuit(c); phys = EdgeTable.from_circuit(circ); t = nominal_template(c)
        bulk = np.flatnonzero(t.v >= 0); pairs = np.stack([t.u[bulk], t.v[bulk]], 1)
        xi, xij = population_moments(circ.detector_error_model(decompose_errors=False), pairs)
        lim = PijEstimator().fit_moments(t, xi, xij, np.inf).table
        tp = {(int(u), int(v)): p for u, v, p in zip(phys.u, phys.v, phys.p)}
        lr = np.array([np.log(lim.p[k] / tp[(int(lim.u[k]), int(lim.v[k]))]) for k in range(lim.num_edges)])
        for cl in np.unique(lim.cls):
            x = lr[lim.cls == cl]; ch = x[np.abs(x) > 1e-3]
            rows.append({"cond_id": cid, "distance": c.distance, "class": CLASS_NAMES[int(cl)], "n": len(x),
                         "frac_changed": len(ch) / len(x), "lr_min": float(ch.min()) if len(ch) else 0.0,
                         "lr_median": float(np.median(ch)) if len(ch) else 0.0, "lr_max": float(ch.max()) if len(ch) else 0.0})
    df = pd.DataFrame(rows); df.to_csv(a.run_dir / "limit_edges.csv", index=False)
    print(df.groupby("class").agg(frac_changed_min=("frac_changed", "min"), frac_changed_med=("frac_changed", "median"),
                                   lr_min=("lr_min", "min"), lr_med=("lr_median", "median"), lr_max=("lr_max", "max")).round(3))


if __name__ == "__main__":
    main()
