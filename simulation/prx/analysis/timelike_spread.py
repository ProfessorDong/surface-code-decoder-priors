"""Spread of time-like edge probabilities: Willow's shipped model, a fit to its syndromes,
and the measurement-only heterogeneity model (App. C).

Statistic: the standard deviation of ln p_e over the time-like edges of the bulk time
layers (layer class "bulk" of the pooling signatures, i.e. excluding the first two and
the last two layers), per patch and basis, on the 13-round calibration run.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from ..config import RESULTS_ROOT, WILLOW_ZIP
from ..dem.edges import EdgeClass, EdgeTable
from ..estimators.pooled import PooledPijEstimator, edge_signatures
from ..experiments.common import Condition, build_condition_circuit
from ..experiments.willow import SI_PATHWAY
from ..willow.reader import WillowArchive


def bulk_time_sd(t: EdgeTable) -> float:
    sig = edge_signatures(t)
    sel = [k for k in range(t.num_edges) if t.cls[k] == int(EdgeClass.TIME) and sig[k][0] == "E" and sig[k][4] == "bulk"]
    return float(np.std(np.log(t.p[sel]), ddof=1)), len(sel)


def main() -> None:
    out = RESULTS_ROOT / "willow_timelike_spread"; out.mkdir(parents=True, exist_ok=True)
    ar = WillowArchive(WILLOW_ZIP)
    rows = []
    for key in ar.leaves(rounds=13):
        tmpl = EdgeTable.from_dem(ar.dem(key, SI_PATHWAY))
        fit = PooledPijEstimator(kappa=0.0).fit(tmpl, ar.detection_events(key)).table
        (s0, n0), (s1, _) = bulk_time_sd(tmpl), bulk_time_sd(fit)
        rows.append({"source": "willow", "leaf": key.leaf_id, "distance": key.distance, "n_edges": n0,
                     "sd_shipped": s0, "sd_fit": s1})
    for d in (3, 5, 7):
        for sigma in (0.35, 0.5, 0.65, 0.8, 1.0):
            for seed in range(1, 11):
                c = Condition("si1000", d, 13, 0.002, extra={"variant": "google2025", "sigma_h": sigma,
                                                               "het_seed": seed, "het_channels": ["meas_flip"]})
                s, n = bulk_time_sd(EdgeTable.from_circuit(build_condition_circuit(c)))
                rows.append({"source": "model", "distance": d, "sigma": sigma, "seed": seed, "n_edges": n, "sd_true": s})
    df = pd.DataFrame(rows); df.to_csv(out / "timelike_spread.csv", index=False)
    w = df[df.source == "willow"]
    print(w.groupby("distance")[["sd_shipped", "sd_fit"]].agg(["min", "median", "max"]).round(3))
    m = df[df.source == "model"]
    print(m.groupby(["distance", "sigma"]).sd_true.median().unstack().round(3))


if __name__ == "__main__":
    main()
