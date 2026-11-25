"""Post hoc hardware sensitivity to the shrinkage strength (kappa protocol, "Hardware").

The Willow analysis used kappa = 5, fixed in the pre-run plan; it stays primary.
This module refits the pooled estimator with the development-selected kappa at the
primary budget for all five calibration draws, exactly as ``willow.py`` does
(calibration prefix r of the 13-round run, signature transfer to the test leaf),
and stores the per-shot vectors under distinct names. As a determinism check,
replicate 0 is also refitted with kappa = 5 and must reproduce the stored vector.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import WILLOW_ZIP
from ..dem.edges import EdgeTable
from ..estimators.pooled import PooledPijEstimator, apply_signature_map, signature_map
from ..results_io import load_error_vector, save_error_vector
from ..willow.reader import LeafKey, WillowArchive
from .common import decode_errors, run_pool
from .willow import SI_PATHWAY


def fit_decode(cal_template, cal_dets, template, dets, actual, kappa):
    res = PooledPijEstimator(kappa=kappa).fit(cal_template, cal_dets)
    table, _ = apply_signature_map(template, signature_map(res.table))
    return decode_errors(table.to_matching(), dets, actual)


def task_fn(t: Dict[str, Any]) -> Dict[str, Any]:
    ar = WillowArchive(t["zip"])
    run_dir = Path(t["run_dir"])
    key: LeafKey = t["test_key"]; cal: LeafKey = t["cal_key"]
    n, kappa, reps = t["n_cal"], t["kappa"], t["replicates"]
    tag = key.leaf_id.replace("/", "_")
    dets, actual = ar.detection_events(key), ar.obs_actual(key)
    template = EdgeTable.from_dem(ar.dem(key, SI_PATHWAY))
    cal_dets = ar.detection_events(cal)
    cal_template = EdgeTable.from_dem(ar.dem(cal, SI_PATHWAY))
    e5 = fit_decode(cal_template, cal_dets[:n], template, dets, actual, 5.0)
    reproduced = bool(np.array_equal(e5, load_error_vector(run_dir, f"errors/{tag}__pooled_shrink_N={n}_rep=0.npz")))
    for r in range(reps):
        e = fit_decode(cal_template, cal_dets[r * n:(r + 1) * n], template, dets, actual, kappa)
        save_error_vector(run_dir, f"{tag}__pooled_shrink_kappa{kappa:g}_N={n}_rep={r}", e)
    return {"leaf_id": key.leaf_id, "kappa5_rep0_reproduced": reproduced}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rows = pd.read_csv(a.run_dir / "rows.csv", usecols=["leaf_id", "distance", "patch", "basis", "rounds"])
    leaves = rows.drop_duplicates().sort_values("leaf_id")
    tasks = [{"test_key": LeafKey(int(l.distance), l.patch, l.basis, int(l.rounds)),
              "cal_key": LeafKey(int(l.distance), l.patch, l.basis, 13), "run_dir": str(a.run_dir),
              "zip": str(WILLOW_ZIP), "n_cal": a.n_cal, "kappa": a.kappa, "replicates": a.replicates}
             for l in leaves.itertuples()]
    out: List[Dict[str, Any]] = []
    for i, res in enumerate(run_pool(tasks, task_fn, workers=a.workers), 1):
        out.append(res)
        print(f"[{i}/{len(tasks)}] {res['leaf_id']} reproduced={res['kappa5_rep0_reproduced']}", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(a.run_dir / f"kappa{a.kappa:g}_sensitivity_leaves.csv", index=False)
    bad = int((~df.kappa5_rep0_reproduced).sum())
    print(f"kappa = 5 replicate 0 reproduced on {len(df) - bad} of {len(df)} leaves")
    if bad:
        raise SystemExit("determinism check failed")


if __name__ == "__main__":
    main()
