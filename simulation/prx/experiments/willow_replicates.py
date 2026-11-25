"""Store per-shot outcomes for every calibration replicate of the primary estimator.

The primary Willow run saved per-shot error vectors only for replicate 0 of each
fitted prior, so a paired test against the RL prior was possible only for that
single calibration draw, while effect sizes averaged five draws. This module
re-fits the pooled estimator with shrinkage at the primary budget for every
replicate, decodes each test leaf, and stores the vectors, so that every
replicate supports its own paired comparison.

Replicate r is fitted on calibration shots [r N, (r + 1) N) of the 13-round run,
exactly as in `willow.py`, so the procedure is deterministic. Replicate 0 is
re-fitted as well and must reproduce the stored vector bit for bit; any mismatch
is reported and the run fails loudly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import WILLOW_ZIP
from ..dem.edges import EdgeTable
from ..estimators.pooled import PooledPijEstimator, apply_signature_map, signature_map
from ..results_io import load_error_vector, save_error_vector
from ..stats import discordant_counts
from ..willow.reader import LeafKey, WillowArchive
from .common import decode_errors, run_pool
from .willow import SI_PATHWAY, RL_PATHWAY

EST = "pooled_shrink"


def task_fn(t: Dict[str, Any]) -> Dict[str, Any]:
    ar = WillowArchive(t["zip"])
    run_dir = Path(t["run_dir"])
    key: LeafKey = t["test_key"]; cal: LeafKey = t["cal_key"]
    n, kappa, reps = t["n_cal"], t["kappa"], t["replicates"]
    tag = key.leaf_id.replace("/", "_")
    dets, actual = ar.detection_events(key), ar.obs_actual(key)
    template = EdgeTable.from_dem(ar.dem(key, SI_PATHWAY))
    si = load_error_vector(run_dir, f"errors/{tag}__si1000_shipped.npz")
    rl = load_error_vector(run_dir, f"errors/{tag}__rl_shipped.npz")
    cal_dets = ar.detection_events(cal)
    cal_template = EdgeTable.from_dem(ar.dem(cal, SI_PATHWAY))
    out: List[Dict[str, Any]] = []
    for r in range(reps):
        cd = cal_dets[r * n:(r + 1) * n]
        res = PooledPijEstimator(kappa=kappa).fit(cal_template, cd)
        table, _ = apply_signature_map(template, signature_map(res.table))
        errs = decode_errors(table.to_matching(), dets, actual)
        name = f"{tag}__{EST}_N={n}_rep={r}"
        stored = run_dir / "errors" / f"{name}.npz"
        reproduced = None
        if r == 0:
            if not stored.exists():
                raise FileNotFoundError(stored)
            reproduced = bool(np.array_equal(load_error_vector(run_dir, f"errors/{name}.npz"), errs))
        else:
            save_error_vector(run_dir, name, errs)
        b_si, c_si = discordant_counts(errs, si)
        b_rl, c_rl = discordant_counts(errs, rl)
        out.append({"leaf_id": key.leaf_id, "patch": key.patch, "basis": key.basis,
                    "distance": key.distance, "rounds": key.rounds, "replicate": r,
                    "n_shots": int(errs.size), "k_est": int(errs.sum()), "k_si": int(si.sum()),
                    "k_rl": int(rl.sum()), "b_vs_si": b_si, "c_vs_si": c_si,
                    "b_vs_rl": b_rl, "c_vs_rl": c_rl, "rep0_reproduced": reproduced,
                    "error_vector_path": f"errors/{name}.npz"})
    return {"rows": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    ar = WillowArchive(str(WILLOW_ZIP))
    rows = pd.read_csv(a.run_dir / "rows.csv")
    leaves = rows[["leaf_id", "distance", "patch", "basis", "rounds"]].drop_duplicates()
    tasks = []
    for lf in leaves.itertuples():
        tk = LeafKey(int(lf.distance), lf.patch, lf.basis, int(lf.rounds))
        ck = LeafKey(int(lf.distance), lf.patch, lf.basis, 13)
        tasks.append({"test_key": tk, "cal_key": ck, "run_dir": str(a.run_dir), "zip": str(WILLOW_ZIP),
                      "n_cal": a.n_cal, "kappa": a.kappa, "replicates": a.replicates})
    tasks.sort(key=lambda t: -(t["test_key"].distance ** 2 * t["test_key"].rounds))
    allrows = []
    for i, res in enumerate(run_pool(tasks, task_fn, workers=a.workers), 1):
        allrows += res["rows"]
        print(f"[{i}/{len(tasks)}] {res['rows'][0]['leaf_id']}", flush=True)
    df = pd.DataFrame(allrows).sort_values(["distance", "leaf_id", "replicate"])
    out = a.run_dir / f"replicates_{EST}_N={a.n_cal}.csv"
    df.to_csv(out, index=False)
    bad = df[(df.replicate == 0) & (df.rep0_reproduced != True)]
    print(f"wrote {out}; replicate-0 reproduced on {int((df.replicate == 0).sum()) - len(bad)}"
          f" of {int((df.replicate == 0).sum())} leaves")
    if len(bad):
        raise SystemExit(f"replicate 0 NOT reproduced on {len(bad)} leaves: determinism broken")


if __name__ == "__main__":
    main()
