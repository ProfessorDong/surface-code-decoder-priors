"""Scale statements of Section III-A, recomputed from one seeded run.

Theorem 1 says that multiplying every matching weight by lambda > 0 changes no
decision, and that multiplying every edge probability by f adds the offset -ln f to
every weight plus a small edge-dependent residual. Corollary 1 says that under the
offset alone the returned edge set cannot get smaller as f grows. This script
measures, on one sampled memory and on two exactly enumerated ones:

  * weight rescaling over twelve orders of magnitude: predictions changed against
    lambda = 1;
  * probability rescaling at f = 4 against its pure offset (weights w_e - ln f): the
    decisions on which they differ, how many of those are between edge sets of equal
    size, and the change in logical errors each causes relative to the physical prior;
  * the mean size of the returned edge set at f = 0.25, 0.5, 1, 2, 4;
  * the decisions the pure offset flips relative to the physical prior, and how many
    of those are between edge sets of equal size (Theorem 1(iv): none can be);
  * the exact change in logical error rate of a d=3, T=2 surface memory at f = 4 and of
    a d=5, T=3 repetition memory for f in [0.1, 2].

Writes research/results/theory/offset_check.json.
usage: python -m simulation.prx.theory.offset_check
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pymatching
import stim

from ..config import RESULTS_ROOT
from ..dem.edges import EdgeTable
from ..dem.priors import build_circuit
from .exact import enumerate_dem, repetition_memory

SEED = 20261125
SHOTS = 200_000
DISTANCE, ROUNDS, P = 5, 5, 0.002
F_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
LAMBDAS = tuple([10.0 ** k for k in range(-6, 7)] + [2e6])


def matching_with_weights(table: EdgeTable, w: np.ndarray) -> pymatching.Matching:
    """The table's matching graph with explicit weights (same edges, same fault ids)."""
    m = pymatching.Matching()
    for k in range(table.num_edges):
        fids = table.fault_ids(k)
        if table.v[k] < 0:
            m.add_boundary_edge(int(table.u[k]), fault_ids=fids, weight=float(w[k]), merge_strategy="disallow")
        else:
            m.add_edge(int(table.u[k]), int(table.v[k]), fault_ids=fids, weight=float(w[k]),
                       merge_strategy="disallow")
    if m.num_nodes < table.num_detectors:
        m.ensure_num_nodes(table.num_detectors)
    return m


def edge_counts(m: pymatching.Matching, dets: np.ndarray) -> np.ndarray:
    """Size of the returned edge set for every shot (0 for an empty syndrome)."""
    out = np.zeros(dets.shape[0], dtype=np.int64)
    for s in np.flatnonzero(dets.any(axis=1)):
        out[s] = len(m.decode_to_edges_array(dets[s]))
    return out


def exact_change(circuit: stim.Circuit, factors) -> dict:
    """Exact relative change of the logical error rate when every edge probability is scaled."""
    dem = circuit.detector_error_model(decompose_errors=True)
    full = enumerate_dem(dem)
    table = EdgeTable.from_dem(dem)
    bits = ((full.syndromes[:, None] >> np.arange(dem.num_detectors, dtype=np.int64)[None, :]) & 1).astype(np.uint8)

    def risk(f: float) -> float:
        pred = table.scaled(f).to_matching().decode_batch(bits)[:, 0].astype(bool)
        return full.decoder_failure(pred)

    r1 = risk(1.0)
    return {"detectors": int(dem.num_detectors), "mass": float(full.total_probability), "risk_f1": r1,
            "change_percent": {str(f): 100.0 * (risk(f) / r1 - 1.0) for f in factors},
            "max_scaled_p": {str(f): float(table.p.max()) * f for f in factors}}


def main() -> None:
    t0 = time.time()
    circuit = build_circuit("si1000", DISTANCE, ROUNDS, P, variant="google2025")
    table = EdgeTable.from_circuit(circuit)
    dets, obs = circuit.compile_detector_sampler(seed=SEED).sample(SHOTS, separate_observables=True)
    obs = obs[:, 0]
    w = table.weight

    # weight rescaling
    base = matching_with_weights(table, w).decode_batch(dets)[:, 0].astype(bool)
    changed = {f"{lam:g}": int((matching_with_weights(table, lam * w).decode_batch(dets)[:, 0].astype(bool)
                                 != base).sum()) for lam in LAMBDAS}

    # probability rescaling and the pure offset
    pred, size, errors = {}, {}, {}
    for f in F_GRID:
        m = table.scaled(f).to_matching()
        pred[f] = m.decode_batch(dets)[:, 0].astype(bool)
        size[f] = edge_counts(m, dets)
        errors[f] = int((pred[f] != obs).sum())
    m_off = matching_with_weights(table, w - np.log(4.0))
    pred_off = m_off.decode_batch(dets)[:, 0].astype(bool)
    size_off = edge_counts(m_off, dets)
    err_off = int((pred_off != obs).sum())

    dis = pred_off != pred[4.0]
    flips = pred_off != pred[1.0]
    res = {
        "config": {"family": "si1000", "variant": "google2025", "distance": DISTANCE, "rounds": ROUNDS, "p": P,
                   "shots": SHOTS, "seed": SEED, "stim": stim.__version__, "pymatching": pymatching.__version__},
        "weight_rescale_predictions_changed": changed,
        "weight_rescale_max_changed": max(changed.values()),
        "errors": {str(f): errors[f] for f in F_GRID} | {"offset_f4": err_off},
        "rescale_f4_change_percent": 100.0 * (errors[4.0] / errors[1.0] - 1.0),
        "offset_f4_change_percent": 100.0 * (err_off / errors[1.0] - 1.0),
        "offset_vs_rescale_disagreements": int(dis.sum()),
        "offset_vs_rescale_disagreements_equal_size": int((dis & (size_off == size[4.0])).sum()),
        "offset_flips_vs_physical": int(flips.sum()),
        "offset_flips_vs_physical_equal_size": int((flips & (size_off == size[1.0])).sum()),
        "mean_edge_set_size_all_shots": {str(f): float(size[f].mean()) for f in F_GRID},
        "max_scaled_edge_probability_f4": float(table.p.max()) * 4.0,
    }
    res["exact_surface_d3_T2_p0.002"] = exact_change(build_circuit("si1000", 3, 2, 0.002, variant="google2025"), (4.0,))
    res["exact_repetition_d5_T3_p0.05"] = exact_change(repetition_memory(5, 3, 0.05, 0.05),
                                                       (0.1, 0.25, 0.5, 0.75, 1.5, 2.0))
    res["seconds"] = time.time() - t0
    out = RESULTS_ROOT / "theory" / "offset_check.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "weight_rescale_predictions_changed"}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
