"""What the pairwise correlation estimator converges to, exactly.

With z_i = (-1)^{x_i} and D_m the detector support of independent mechanism m of the
*undecomposed* detector error model, the population moments factorize,

    E z_i      = prod_{m : i in D_m} (1 - 2 p_m),
    E z_i z_j  = prod_{m : |D_m cap {i,j}| = 1} (1 - 2 p_m),

so the pairwise estimator's limit obeys

    1 - 2 phat_ij,inf = sqrt( E z_i E z_j / E z_i z_j ) = prod_{m : i,j in D_m} (1 - 2 p_m).   (1)

It therefore estimates the parity aggregate of every mechanism containing BOTH detectors,
hyperedges included, which is not the edge probability of the graph the decoder is given.

For a boundary edge the estimator divides the detector's marginal by the bulk edges at it.
When no mechanism at i meets two or more graph neighbours of i -- true for every model
here -- that leaves

    1 - 2 phat_i,bdry,inf = prod_{m : i in D_m, D_m cap N(i) = empty} (1 - 2 p_m).            (2)

Any mechanism at i that also touches a graph neighbour is attributed to that bulk pair and
drops out of the boundary estimate, even when the decomposition gave it a boundary
component at i. That is why the boundary limits come out below the graph prior.

Both are statements about the raw formula. The estimator as implemented also clips
its output into [P_MIN, P_MAX] and falls back to the reference prior wherever an
estimate is not finite or not positive, so its population output obeys (1) and (2)
only where neither rule fires. This script therefore checks (1) and (2) against the
DEMs used in the paper AND against the implemented estimator fed exact population
moments, recording the fallback counts and the margin to the clipping range, and
writes theory/population_target_identity.csv.

    python -m simulation.prx.theory.population_target
"""
from __future__ import annotations

import collections
import csv

import numpy as np

from ..dem.edges import EdgeTable
from ..config import P_MAX, P_MIN
from ..estimators.pij import PijEstimator
from ..results_io import RESULTS_ROOT
from .certify_class_family import build_circuit

ROUNDS = (2, 3)
RATES = (0.0005, 0.001, 0.002, 0.004, 0.008)


def mechanisms(circ):
    """(probability, detector support) of every mechanism of the undecomposed model."""
    out = []
    for inst in circ.detector_error_model(decompose_errors=False).flattened():
        if inst.type == "error":
            out.append((inst.args_copy()[0],
                        frozenset(t.val for t in inst.targets_copy() if t.is_relative_detector_id())))
    return out


def analyse(T: int, p: float) -> dict:
    circ = build_circuit("si1000", 3, T, p, variant="google2025")
    tb = EdgeTable.from_circuit(circ)
    mech = mechanisms(circ)
    bulk = np.flatnonzero(tb.v >= 0)
    nbr = collections.defaultdict(set)
    for k in bulk:
        nbr[int(tb.u[k])].add(int(tb.v[k])); nbr[int(tb.v[k])].add(int(tb.u[k]))

    def prod(pred):
        v = 1.0
        for pm, d in mech:
            if pred(d):
                v *= (1 - 2 * pm)
        return v

    err_bulk = 0.0
    above = 0
    for k in bulk:
        i, j = int(tb.u[k]), int(tb.v[k])
        lhs = np.sqrt(prod(lambda d, i=i: i in d) * prod(lambda d, j=j: j in d)
                      / prod(lambda d, i=i, j=j: (i in d) != (j in d)))
        rhs = prod(lambda d, i=i, j=j: i in d and j in d)
        err_bulk = max(err_bulk, abs(lhs - rhs))
        if (1 - lhs) / 2 > tb.p[k] + 1e-15:
            above += 1

    err_bdry = 0.0
    below = 0
    multi = 0
    bdry = np.flatnonzero(tb.v < 0)
    for k in bdry:
        i = int(tb.u[k])
        den = 1.0
        for j in nbr[i]:
            den *= prod(lambda d, i=i, j=j: i in d and j in d)
        lhs = prod(lambda d, i=i: i in d) / den
        rhs = prod(lambda d, i=i: i in d and not (d & nbr[i]))
        err_bdry = max(err_bdry, abs(lhs - rhs))
        multi += sum(1 for pm, d in mech if i in d and len(d & nbr[i]) >= 2)
        if (1 - lhs) / 2 < tb.p[k] - 1e-15:
            below += 1

    # the same targets against the estimator itself, fed exact population moments
    def ez(S):
        v = 1.0
        for pm, d in mech:
            if len(d & S) % 2 == 1:
                v *= (1 - 2 * pm)
        return v

    xi = np.array([(1 - ez(frozenset({i}))) / 2 for i in range(tb.num_detectors)])
    pairs = np.stack([tb.u[bulk], tb.v[bulk]], axis=1) if bulk.size else np.zeros((0, 2), int)
    xij = np.array([(1 - ez(frozenset({int(u)})) - ez(frozenset({int(v)}))
                     + ez(frozenset({int(u), int(v)}))) / 4 for u, v in pairs])
    fit = PijEstimator().fit_moments(tb, xi, xij, np.inf)
    est_bulk = max((abs((1 - prod(lambda d, i=int(u), j=int(v): i in d and j in d)) / 2 - fit.table.p[k])
                    for k, (u, v) in zip(bulk, pairs)), default=0.0)
    groups: dict = collections.defaultdict(list)
    for k in bdry:
        groups[int(tb.u[k])].append(k)
    est_bdry = max((abs((1 - prod(lambda d, i=i: i in d and not (d & nbr[i]))) / 2
                        - float(fit.table.p[ks].sum())) for i, ks in groups.items()), default=0.0)
    return {"T": T, "p": p, "mechanisms": len(mech),
            "hyperedges": sum(1 for _, d in mech if len(d) >= 3),
            "bulk_edges": len(bulk), "bulk_identity_max_err": err_bulk,
            "bulk_limit_above_graph": above,
            "boundary_edges": len(bdry), "boundary_identity_max_err": err_bdry,
            "boundary_limit_below_graph": below, "mech_touching_two_neighbours": multi,
            "boundary_detectors": len(groups),
            "estimator_bulk_max_err": float(est_bulk), "estimator_boundary_max_err": float(est_bdry),
            "n_fallback_bulk": fit.params["n_fallback_bulk"],
            "n_fallback_boundary": fit.params["n_fallback_boundary"],
            "target_min": float(fit.table.p.min()), "target_max": float(fit.table.p.max()),
            "clip_lo": P_MIN, "clip_hi": P_MAX}


def main() -> None:
    out = RESULTS_ROOT / "theory" / "population_target_identity.csv"
    rows = [analyse(T, p) for T in ROUNDS for p in RATES]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
            print(f"T={r['T']} p={r['p']:<7} bulk id err {r['bulk_identity_max_err']:.1e}, "
                  f"boundary id err {r['boundary_identity_max_err']:.1e}, "
                  f"vs estimator {max(r['estimator_bulk_max_err'], r['estimator_boundary_max_err']):.1e} "
                  f"({r['n_fallback_bulk'] + r['n_fallback_boundary']} fallbacks), "
                  f"{r['bulk_limit_above_graph']}/{r['bulk_edges']} bulk limits above the graph prior, "
                  f"{r['boundary_limit_below_graph']}/{r['boundary_edges']} boundary limits below")
    print("wrote", out)


if __name__ == "__main__":
    main()
