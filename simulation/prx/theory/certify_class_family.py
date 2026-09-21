"""Certified minimum of the matching risk over the continuous per-class prior family.

Family: theta(s) with p_e(s) = s_{c(e)} p_e, one scale per edge class c in
{space, time, diag, boundary}, s in a declared box B = [lo, hi]^4 (log scale).

Why a certificate is possible. By Lemma 1 matching returns the logical class of the
minimum-weight fault set of the graphlike model it is given, so its decision for
syndrome x is argmin_l W_l(x; s), where W_l(x; s) is the minimum total weight
w_e(s) = ln[(1 - p_e(s)) / p_e(s)] of an edge set with syndrome x and logical class l.
W_l is computed for every syndrome at once by the XOR recursion of Proposition 1 in
the (min, +) semiring. Every w_e(s) is strictly decreasing in its class scale, hence
W_l(x; s) is non-increasing in every scale, and over a box [a, b]

    W_l(x; b) <= W_l(x; s) <= W_l(x; a)   for all s in [a, b].

If W_0(x; a) < W_1(x; b), matching decides class 0 for x everywhere in the box (and
symmetrically); otherwise x is ambiguous. A lower bound on the risk over the box is

    LB = sum over decided x of pi_true[x, 1 - l(x)] + sum over ambiguous x of min_l pi_true[x, l],

and the risk at any point s is bounded above by evaluating the decision there, counting
exact ties as failures. Best-first branch and bound over boxes then returns an interval
[LB*, UB*] containing the minimum of the risk over B; UB* is attained (up to ties) by a
recorded prior. Everything is evaluated over every syndrome of the enumerated model
(no truncation). A margin ``tol`` on weight comparisons makes near-ties ambiguous, which
only lowers LB, so floating-point error cannot make the bound invalid.
"""
from __future__ import annotations

import argparse
import heapq
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESULTS_ROOT
from ..dem.edges import CLASS_NAMES, EdgeTable
from ..dem.priors import build_circuit
from .exact import enumerate_dem

CLASSES = ("space", "time", "diag", "boundary")


class Problem:
    def __init__(self, T: int, p: float, distance: int = 3):
        dem = build_circuit("si1000", distance, T, p, variant="google2025").detector_error_model(decompose_errors=True)
        self.D = dem.num_detectors
        self.size = 1 << self.D
        full = enumerate_dem(dem)
        self.pi = np.zeros((self.size, 2))
        self.pi[full.syndromes, 0] = full.p_logical0
        self.pi[full.syndromes, 1] = full.p_logical1
        self.mass_deficit = 1.0 - float(self.pi.sum())
        t = EdgeTable.from_dem(dem)
        self.table = t
        self.masks = np.array([(1 << int(u)) | ((1 << int(v)) if int(v) >= 0 else 0) for u, v in zip(t.u, t.v)],
                              dtype=np.int64)
        self.flip = (t.fault_mask.astype(np.int64) & 1).astype(bool)
        names = [CLASS_NAMES[int(c)] for c in t.cls]
        assert set(names) <= set(CLASSES), names
        self.cidx = np.array([CLASSES.index(n) for n in names])
        self.p0 = t.p.astype(float)
        self.ar = np.arange(self.size, dtype=np.int64)
        self.flip_axes = []
        for u_, v_, f in zip(t.u, t.v, self.flip):
            ax = [self.D - 1 - int(u_)] + ([self.D - 1 - int(v_)] if int(v_) >= 0 else [])
            self.flip_axes.append(tuple(ax + ([self.D] if f else [])))
        self.r_phys = self.risk_upper(np.zeros(4))
        self.p_rep = np.array([np.exp(np.log(self.p0[self.cidx == c]).mean()) if (self.cidx == c).any() else np.nan
                               for c in range(4)])           # representative edge of each class

    def split(self, a: np.ndarray, b: np.ndarray, rule: str = "wspan"):
        """Which class to branch on, and where. ``longest``: longest edge of the log-scale
        box, halved. ``wspan``: the class whose edge weights move most over the box, split
        at the point halving that weight range (branching guided by the nonlinearity)."""
        if rule == "longest":
            k = int(np.argmax(b - a))
            return k, float((a[k] + b[k]) / 2)
        w = lambda u: np.log1p(-self.p_rep * np.exp(u)) - np.log(self.p_rep * np.exp(u))
        k = int(np.argmax(np.where(b - a > 0, w(a) - w(b), -np.inf)))
        wm = 0.5 * (w(a)[k] + w(b)[k])
        u = float(np.log(1.0 / (1.0 + np.exp(wm)) / self.p_rep[k]))
        return k, float(np.clip(u, a[k] + 1e-12, b[k] - 1e-12))

    def weights(self, u: np.ndarray) -> np.ndarray:
        p = self.p0 * np.exp(u[self.cidx])
        if np.any(p >= 0.5) or np.any(p <= 0):
            raise ValueError("scaled probability outside (0, 1/2)")
        return np.log1p(-p) - np.log(p)

    def min_weights(self, u: np.ndarray) -> np.ndarray:
        """W[x, l]: minimum weight of an edge set with syndrome x and logical class l."""
        w = self.weights(u)
        W = np.full((2,) * self.D + (2,), np.inf)       # axis D-1-j holds detector bit j; last axis: logical
        W[(0,) * self.D + (0,)] = 0.0
        for axes, we in zip(self.flip_axes, w):
            # XOR by the edge's detector mask = reversal along its (at most two) bit axes
            np.minimum(W, np.flip(W, axis=axes) + we, out=W)
        return W.reshape(self.size, 2)

    def lower_bound(self, a: np.ndarray, b: np.ndarray, tol: float):
        Wa, Wb = self.min_weights(a), self.min_weights(b)   # a: lower corner (largest weights)
        d0 = Wa[:, 0] < Wb[:, 1] - tol * (1 + np.abs(Wb[:, 1]))
        d1 = Wa[:, 1] < Wb[:, 0] - tol * (1 + np.abs(Wb[:, 0]))
        amb = ~(d0 | d1)
        lb = self.pi[d0, 1].sum() + self.pi[d1, 0].sum() + np.minimum(self.pi[amb, 0], self.pi[amb, 1]).sum()
        return float(lb), float(self.pi[amb].sum())

    def risk_upper(self, u: np.ndarray, tol: float = 1e-12) -> float:
        W = self.min_weights(u)
        diff = W[:, 0] - W[:, 1]
        tie = np.abs(diff) <= tol * (1 + np.abs(W[:, 0]))
        dec1 = diff > 0
        r = np.where(dec1, self.pi[:, 0], self.pi[:, 1])
        r = np.where(tie, np.maximum(self.pi[:, 0], self.pi[:, 1]), r)
        return float(r.sum())

    def risk_matching(self, u: np.ndarray) -> float:
        """Risk of PyMatching itself at u, decoding every syndrome with nonzero mass (check)."""
        scaled = self.table.with_probabilities(self.p0 * np.exp(u[self.cidx]))
        idx = np.flatnonzero(self.pi.sum(axis=1) > 0)
        bits = ((idx[:, None] >> np.arange(self.D)[None, :]) & 1).astype(np.uint8)
        pred = scaled.to_matching().decode_batch(bits)[:, 0].astype(bool)
        return float(np.where(pred, self.pi[idx, 0], self.pi[idx, 1]).sum())


def certify(prob: Problem, lo: float, hi: float, rel_gap: float, max_boxes: int, tol: float = 1e-9, log=print,
            rule: str = "wspan"):
    """Best-first branch and bound over the box [ln lo, ln hi]^4. Returns a result dict."""
    a0, b0 = np.full(4, np.log(lo)), np.full(4, np.log(hi))
    t0 = time.time()
    lb, amb = prob.lower_bound(a0, b0, tol)
    c = (a0 + b0) / 2
    best_ub, best_u = prob.risk_upper(c), c.copy()
    for cand in (np.zeros(4),):                       # the physical prior is inside the box
        r = prob.risk_upper(cand)
        if r < best_ub:
            best_ub, best_u = r, cand.copy()
    heap = [(lb, 0, a0, b0)]
    n_eval, counter = 1, 1
    pruned_min = np.inf          # smallest lower bound among discarded boxes (they still count)
    target = rel_gap * prob.r_phys
    while heap:
        lb, _, a, b = heapq.heappop(heap)
        if lb >= best_ub - target:
            heapq.heappush(heap, (lb, 0, a, b))
            break
        if n_eval >= max_boxes:
            heapq.heappush(heap, (lb, 0, a, b))
            break
        k, mid = prob.split(a, b, rule)
        for ca, cb in ((a, np.where(np.arange(4) == k, mid, b)), (np.where(np.arange(4) == k, mid, a), b)):
            ca, cb = ca.copy(), cb.copy()
            clb, camb = prob.lower_bound(ca, cb, tol)
            n_eval += 1
            cc = (ca + cb) / 2
            ub = prob.risk_upper(cc)
            if ub < best_ub:
                best_ub, best_u = ub, cc.copy()
            if clb < best_ub - target:
                counter += 1
                heapq.heappush(heap, (clb, counter, ca, cb))
            else:
                pruned_min = min(pruned_min, clb)
        if n_eval % 2000 < 2:
            log(f"    boxes {n_eval}, open {len(heap)}, LB {heap[0][0] if heap else float('nan'):.6e}, "
                f"UB {best_ub:.6e}, {time.time() - t0:.0f}s")
    # every point of the box lies in an open box or a discarded one
    global_lb = min(best_ub, heap[0][0] if heap else np.inf, pruned_min)
    return {"lo": lo, "hi": hi, "LB": global_lb, "UB": best_ub, "best_scales": np.exp(best_u).tolist(),
            "boxes": n_eval, "open_boxes": len(heap), "gap_rel_to_phys": (best_ub - global_lb) / prob.r_phys,
            "converged": (best_ub - global_lb) <= target, "seconds": time.time() - t0}

_PROB = None


def _init(T, p, distance):
    global _PROB
    _PROB = Problem(T, p, distance)


def _eval(args):
    a, b, tol = args
    lb, _ = _PROB.lower_bound(a, b, tol)
    c = (a + b) / 2
    return lb, _PROB.risk_upper(c), c


def certify_parallel(prob: Problem, T: int, p: float, lo: float, hi: float, rel_gap: float, max_boxes: int,
                     workers: int, batch: int = 256, tol: float = 1e-9, log=print, rule: str = "wspan"):
    """Same bounds and pruning as ``certify``, with boxes evaluated in parallel batches."""
    import multiprocessing as mp
    a0, b0 = np.full(4, np.log(lo)), np.full(4, np.log(hi))
    t0 = time.time()
    lb, _ = prob.lower_bound(a0, b0, tol)
    best_ub, best_u = prob.risk_upper(np.zeros(4)), np.zeros(4)
    c = (a0 + b0) / 2
    r = prob.risk_upper(c)
    if r < best_ub:
        best_ub, best_u = r, c
    target = rel_gap * prob.r_phys
    heap, counter, n_eval, pruned_min, last = [(lb, 0, a0, b0)], 1, 1, np.inf, time.time()
    with mp.get_context("forkserver").Pool(workers, initializer=_init, initargs=(T, p, 3)) as pool:
        while heap and heap[0][0] < best_ub - target and n_eval < max_boxes:
            parents = []
            while heap and len(parents) < batch and heap[0][0] < best_ub - target:
                parents.append(heapq.heappop(heap))
            jobs = []
            for _, _, a, b in parents:
                k, mid = prob.split(a, b, rule)
                e = np.arange(4) == k
                jobs += [(a.copy(), np.where(e, mid, b), tol), (np.where(e, mid, a), b.copy(), tol)]
            for (ca, cb, _), (clb, ub, cc) in zip(jobs, pool.map(_eval, jobs, chunksize=4)):
                n_eval += 1
                if ub < best_ub:
                    best_ub, best_u = ub, cc
                if clb < best_ub - target:
                    counter += 1
                    heapq.heappush(heap, (clb, counter, ca, cb))
                else:
                    pruned_min = min(pruned_min, clb)
            # progress on wall-clock, so short runs report too
            if time.time() - last >= 300:
                last = time.time()
                log(f"    boxes {n_eval}, open {len(heap)}, LB {heap[0][0] if heap else best_ub:.8e}, "
                    f"UB {best_ub:.8e}, gap {(best_ub - (heap[0][0] if heap else best_ub)) / prob.r_phys:.2e}, "
                    f"{time.time() - t0:.0f}s")
    open_min = min((h[0] for h in heap), default=np.inf)
    global_lb = min(best_ub, open_min, pruned_min)
    return {"lo": lo, "hi": hi, "LB": global_lb, "UB": best_ub, "best_scales": np.exp(best_u).tolist(),
            "boxes": n_eval, "open_boxes": len(heap), "gap_rel_to_phys": (best_ub - global_lb) / prob.r_phys,
            "converged": (best_ub - global_lb) <= target, "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, nargs="+", default=[2])
    ap.add_argument("--p", type=float, nargs="+", default=[0.0005, 0.001, 0.002, 0.004, 0.008])
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=10.0)
    ap.add_argument("--rel-gap", type=float, default=1e-6)
    ap.add_argument("--max-boxes", type=int, default=400_000)
    ap.add_argument("--out", default="certified_class_family.csv")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--branch", choices=["wspan", "longest"], default="wspan")
    a = ap.parse_args()
    rows = []
    out = RESULTS_ROOT / "theory" / a.out
    for T in a.rounds:
        for p in a.p:
            prob = Problem(T, p)
            hi = min(a.hi, 0.49 / float(prob.p0.max()))
            print(f"T={T} p={p:g}: D={prob.D}, R_phys={prob.r_phys:.10e}, mass deficit {prob.mass_deficit:.1e}, box [{a.lo}, {hi:.3g}]^4", flush=True)
            if a.workers > 1:
                res = certify_parallel(prob, T, p, a.lo, hi, a.rel_gap, a.max_boxes, a.workers, rule=a.branch)
            else:
                res = certify(prob, a.lo, hi, a.rel_gap, a.max_boxes, rule=a.branch)
            r_match = prob.risk_matching(np.log(np.array(res["best_scales"])))
            res.update({"T": T, "p": p, "branch": a.branch, "R_phys": prob.r_phys, "R_best_matching": r_match,
                        "gain_UB_certified": 1 - res["UB"] / prob.r_phys,   # achieved (lower bound on headroom)
                        "gain_LB_certified_max": 1 - res["LB"] / prob.r_phys})  # no prior in the box does better
            rows.append(res)
            print(f"   certified min risk in [{res['LB']:.10e}, {res['UB']:.10e}]  headroom in "
                  f"[{100*res['gain_UB_certified']:.4f}%, {100*res['gain_LB_certified_max']:.4f}%]  "
                  f"boxes {res['boxes']} converged {res['converged']}  (PyMatching at best: {r_match:.10e})", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
