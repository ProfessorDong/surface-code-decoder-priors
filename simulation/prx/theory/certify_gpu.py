"""GPU backend for the per-class certificate, so that three rounds are reachable.

The bound of :mod:`certify_class_family` is memory bound, not compute bound: one
evaluation streams the whole :math:`(2, 2^D)` table of minimum weights once per edge.
At three rounds that table is 268 MB, and on the CPU the recursion runs at about
2 GB/s per core and *slows down* as workers are added, because they share one memory
bus. Eighteen workers bought a factor of about 2.5. A GPU streams the same table at
about 900 GB/s, which is what makes three rounds possible at all.

Everything else is unchanged, deliberately. The arithmetic is float64, the edge order
is the edge order of the table, the comparisons carry the same margin, and the bound
and its proof are those of :mod:`certify_class_family`. ``check_against_cpu`` asserts
agreement with the CPU implementation: the tables of minimum weights bit for bit, and
the summed risks and bounds to floating-point rounding, the two summations differing
only in the order they add the same terms.

Layout: ``W[l, s]`` is the minimum weight of an edge set with syndrome ``s`` and
logical class ``l``, with bit ``j`` of ``s`` holding detector ``j``. An edge flipping
detectors ``u < v`` and logical class ``f`` maps ``(l, s)`` to ``(l ^ f, s ^ mask)``,
which is a swap of halves along each of the at most two bit axes, so the update is
done with slice views and no gathers.
"""
from __future__ import annotations

import numpy as np

from ..dem.edges import CLASS_NAMES, EdgeTable
from ..dem.priors import build_circuit
from .exact import enumerate_dem

CLASSES = ("space", "time", "diag", "boundary")


class GpuProblem:
    """The certificate's bound, evaluated on the GPU."""

    def __init__(self, T: int, p: float, distance: int = 3, device: str = "cuda", dtype=None):
        import torch

        self.torch = torch
        self.dtype = dtype or torch.float64
        self.device = torch.device(device)
        dem = build_circuit("si1000", distance, T, p, variant="google2025").detector_error_model(decompose_errors=True)
        self.D = dem.num_detectors
        self.size = 1 << self.D
        full = enumerate_dem(dem)
        pi = np.zeros((2, self.size))
        pi[0, full.syndromes] = full.p_logical0
        pi[1, full.syndromes] = full.p_logical1
        self.mass_deficit = 1.0 - float(pi.sum())
        self.pi = torch.from_numpy(pi).to(self.device, torch.float64)
        self.pi_min = torch.minimum(self.pi[0], self.pi[1])
        self.pi_max = torch.maximum(self.pi[0], self.pi[1])
        t = EdgeTable.from_dem(dem)
        self.table = t
        names = [CLASS_NAMES[int(c)] for c in t.cls]
        assert set(names) <= set(CLASSES), names
        self.cidx = np.array([CLASSES.index(n) for n in names])
        self.p0 = t.p.astype(float)
        self.p_rep = np.array([np.exp(np.log(self.p0[self.cidx == c]).mean()) if (self.cidx == c).any() else np.nan
                               for c in range(4)])
        flip = (t.fault_mask.astype(np.int64) & 1).astype(bool)
        self.edges = []                      # (sorted bit positions, logical flip)
        for u_, v_, f in zip(t.u, t.v, flip):
            bits = [int(u_)] + ([int(v_)] if int(v_) >= 0 else [])
            self.edges.append((tuple(sorted(bits)), bool(f)))
        self._buf = [torch.empty((2, self.size), device=self.device, dtype=self.dtype) for _ in range(2)]
        self.r_phys = self.risk_upper(np.zeros(4))

    # ---------------------------------------------------------------- weights
    def weights(self, u: np.ndarray) -> np.ndarray:
        p = self.p0 * np.exp(u[self.cidx])
        if np.any(p >= 0.5) or np.any(p <= 0):
            raise ValueError("scaled probability outside (0, 1/2)")
        return np.log1p(-p) - np.log(p)

    def _views(self, flat, bits):
        """Split the syndrome axis at the edge's bit positions: returns a view and the
        index of the axis (or axes) that the edge swaps."""
        D = self.D
        if len(bits) == 1:
            j = bits[0]
            return flat.view(2, 1 << (D - 1 - j), 2, 1 << j), (2,)
        j, k = bits                                   # j < k
        return flat.view(2, 1 << (D - 1 - k), 2, 1 << (k - j - 1), 2, 1 << j), (2, 4)

    def min_weights(self, u: np.ndarray):
        """W[l, s] for every syndrome at once, by the (min, +) recursion over edges."""
        torch = self.torch
        w = self.weights(u)
        src, dst = self._buf
        src.fill_(float("inf"))
        src[0, 0] = 0.0
        for (bits, f), we in zip(self.edges, w):
            S, axes = self._views(src, bits)
            Dv, _ = self._views(dst, bits)
            we = float(we)
            if len(bits) == 1:
                for l in (0, 1):
                    m = l ^ int(f)
                    for b in (0, 1):
                        a = Dv[l, :, b, :]
                        torch.add(S[m, :, 1 - b, :], we, out=a)
                        torch.minimum(a, S[l, :, b, :], out=a)
            else:
                for l in (0, 1):
                    m = l ^ int(f)
                    for b1 in (0, 1):
                        for b2 in (0, 1):
                            a = Dv[l, :, b1, :, b2, :]
                            torch.add(S[m, :, 1 - b1, :, 1 - b2, :], we, out=a)
                            torch.minimum(a, S[l, :, b1, :, b2, :], out=a)
            src, dst = dst, src
        self._buf = [src, dst]
        return src                                    # live until the next call

    # ----------------------------------------------------------------- bounds
    def lower_bound(self, a: np.ndarray, b: np.ndarray, tol: float = 1e-9):
        """Lower bound on the risk over the box [a, b] of log-scales, and its ambiguous mass."""
        torch = self.torch
        Wa = self.min_weights(a).clone()              # a: lower corner, largest weights
        Wb = self.min_weights(b)
        d0 = Wa[0] < Wb[1] - tol * (1 + Wb[1].abs())  # class 0 decided everywhere in the box
        d1 = Wa[1] < Wb[0] - tol * (1 + Wb[0].abs())
        amb = ~(d0 | d1)
        lb = (torch.where(d0, self.pi[1], torch.zeros((), device=self.device, dtype=torch.float64)).sum()
              + torch.where(d1, self.pi[0], torch.zeros((), device=self.device, dtype=torch.float64)).sum()
              + torch.where(amb, self.pi_min, torch.zeros((), device=self.device, dtype=torch.float64)).sum())
        return float(lb), float(torch.where(amb, self.pi[0] + self.pi[1],
                                            torch.zeros((), device=self.device, dtype=torch.float64)).sum())

    def risk_upper(self, u: np.ndarray, tol: float = 1e-12) -> float:
        """Risk of matching at the single point u, with exact ties counted as failures."""
        torch = self.torch
        W = self.min_weights(u)
        diff = W[0] - W[1]
        tie = diff.abs() <= tol * (1 + W[0].abs())
        r = torch.where(diff > 0, self.pi[0], self.pi[1])
        r = torch.where(tie, self.pi_max, r)
        return float(r.sum())

    def split(self, a: np.ndarray, b: np.ndarray, rule: str = "wspan"):
        """Identical to the CPU rule: branch the class whose weights move most, at the
        scale halving that weight range."""
        if rule == "longest":
            k = int(np.argmax(b - a))
            return k, float((a[k] + b[k]) / 2)
        w = lambda u: np.log1p(-self.p_rep * np.exp(u)) - np.log(self.p_rep * np.exp(u))
        k = int(np.argmax(np.where(b - a > 0, w(a) - w(b), -np.inf)))
        wm = 0.5 * (w(a)[k] + w(b)[k])
        u = float(np.log(1.0 / (1.0 + np.exp(wm)) / self.p_rep[k]))
        return k, float(np.clip(u, a[k] + 1e-12, b[k] - 1e-12))


def check_against_cpu(T: int, p: float, points: int = 3, seed: int = 0, tol: float = 1e-9) -> dict:
    """Assert that the GPU bound reproduces the CPU bound of ``certify_class_family``.

    Compares the minimum-weight tables entry by entry, the risk at sampled points, and
    the lower bound over sampled boxes. Returns the largest disagreements seen.
    """
    from .certify_class_family import Problem

    cpu, gpu = Problem(T, p), GpuProblem(T, p)
    rng = np.random.default_rng(seed)
    worst_w = worst_r = worst_lb = 0.0
    for _ in range(points):
        u = rng.uniform(np.log(0.4), np.log(2.5), 4)
        Wc = cpu.min_weights(u)
        Wg = gpu.min_weights(u).cpu().numpy().T
        finite = np.isfinite(Wc)
        assert np.array_equal(finite, np.isfinite(Wg)), "reachability differs"
        worst_w = max(worst_w, float(np.abs(Wc[finite] - Wg[finite]).max()))
        worst_r = max(worst_r, abs(cpu.risk_upper(u) - gpu.risk_upper(u)))
        a = np.minimum(u, rng.uniform(np.log(0.4), np.log(2.5), 4))
        b = np.maximum(a + 1e-3, rng.uniform(np.log(0.4), np.log(2.5), 4))
        worst_lb = max(worst_lb, abs(cpu.lower_bound(a, b, tol)[0] - gpu.lower_bound(a, b, tol)[0]))
    return {"max_weight_diff": worst_w, "max_risk_diff": worst_r, "max_lower_bound_diff": worst_lb,
            "r_phys_cpu": cpu.r_phys, "r_phys_gpu": gpu.r_phys}


# ---------------------------------------------------------------------------
# Branch and bound with corner reuse ("plunging")
# ---------------------------------------------------------------------------

def _bound_from(prob, Wa, Wb, tol):
    """The box lower bound from the two corner tables, which the caller owns."""
    torch = prob.torch
    zero = torch.zeros((), device=prob.device, dtype=torch.float64)
    d0 = Wa[0] < Wb[1] - tol * (1 + Wb[1].abs())
    d1 = Wa[1] < Wb[0] - tol * (1 + Wb[0].abs())
    amb = ~(d0 | d1)
    lb = (torch.where(d0, prob.pi[1], zero).sum() + torch.where(d1, prob.pi[0], zero).sum()
          + torch.where(amb, prob.pi_min, zero).sum())
    return float(lb)


def certify_gpu(prob, lo: float, hi: float, rel_gap: float, max_boxes: int, tol: float = 1e-9,
                dive: int = 8, log=print, checkpoint=None, checkpoint_every: int = 2000,
                state: dict | None = None):
    """Best-first branch and bound, diving from each popped box so that corner tables
    are reused instead of recomputed.

    Each evaluated corner costs one pass over the whole table, which dominates
    everything else, so the cost of the search is counted in corner evaluations. A dive
    keeps the tables of the box it descends into, which brings the steady-state cost
    down to about one evaluation per box, against two without it.

    ``checkpoint`` is a path; the search state is written there periodically and
    restored by passing the loaded dictionary as ``state``, so a long run can be
    interrupted without losing its bound.
    """
    import heapq
    import time

    torch = prob.torch
    a0, b0 = np.full(4, np.log(lo)), np.full(4, np.log(hi))
    target = rel_gap * prob.r_phys
    t0 = time.time()
    if state is None:
        best_ub, best_u = prob.risk_upper(np.zeros(4)), np.zeros(4)
        c = (a0 + b0) / 2
        r = prob.risk_upper(c)
        if r < best_ub:
            best_ub, best_u = r, c
        Wa, Wb = prob.min_weights(a0).clone(), prob.min_weights(b0).clone()
        heap = [(_bound_from(prob, Wa, Wb, tol), 0, a0, b0)]
        counter, n_eval, n_box, pruned_min = 1, 4, 1, np.inf
        del Wa, Wb
    else:
        heap = [(float(l), int(c), np.asarray(a), np.asarray(b)) for l, c, a, b in state["heap"]]
        heapq.heapify(heap)
        best_ub, best_u = float(state["best_ub"]), np.asarray(state["best_u"])
        counter, n_eval, n_box = int(state["counter"]), int(state["n_eval"]), int(state["n_box"])
        pruned_min = float(state["pruned_min"])
        log(f"    resumed with {len(heap)} open boxes, {n_box} boxes done, UB {best_ub:.8e}")

    def save():
        if checkpoint is None:
            return
        gl = min([best_ub] + ([heap[0][0]] if heap else []) + [pruned_min])
        np.savez_compressed(checkpoint, heap=np.array([(l, c, a, b) for l, c, a, b in heap], dtype=object),
                            best_ub=best_ub, best_u=best_u, counter=counter, n_eval=n_eval, n_box=n_box,
                            pruned_min=pruned_min, global_lb=gl, seconds=time.time() - t0)

    last_log = time.time()
    while heap and heap[0][0] < best_ub - target and n_box < max_boxes:
        lb, _, a, b = heapq.heappop(heap)
        Wa = prob.min_weights(a).clone()
        Wb = prob.min_weights(b).clone()
        n_eval += 2
        for _ in range(dive):
            k, mid = prob.split(a, b)
            e = np.arange(4) == k
            b1 = np.where(e, mid, b)
            a2 = np.where(e, mid, a)
            Wb1 = prob.min_weights(b1).clone()
            Wa2 = prob.min_weights(a2).clone()
            n_eval += 2
            lb1 = _bound_from(prob, Wa, Wb1, tol)
            lb2 = _bound_from(prob, Wa2, Wb, tol)
            n_box += 2
            if lb1 <= lb2:
                keep = (a, b1, Wa, Wb1)
                other, other_lb = (a2, b), lb2
            else:
                keep = (a2, b, Wa2, Wb)
                other, other_lb = (a, b1), lb1
            if other_lb < best_ub - target:
                counter += 1
                heapq.heappush(heap, (other_lb, counter, other[0], other[1]))
            else:
                pruned_min = min(pruned_min, other_lb)
            a, b, Wa, Wb = keep
            kept_lb = min(lb1, lb2)
            if kept_lb >= best_ub - target:
                pruned_min = min(pruned_min, kept_lb)
                a = None
                break
            if n_box >= max_boxes:
                break
        if a is not None:
            r = prob.risk_upper((a + b) / 2)          # one incumbent test per dive
            n_eval += 1
            if r < best_ub:
                best_ub, best_u = r, (a + b) / 2
            counter += 1
            heapq.heappush(heap, (kept_lb, counter, a, b))
        del Wa, Wb
        if n_box % checkpoint_every < 2 * dive:
            save()
        if time.time() - last_log >= 300:
            last_log = time.time()
            gl = min([best_ub] + ([heap[0][0]] if heap else []) + [pruned_min])
            log(f"    boxes {n_box}, evals {n_eval}, open {len(heap)}, LB {gl:.8e}, UB {best_ub:.8e}, "
                f"ceiling {100 * (1 - gl / prob.r_phys):.4f}%, {time.time() - t0:.0f}s")
    global_lb = min([best_ub] + ([h[0] for h in heap] or [np.inf]) + [pruned_min])
    save()
    return {"lo": lo, "hi": hi, "LB": global_lb, "UB": best_ub, "best_scales": np.exp(best_u).tolist(),
            "boxes": n_box, "evals": n_eval, "open_boxes": len(heap),
            "gap_rel_to_phys": (best_ub - global_lb) / prob.r_phys,
            "converged": (best_ub - global_lb) <= target, "seconds": time.time() - t0}


def main() -> None:
    import argparse
    import pandas as pd
    from ..config import RESULTS_ROOT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--p", type=float, nargs="+", default=[0.0005])
    ap.add_argument("--lo", type=float, default=0.4)
    ap.add_argument("--hi", type=float, default=3.2)
    ap.add_argument("--rel-gap", type=float, default=1e-6)
    ap.add_argument("--max-boxes", type=int, default=100_000)
    ap.add_argument("--dive", type=int, default=8)
    ap.add_argument("--out", default="certified_class_family_T3_gpu.csv")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = RESULTS_ROOT / "theory" / a.out
    rows = []
    for p in a.p:
        prob = GpuProblem(a.rounds, p)
        hi = min(a.hi, 0.49 / float(prob.p0.max()))
        ck = out.with_name(f"{out.stem}_T{a.rounds}_p{p:g}.npz")
        state = None
        if a.resume and ck.exists():
            z = np.load(ck, allow_pickle=True)
            state = {k: z[k] for k in z.files}
        print(f"T={a.rounds} p={p:g}: D={prob.D}, R_phys={prob.r_phys:.10e}, mass deficit "
              f"{prob.mass_deficit:.1e}, box [{a.lo}, {hi:.3g}]^4", flush=True)
        res = certify_gpu(prob, a.lo, hi, a.rel_gap, a.max_boxes, dive=a.dive, checkpoint=ck, state=state)
        res.update({"T": a.rounds, "p": p, "R_phys": prob.r_phys,
                    "gain_UB_certified": 1 - res["UB"] / prob.r_phys,
                    "gain_LB_certified_max": 1 - res["LB"] / prob.r_phys})
        rows.append(res)
        print(f"   certified min risk in [{res['LB']:.10e}, {res['UB']:.10e}]  headroom in "
              f"[{100 * res['gain_UB_certified']:.4f}%, {100 * res['gain_LB_certified_max']:.4f}%]  "
              f"boxes {res['boxes']} evals {res['evals']} converged {res['converged']}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        del prob
    print("wrote", out)


if __name__ == "__main__":
    main()
