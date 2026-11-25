"""Compare branching rules for the per-class certificate at equal box budget.

Rules (branching heuristics from the NN-verification literature, adapted):
  longest : split the longest edge of the log-scale box at its midpoint  [what the paper ran]
  wspan   : split the class whose EDGE WEIGHTS move most over the box, at the point that
            halves that weight range (branching points chosen by the nonlinearity, GenBaB)
  fsb     : evaluate all four candidate splits and keep the one whose weaker child has the
            largest lower bound (filtered smart branching)
"""
import ast
import heapq
import sys
import time

import numpy as np
import pandas as pd

from ..config import RESULTS_ROOT
from .certify_class_family import Problem

TOL = 1e-9


def rep_p(prob):
    return np.array([np.exp(np.log(prob.p0[prob.cidx == c]).mean()) for c in range(4)])


def w_of(p0c, u):
    p = p0c * np.exp(u)
    return np.log1p(-p) - np.log(p)


def split_wspan(prob, pr, a, b):
    span = w_of(pr, a) - w_of(pr, b)              # weights decrease with u
    k = int(np.argmax(span))
    wm = 0.5 * (w_of(pr[k], a[k]) + w_of(pr[k], b[k]))
    u = np.log(1.0 / (1.0 + np.exp(wm)) / pr[k])
    return k, float(np.clip(u, a[k] + 1e-9, b[k] - 1e-9))


def children(a, b, k, mid):
    e = np.arange(4) == k
    return (a.copy(), np.where(e, mid, b)), (np.where(e, mid, a), b.copy())


def run(prob, rule, lo, hi, max_boxes, log_every=2000):
    pr = rep_p(prob)
    a0, b0 = np.full(4, np.log(lo)), np.full(4, np.log(hi))
    t0 = time.time()
    lb, _ = prob.lower_bound(a0, b0, TOL)
    best_ub, best_u = prob.risk_upper(np.zeros(4)), np.zeros(4)
    c = (a0 + b0) / 2
    if prob.risk_upper(c) < best_ub:
        best_ub, best_u = prob.risk_upper(c), c
    heap, counter, n_eval, pruned_min = [(lb, 0, a0, b0)], 1, 1, np.inf
    while heap and n_eval < max_boxes:
        lb, _, a, b = heapq.heappop(heap)
        if lb >= best_ub:
            heapq.heappush(heap, (lb, 0, a, b)); break
        if rule == "longest":
            k = int(np.argmax(b - a)); mid = float((a[k] + b[k]) / 2)
            kids = children(a, b, k, mid)
        elif rule == "wspan":
            k, mid = split_wspan(prob, pr, a, b)
            kids = children(a, b, k, mid)
        else:  # fsb
            best_score, kids = -np.inf, None
            for k in range(4):
                mid = float((a[k] + b[k]) / 2)
                cand = children(a, b, k, mid)
                lbs = [prob.lower_bound(ca, cb, TOL)[0] for ca, cb in cand]
                n_eval += 2
                if min(lbs) > best_score:
                    best_score, kids = min(lbs), cand
        for ca, cb in kids:
            clb, _ = prob.lower_bound(ca, cb, TOL)
            n_eval += 1
            ub = prob.risk_upper((ca + cb) / 2)
            if ub < best_ub:
                best_ub, best_u = ub, (ca + cb) / 2
            if clb < best_ub:
                counter += 1
                heapq.heappush(heap, (clb, counter, ca, cb))
            else:
                pruned_min = min(pruned_min, clb)
        if n_eval % log_every < 3:
            gl = min(best_ub, heap[0][0] if heap else np.inf, pruned_min)
            print(f"  [{rule}] boxes {n_eval} LB {gl:.8e} UB {best_ub:.8e} "
                  f"headroom<= {100*(1-gl/prob.r_phys):.4f}% {time.time()-t0:.0f}s", flush=True)
    gl = min(best_ub, min((h[0] for h in heap), default=np.inf), pruned_min)
    return {"rule": rule, "boxes": n_eval, "LB": gl, "UB": best_ub,
            "ceiling_%": 100 * (1 - gl / prob.r_phys), "achieved_%": 100 * (1 - best_ub / prob.r_phys),
            "seconds": time.time() - t0}


def main() -> None:
    p = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    prob = Problem(2, p)
    hi = min(5.0, 0.49 / float(prob.p0.max()))
    print(f"T=2 p={p:g} box [0.15, {hi:.3g}]^4  R_phys={prob.r_phys:.8e}  budget {budget} boxes", flush=True)
    rows = []
    for rule in ("longest", "wspan", "fsb"):
        r = run(prob, rule, 0.15, hi, budget)
        print(f"RESULT {r}", flush=True)
        rows.append(dict(r, T=2, p=p, lo=0.15, hi=hi, R_phys=prob.r_phys))
    out = RESULTS_ROOT / "theory" / "branch_rule_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
