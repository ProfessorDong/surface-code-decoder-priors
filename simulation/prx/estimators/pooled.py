"""p_ij with pooling over time-translated edges and shrinkage toward a reference.

Edge signature = (spatial coordinates of both endpoints, time offset, layer
class) where the layer class distinguishes the first, second, bulk,
penultimate and last time layers. Moments are pooled over all edges sharing a
signature (equivalent to more shots), then the p_ij formula is applied.
Shrinkage: logit p = w logit p_pooled + (1-w) logit p_ref, w = n/(n+kappa) with
n the pooled coincidence count (bulk) or event count (boundary).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import P_MAX, P_MIN
from ..dem.edges import EdgeTable, space_time
from .base import Estimator, FitResult
from .pij import combine_odd, pairwise_moments, pij_from_moments


def _layer_class(t: float, t_max: float) -> str:
    if t <= 0:
        return "first"
    if t == 1:
        return "second"
    if t >= t_max:
        return "last"
    if t == t_max - 1:
        return "penult"
    return "bulk"


def edge_signatures(table: EdgeTable, pool_time: bool = True) -> List[tuple]:
    ts = [space_time(c)[1] for c in table.coords.values()]
    t_max = max(ts) if ts else 0.0
    sigs = []
    for k in range(table.num_edges):
        su, tu = space_time(table.coords[int(table.u[k])])
        if table.v[k] < 0:
            key = ("B", tuple(su), _layer_class(tu, t_max) if pool_time else tu, int(table.fault_mask[k]))
        else:
            sv, tv = space_time(table.coords[int(table.v[k])])
            (sa, ta), (sb, tb) = ((su, tu), (sv, tv)) if tu <= tv else ((sv, tv), (su, tu))
            key = ("E", tuple(sa), tuple(sb), tb - ta,
                   _layer_class(ta, t_max) if pool_time else ta, int(table.fault_mask[k]))
        sigs.append(key)
    return sigs


def _logit(p):
    p = np.clip(p, P_MIN, P_MAX)
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def eb_kappa_mle(delta: np.ndarray, n_eff: np.ndarray, iters: int = 200) -> float:
    """Efron-Morris maximum-likelihood shrinkage strength for unequal sampling variances.

    Their toxoplasmosis model is ours: X_g ~ N(theta_g, D_g) with D_g known and unequal
    (here D_g = 1/n_g, an estimate from n_g events), theta_g ~ N(0, A) about the reference.
    A is the root of A = sum_g (X_g^2 - D_g) I_g(A) / sum_g I_g(A) with Fisher information
    I_g(A) = 1/[2(A + D_g)^2], reached here by fixed-point iteration from the moment
    estimate and truncated at zero. Returns kappa = 1/A, the weight in n/(n + kappa).
    """
    ok = np.isfinite(delta) & np.isfinite(n_eff) & (n_eff > 0)
    if ok.sum() < 3:                      # A is not identified from one or two signatures
        return np.inf
    d2, D = delta[ok] ** 2, 1.0 / n_eff[ok]
    A = max(float(np.mean(d2 - D)), 1e-12)
    for _ in range(iters):
        w = 1.0 / (A + D) ** 2
        A_new = float(np.sum((d2 - D) * w) / np.sum(w))
        A_new = max(A_new, 0.0)
        if abs(A_new - A) <= 1e-12 * max(A, 1e-12):
            A = A_new
            break
        A = A_new
    return 1.0 / A if A > 0 else np.inf


def eb_kappa(delta: np.ndarray, n_eff: np.ndarray) -> float:
    """Empirical-Bayes shrinkage strength from the signature-level deviations.

    With logit p_hat_g ~ N(logit p_g, 1/n_g) (a count of n_g events) and the true
    logits dispersed around the reference with variance tau^2, the risk-optimal
    weight on the data is tau^2 / (tau^2 + 1/n_g) = n_g / (n_g + 1/tau^2), i.e. the
    kappa of this class with kappa = 1/tau^2. The moment estimate of tau^2 is the
    mean of (deviation^2 - sampling variance), truncated at zero (James-Stein);
    tau^2 = 0 means shrink all the way to the reference.
    """
    ok = np.isfinite(delta) & np.isfinite(n_eff) & (n_eff > 0)
    if not ok.any():
        return np.inf
    tau2 = float(np.mean(delta[ok] ** 2 - 1.0 / n_eff[ok]))
    return 1.0 / tau2 if tau2 > 0 else np.inf


@dataclass
class PooledPijEstimator(Estimator):
    """Time-pooled p_ij with optional shrinkage toward the template (kappa events).

    ``eb`` replaces the fixed kappa by a moment estimate from the same syndromes:
    "global" fits one kappa, "per_class" one per edge class.
    """

    kappa: float = 0.0
    pool_time: bool = True
    eb: str = ""
    name: str = "pooled_pij"
    needs_labels: bool = False

    def fit(self, template: EdgeTable, dets_packed: np.ndarray, obs_packed=None) -> FitResult:
        D = template.num_detectors
        bulk = np.flatnonzero(template.v >= 0)
        bnd = np.flatnonzero(template.v < 0)
        pairs = np.stack([template.u[bulk], template.v[bulk]], axis=1)
        xi, xij, n = pairwise_moments(dets_packed, D, pairs)
        return self.fit_moments(template, xi, xij, n)

    def fit_moments(self, template: EdgeTable, xi: np.ndarray, xij: np.ndarray, n: float) -> FitResult:
        """The estimator applied to moments <x_i> and <x_u x_v> (bulk edges in template order).

        ``n`` is the number of shots behind the moments; with ``np.inf`` every
        shrinkage weight is one, which gives the estimator's population limit."""
        bulk = np.flatnonzero(template.v >= 0)
        bnd = np.flatnonzero(template.v < 0)
        pairs = np.stack([template.u[bulk], template.v[bulk]], axis=1)
        sigs = edge_signatures(template, self.pool_time)
        p_new = template.p.copy()
        n_eff = np.zeros(template.num_edges)
        # ---- bulk: pool moments per signature
        groups: Dict[tuple, List[int]] = {}
        for idx, k in enumerate(bulk):
            groups.setdefault(sigs[k], []).append(idx)
        n_fallback = 0
        for sig, idxs in groups.items():
            idxs = np.asarray(idxs)
            xu = xi[pairs[idxs, 0]].mean()
            xv = xi[pairs[idxs, 1]].mean()
            xuv = xij[idxs].mean()
            est = float(pij_from_moments(xu, xv, xuv))
            cnt = float(xij[idxs].sum() * n)
            ks = bulk[idxs]
            if not np.isfinite(est) or est <= 0:
                n_fallback += len(ks)
                continue
            p_new[ks] = est
            n_eff[ks] = cnt
        # ---- boundary: residual per detector, then pool by signature
        incident: Dict[int, list] = {}
        for k in bulk:
            incident.setdefault(int(template.u[k]), []).append(p_new[k])
            incident.setdefault(int(template.v[k]), []).append(p_new[k])
        raw_b = np.full(template.num_edges, np.nan)
        per_det: Dict[int, List[int]] = {}
        for k in bnd:
            per_det.setdefault(int(template.u[k]), []).append(k)
        for det, ks in per_det.items():
            p_sum = combine_odd(incident.get(det, []))
            den = 1 - 2 * p_sum
            est = (xi[det] - p_sum) / den if den > 0 else np.nan
            tot = float(template.p[ks].sum())
            for k in ks:
                share = template.p[k] / tot if tot > 0 else 1.0 / len(ks)
                raw_b[k] = est * share if np.isfinite(est) and est > 0 else np.nan
        bgroups: Dict[tuple, List[int]] = {}
        for k in bnd:
            bgroups.setdefault(sigs[k], []).append(k)
        for sig, ks in bgroups.items():
            ks = np.asarray(ks)
            vals = raw_b[ks]
            ok = np.isfinite(vals)
            if not ok.any():
                n_fallback += len(ks)
                continue
            est = float(vals[ok].mean())
            p_new[ks] = max(est, P_MIN)
            n_eff[ks] = float(xi[template.u[ks]].sum() * n)
        # ---- shrinkage toward the template
        kappa_used = {}
        if self.eb and np.isfinite(n):
            # one deviation and one count per signature (the unit the estimate is pooled over)
            reps = [np.asarray(bulk[np.asarray(i)]) for i in groups.values()] + \
                   [np.asarray(k) for k in bgroups.values()]
            dz = np.array([float(np.mean(_logit(p_new[ks]) - _logit(template.p[ks]))) for ks in reps])
            ne = np.array([float(np.mean(n_eff[ks])) for ks in reps])
            cl = np.array([int(template.cls[ks[0]]) for ks in reps])
            est = eb_kappa_mle if self.eb.endswith("mle") else eb_kappa
            if self.eb == "per_class":
                kap = {int(c): est(dz[cl == c], ne[cl == c]) for c in np.unique(cl)}
                k_edge = np.array([kap.get(int(c), np.inf) for c in template.cls])
                kappa_used = {int(c): float(v) for c, v in kap.items()}
            else:
                k = est(dz, ne)
                k_edge = np.full(template.num_edges, k)
                kappa_used = {"global": float(k)}
            w = np.where(np.isfinite(k_edge), n_eff / (n_eff + np.where(np.isfinite(k_edge), k_edge, 1.0)), 0.0)
            p_new = _expit(w * _logit(p_new) + (1 - w) * _logit(template.p))
        elif self.kappa > 0 and np.isfinite(n):
            w = n_eff / (n_eff + self.kappa)
            p_new = _expit(w * _logit(p_new) + (1 - w) * _logit(template.p))
        table = template.with_probabilities(np.clip(p_new, P_MIN, P_MAX))
        params = {"n_cal_shots": int(n) if np.isfinite(n) else -1, "kappa": self.kappa, "pool_time": self.pool_time,
                  "eb": self.eb, "kappa_eb_json": json.dumps(kappa_used),
                  "n_signatures": len(groups) + len(bgroups), "n_fallback": n_fallback}
        return FitResult(table=table, params=params, p_hat=float(xi.mean()))


@dataclass
class TimeWindowPijEstimator(Estimator):
    """Sliding-window pooling over rounds: each edge pools with same-geometry edges within +-half_width layers.

    This is the within-circuit analogue of a windowed drift estimator; with
    half_width -> infinity it becomes plain time pooling, with 0 the raw p_ij.
    """

    half_width: int = 2
    kappa: float = 0.0
    name: str = "window_pij"
    needs_labels: bool = False

    def fit(self, template: EdgeTable, dets_packed: np.ndarray, obs_packed=None) -> FitResult:
        D = template.num_detectors
        bulk = np.flatnonzero(template.v >= 0)
        pairs = np.stack([template.u[bulk], template.v[bulk]], axis=1)
        xi, xij, n = pairwise_moments(dets_packed, D, pairs)
        sigs = edge_signatures(template, pool_time=False)   # exact time in the key
        # geometry key without time, plus the time of the earlier endpoint
        geo: Dict[tuple, List[Tuple[float, int]]] = {}
        t_of = {}
        for idx, k in enumerate(bulk):
            s = sigs[k]
            gkey = (s[0], s[1], s[2], s[3], s[5])   # geometry + fault mask, without the time entry
            geo.setdefault(gkey, []).append((s[4], idx))
            t_of[idx] = s[4]
        p_new = template.p.copy()
        n_fallback = 0
        for gkey, members in geo.items():
            times = np.array([m[0] for m in members])
            idxs = np.array([m[1] for m in members])
            for t, idx in members:
                sel = idxs[np.abs(times - t) <= self.half_width]
                xu = xi[pairs[sel, 0]].mean(); xv = xi[pairs[sel, 1]].mean(); xuv = xij[sel].mean()
                est = float(pij_from_moments(xu, xv, xuv))
                if np.isfinite(est) and est > 0:
                    p_new[bulk[idx]] = est
                else:
                    n_fallback += 1
        # boundary edges: residual per detector using the windowed bulk estimates
        incident: Dict[int, list] = {}
        for k in bulk:
            incident.setdefault(int(template.u[k]), []).append(p_new[k])
            incident.setdefault(int(template.v[k]), []).append(p_new[k])
        for k in np.flatnonzero(template.v < 0):
            det = int(template.u[k])
            p_sum = combine_odd(incident.get(det, []))
            den = 1 - 2 * p_sum
            est = (xi[det] - p_sum) / den if den > 0 else np.nan
            if np.isfinite(est) and est > 0:
                p_new[k] = est * (template.p[k] / max(template.p[template.u == det][template.v[template.u == det] < 0].sum(), P_MIN))
            else:
                n_fallback += 1
        table = template.with_probabilities(np.clip(p_new, P_MIN, P_MAX))
        return FitResult(table=table, params={"n_cal_shots": int(n), "half_width": self.half_width, "n_fallback": n_fallback},
                         p_hat=float(xi.mean()))


# ----------------------------------------------------------------------------
# Transfer of a fitted prior to another matching graph (e.g. a different round count)
# ----------------------------------------------------------------------------

def signature_map(table: EdgeTable, pool_time: bool = True) -> Dict[tuple, float]:
    """signature -> mean probability over the table's edges with that signature."""
    sigs = edge_signatures(table, pool_time)
    acc: Dict[tuple, List[float]] = {}
    for s, p in zip(sigs, table.p):
        acc.setdefault(s, []).append(float(p))
    return {s: float(np.exp(np.mean(np.log(v)))) for s, v in acc.items()}


def apply_signature_map(target: EdgeTable, smap: Dict[tuple, float], pool_time: bool = True) -> Tuple[EdgeTable, int]:
    """Assign probabilities to ``target`` edges by signature; unmatched edges keep their own. Returns (table, n_unmatched)."""
    sigs = edge_signatures(target, pool_time)
    p_new = target.p.copy()
    missing = 0
    for k, s in enumerate(sigs):
        if s in smap:
            p_new[k] = smap[s]
        else:
            missing += 1
    return target.with_probabilities(np.clip(p_new, P_MIN, P_MAX)), missing
