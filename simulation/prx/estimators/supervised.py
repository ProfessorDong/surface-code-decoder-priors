"""Label-requiring estimators: per-class scale search minimising failures on calibration shots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..dem.edges import EdgeTable
from ..experiments.common import decode_errors
from .base import Estimator, FitResult

CLASSES = ("space", "time", "diag", "boundary")


def coordinate_search(base: EdgeTable, dets: np.ndarray, obs: np.ndarray, grid: Sequence[float], sweeps: int,
                      classes: Sequence[str] = CLASSES, log: Optional[List[Dict[str, Any]]] = None) -> Tuple[Dict[str, float], int]:
    scales = {c: 1.0 for c in classes}
    best = int(decode_errors(base.scaled_by_class(scales).to_matching(), dets, obs).sum())
    if log is not None:
        log.append({"sweep": 0, "class": "init", "scale": 1.0, "n_errors": best, "scales": dict(scales)})
    for sweep in range(1, sweeps + 1):
        improved = False
        for c in classes:
            cand = scales[c]
            for s in grid:
                if abs(s - scales[c]) < 1e-12:
                    continue
                trial = dict(scales); trial[c] = float(s)
                fail = int(decode_errors(base.scaled_by_class(trial).to_matching(), dets, obs).sum())
                if log is not None:
                    log.append({"sweep": sweep, "class": c, "scale": float(s), "n_errors": fail, "scales": trial})
                if fail < best:
                    best, cand, improved = fail, float(s), True
            scales[c] = cand
        if not improved:
            break
    return scales, best


@dataclass
class ClassScaleSearch(Estimator):
    grid: Sequence[float] = field(default_factory=lambda: [float(x) for x in np.geomspace(0.35, 2.85, 9)])
    sweeps: int = 2
    classes: Sequence[str] = CLASSES
    name: str = "class_scale_search"
    needs_labels: bool = True

    def fit(self, template: EdgeTable, dets_packed: np.ndarray, obs_packed=None) -> FitResult:
        if obs_packed is None:
            raise ValueError("ClassScaleSearch needs logical labels")
        log: List[Dict[str, Any]] = []
        scales, best = coordinate_search(template, dets_packed, obs_packed, self.grid, self.sweeps, self.classes, log)
        table = template.scaled_by_class(scales)
        params = {"scales": scales, "best_cal_errors": best, "n_cal_shots": int(dets_packed.shape[0]),
                  "n_evaluations": len(log)}
        return FitResult(table=table, params=params)


@dataclass
class UniformScaleSearch(Estimator):
    grid: Sequence[float] = field(default_factory=lambda: [float(x) for x in np.geomspace(0.25, 4.0, 13)])
    name: str = "uniform_scale_search"
    needs_labels: bool = True

    def fit(self, template: EdgeTable, dets_packed: np.ndarray, obs_packed=None) -> FitResult:
        if obs_packed is None:
            raise ValueError("UniformScaleSearch needs logical labels")
        best_s, best = 1.0, int(decode_errors(template.to_matching(), dets_packed, obs_packed).sum())
        for s in self.grid:
            fail = int(decode_errors(template.scaled(s).to_matching(), dets_packed, obs_packed).sum())
            if fail < best:
                best, best_s = fail, float(s)
        return FitResult(table=template.scaled(best_s), params={"s": best_s, "best_cal_errors": best, "n_cal_shots": int(dets_packed.shape[0])})
