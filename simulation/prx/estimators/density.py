"""Scalar estimator: detection density -> physical rate via a simulated calibration curve.

Replaces the naive ``p = rate / 4`` mapping, which is biased by about 2.7x.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from ..dem.edges import EdgeTable, join_probabilities
from ..dem.priors import build_circuit, physical_prior
from .base import Estimator, FitResult


def detection_fraction(dets_packed: np.ndarray, num_detectors: int) -> float:
    counts = np.bitwise_count(dets_packed).sum(dtype=np.int64)
    return float(counts) / (dets_packed.shape[0] * num_detectors)


@dataclass
class DensityCalibrationCurve:
    family: str
    distance: int
    rounds: int
    basis: str = "Z"
    variant: Optional[str] = None
    p_grid: np.ndarray = field(default_factory=lambda: np.geomspace(3e-4, 4e-2, 25))
    shots_per_point: int = 20_000
    seed: int = 1234
    _log_p: np.ndarray = field(default=None, repr=False)
    _f: np.ndarray = field(default=None, repr=False)

    def build(self) -> "DensityCalibrationCurve":
        fs = []
        kw = {"variant": self.variant} if self.variant else {}
        for i, p in enumerate(self.p_grid):
            c = build_circuit(self.family, self.distance, self.rounds, float(p), self.basis, **kw)
            d, _ = c.compile_detector_sampler(seed=self.seed + i).sample(self.shots_per_point, separate_observables=True, bit_packed=True)
            fs.append(detection_fraction(d, c.num_detectors))
        self._log_p = np.log(self.p_grid)
        self._f = np.asarray(fs)
        if not np.all(np.diff(self._f) > 0):
            raise RuntimeError("detection fraction is not monotone on the grid; increase shots_per_point")
        return self

    def invert(self, f_obs: float) -> float:
        if self._f is None:
            self.build()
        f = float(np.clip(f_obs, self._f[0], self._f[-1]))
        return float(np.exp(np.interp(f, self._f, self._log_p)))


@dataclass
class DensityEstimator(Estimator):
    curve: DensityCalibrationCurve
    name: str = "density"
    needs_labels: bool = False

    def fit(self, template: EdgeTable, dets_packed: np.ndarray, obs_packed=None) -> FitResult:
        f = detection_fraction(dets_packed, template.num_detectors)
        p_hat = self.curve.invert(f)
        kw = {"variant": self.curve.variant} if self.curve.variant else {}
        phys = physical_prior(self.curve.family, self.curve.distance, self.curve.rounds, p_hat, self.curve.basis, **kw)
        table = template.with_probabilities(join_probabilities(template, phys))
        return FitResult(table=table, params={"detection_fraction": f, "n_cal_shots": int(dets_packed.shape[0])}, p_hat=p_hat)
