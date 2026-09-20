"""Oracles: the physical prior (needs the true circuit) and the identity (static reference)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dem.edges import EdgeTable
from .base import Estimator, FitResult


@dataclass
class StaticPrior(Estimator):
    """Returns the template unchanged (the static, uncalibrated reference)."""

    name: str = "static"
    needs_labels: bool = False

    def fit(self, template: EdgeTable, dets_packed: np.ndarray, obs_packed=None) -> FitResult:
        return FitResult(table=template, params={"n_cal_shots": 0})


@dataclass
class PhysicalOracle(Estimator):
    """Returns the true circuit's matching graph mapped onto the template edges."""

    physical: EdgeTable
    name: str = "oracle_physical"
    needs_labels: bool = False

    def fit(self, template: EdgeTable, dets_packed: np.ndarray, obs_packed=None) -> FitResult:
        from ..dem.edges import join_probabilities
        return FitResult(table=template.with_probabilities(join_probabilities(template, self.physical)), params={"n_cal_shots": 0})
