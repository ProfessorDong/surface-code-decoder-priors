"""Common interface for prior estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..dem.edges import EdgeTable


@dataclass
class FitResult:
    table: EdgeTable
    params: Dict[str, Any] = field(default_factory=dict)   # serialisable summary
    p_hat: Optional[float] = None                           # scalar estimate where meaningful


class Estimator:
    """fit(template, dets_packed, obs_packed=None) -> FitResult."""

    name: str = "estimator"
    needs_labels: bool = False

    def fit(self, template: EdgeTable, dets_packed: np.ndarray,
            obs_packed: Optional[np.ndarray] = None) -> FitResult:  # pragma: no cover - interface
        raise NotImplementedError
