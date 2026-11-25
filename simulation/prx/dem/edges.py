"""Edge tables: the matching graph as data, with edge classes and rebuild.

An ``EdgeTable`` is the 2-detector matching graph PyMatching derives from a
detector error model (parallel edges already merged by PyMatching), stored as
arrays so that probabilities can be edited per edge or per class and the
``pymatching.Matching`` object rebuilt. Rebuilding an unmodified table gives
bit-identical predictions to ``Matching.from_detector_error_model``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pymatching
import stim

from ..config import P_MAX, P_MIN


class EdgeClass(IntEnum):
    SPACE = 0      # same time layer, different location
    TIME = 1       # same location, different time layer
    DIAG = 2       # different location and different time layer (hook / space-time)
    BOUNDARY = 3   # single-detector mechanism (edge to the virtual boundary)


CLASS_NAMES = {c: c.name.lower() for c in EdgeClass}


def weight_from_probability(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), P_MIN, P_MAX)
    return np.log((1.0 - p) / p)


def space_time(coord: Sequence[float]) -> Tuple[np.ndarray, float]:
    """Split a detector coordinate into (spatial vector, time).

    Stim writes (x, y, t) for 2D codes and (x, t) for the repetition code;
    Google's Willow DEMs carry six components (x, y, t, x, y, t-1) of which only
    the first three are the detector's own position.
    """
    c = np.asarray(coord, dtype=float)
    if c.size >= 3:
        return c[:2], float(c[2])
    if c.size == 2:
        return c[:1], float(c[1])
    raise ValueError("detector coordinates need at least two components")


def classify_edges(u: np.ndarray, v: np.ndarray, coords: Mapping[int, Sequence[float]]) -> np.ndarray:
    """Assign an EdgeClass to every edge from detector coordinates."""
    out = np.empty(len(u), dtype=np.int8)
    for k, (a, b) in enumerate(zip(u, v)):
        if b < 0:
            out[k] = EdgeClass.BOUNDARY
            continue
        sa, ta = space_time(coords[int(a)])
        sb, tb = space_time(coords[int(b)])
        dt = abs(ta - tb)
        dxy = float(np.linalg.norm(sa - sb))
        if dt < 0.5:
            out[k] = EdgeClass.SPACE
        elif dxy < 1e-9:
            out[k] = EdgeClass.TIME
        else:
            out[k] = EdgeClass.DIAG
    return out


@dataclass
class EdgeTable:
    u: np.ndarray                 # int64, detector index
    v: np.ndarray                 # int64, detector index or -1 for the boundary
    p: np.ndarray                 # float64, edge error probability
    fault_mask: np.ndarray        # uint64 bitmask over logical observables
    cls: np.ndarray               # int8 EdgeClass
    num_detectors: int
    coords: Dict[int, Tuple[float, ...]]

    # ------------------------------------------------------------------ build
    @classmethod
    def from_matching(cls, matching: pymatching.Matching,
                      coords: Mapping[int, Sequence[float]], num_detectors: Optional[int] = None) -> "EdgeTable":
        us, vs, ps, masks = [], [], [], []
        for a, b, attrs in matching.edges():
            us.append(int(a))
            vs.append(-1 if b is None else int(b))
            prob = attrs.get("error_probability")
            if prob is None or not np.isfinite(prob):
                # PyMatching stores -1 when no probability was supplied; derive from the weight.
                prob = 1.0 / (1.0 + np.exp(float(attrs["weight"])))
            ps.append(float(prob))
            mask = 0
            for f in attrs.get("fault_ids", set()) or set():
                mask |= 1 << int(f)
            masks.append(mask)
        u = np.asarray(us, dtype=np.int64)
        v = np.asarray(vs, dtype=np.int64)
        n_det = int(num_detectors if num_detectors is not None else matching.num_detectors)
        coords_d = {int(k): tuple(float(x) for x in val) for k, val in coords.items()}
        table = cls(u=u, v=v, p=np.asarray(ps, dtype=float),
                    fault_mask=np.asarray(masks, dtype=np.uint64),
                    cls=np.zeros(len(u), dtype=np.int8), num_detectors=n_det, coords=coords_d)
        table.cls = classify_edges(u, v, coords_d)
        return table

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "EdgeTable":
        matching = pymatching.Matching.from_detector_error_model(dem)
        return cls.from_matching(matching, dem.get_detector_coordinates(), num_detectors=dem.num_detectors)

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit, decompose_errors: bool = True) -> "EdgeTable":
        return cls.from_dem(circuit.detector_error_model(decompose_errors=decompose_errors))

    # --------------------------------------------------------------- derived
    @property
    def weight(self) -> np.ndarray:
        return weight_from_probability(self.p)

    @property
    def num_edges(self) -> int:
        return int(self.u.size)

    def fault_ids(self, k: int) -> set:
        mask = int(self.fault_mask[k])
        return {i for i in range(64) if (mask >> i) & 1}

    def class_counts(self) -> Dict[str, int]:
        return {CLASS_NAMES[c]: int(np.count_nonzero(self.cls == c)) for c in EdgeClass}

    def edge_key(self) -> np.ndarray:
        """Canonical (min, max) detector pair per edge, -1 for the boundary, for joins."""
        lo = np.minimum(self.u, np.where(self.v < 0, self.u, self.v))
        hi = np.maximum(self.u, np.where(self.v < 0, -1, self.v))
        return np.stack([lo, hi], axis=1)

    # --------------------------------------------------------------- editing
    def with_probabilities(self, p_new: np.ndarray) -> "EdgeTable":
        p_arr = np.clip(np.asarray(p_new, dtype=float), P_MIN, P_MAX)
        if p_arr.shape != self.p.shape:
            raise ValueError("probability array must match the number of edges")
        return replace(self, p=p_arr)

    def scaled(self, factor: np.ndarray | float) -> "EdgeTable":
        return self.with_probabilities(self.p * np.asarray(factor, dtype=float))

    def scaled_by_class(self, scales: Mapping[EdgeClass | int | str, float]) -> "EdgeTable":
        factor = np.ones(self.num_edges)
        for key, s in scales.items():
            c = _class_from_key(key)
            factor[self.cls == c] = float(s)
        return self.scaled(factor)

    # ----------------------------------------------------------------- build
    def to_matching(self) -> pymatching.Matching:
        m = pymatching.Matching()
        w = self.weight
        for k in range(self.num_edges):
            fids = self.fault_ids(k)
            if self.v[k] < 0:
                m.add_boundary_edge(int(self.u[k]), fault_ids=fids, weight=float(w[k]),
                                    error_probability=float(self.p[k]), merge_strategy="disallow")
            else:
                m.add_edge(int(self.u[k]), int(self.v[k]), fault_ids=fids, weight=float(w[k]),
                           error_probability=float(self.p[k]), merge_strategy="disallow")
        if m.num_nodes < self.num_detectors:
            m.ensure_num_nodes(self.num_detectors)
        return m


def _class_from_key(key: EdgeClass | int | str) -> int:
    if isinstance(key, str):
        return int(EdgeClass[key.upper()])
    return int(key)


def join_probabilities(target: EdgeTable, source: EdgeTable, fill: Optional[np.ndarray] = None) -> np.ndarray:
    """Return probabilities for ``target`` edges taken from ``source`` by (u, v) key.

    Edges missing in ``source`` take ``fill`` (default: target's own probabilities).
    """
    src = {tuple(k): p for k, p in zip(source.edge_key().tolist(), source.p.tolist())}
    base = target.p if fill is None else np.asarray(fill, dtype=float)
    out = np.array([src.get(tuple(k), b) for k, b in zip(target.edge_key().tolist(), base.tolist())])
    return out
