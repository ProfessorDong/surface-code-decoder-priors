"""Write a fitted graphlike prior back into a detector error model.

PyMatching's correlated matching (``enable_correlations=True``) needs a DEM,
not an edge list, because it uses the hyperedge structure. To evaluate a prior
that was fitted on the graphlike matching graph under a hyperedge-aware
decoder, we transfer the fit multiplicatively:

    for every DEM error instruction, split it at the separators into components;
    every component with one or two detectors identifies a matching edge;
    the instruction's probability is multiplied by the geometric mean of the
    fitted/template probability ratios of those edges.

With ``fitted == template`` every ratio is one and the DEM is returned
unchanged, so the construction is a strict generalisation of the identity.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import stim

from ..config import P_MAX, P_MIN
from .edges import EdgeTable


def _components(instruction: stim.DemInstruction) -> List[Tuple[int, ...]]:
    """Detector-index tuples of each separator-delimited component."""
    comps: List[Tuple[int, ...]] = []
    cur: List[int] = []
    for t in instruction.targets_copy():
        if t.is_separator():
            comps.append(tuple(sorted(cur)))
            cur = []
        elif t.is_relative_detector_id():
            cur.append(t.val)
    comps.append(tuple(sorted(cur)))
    return comps


def edge_ratio_map(template: EdgeTable, fitted: EdgeTable) -> Dict[Tuple[int, ...], float]:
    """(u,) or (u, v) key -> fitted/template probability ratio."""
    out: Dict[Tuple[int, ...], float] = {}
    for k in range(template.num_edges):
        u, v = int(template.u[k]), int(template.v[k])
        key = (u,) if v < 0 else tuple(sorted((u, v)))
        r = float(fitted.p[k] / template.p[k]) if template.p[k] > 0 else 1.0
        out[key] = out.get(key, 1.0) * r if key in out else r
    return out


def reweight_dem(dem: stim.DetectorErrorModel, template: EdgeTable, fitted: EdgeTable) -> stim.DetectorErrorModel:
    ratios = edge_ratio_map(template, fitted)
    out = stim.DetectorErrorModel()
    for ins in dem.flattened():
        if ins.type != "error":
            out.append(ins)
            continue
        p = ins.args_copy()[0]
        logs = []
        for comp in _components(ins):
            if 1 <= len(comp) <= 2 and comp in ratios:
                logs.append(np.log(ratios[comp]))
        if logs:
            p = float(np.clip(p * np.exp(float(np.mean(logs))), P_MIN, P_MAX))
        out.append(stim.DemInstruction("error", [p], ins.targets_copy()))
    return out
