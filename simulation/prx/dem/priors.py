"""Prior parametrizations for matching decoders and the circuit-family registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping

import stim

from .edges import EdgeClass, EdgeTable

# ----------------------------------------------------------------------------
# Circuit families (how a physical rate p maps to a Stim circuit)
# ----------------------------------------------------------------------------

_FAMILIES: Dict[str, Callable[..., stim.Circuit]] = {}


def register_family(name: str):
    def deco(fn):
        _FAMILIES[name] = fn
        return fn
    return deco


def families():
    return sorted(_FAMILIES)


def build_circuit(family: str, distance: int, rounds: int, p: float, basis: str = "Z", **kwargs) -> stim.Circuit:
    if family not in _FAMILIES:
        raise KeyError(f"unknown circuit family {family!r}; known: {families()}")
    return _FAMILIES[family](distance=distance, rounds=rounds, p=p, basis=basis, **kwargs)


@register_family("legacy_from_physical_rate")
def _legacy(distance: int, rounds: int, p: float, basis: str = "Z", **_: Any) -> stim.Circuit:
    """Legacy family: Stim generator with 2q = meas = reset = p, data idle = p/3.

    Kept only to reproduce the submitted Table V; not a physically motivated model.
    """
    task = f"surface_code:rotated_memory_{basis.lower()}"
    return stim.Circuit.generated(
        task, distance=distance, rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p / 3.0,
    )


@register_family("si1000")
def _si1000(distance: int, rounds: int, p: float, basis: str = "Z", variant: str = "gidney2021", **kw: Any) -> stim.Circuit:
    from ..noise.si1000 import SI1000Rates, noiseless_memory_circuit, inject_si1000

    rates = SI1000Rates.from_variant(variant, p)
    return inject_si1000(noiseless_memory_circuit(distance, rounds, basis), rates, **kw)


# ----------------------------------------------------------------------------
# Priors
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PriorSpec:
    """Serialisable description of a prior for result rows."""

    prior_id: str
    params: Dict[str, Any] = field(default_factory=dict)

    def scales(self) -> Dict[str, float]:
        return {k: float(self.params.get(f"scale_{k}", 1.0)) for k in ("space", "time", "diag", "boundary")}


def physical_prior(family: str, distance: int, rounds: int, p: float, basis: str = "Z", **kw) -> EdgeTable:
    """Matching graph of the circuit itself (the 'physical' or oracle prior)."""
    return EdgeTable.from_circuit(build_circuit(family, distance, rounds, p, basis, **kw))


def stale_prior(family: str, distance: int, rounds: int, p_cal: float, basis: str = "Z", **kw) -> EdgeTable:
    """Matching graph of the same family calibrated at ``p_cal`` (the Table V construction)."""
    return physical_prior(family, distance, rounds, p_cal, basis, **kw)


def uniform_scale_prior(table: EdgeTable, s: float) -> EdgeTable:
    return table.scaled(float(s))


def class_scale_prior(table: EdgeTable, scales: Mapping[str, float]) -> EdgeTable:
    return table.scaled_by_class({EdgeClass[k.upper()]: float(v) for k, v in scales.items()})
