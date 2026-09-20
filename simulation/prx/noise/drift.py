"""Within-circuit drift injected per syndrome round (correctly indexed by round).

Replaces ``simulation.data.noise_models.apply_temporal_drift``, which indexed
its schedule by TICK count (about seven TICKs per round) and therefore drifted
only the first few rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import stim

from .si1000 import (MEASURE_GATES, NOISE_OPS, SI1000Rates, _qubit_targets, _split_layers,
                     inject_si1000, noiseless_memory_circuit)


@dataclass(frozen=True)
class RoundSchedule:
    """Physical rate p(t) for t = 0 .. rounds-1."""

    values: np.ndarray

    @property
    def rounds(self) -> int:
        return int(self.values.size)

    @classmethod
    def constant(cls, rounds: int, p0: float) -> "RoundSchedule":
        return cls(np.full(rounds, float(p0)))

    @classmethod
    def sinusoid(cls, rounds: int, p0: float, dp: float, period: float, onset: int = 0) -> "RoundSchedule":
        t = np.arange(rounds)
        v = np.full(rounds, float(p0))
        m = t >= onset
        v[m] = p0 + dp * np.sin(2 * np.pi * (t[m] - onset) / period)
        return cls(np.clip(v, 1e-6, 0.5))

    @classmethod
    def step(cls, rounds: int, p0: float, p1: float, t_step: int) -> "RoundSchedule":
        v = np.full(rounds, float(p0))
        v[t_step:] = p1
        return cls(v)

    @classmethod
    def burst(cls, rounds: int, p0: float, p_burst: float, t0: int, width: int) -> "RoundSchedule":
        v = np.full(rounds, float(p0))
        v[t0:t0 + width] = p_burst
        return cls(v)

    @classmethod
    def ornstein_uhlenbeck(cls, rounds: int, p0: float, sigma_log: float, tau: float, seed: int) -> "RoundSchedule":
        """log p follows an OU process around log p0 with relaxation time tau (rounds)."""
        rng = np.random.default_rng(seed)
        x = np.zeros(rounds)
        a = np.exp(-1.0 / tau)
        noise_sd = sigma_log * np.sqrt(1 - a * a)
        x[0] = rng.normal(0, sigma_log)
        for t in range(1, rounds):
            x[t] = a * x[t - 1] + rng.normal(0, noise_sd)
        return cls(np.clip(p0 * np.exp(x), 1e-6, 0.5))


def drifting_circuit(distance: int, rounds: int, schedule: RoundSchedule, variant: str = "google2025",
                     basis: str = "Z", qubit_scale=None) -> stim.Circuit:
    if schedule.rounds != rounds:
        raise ValueError("schedule length must equal rounds")
    rates = [SI1000Rates.from_variant(variant, float(p)) for p in schedule.values]
    return inject_si1000(noiseless_memory_circuit(distance, rounds, basis), rates, qubit_scale=qubit_scale)


def scale_noise_by_round(noisy: stim.Circuit, factors: Sequence[float]) -> stim.Circuit:
    """Rescale every noise probability in round t by ``factors[t]`` (any circuit family).

    A round ends at each layer containing a measurement; layers after the last
    listed round use the last factor. Measurement-flip arguments are scaled too.
    """
    layers = _split_layers(noisy)
    out = stim.Circuit()
    round_idx = 0
    f = list(map(float, factors))
    for li, layer in enumerate(layers):
        s = f[min(round_idx, len(f) - 1)]
        measured_here = False
        for ins in layer:
            args = ins.gate_args_copy()
            if (ins.name in NOISE_OPS or ins.name in MEASURE_GATES) and args:
                new_args = [min(max(a * s, 0.0), 1.0) for a in args]
                out.append(stim.CircuitInstruction(ins.name, ins.targets_copy(), new_args))
            else:
                out.append(ins)
            if ins.name in MEASURE_GATES and _qubit_targets(ins):
                measured_here = True
        if measured_here:
            round_idx += 1
        if li < len(layers) - 1:
            out.append(stim.CircuitInstruction("TICK", [], []))
    return out


def parse_round_rates(circuit: stim.Circuit, op: str = "DEPOLARIZE2") -> np.ndarray:
    """Per-round argument of ``op`` (first occurrence in each round), for tests."""
    layers = _split_layers(circuit)
    rates: List[float] = []
    current: Optional[float] = None
    for layer in layers:
        measured_here = False
        for ins in layer:
            if ins.name == op and current is None:
                current = float(ins.gate_args_copy()[0])
            if ins.name in MEASURE_GATES and _qubit_targets(ins):
                measured_here = True
        if measured_here:
            rates.append(float("nan") if current is None else current)
            current = None
    return np.asarray(rates)
