"""SI1000-style circuit-level noise injected layer by layer into a noiseless circuit.

Rules (derived from Google's shipped ``circuit_noisy_si1000.stim`` files of the
Willow release, variant ``google2025``, and checked by reproducing their
detector error models exactly):

* Layers are delimited by ``TICK``. Within a layer:
  - every qubit that is a target of a two-qubit Clifford gate (CZ, CX between
    two qubits) receives ``DEPOLARIZE2(two_qubit)`` applied to the gate's pair;
  - every measured qubit is measured with flip probability ``meas_flip``
    (``M(p)``/``MR(p)``) and then receives ``DEPOLARIZE1(post_meas_depol)``;
  - every reset qubit (``R``, ``MR``) receives ``X_ERROR(reset_flip)`` after the reset;
  - every other qubit (idle, single-qubit-gate target, sweep-controlled CX
    target) receives ``DEPOLARIZE1(one_qubit)``;
  - in a layer containing a measurement or reset, every gate-free qubit
    additionally receives ``DEPOLARIZE1(idle_during_mr)``.
* Detector, observable, and coordinate annotations pass through unchanged.

Variant ``gidney2021`` (the original SI1000 of arXiv:2108.10457, Tables 1-2)
uses the same rules without the ``post_meas_depol`` channel, which Google's
2025 release adds (``DEPOLARIZE1(p)`` on every measured qubit). Measurement
flips are emitted as the ``M(p)``/``MR(p)`` argument, equivalent for detector
statistics to the paper's ``X_ERROR(5p)`` before the measurement.

Per-round rates: pass a sequence of ``SI1000Rates`` (one per syndrome round).
A round ends at every layer that measures at least one qubit; the final
data-qubit measurement layer uses the last round's rates. Per-qubit
heterogeneity: ``qubit_scale`` multiplies every single-qubit channel on qubit
q by ``scale[q]`` and every two-qubit channel by the mean of the two scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import stim

TWO_QUBIT_GATES = {"CZ", "CX", "CY", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "SWAP", "ISWAP", "ISWAP_DAG", "SQRT_XX", "SQRT_YY", "SQRT_ZZ"}
MEASURE_GATES = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}
RESET_GATES = {"R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"}
ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"}
NOISE_OPS = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"}


@dataclass(frozen=True)
class SI1000Rates:
    p: float
    two_qubit: float
    one_qubit: float
    meas_flip: float
    reset_flip: float
    idle_during_mr: float
    post_meas_depol: float

    @classmethod
    def from_variant(cls, variant: str, p: float) -> "SI1000Rates":
        if variant in ("google2025", "willow"):
            return cls(p=p, two_qubit=p, one_qubit=p / 10, meas_flip=5 * p, reset_flip=2 * p,
                       idle_during_mr=2 * p, post_meas_depol=p)
        if variant == "gidney2021":
            # Honeycomb-memory paper (Quantum 5, 605) Tables 1-2: no post-measurement depolarization.
            return cls(p=p, two_qubit=p, one_qubit=p / 10, meas_flip=5 * p, reset_flip=2 * p,
                       idle_during_mr=2 * p, post_meas_depol=0.0)
        raise KeyError(f"unknown SI1000 variant {variant!r}")

    def scaled(self, factor: float) -> "SI1000Rates":
        f = float(factor)
        return SI1000Rates(p=self.p * f, two_qubit=self.two_qubit * f, one_qubit=self.one_qubit * f,
                           meas_flip=self.meas_flip * f, reset_flip=self.reset_flip * f,
                           idle_during_mr=self.idle_during_mr * f, post_meas_depol=self.post_meas_depol * f)


def noiseless_memory_circuit(distance: int, rounds: int, basis: str = "Z") -> stim.Circuit:
    task = f"surface_code:rotated_memory_{basis.lower()}"
    return stim.Circuit.generated(task, distance=distance, rounds=rounds)


def strip_noise(circuit: stim.Circuit) -> stim.Circuit:
    """Remove all noise channels and measurement-flip arguments (keeps structure)."""
    out = stim.Circuit()
    for ins in circuit.flattened():
        if ins.name in NOISE_OPS:
            continue
        if ins.name in MEASURE_GATES and ins.gate_args_copy():
            out.append(stim.CircuitInstruction(ins.name, ins.targets_copy(), []))
        else:
            out.append(ins)
    return out


def _split_layers(circuit: stim.Circuit) -> List[List[stim.CircuitInstruction]]:
    layers: List[List[stim.CircuitInstruction]] = [[]]
    for ins in circuit.flattened():
        if ins.name == "TICK":
            layers.append([])
        else:
            layers[-1].append(ins)
    return layers


def _qubit_targets(ins: stim.CircuitInstruction) -> List[int]:
    return [t.value for t in ins.targets_copy() if t.is_qubit_target]


def _two_qubit_pairs(ins: stim.CircuitInstruction) -> List[tuple]:
    """Pairs of real qubits for a two-qubit gate; sweep/record-controlled pairs are skipped."""
    ts = ins.targets_copy()
    pairs = []
    for a, b in zip(ts[::2], ts[1::2]):
        if a.is_qubit_target and b.is_qubit_target:
            pairs.append((a.value, b.value))
    return pairs


def _clip(p: float) -> float:
    return float(min(max(p, 0.0), 1.0))


CHANNELS = ("two_qubit", "one_qubit", "meas_flip", "reset_flip", "idle_during_mr", "post_meas_depol")


def _scale_lookup(qubit_scale, all_qubits):
    """Normalise ``qubit_scale`` into {channel: {qubit: factor}}.

    Accepts a flat mapping (applied to every channel) or a mapping keyed by
    channel name, so heterogeneity can be confined to specific error channels
    (for example only the measurement flips, which is what a homogeneous
    reference model most often gets wrong).
    """
    base = {c: {q: 1.0 for q in all_qubits} for c in CHANNELS}
    if not qubit_scale:
        return base
    is_channel_form = any(isinstance(v, Mapping) for v in qubit_scale.values())
    if is_channel_form:
        unknown = set(qubit_scale) - set(CHANNELS)
        if unknown:
            raise KeyError(f"unknown channels in qubit_scale: {sorted(unknown)}")
        for ch, mapping in qubit_scale.items():
            for q, v in mapping.items():
                base[ch][int(q)] = float(v)
    else:
        for q, v in qubit_scale.items():
            for ch in CHANNELS:
                base[ch][int(q)] = float(v)
    return base


def inject_si1000(noiseless: stim.Circuit,
                  rates: Union[SI1000Rates, Sequence[SI1000Rates]],
                  qubit_scale: Optional[Mapping] = None) -> stim.Circuit:
    """Return a noisy copy of ``noiseless`` following the module's SI1000 rules."""
    circuit = strip_noise(noiseless)
    layers = _split_layers(circuit)
    # A trailing layer without any qubit operation (a final TICK) receives no noise.
    while len(layers) > 1 and not any(ins.name not in ANNOTATIONS for ins in layers[-1]):
        layers.pop()
    all_qubits = list(range(circuit.num_qubits))
    per_round = isinstance(rates, (list, tuple))
    rate_seq: List[SI1000Rates] = list(rates) if per_round else [rates]  # type: ignore[list-item]
    n_rounds = len(rate_seq)
    sc = _scale_lookup(qubit_scale, all_qubits)

    out = stim.Circuit()
    round_idx = 0
    for li, layer in enumerate(layers):
        r = rate_seq[min(round_idx, n_rounds - 1)]
        two_q: List[tuple] = []
        measured: List[int] = []
        reset: List[int] = []
        gated: set = set()
        has_mr = False
        emitted: List[stim.CircuitInstruction] = []
        for ins in layer:
            name = ins.name
            if name in TWO_QUBIT_GATES:
                pairs = _two_qubit_pairs(ins)
                two_q.extend(pairs)
                for a, b in pairs:
                    gated.add(a); gated.add(b)
                # sweep-controlled CX: target qubit counts as gated but not as a 2q qubit
                for t in _qubit_targets(ins):
                    gated.add(t)
                emitted.append(ins)
            elif name in MEASURE_GATES:
                qs = _qubit_targets(ins)
                measured.extend(qs)
                gated.update(qs)
                has_mr = True
                # measurement flip enters as the gate argument, per qubit scale (single arg allowed
                # only if all scales equal; otherwise split into groups)
                groups: Dict[float, List[int]] = {}
                for q in qs:
                    groups.setdefault(_clip(r.meas_flip * sc["meas_flip"][q]), []).append(q)
                for pflip, qq in groups.items():
                    emitted.append(stim.CircuitInstruction(name, [stim.GateTarget(q) for q in qq], [pflip]))
                if name in RESET_GATES:
                    reset.extend(qs)
            elif name in RESET_GATES:
                qs = _qubit_targets(ins)
                reset.extend(qs)
                gated.update(qs)
                has_mr = True
                emitted.append(ins)
            elif name in ANNOTATIONS:
                emitted.append(ins)
            else:  # single-qubit gates and anything else
                gated.update(_qubit_targets(ins))
                emitted.append(ins)

        for ins in emitted:
            out.append(ins)

        two_q_qubits = {q for pair in two_q for q in pair}
        # DEPOLARIZE2 on gate pairs
        for a, b in two_q:
            f = 0.5 * (sc["two_qubit"][a] + sc["two_qubit"][b])
            out.append(stim.CircuitInstruction("DEPOLARIZE2", [stim.GateTarget(a), stim.GateTarget(b)], [_clip(r.two_qubit * f)]))
        # post-measurement depolarization
        for q in measured:
            out.append(stim.CircuitInstruction("DEPOLARIZE1", [stim.GateTarget(q)], [_clip(r.post_meas_depol * sc["post_meas_depol"][q])]))
        # reset flips
        for q in reset:
            out.append(stim.CircuitInstruction("X_ERROR", [stim.GateTarget(q)], [_clip(r.reset_flip * sc["reset_flip"][q])]))
        # everything else: single-qubit depolarization
        others = [q for q in all_qubits if q not in two_q_qubits and q not in measured and q not in reset]
        for q in others:
            out.append(stim.CircuitInstruction("DEPOLARIZE1", [stim.GateTarget(q)], [_clip(r.one_qubit * sc["one_qubit"][q])]))
        if has_mr:
            for q in all_qubits:
                if q not in gated:
                    out.append(stim.CircuitInstruction("DEPOLARIZE1", [stim.GateTarget(q)], [_clip(r.idle_during_mr * sc["idle_during_mr"][q])]))
        if measured:
            round_idx += 1
        if li < len(layers) - 1:
            out.append(stim.CircuitInstruction("TICK", [], []))
    return out


def lognormal_qubit_scales(num_qubits: int, sigma: float, seed: int) -> Dict[int, float]:
    """Heterogeneous per-qubit multipliers lambda_q = exp(sigma * xi_q), xi_q ~ N(0, 1)."""
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal(num_qubits)
    return {q: float(np.exp(sigma * xi[q])) for q in range(num_qubits)}


def noise_op_histogram(circuit: stim.Circuit) -> Dict[tuple, int]:
    """(name, rounded arg) -> number of qubit targets; used by tests."""
    hist: Dict[tuple, int] = {}
    for ins in circuit.flattened():
        if ins.name in NOISE_OPS or (ins.name in MEASURE_GATES and ins.gate_args_copy()):
            key = (ins.name, tuple(round(a, 9) for a in ins.gate_args_copy()))
            hist[key] = hist.get(key, 0) + len(_qubit_targets(ins))
    return hist
