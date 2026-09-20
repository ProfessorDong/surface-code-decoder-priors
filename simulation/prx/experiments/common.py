"""Shared machinery: conditions, seeded sampling, paired decoding, and a process pool."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pymatching
import stim

from ..config import SeedPlan, worker_init
from ..dem.edges import EdgeTable
from ..dem.priors import build_circuit
from ..noise.si1000 import lognormal_qubit_scales
from ..stats import clopper_pearson, discordant_counts, exact_mcnemar, paired_delta_ci


@dataclass(frozen=True)
class Condition:
    family: str
    distance: int
    rounds: int
    p: float
    basis: str = "Z"
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def cond_id(self) -> str:
        extra = json.dumps(self.extra, sort_keys=True) if self.extra else ""
        return f"{self.family}|d{self.distance}|T{self.rounds}|{self.basis}|p{self.p:.6g}|{extra}"

    def row_fields(self) -> Dict[str, Any]:
        return {
            "code_family": f"rotated_surface_memory_{self.basis}",
            "distance": self.distance,
            "rounds": self.rounds,
            "basis": self.basis,
            "noise_family": self.family + (":" + self.extra["variant"] if "variant" in self.extra else ""),
            "p_true": self.p,
            "noise_extra_json": json.dumps(self.extra, sort_keys=True),
        }


def build_condition_circuit(cond: Condition) -> stim.Circuit:
    kw: Dict[str, Any] = {}
    extra = cond.extra
    if "variant" in extra:
        kw["variant"] = extra["variant"]
    if extra.get("sigma_h", 0.0) > 0:
        # number of qubits is family-dependent; build once to size the scale map
        probe = build_circuit(cond.family, cond.distance, cond.rounds, cond.p, cond.basis, **kw)
        scales = lognormal_qubit_scales(probe.num_qubits, float(extra["sigma_h"]), int(extra["het_seed"]))
        channels = extra.get("het_channels")
        kw["qubit_scale"] = {c: scales for c in channels} if channels else scales
    return build_circuit(cond.family, cond.distance, cond.rounds, cond.p, cond.basis, **kw)


def sample_shots(circuit: stim.Circuit, shots: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bit-packed detector events and observable flips (seeded, reproducible)."""
    sampler = circuit.compile_detector_sampler(seed=int(seed))
    dets, obs = sampler.sample(int(shots), separate_observables=True, bit_packed=True)
    return dets, obs


def decode_errors(matching: pymatching.Matching, dets_packed: np.ndarray, obs_packed: np.ndarray) -> np.ndarray:
    """Per-shot logical failure indicator for observable 0."""
    preds = matching.decode_batch(dets_packed, bit_packed_shots=True, bit_packed_predictions=True)
    return ((preds[:, 0] ^ obs_packed[:, 0]) & 1).astype(bool)


def paired_summary(errors: np.ndarray, ref_errors: Optional[np.ndarray]) -> Dict[str, Any]:
    n = int(errors.size)
    k = int(errors.sum())
    lo, hi = clopper_pearson(k, n)
    out: Dict[str, Any] = {"n_test_shots": n, "n_errors": k, "ler": k / n, "ler_cp_lo": lo, "ler_cp_hi": hi}
    if ref_errors is not None:
        b, c = discordant_counts(errors, ref_errors)
        d, dlo, dhi = paired_delta_ci(errors, ref_errors)
        out.update({"mcnemar_b": b, "mcnemar_c": c, "mcnemar_p_exact": exact_mcnemar(b, c),
                    "delta_ler_vs_ref": d, "delta_ci_lo": dlo, "delta_ci_hi": dhi})
    return out


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def run_pool(tasks: List[Any], fn: Callable[[Any], Any], workers: int = 16) -> Iterator[Any]:
    """Run ``fn`` over ``tasks`` in a forkserver pool, yielding results as they finish."""
    if workers <= 1:
        for t in tasks:
            yield fn(t)
        return
    ctx = mp.get_context("forkserver")
    with ctx.Pool(processes=workers, initializer=worker_init, maxtasksperchild=8) as pool:
        for res in pool.imap_unordered(fn, tasks, chunksize=1):
            yield res


def geometric_mean_ratio(a: EdgeTable, b: EdgeTable) -> float:
    """exp(mean(log(a.p / b.p))) over matched edges."""
    return float(np.exp(np.mean(np.log(a.p) - np.log(b.p))))
