"""Paths, seeds, worker setup, and provenance for the PRX rework."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRX_ROOT = PROJECT_ROOT / "research" / "prx_quantum"
RESULTS_ROOT = PRX_ROOT / "results"
WILLOW_ZIP = PROJECT_ROOT / "data" / "willow_hardware" / "google_105Q_surface_code_d3_d5_d7.zip"
WILLOW_EXTRACTED = PROJECT_ROOT / "data" / "willow_hardware" / "google_105Q_surface_code_d3_d5_d7"

# Maximum probability admitted into a matching graph (weights must stay finite and positive).
P_MAX = 0.5 - 1e-9
P_MIN = 1e-12


@dataclass(frozen=True)
class SeedPlan:
    """Deterministic, collision-resistant seeds derived from one base seed and tags.

    Every stochastic call in the pipeline (Stim samplers, numpy generators,
    torch) must obtain its seed from ``SeedPlan.for_(...)`` so a manifest can
    reproduce any run from ``base_seed`` plus the tag path.
    """

    base_seed: int

    def for_(self, *tags: object) -> int:
        payload = repr((int(self.base_seed),) + tuple(str(t) for t in tags)).encode()
        digest = hashlib.sha256(payload).digest()
        # Stim accepts seeds up to 2**64-1; numpy accepts arbitrary ints. Keep below 2**62.
        return int.from_bytes(digest[:8], "little") & ((1 << 62) - 1)


def worker_init() -> None:
    """Pin every worker to one thread; the pool provides the parallelism."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch  # noqa: WPS433

        torch.set_num_threads(1)
    except Exception:  # torch is optional for the CPU pipeline
        pass


def source_hash() -> str:
    """SHA-256 over all Python sources in simulation/prx (sorted by path)."""
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def versions() -> Dict[str, str]:
    out: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "scipy", "stim", "pymatching", "torch", "torch_geometric", "pandas"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "?"))
        except Exception:
            out[name] = "absent"
    return out
