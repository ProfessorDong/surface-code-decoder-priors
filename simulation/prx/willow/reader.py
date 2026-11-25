"""Random-access reader for the Willow surface-code archive (zip, no extraction).

Layout inside the zip::

    google_105Q_surface_code_d3_d5_d7/<patch>/<basis>/r<NN>/{circuit_ideal.stim,
        circuit_noisy_si1000.stim, detection_events.b8, obs_flips_actual.b8,
        measurements.b8, sweep_bits.b8, metadata.json,
        decoding_results/<pathway>/{error_model.dem, obs_flips_predicted.b8}}

Detector-event files are Stim ``b8``: one row per shot, bits little-endian,
each row padded to whole bytes. We keep everything bit-packed.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import stim

from ..config import WILLOW_ZIP

ROOT = "google_105Q_surface_code_d3_d5_d7"
PATHWAYS = (
    "correlated_matching_decoder_with_si1000_prior",
    "correlated_matching_decoder_with_rl_optimized_prior",
    "harmony_decoder_with_si1000_prior",
    "harmony_decoder_with_rl_optimized_prior",
    "libra_decoder_with_rl_optimized_prior",
)
_LEAF_RE = re.compile(rf"^{ROOT}/(d(\d)_at_q(\d+)_(\d+))/([XZ])/r(\d+)/(.+)$")


@dataclass(frozen=True, order=True)
class LeafKey:
    distance: int
    patch: str
    basis: str
    rounds: int

    @property
    def leaf_id(self) -> str:
        return f"{self.patch}/{self.basis}/r{self.rounds:02d}"

    @property
    def prefix(self) -> str:
        return f"{ROOT}/{self.leaf_id}/"


class WillowArchive:
    def __init__(self, zip_path=WILLOW_ZIP):
        self.zip_path = str(zip_path)
        self._zf: Optional[zipfile.ZipFile] = None
        self._index: Dict[LeafKey, Dict[str, str]] = {}
        with zipfile.ZipFile(self.zip_path) as z:
            for name in z.namelist():
                m = _LEAF_RE.match(name)
                if not m:
                    continue
                patch, d, _, _, basis, r, rest = m.groups()
                key = LeafKey(int(d), patch, basis, int(r))
                self._index.setdefault(key, {})[rest] = name

    # ------------------------------------------------------------- plumbing
    def _z(self) -> zipfile.ZipFile:
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return self._zf

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zf"] = None      # re-open lazily in each worker process
        return state

    def _read(self, key: LeafKey, rest: str) -> bytes:
        try:
            name = self._index[key][rest]
        except KeyError as exc:
            raise FileNotFoundError(f"{key.leaf_id}/{rest}") from exc
        return self._z().read(name)

    # -------------------------------------------------------------- listing
    def leaves(self, distance: Optional[int] = None, basis: Optional[str] = None,
               rounds: Optional[int] = None) -> List[LeafKey]:
        keys = [k for k in self._index
                if (distance is None or k.distance == distance)
                and (basis is None or k.basis == basis)
                and (rounds is None or k.rounds == rounds)]
        return sorted(keys)

    def has_pathway(self, key: LeafKey, pathway: str) -> bool:
        return f"decoding_results/{pathway}/obs_flips_predicted.b8" in self._index.get(key, {})

    def patches(self, distance: Optional[int] = None) -> List[str]:
        return sorted({k.patch for k in self.leaves(distance)})

    # --------------------------------------------------------------- content
    def metadata(self, key: LeafKey) -> dict:
        return json.loads(self._read(key, "metadata.json").decode())

    def circuit(self, key: LeafKey, kind: str = "noisy_si1000") -> stim.Circuit:
        return stim.Circuit(self._read(key, f"circuit_{kind}.stim").decode())

    def num_detectors(self, key: LeafKey) -> int:
        return self.circuit(key, "ideal").num_detectors

    def detection_events(self, key: LeafKey) -> np.ndarray:
        """Bit-packed uint8[shots, ceil(D/8)]."""
        raw = np.frombuffer(self._read(key, "detection_events.b8"), dtype=np.uint8)
        d = self.num_detectors(key)
        row = (d + 7) // 8
        if raw.size % row != 0:
            raise ValueError(f"{key.leaf_id}: {raw.size} bytes not divisible by {row}")
        return raw.reshape(-1, row)

    def obs_actual(self, key: LeafKey) -> np.ndarray:
        raw = np.frombuffer(self._read(key, "obs_flips_actual.b8"), dtype=np.uint8)
        return raw.reshape(-1, 1)

    def obs_predicted(self, key: LeafKey, pathway: str) -> np.ndarray:
        raw = np.frombuffer(self._read(key, f"decoding_results/{pathway}/obs_flips_predicted.b8"), dtype=np.uint8)
        return raw.reshape(-1, 1)

    def dem(self, key: LeafKey, pathway: str) -> stim.DetectorErrorModel:
        return stim.DetectorErrorModel(self._read(key, f"decoding_results/{pathway}/error_model.dem").decode())

    def sweep_bits(self, key: LeafKey) -> np.ndarray:
        raw = np.frombuffer(self._read(key, "sweep_bits.b8"), dtype=np.uint8)
        n_sweep = self.circuit(key, "ideal").num_sweep_bits
        row = (n_sweep + 7) // 8
        return raw.reshape(-1, row)


def google_ler(pred_packed: np.ndarray, actual_packed: np.ndarray) -> Tuple[int, int]:
    """(n_errors, n_shots) from bit-packed single-observable prediction and truth."""
    diff = (pred_packed[:, 0] ^ actual_packed[:, 0]) & 1
    return int(diff.sum()), int(diff.size)


def shot_errors(pred_packed: np.ndarray, actual_packed: np.ndarray) -> np.ndarray:
    return (((pred_packed[:, 0] ^ actual_packed[:, 0]) & 1) != 0)


def prior_name(pathway: str) -> str:
    return "rl_optimized" if "rl_optimized" in pathway else "si1000"
