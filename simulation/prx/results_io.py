"""Tidy result rows, run directories, manifests, and packed error vectors."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from .config import RESULTS_ROOT, source_hash, versions

SCHEMA = [
    "experiment", "run_id", "created_utc", "code_family", "distance", "rounds", "basis",
    "noise_family", "p_true", "noise_extra_json", "drift_id", "patch", "leaf_id",
    "seed_cal", "seed_test", "replicate", "estimator", "estimator_params_json", "needs_labels",
    "n_cal_shots", "n_cal_rounds", "prior_id", "scale_space", "scale_time", "scale_diag",
    "scale_boundary", "p_hat", "decoder", "decoder_params_json", "n_test_shots", "n_errors",
    "ler", "ler_cp_lo", "ler_cp_hi", "ler_per_round_google", "ler_per_round_naive",
    "ref_prior_id", "delta_ler_vs_ref", "delta_ci_lo", "delta_ci_hi", "mcnemar_b", "mcnemar_c",
    "mcnemar_p_exact", "error_vector_path", "fit_seconds", "decode_seconds", "stim_version",
    "pymatching_version", "code_hash",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_dir(experiment: str, root: Path = RESULTS_ROOT, label: str = "") -> Path:
    """Create research/prx_quantum/results/<experiment>/<run_id>/ and return it."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    salt = hashlib.sha1(f"{stamp}{os.getpid()}{time.time_ns()}".encode()).hexdigest()[:6]
    run_id = f"{stamp}_{salt}" + (f"_{label}" if label else "")
    run_dir = Path(root) / experiment / run_id
    (run_dir / "errors").mkdir(parents=True, exist_ok=False)
    return run_dir


def write_manifest(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    data = dict(payload)
    data.setdefault("created_utc", utc_now())
    data.setdefault("versions", versions())
    data.setdefault("code_hash", source_hash())
    path = Path(run_dir) / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default))
    os.replace(tmp, path)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


class RowWriter:
    """Append tidy rows to rows.csv under an exclusive file lock (safe across processes)."""

    def __init__(self, run_dir: Path, filename: str = "rows.csv", schema: Iterable[str] = SCHEMA):
        self.path = Path(run_dir) / filename
        self.schema = list(schema)
        if not self.path.exists():
            with open(self.path, "w", newline="") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                csv.DictWriter(fh, fieldnames=self.schema).writeheader()

    def write(self, row: Mapping[str, Any]) -> None:
        unknown = set(row) - set(self.schema)
        if unknown:
            raise KeyError(f"row has columns outside the schema: {sorted(unknown)}")
        clean = {k: _scalar(row.get(k, "")) for k in self.schema}
        line = _format_line(clean, self.schema)
        with open(self.path, "a", newline="") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def write_many(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            self.write(row)


def _scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=_json_default, sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _format_line(row: Dict[str, Any], schema: Iterable[str]) -> str:
    import io

    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=list(schema)).writerow(row)
    return buf.getvalue()


def read_rows(path: Path):
    """Load rows.csv into a pandas DataFrame with numeric coercion where sensible."""
    import pandas as pd

    df = pd.read_csv(path)
    for col in ("distance", "rounds", "n_cal_shots", "n_cal_rounds", "n_test_shots", "n_errors",
                "mcnemar_b", "mcnemar_c", "replicate"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_error_vector(run_dir: Path, name: str, errors: np.ndarray) -> str:
    """Store a boolean per-shot error vector bit-packed; returns the relative path."""
    arr = np.asarray(errors, dtype=bool)
    packed = np.packbits(arr)
    rel = Path("errors") / f"{name}.npz"
    path = Path(run_dir) / rel
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, packed=packed, n=np.int64(arr.size))
    os.replace(tmp, path)
    return str(rel)


def load_error_vector(run_dir: Path, rel_path: str) -> np.ndarray:
    with np.load(Path(run_dir) / rel_path) as data:
        n = int(data["n"])
        return np.unpackbits(data["packed"])[:n].astype(bool)
