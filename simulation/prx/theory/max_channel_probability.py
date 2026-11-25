"""Largest channel probability applied in any simulated circuit the paper reports.

Section II states that per-qubit multipliers are clipped to [0, 1] and that the clip
never binds; this records the largest probability argument of any noise channel or
noisy measurement over the census inventory (every simulated model of the paper), with
the run that attains it and the largest value per run.

Writes research/results/theory/max_channel_probability.json.
usage: python -m simulation.prx.theory.max_channel_probability
"""
from __future__ import annotations

import json

import stim

from ..config import RESULTS_ROOT
from .decomposition_census import draw_study_models, exact_models, manifest_models

NOISY = ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "M", "MR", "MX", "MY", "MZ", "MRX", "MRY", "MRZ")


def largest(circ: stim.Circuit) -> tuple[float, str]:
    best, kind = 0.0, ""
    for ins in circ.flattened():
        args = ins.gate_args_copy()
        if args and ins.name in NOISY and max(args) > best:
            best, kind = float(max(args)), ins.name
    return best, kind


def main() -> None:
    per_run: dict[str, dict] = {}
    n = 0
    for gen in (exact_models(), manifest_models(), draw_study_models()):
        for name, circ in gen:
            n += 1
            run = name.split(":")[0]
            v, kind = largest(circ)
            if v > per_run.get(run, {}).get("largest", -1.0):
                per_run[run] = {"largest": v, "instruction": kind, "condition": name}
    top = max(per_run.values(), key=lambda r: r["largest"])
    out = {"models": n, "largest_channel_probability": top["largest"], "instruction": top["instruction"],
           "condition": top["condition"], "per_run": per_run}
    f = RESULTS_ROOT / "theory" / "max_channel_probability.json"
    f.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_run"}, indent=2))
    print("wrote", f)


if __name__ == "__main__":
    main()
