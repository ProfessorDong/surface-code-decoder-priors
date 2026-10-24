"""How much the graphlike projection actually throws away.

PyMatching builds its matching graph from the graphlike components Stim produces,
merging parallel edges and discarding any component it cannot represent.  Three
distinct losses hide behind that sentence, and the paper's attribution depends on
telling them apart:

  (i)   a component with three or more detectors has no edge to become, so it is
        dropped outright;
  (ii)  two parallel components -- the same detector pair -- that flip different
        observables are merged into one edge, which does not preserve the joint
        law of syndrome and logical flip;
  (iii) the components of one mechanism are handed to the decoder as independent
        edges, so their joint law is lost even though every component survives.

This script censuses the detector error models the paper uses and records, for
each, the component-size histogram, the number of components with no detector at
all (a pure-logical component, which the matching graph cannot carry either), and
the number of parallel component groups whose members disagree on the observables
they flip.  Loss (i) occurs only if some component has three or more detectors,
loss (ii) only if some parallel group disagrees.

Coverage is tied to the inventory the paper actually reports, not to a sample:
the exact-risk models, every condition of every archived run behind Secs. III-V
(the prior sweeps, all three heterogeneity frontiers, both shrinkage-development
runs, the variance check and the many-draw heterogeneity study), and every Willow
circuit the hardware protocol calibrates or tests on.

Two different conditions can specify the same physical circuit -- the frontier
runs differ in the reference prior they misspecify, not in the device -- so rows
are deduplicated on the decomposed model itself, by hashing the detector error
model. Each row lists every run and condition that produced that model, so the
coverage can be checked against the manifests rather than taken on trust.

    python -m simulation.prx.theory.decomposition_census [--quick]
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json

import pandas as pd
import stim

from ..config import WILLOW_EXTRACTED
from ..experiments.common import Condition, build_condition_circuit
from ..experiments.hetero_seeds import qubit_scales
from ..noise.si1000 import SI1000Rates, inject_si1000, noiseless_memory_circuit
from ..results_io import RESULTS_ROOT
from .certify_class_family import build_circuit

ROUNDS = (2, 3)
RATES = (0.0005, 0.001, 0.002, 0.004, 0.008)

# Every archived run the paper reports, by the same globs verify_numbers.py uses to
# find them. A run whose manifest lists `conditions` is censused from that list; the
# many-draw heterogeneity study has its own schema and is censused from its rows.
REPORTED_RUNS = ("prior_sweep/*_gate0", "prior_sweep/*_gate0_deep",
                 "frontier/*device_like", "frontier/*_draws", "frontier/*_draws_dgr",
                 "frontier/*_draws_kappa20", "kappa_dev/*_dev", "eb_variance/*_check")
DRAW_STUDY = "hetero_seeds/*_main"
# the Willow leaves the hardware protocol uses: r13 for calibration, these for test
WILLOW_TEST_ROUNDS = {3: (10, 30, 50, 70), 5: (10, 30, 50, 70, 90, 110),
                      7: (10, 30, 50, 70, 90, 110, 130, 150)}


def census(dem: stim.DetectorErrorModel) -> dict:
    """Component census for one decomposed model.

    Besides the size histogram this counts components with no detector and
    parallel bulk components -- same unordered detector pair -- that flip
    different observables, the two conditions under which the merge into a single
    weighted edge is not faithful.
    """
    sizes: collections.Counter = collections.Counter()
    mechanisms = hyperedges = split = empty = 0
    obs_by_support: dict = collections.defaultdict(set)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comps: list = [([], [])]
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append(([], []))
            elif t.is_relative_detector_id():
                comps[-1][0].append(t.val)
            elif t.is_logical_observable_id():
                comps[-1][1].append(t.val)
        mechanisms += 1
        parts = [len(d) for d, _ in comps if d]
        if sum(parts) >= 3:
            hyperedges += 1
        if len(parts) > 1:
            split += 1
        for dets, obs in comps:
            if not dets:
                # a component carrying observables but no detector is invisible to
                # the decoder; an entirely empty component is just padding
                if obs:
                    empty += 1
                continue
            sizes[len(dets)] += 1
            obs_by_support[frozenset(dets)].add(frozenset(obs))
    return {"mechanisms": mechanisms, "hyperedges": hyperedges,
            "mechanisms_split": split,
            "components_1det": sizes[1], "components_2det": sizes[2],
            "components_3det_or_more": sum(n for s, n in sizes.items() if s >= 3),
            "max_component_size": max(sizes) if sizes else 0,
            "components_0det_with_logical": empty,
            "parallel_groups_conflicting_logical": sum(1 for v in obs_by_support.values() if len(v) > 1)}


def exact_models():
    """The models of the exact-risk study (Sec. III)."""
    for T in ROUNDS:
        for p in RATES:
            yield f"si1000_d3_T{T}_p{p:g}", build_circuit("si1000", 3, T, p, variant="google2025")


def _runs(pattern: str):
    """Every archived run directory matching one of the reported globs."""
    return sorted(d for d in RESULTS_ROOT.glob(pattern) if (d / "manifest.json").is_file())


def manifest_models(quick: bool = False):
    """Every condition of every reported run that records a condition list."""
    for pattern in REPORTED_RUNS:
        for run in _runs(pattern):
            conds = json.loads((run / "manifest.json").read_text()).get("conditions", [])
            if quick:
                conds = conds[:2]
            for cid in conds:
                family, d, T, basis, p, extra = cid.split("|", 5)
                cond = Condition(family, int(d[1:]), int(T[1:]), float(p[1:]), basis,
                                 json.loads(extra) if extra else {})
                yield f"{run.parent.name}/{run.name}:{cid}", build_condition_circuit(cond)


def draw_study_models(quick: bool = False):
    """The many-draw heterogeneity study, whose rows carry the draw rather than a condition."""
    for run in _runs(DRAW_STUDY):
        rows = pd.read_csv(run / "rows.csv",
                           usecols=["distance", "rounds", "p_true", "noise_extra_json"])
        rows = rows.drop_duplicates()
        if quick:
            rows = rows.head(2)
        for _, r in rows.iterrows():
            x = json.loads(r.noise_extra_json)
            d, T, p = int(r.distance), int(r.rounds), float(r.p_true)
            noiseless = noiseless_memory_circuit(d, T)
            rng_sigma, seed, mp = float(x["sigma"]), int(x["seed"]), bool(x["mean_preserving"])
            scales = (qubit_scales(noiseless.num_qubits, rng_sigma, seed, mp)
                      if rng_sigma > 0 else None)
            chans = x.get("channels") or []
            qs = {c: scales for c in chans} if (scales and chans) else scales
            circ = inject_si1000(noiseless, SI1000Rates.from_variant("google2025", p), qubit_scale=qs)
            tag = f"d{d}|T{T}|p{p:g}|sigma{rng_sigma:g}|seed{seed}|{'mp' if mp else 'raw'}"
            yield f"{run.parent.name}/{run.name}:{tag}", circ


def willow_circuits(quick: bool = False):
    """Every Willow circuit the hardware protocol calibrates or tests on."""
    for d, rounds in WILLOW_TEST_ROUNDS.items():
        for patch in sorted(WILLOW_EXTRACTED.glob(f"d{d}_at_*")):
            for basis in ("X", "Z"):
                for r in (13,) + (rounds[:1] if quick else rounds):
                    pth = patch / basis / f"r{r:02d}" / "circuit_noisy_si1000.stim"
                    if pth.is_file():
                        yield f"willow_{patch.name}_{basis}_r{r}", stim.Circuit.from_file(pth)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="a few models per family, for tests")
    a = ap.parse_args()
    seen: dict = {}
    order: list = []
    groups = [("exact", exact_models()), ("manifest", manifest_models(a.quick)),
              ("draws", draw_study_models(a.quick))]
    if WILLOW_EXTRACTED.is_dir():
        groups.append(("willow", willow_circuits(a.quick)))
    n_eval = 0
    for group, it in groups:
        for name, circ in it:
            n_eval += 1
            dem = circ.detector_error_model(decompose_errors=True)
            key = hashlib.sha256(str(dem).encode()).hexdigest()[:16]
            if key in seen:                       # the same physical model, specified twice
                seen[key]["sources"].append(name)
                continue
            r = {"group": group, "model": name, "dem_sha16": key, "sources": [name]}
            r.update(census(dem))
            seen[key] = r
            order.append(key)
            print(f"[{len(order)}/{n_eval}] {group} {name[:64]}: "
                  f"{r['components_3det_or_more']} oversize, {r['components_0det_with_logical']} detectorless, "
                  f"{r['parallel_groups_conflicting_logical']} conflicting parallel", flush=True)

    rows = [seen[k] for k in order]
    for r in rows:
        r["n_conditions"] = len(r["sources"])
        # a condition id contains spaces, so join on a separator that cannot occur in one
        r["sources"] = ";;".join(r.pop("sources"))
    out = RESULTS_ROOT / "theory" / "decomposition_census.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    worst = max(r["max_component_size"] for r in rows)
    print(f"{n_eval} evaluations over {len(rows)} distinct models; largest component {worst} detectors; "
          f"{sum(r['components_3det_or_more'] for r in rows)} components too large to represent; "
          f"{sum(r['components_0det_with_logical'] for r in rows)} detectorless logical components; "
          f"{sum(r['parallel_groups_conflicting_logical'] for r in rows)} parallel groups disagreeing on the logical flip")
    print("wrote", out)


if __name__ == "__main__":
    main()
