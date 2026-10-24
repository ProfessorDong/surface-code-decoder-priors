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
the exact-risk models, every condition in the manifests of the archived runs
behind Secs. IV and V, and every Willow circuit the hardware protocol calibrates
or tests on.  Each row carries the identifier of the model it censused so the
coverage can be checked against those manifests.

    python -m simulation.prx.theory.decomposition_census [--quick]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json

import stim

from ..config import WILLOW_EXTRACTED
from ..experiments.common import Condition, build_condition_circuit
from ..results_io import RESULTS_ROOT
from .certify_class_family import build_circuit

ROUNDS = (2, 3)
RATES = (0.0005, 0.001, 0.002, 0.004, 0.008)

# the archived runs whose conditions the paper reports; every condition in each
# manifest is censused, so the census covers the inventory rather than a sample
RUN_MANIFESTS = (
    "prior_sweep/20260917T142535Z_fbc712_gate0",
    "prior_sweep/20260917T175304Z_d2f74a_gate0_deep",
    "frontier/20260918T054857Z_6e4d15_device_like",
    "frontier/20260919T131450Z_4e7c86_draws_kappa20",
)
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


def manifest_models(quick: bool = False):
    """Every condition in the archived manifests of the runs the paper reports."""
    for run in RUN_MANIFESTS:
        path = RESULTS_ROOT / run / "manifest.json"
        if not path.is_file():
            continue
        conds = json.loads(path.read_text()).get("conditions", [])
        if quick:
            conds = conds[:2]
        for cid in conds:
            family, d, T, basis, p, extra = cid.split("|", 5)
            cond = Condition(family, int(d[1:]), int(T[1:]), float(p[1:]), basis,
                             json.loads(extra) if extra else {})
            yield f"{run.split('/')[0]}:{cid}", build_condition_circuit(cond)


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
    rows = []
    sources = [("exact", exact_models()), ("manifest", manifest_models(a.quick))]
    if WILLOW_EXTRACTED.is_dir():
        sources.append(("willow", willow_circuits(a.quick)))
    for group, it in sources:
        for name, circ in it:
            r = {"group": group, "model": name}
            r.update(census(circ.detector_error_model(decompose_errors=True)))
            rows.append(r)
            print(f"[{len(rows)}] {group} {name[:70]}: "
                  f"{r['components_3det_or_more']} oversize, {r['components_0det_with_logical']} detectorless, "
                  f"{r['parallel_groups_conflicting_logical']} conflicting parallel", flush=True)

    out = RESULTS_ROOT / "theory" / "decomposition_census.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    worst = max(r["max_component_size"] for r in rows)
    print(f"{len(rows)} models; largest component {worst} detectors; "
          f"{sum(r['components_3det_or_more'] for r in rows)} components too large to represent; "
          f"{sum(r['components_0det_with_logical'] for r in rows)} detectorless logical components; "
          f"{sum(r['parallel_groups_conflicting_logical'] for r in rows)} parallel groups disagreeing on the logical flip")
    print("wrote", out)


if __name__ == "__main__":
    main()
