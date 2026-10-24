"""Surface-code series for the exact-theory figure (fig2_exact_theory).

Panel (a) plots the logical error of the best per-class prior found and of the
most-likely-coset decoder, relative to matching with the physical prior; panel
(b) plots the share of the matching-to-ML gap that the best prior recovers.

"Best found" means the same thing here as in Table II and in the ceiling figure:
the maximum over every per-class prior whose exact risk we have evaluated, which
is the coordinate scan, its robustness re-test, and the incumbent of the
branch-and-bound run. The incumbent beats the scan at every error rate, so a
figure built from the scan alone understates the best known prior. Only the
surface-code series are written here; the repetition-code curves come straight
from rep_untruncated/ and are unaffected.

Run from the repository root:
    python -m simulation.prx.plotting.tikz.gen_exact_fig
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "research" / "prx_quantum" / "results" / "theory"
OUT = ROOT / "manuscript" / "figdata"

CERT = {2: "certified_class_family_T2_wspan.csv", 3: "certified_class_family_T3_gpu.csv"}
# series indices as fig2_exact_theory.tex references them
BEST_PANEL_A = {2: "p0_s2", 3: "p0_s4"}
ML_PANEL_A = {2: "p0_s3", 3: "p0_s5"}
SHARE_PANEL_B = {2: "p1_s1", 3: "p1_s2"}


def write(name, rows):
    path = OUT / f"fig2_exact_theory_{name}.dat"
    with path.open("w") as fh:
        fh.write("x y\n")
        for x, y in rows:
            fh.write(f"{x:g} {y:.10g}\n")
    print(f"wrote {path.name} ({len(rows)} points)")


def main():
    exact = pd.read_csv(RES / "exact_candidates_untruncated.csv")
    for T, fname in CERT.items():
        cert = pd.read_csv(RES / fname).set_index("p").sort_index()
        ex = exact[exact["T"] == T]
        scan = ex.groupby("p").gain.max()
        # the exact risks come from the enumeration table, which covers the rates the
        # branch-and-bound search reached as well as the ones the scan visited
        lim = pd.read_csv(RES / "moment_limit_exact.csv").set_index(["T", "p"]).sort_index()
        rml = lim.loc[T].R_ml
        rph = lim.loc[T].R_phys

        rates = [float(p) for p in rml.index]
        best = {p: max(float(scan.get(p, -1.0)), float(cert.gain_UB_certified[p])) for p in rates}

        write(BEST_PANEL_A[T], [(p, 100.0 * (1.0 - best[p])) for p in rates])
        write(ML_PANEL_A[T], [(p, 100.0 * rml[p] / rph[p]) for p in rates])
        write(SHARE_PANEL_B[T],
              [(p, 100.0 * best[p] * rph[p] / (rph[p] - rml[p])) for p in rates])


if __name__ == "__main__":
    main()
