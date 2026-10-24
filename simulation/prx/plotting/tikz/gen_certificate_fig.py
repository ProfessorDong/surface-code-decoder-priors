"""Data for the certified-ceiling figure (fig_ceiling).

Writes one .dat per series into manuscript/figdata/. Every quantity is derived
here from the archived result tables with the same formulas the number checker
uses, so the figure cannot drift from the text:

    certified ceiling (%) = 100 * (1 - LB / R_phys)
    best gain found  (%) = 100 * max gain over every per-class prior evaluated, which is
                           the coordinate scan, its robustness re-test, and the incumbent
                           of the branch-and-bound run (an exact risk at a point of the
                           same family); the branch and bound improves on the scan at
                           every error rate, so the maximum is the one to plot
    gap to ML        (%) = 100 * (R_phys - R_ml) / R_phys

Run from the repository root:
    python -m simulation.prx.plotting.tikz.gen_certificate_fig
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "research" / "prx_quantum" / "results" / "theory"
OUT = ROOT / "manuscript" / "figdata"

CERT = {2: "certified_class_family_T2_wspan.csv", 3: "certified_class_family_T3_gpu.csv"}


def write(name, rows):
    path = OUT / f"fig_ceiling_{name}.dat"
    with path.open("w") as fh:
        fh.write("x y\n")
        for x, y in rows:
            fh.write(f"{x:g} {y:.6f}\n")
    print(f"wrote {path.name} ({len(rows)} points)")


def main():
    OUT.mkdir(exist_ok=True)
    exact = pd.read_csv(RES / "exact_candidates_untruncated.csv")
    # the maximum-likelihood risk at a point the coordinate scan never visited is still
    # exact, and the attribution run computes it, so take R_ml from both sources
    att = pd.read_csv(RES / "reweight_mechanism.csv")
    for T, fname in CERT.items():
        cert = pd.read_csv(RES / fname).set_index("p").sort_index()
        ex = exact[exact["T"] == T]
        scan = ex.groupby("p").gain.max()
        rml = ex.groupby("p").R_ml.first()
        extra = att[att["T"] == T].set_index("p").R_ml_full
        rml = pd.concat([rml, extra[~extra.index.isin(rml.index)]]).sort_index()
        best = {p: max(float(scan.get(p, -1.0)), float(cert.gain_UB_certified[p])) for p in cert.index}

        write(f"T{T}_ceiling", [(p, 100.0 * (1.0 - cert.LB[p] / cert.R_phys[p])) for p in cert.index])
        write(f"T{T}_best", [(p, 100.0 * best[p]) for p in cert.index])
        write(f"T{T}_mlgap",
              [(p, 100.0 * (cert.R_phys[p] - rml[p]) / cert.R_phys[p])
               for p in cert.index if p in rml.index])

        # the open interval, on the error rates where both ends exist
        common = list(cert.index)
        write(f"T{T}_band_hi", [(p, 100.0 * (1.0 - cert.LB[p] / cert.R_phys[p])) for p in common])
        write(f"T{T}_band_lo", [(p, 100.0 * best[p]) for p in common])


if __name__ == "__main__":
    main()
