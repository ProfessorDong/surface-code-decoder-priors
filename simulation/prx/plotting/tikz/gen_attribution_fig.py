"""Data for the decision-attribution figure (fig_attribution).

Two priors reach a similar gain over the physical prior: the best per-class prior
found by search, and the population limit of the pairwise correlation fit, which
needs no logical labels. The exact attribution of Appendix B splits each gain into
the part carried by syndromes where the new decision agrees with the decomposed
model's most-probable-coset rule ("toward") and the part carried by syndromes where
the old decision did ("away"). The two splits are not the same, which is the point
of the figure: a gain of a given size does not imply a particular mechanism.

Everything is a percentage of the physical prior's risk, so the toward and away
contributions add to the total gain. The away contribution is negative at one
operating point, so the figure plots the total and the toward part as overlaid bars
rather than stacking them: where the toward bar overtops the total, the away moves
cost risk on net.

Run from the repository root:
    python -m simulation.prx.plotting.tikz.gen_attribution_fig
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "research" / "prx_quantum" / "results" / "theory"
OUT = ROOT / "manuscript" / "figdata"


def write(name, rows):
    path = OUT / f"fig_attribution_{name}.dat"
    with path.open("w") as fh:
        fh.write("x y\n")
        for x, y in rows:
            fh.write(f"{x:g} {y:.6f}\n")
    print(f"wrote {path.name} ({len(rows)} points)")


def main():
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(RES / "reweight_mechanism.csv")
    for T in (2, 3):
        sub = d[d["T"] == T].sort_values("p").reset_index(drop=True)
        idx = list(range(len(sub)))
        write(f"T{T}_search_total", [(i, 100 * g) for i, g in zip(idx, sub.gain_best)])
        write(f"T{T}_search_toward", [(i, 100 * g) for i, g in zip(idx, sub.gain_toward)])
        write(f"T{T}_limit_total", [(i, 100 * g) for i, g in zip(idx, sub.gain_limit)])
        write(f"T{T}_limit_toward", [(i, 100 * g) for i, g in zip(idx, sub.gain_limit_toward)])
        print(f"  T={T} rates {[f'{p:g}' for p in sub.p]}")


if __name__ == "__main__":
    main()
