"""Does the homogeneous-prior penalty track the worst qubit? (ten-draw study, Sec. IV)

The multiplier map is drawn over every index below Stim's ``num_qubits``, but a
generated rotated-surface-code circuit does not use every index. The worst qubit is
therefore taken over the qubits the noiseless circuit acts on. Spearman's rho is
tested by exact enumeration of all 10! permutations of the ranks.
"""
from __future__ import annotations
import itertools, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from ..config import RESULTS_ROOT
from ..experiments.hetero_seeds import noiseless_memory_circuit, qubit_scales


def main() -> None:
    run = sorted((RESULTS_ROOT / "hetero_seeds").glob("*_main"))[-1]
    h = pd.read_csv(run / "rows.csv"); h["x"] = h.noise_extra_json.map(json.loads)
    h["sig"] = h.x.map(lambda e: e["sigma"]); h["mp"] = h.x.map(lambda e: e["mean_preserving"])
    h["seed"] = h.x.map(lambda e: e["seed"]); h["pen"] = h.estimator_params_json.map(lambda s: json.loads(s)["penalty_vs_oracle"])
    out = []
    for sig in (0.6, 1.0):
        for d in (3, 5, 7):
            g = h[(h.sig == sig) & (~h.mp) & (h.distance == d)].sort_values("seed")
            circ = noiseless_memory_circuit(d, int(g.rounds.iloc[0]))
            used = sorted({t.value for ins in circ.flattened() for t in ins.targets_copy() if t.is_qubit_target})
            act = [max(qubit_scales(circ.num_qubits, sig, int(s), False)[q] for q in used) for s in g.seed]
            allm = [e["lambda_max"] for e in g.x]
            ra, rp = rankdata(act), rankdata(g.pen.values)
            rho = float(np.corrcoef(ra, rp)[0, 1])
            n = len(ra)
            perms = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
            rap = ra[perms]; rpc = rp - rp.mean(); rapc = rap - rap.mean(1, keepdims=True)
            rhos = (rapc * rpc).sum(1) / np.sqrt((rapc ** 2).sum(1) * (rpc ** 2).sum())
            out.append({"sigma": sig, "distance": d, "n": n, "active_qubits": len(used), "all_indices": circ.num_qubits,
                        "rho_all_indices": float(np.corrcoef(rankdata(allm), rp)[0, 1]), "rho_active": rho,
                        "p_exact_two_sided": float(np.mean(np.abs(rhos) >= abs(rho) - 1e-12)),
                        "argmax_on_unused_index": int(sum(abs(a - b) > 1e-12 for a, b in zip(act, allm)))})
    df = pd.DataFrame(out); df.to_csv(run / "worst_qubit_active.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
