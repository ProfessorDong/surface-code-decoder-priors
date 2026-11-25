import unittest

import numpy as np

from simulation.prx.dem.edges import EdgeTable
from simulation.prx.dem.priors import build_circuit
from simulation.prx.estimators.pooled import apply_signature_map, signature_map
from simulation.prx.suppression import average_eps, fit_eps_groups, lambda_regression


class TransferTests(unittest.TestCase):
    def test_signature_transfer_across_round_counts(self):
        src = EdgeTable.from_circuit(build_circuit("si1000", 3, 9, 0.003, variant="google2025"))
        dst = EdgeTable.from_circuit(build_circuit("si1000", 3, 15, 0.003, variant="google2025"))
        smap = signature_map(src)
        out, missing = apply_signature_map(dst, smap)
        self.assertEqual(missing, 0)
        # same physics at both round counts => transferred probabilities match the target's own
        self.assertTrue(np.allclose(out.p, dst.p, rtol=1e-6))

    def test_lambda_pipeline(self):
        eps = {3: 0.008, 5: 0.004, 7: 0.002}
        rows = []
        for d, e in eps.items():
            for g in ("pA/X", "pA/Z"):
                for r in (10, 30, 50):
                    p = 0.5 * (1 - (1 - 2 * e) ** r)
                    rows.append((g, d, r, int(round(p * 50000)), 50000))
        fits = fit_eps_groups(rows)
        avg = average_eps(fits)
        for d, e in eps.items():
            self.assertAlmostEqual(avg[d][0], e, delta=2e-5)
        lam, dlam = lambda_regression(avg)
        self.assertAlmostEqual(lam, 2.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
