import unittest

import numpy as np

from simulation.prx import stats as S
from simulation.evaluation.metrics import clopper_pearson_ci


class StatsTests(unittest.TestCase):
    def test_clopper_pearson_matches_legacy_and_edges(self):
        for k, n in [(0, 100), (100, 100), (7, 250), (1, 5)]:
            lo, hi = S.clopper_pearson(k, n)
            lo2, hi2 = clopper_pearson_ci(k, n)
            self.assertAlmostEqual(lo, lo2, places=12)
            self.assertAlmostEqual(hi, hi2, places=12)
        self.assertEqual(S.clopper_pearson(0, 10)[0], 0.0)
        self.assertEqual(S.clopper_pearson(10, 10)[1], 1.0)

    def test_mcnemar_symmetric_and_exact(self):
        self.assertEqual(S.exact_mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(S.exact_mcnemar(5, 5), 1.0)
        p = S.exact_mcnemar(40, 10)
        self.assertLess(p, 1e-4)
        a = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
        b = np.array([1, 0, 0, 1, 0, 0], dtype=bool)
        self.assertEqual(S.discordant_counts(a, b), (2, 1))

    def test_paired_delta_ci_contains_truth(self):
        rng = np.random.default_rng(0)
        n = 200_000
        a = rng.random(n) < 0.05
        b = a.copy()
        flip = rng.random(n) < 0.01
        b[flip] = ~b[flip]
        d, lo, hi = S.paired_delta_ci(a, b)
        self.assertLessEqual(lo, d)
        self.assertGreaterEqual(hi, d)
        self.assertLess(hi - lo, 0.005)

    def test_block_and_cluster_bootstrap_run(self):
        rng = np.random.default_rng(1)
        a = rng.random(5000) < 0.1
        b = rng.random(5000) < 0.1
        d, lo, hi = S.block_bootstrap_delta(a, b, block_len=50, n_boot=200, seed=2)
        self.assertLessEqual(lo, hi)
        pairs = [(rng.random(1000) < 0.1, rng.random(1000) < 0.1) for _ in range(5)]
        d2, lo2, hi2 = S.cluster_bootstrap_delta(pairs, n_boot=200, seed=3)
        self.assertLessEqual(lo2, hi2)

    def test_holm(self):
        adj = S.holm([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adj[0], 0.03)
        self.assertTrue(all(0 <= x <= 1 for x in adj))
        self.assertEqual(S.holm([]), [])

    def test_per_round_and_fit(self):
        eps = 0.01
        rounds = np.array([10, 30, 50, 70])
        p = 0.5 * (1 - (1 - 2 * eps) ** rounds)
        self.assertAlmostEqual(S.per_round_ler_google(p[1], 30), eps, places=12)
        fit, se, n = S.fit_eps_per_round(rounds, p, n_shots=[50000] * 4)
        self.assertAlmostEqual(fit, eps, places=9)
        self.assertEqual(n, 4)
        self.assertAlmostEqual(S.lambda_factor(0.01, 0.005), 2.0)
        self.assertGreater(S.mcnemar_shots_required(0.002, 2.8e-4), 1e5)


if __name__ == "__main__":
    unittest.main()
