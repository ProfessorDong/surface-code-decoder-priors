import unittest

import numpy as np

from simulation.prx.dem.edges import EdgeClass, EdgeTable
from simulation.prx.dem.priors import build_circuit
from simulation.prx.estimators.pij import PijEstimator, pij_from_moments, combine_odd
from simulation.prx.experiments.common import decode_errors, sample_shots
from simulation.prx.stats import clopper_pearson, discordant_counts, exact_mcnemar


class PijTests(unittest.TestCase):
    def test_formula_on_synthetic_pair(self):
        # two detectors sharing one edge with probability q, each also with a private edge r
        rng = np.random.default_rng(0)
        n = 400_000
        q, r = 0.01, 0.03
        e = rng.random(n) < q
        a = rng.random(n) < r
        b = rng.random(n) < r
        xi_ = e ^ a
        xj_ = e ^ b
        p = pij_from_moments(xi_.mean(), xj_.mean(), (xi_ & xj_).mean())
        self.assertAlmostEqual(float(p), q, delta=0.0015)
        self.assertAlmostEqual(combine_odd([0.1, 0.2]), 0.1 + 0.2 - 2 * 0.1 * 0.2, places=12)

    def test_recovers_dem_and_matches_physical_prior_ler(self):
        c = build_circuit("si1000", 3, 5, 0.003)
        phys = EdgeTable.from_circuit(c)
        cal_d, cal_o = sample_shots(c, 1_000_000, seed=5)
        res = PijEstimator().fit(phys, cal_d)
        est = res.table
        bulk = phys.v >= 0
        rel = np.abs(est.p[bulk] - phys.p[bulk]) / phys.p[bulk]
        # hyperedge decompositions bias a few edges; require the bulk median to be close
        self.assertLess(float(np.median(rel)), 0.15)
        self.assertLess(res.params["n_fallback_bulk"], 0.05 * bulk.sum())
        test_d, test_o = sample_shots(c, 500_000, seed=6)
        e_phys = decode_errors(phys.to_matching(), test_d, test_o)
        e_est = decode_errors(est.to_matching(), test_d, test_o)
        b, cc = discordant_counts(e_est, e_phys)
        # the fitted prior should not be significantly worse than the physical prior
        worse = (b - cc) / e_phys.size
        self.assertLess(worse, 0.02 * e_phys.mean())


if __name__ == "__main__":
    unittest.main()
