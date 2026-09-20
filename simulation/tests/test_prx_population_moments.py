"""Exact population moments: agreement with sampling, and exactness of the pairwise
estimator where its formula is exact (a graphlike model)."""
import unittest

import numpy as np

from simulation.prx.dem.edges import EdgeTable
from simulation.prx.estimators.pij import PijEstimator, pairwise_moments
from simulation.prx.experiments.common import Condition, build_condition_circuit, sample_shots
from simulation.prx.theory.exact import repetition_memory
from simulation.prx.theory.population_moments import population_moments


class TestPopulationMoments(unittest.TestCase):
    def test_moments_match_sampling(self):
        cond = Condition("si1000", 3, 3, 0.003, extra={"variant": "google2025", "sigma_h": 1.0, "het_seed": 201})
        circ = build_condition_circuit(cond)
        dem = circ.detector_error_model(decompose_errors=False)
        table = EdgeTable.from_circuit(circ)
        bulk = np.flatnonzero(table.v >= 0)
        pairs = np.stack([table.u[bulk], table.v[bulk]], axis=1)
        xi, xij = population_moments(dem, pairs)
        n = 400_000
        dets, _ = sample_shots(circ, n, 7)
        mi, mij, _ = pairwise_moments(dets, dem.num_detectors, pairs)
        # binomial standard errors; 6 sigma over a few hundred comparisons
        se_i = np.sqrt(xi * (1 - xi) / n)
        se_ij = np.sqrt(xij * (1 - xij) / n)
        self.assertLess(np.max(np.abs(mi - xi) / se_i), 6.0)
        self.assertLess(np.max(np.abs(mij - xij) / np.maximum(se_ij, 1e-12)), 6.0)

    def test_pairwise_estimator_is_exact_on_a_graphlike_model(self):
        circ = repetition_memory(5, 5, 0.01, 0.02)          # every mechanism flips at most two detectors
        dem = circ.detector_error_model()
        table = EdgeTable.from_dem(dem)
        bulk = np.flatnonzero(table.v >= 0)
        pairs = np.stack([table.u[bulk], table.v[bulk]], axis=1)
        xi, xij = population_moments(dem, pairs)
        fit = PijEstimator().fit_moments(table.with_probabilities(np.full(table.num_edges, 0.1)), xi, xij, np.inf)
        np.testing.assert_allclose(fit.table.p, table.p, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
