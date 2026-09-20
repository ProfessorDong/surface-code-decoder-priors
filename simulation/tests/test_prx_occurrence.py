"""Occurrence (DGR alignment) estimator: recovers isolated-edge rates at low noise,
and every matched edge maps to a template edge."""
import unittest

import numpy as np

from simulation.prx.dem.edges import EdgeTable
from simulation.prx.estimators.occurrence import OccurrenceEstimator
from simulation.prx.experiments.common import sample_shots
from simulation.prx.theory.exact import repetition_memory


class TestOccurrence(unittest.TestCase):
    def test_recovers_rates_at_low_noise(self):
        circ = repetition_memory(5, 5, 0.002, 0.004)
        true = EdgeTable.from_dem(circ.detector_error_model())
        dets, _ = sample_shots(circ, 200_000, 11)
        fit = OccurrenceEstimator().fit(true.with_probabilities(np.full(true.num_edges, 0.01)), dets)
        self.assertEqual(fit.params["n_unmatched_edge_ids"], 0)
        # isolated errors dominate at this noise level: frequencies track the true rates
        ratio = fit.table.p / true.p
        self.assertLess(np.median(np.abs(np.log(ratio))), 0.1)

    def test_unseen_edges_keep_reference(self):
        circ = repetition_memory(3, 3, 0.001, 0.001)
        t = EdgeTable.from_dem(circ.detector_error_model())
        dets, _ = sample_shots(circ, 50, 3)
        fit = OccurrenceEstimator().fit(t, dets)
        unseen = fit.params["n_edges"] - fit.params["n_edges_seen"]
        self.assertGreater(unseen, 0)
        self.assertEqual(int(np.isclose(fit.table.p, t.p).sum()) >= unseen, True)


if __name__ == "__main__":
    unittest.main()
