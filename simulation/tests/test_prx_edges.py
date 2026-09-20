import unittest

import numpy as np
import pymatching
import stim

from simulation.prx.dem.edges import EdgeClass, EdgeTable, join_probabilities
from simulation.prx.dem.priors import build_circuit, physical_prior, stale_prior


def _circuit(d, T, p=0.005):
    return build_circuit("legacy_from_physical_rate", d, T, p)


class EdgeTableTests(unittest.TestCase):
    def test_round_trip_predictions_identical(self):
        for d in (3, 5):
            c = _circuit(d, d)
            dem = c.detector_error_model(decompose_errors=True)
            ref = pymatching.Matching.from_detector_error_model(dem)
            table = EdgeTable.from_dem(dem)
            rebuilt = table.to_matching()
            self.assertEqual(rebuilt.num_edges, ref.num_edges)
            dets, obs = c.compile_detector_sampler(seed=11).sample(100_000, separate_observables=True, bit_packed=True)
            pa = ref.decode_batch(dets, bit_packed_shots=True, bit_packed_predictions=True)
            pb = rebuilt.decode_batch(dets, bit_packed_shots=True, bit_packed_predictions=True)
            self.assertTrue(np.array_equal(pa, pb))
            # unchanged probabilities => identical table
            same = table.with_probabilities(table.p)
            self.assertTrue(np.array_equal(same.p, table.p))

    def test_class_census_d5(self):
        table = EdgeTable.from_circuit(_circuit(5, 5))
        counts = table.class_counts()
        self.assertEqual(counts, {"space": 150, "time": 96, "diag": 184, "boundary": 72})
        logical_boundary = int(np.count_nonzero((table.cls == EdgeClass.BOUNDARY) & (table.fault_mask != 0)))
        self.assertEqual(logical_boundary, 18)

    def test_class_scaling_touches_only_that_class(self):
        table = EdgeTable.from_circuit(_circuit(3, 3))
        scaled = table.scaled_by_class({"time": 2.0})
        mask = table.cls == EdgeClass.TIME
        self.assertTrue(np.allclose(scaled.p[mask], np.minimum(2 * table.p[mask], 0.5 - 1e-9)))
        self.assertTrue(np.array_equal(scaled.p[~mask], table.p[~mask]))
        # weights move monotonically
        self.assertTrue(np.all(scaled.weight[mask] < table.weight[mask]))

    def test_join_and_stale_prior(self):
        phys = physical_prior("legacy_from_physical_rate", 3, 3, 0.01)
        stale = stale_prior("legacy_from_physical_rate", 3, 3, 0.005)
        self.assertEqual(phys.num_edges, stale.num_edges)
        joined = join_probabilities(phys, stale)
        self.assertTrue(np.allclose(joined, stale.p))
        self.assertTrue(np.all(stale.p < phys.p))


if __name__ == "__main__":
    unittest.main()
