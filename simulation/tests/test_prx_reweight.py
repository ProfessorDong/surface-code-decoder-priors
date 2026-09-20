import unittest

import numpy as np
import pymatching
import stim

from simulation.prx.dem.edges import EdgeTable
from simulation.prx.dem.priors import build_circuit
from simulation.prx.dem.reweight import reweight_dem
from simulation.prx.experiments.common import sample_shots, decode_errors


class ReweightTests(unittest.TestCase):
    def test_identity_when_unchanged(self):
        c = build_circuit("si1000", 3, 5, 0.003, variant="google2025")
        dem = c.detector_error_model(decompose_errors=True)
        t = EdgeTable.from_dem(dem)
        out = reweight_dem(dem, t, t)
        self.assertTrue(out.approx_equals(dem.flattened(), atol=1e-12))

    def test_scaling_propagates_and_keeps_structure(self):
        c = build_circuit("si1000", 3, 5, 0.003, variant="google2025")
        dem = c.detector_error_model(decompose_errors=True)
        t = EdgeTable.from_dem(dem)
        scaled = t.scaled(2.0)
        out = reweight_dem(dem, t, scaled)
        self.assertEqual(out.num_detectors, dem.num_detectors)
        self.assertEqual(len([i for i in out if i.type == "error"]), len([i for i in dem.flattened() if i.type == "error"]))
        # graphlike edges recovered from the new DEM are ~2x the originals
        t2 = EdgeTable.from_dem(out)
        ratio = np.median(t2.p / t.p)
        self.assertGreater(ratio, 1.8)
        self.assertLess(ratio, 2.2)
        # correlated matching runs on the reweighted DEM
        m = pymatching.Matching.from_detector_error_model(out, enable_correlations=True)
        d, o = sample_shots(c, 2000, seed=3)
        preds = m.decode_batch(d, bit_packed_shots=True, bit_packed_predictions=True, enable_correlations=True)
        self.assertEqual(preds.shape[0], 2000)


if __name__ == "__main__":
    unittest.main()
