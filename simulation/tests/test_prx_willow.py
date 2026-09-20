import unittest

import numpy as np
import pymatching
import stim

from simulation.prx.config import WILLOW_ZIP, WILLOW_EXTRACTED
from simulation.prx.willow.reader import PATHWAYS, LeafKey, WillowArchive, google_ler


@unittest.skipUnless(WILLOW_ZIP.exists(), "Willow archive absent")
class WillowReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ar = WillowArchive()

    def test_inventory(self):
        leaves = self.ar.leaves()
        self.assertEqual(len(leaves), 420)
        self.assertEqual(len(self.ar.leaves(distance=3)), 9 * 2 * 15)
        self.assertEqual(len(self.ar.leaves(distance=7)), 30)
        counts = {pw: sum(self.ar.has_pathway(k, pw) for k in leaves) for pw in PATHWAYS}
        self.assertEqual(counts["libra_decoder_with_rl_optimized_prior"], 364)
        self.assertTrue(all(counts[pw] == 420 for pw in PATHWAYS[:4]))

    def test_matches_extracted_leaf_and_google_ler(self):
        key = LeafKey(3, "d3_at_q4_5", "X", 13)
        dets = self.ar.detection_events(key)
        self.assertEqual(dets.shape[0], 50000)
        if WILLOW_EXTRACTED.exists():
            leaf = WILLOW_EXTRACTED / key.patch / key.basis / f"r{key.rounds}"
            circuit = stim.Circuit.from_file(str(leaf / "circuit_noisy_si1000.stim"))
            ref = stim.read_shot_data_file(path=str(leaf / "detection_events.b8"), format="b8",
                                           num_detectors=circuit.num_detectors, num_observables=0, bit_packed=True)
            self.assertTrue(np.array_equal(ref, dets))
        actual = self.ar.obs_actual(key)
        k, n = google_ler(self.ar.obs_predicted(key, "correlated_matching_decoder_with_si1000_prior"), actual)
        self.assertEqual(n, 50000)
        self.assertAlmostEqual(k / n, 0.0938, places=3)
        k2, _ = google_ler(self.ar.obs_predicted(key, "correlated_matching_decoder_with_rl_optimized_prior"), actual)
        self.assertAlmostEqual(k2 / n, 0.0839, places=3)
        # plain PyMatching on the shipped DEMs
        for pw, expect in (("correlated_matching_decoder_with_si1000_prior", 0.1114),
                           ("correlated_matching_decoder_with_rl_optimized_prior", 0.0999)):
            m = pymatching.Matching.from_detector_error_model(self.ar.dem(key, pw))
            preds = m.decode_batch(dets, bit_packed_shots=True, bit_packed_predictions=True)
            kk, nn = google_ler(preds, actual)
            self.assertAlmostEqual(kk / nn, expect, places=3)

    def test_d7_r250_memory_bound(self):
        key = LeafKey(7, "d7_at_q6_7", "Z", 250)
        dets = self.ar.detection_events(key)
        self.assertEqual(dets.shape, (50000, 1500))
        self.assertLess(dets.nbytes, 100e6)


if __name__ == "__main__":
    unittest.main()
