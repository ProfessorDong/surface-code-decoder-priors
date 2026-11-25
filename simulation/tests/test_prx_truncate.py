"""truncate(tol=0) must keep every enumerated syndrome."""
import unittest

from simulation.prx.theory.exact import enumerate_dem, repetition_memory, truncate


class TestTruncateZero(unittest.TestCase):
    def test_zero_tolerance_keeps_everything(self):
        dem = repetition_memory(3, 7, 0.002, 0.002).detector_error_model()
        full = enumerate_dem(dem)
        self.assertEqual(len(full.syndromes), 2 ** dem.num_detectors)
        self.assertEqual(len(truncate(full, 0.0).bits), len(full.syndromes))

    def test_positive_tolerance_still_truncates(self):
        full = enumerate_dem(repetition_memory(3, 7, 0.002, 0.002).detector_error_model())
        self.assertLess(len(truncate(full, 1e-9).bits), len(full.syndromes))


if __name__ == "__main__":
    unittest.main()
