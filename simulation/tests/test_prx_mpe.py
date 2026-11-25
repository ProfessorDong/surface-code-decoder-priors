"""The independent most-probable-error reference must agree with matching, and
must be able to disagree with it.

A check that cannot fail is worthless: an earlier version of the three-way split
compared matching with itself. These tests pin down both directions.
"""
import unittest

from simulation.prx.dem.edges import EdgeTable
from simulation.prx.dem.priors import build_circuit
from simulation.prx.theory.exact import enumerate_dem, mwpm_predictions
from simulation.prx.theory.mpe import check_matching_is_mpe, max_product
from simulation.prx.theory.three_way import decomposed_dem


class TestMaxProductMPE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dem = build_circuit("si1000", 3, 2, 1e-3, variant="google2025").detector_error_model(
            decompose_errors=True)
        cls.D = dem.num_detectors
        cls.table = EdgeTable.from_dem(dem)
        cls.dec = enumerate_dem(decomposed_dem(cls.table))

    def test_matching_agrees_with_max_product(self):
        pred = mwpm_predictions(self.dec, self.table.to_matching())
        r = check_matching_is_mpe(self.table, self.D, self.dec.syndromes, pred)
        self.assertEqual(r["n_disagree"], 0)
        self.assertEqual(r["n_checked"], len(self.dec.syndromes))

    def test_check_detects_a_wrong_prior(self):
        wrong = self.table.scaled_by_class({"diag": 3.0}).to_matching()
        r = check_matching_is_mpe(self.table, self.D, self.dec.syndromes, mwpm_predictions(self.dec, wrong))
        self.assertGreater(r["n_disagree"], 100)

    def test_reference_does_not_call_matching(self):
        import inspect
        from simulation.prx.theory import mpe
        src = inspect.getsource(mpe.max_product)
        self.assertNotIn("to_matching", src)
        self.assertNotIn("decode", src)

    def test_trivial_syndrome_is_trivial(self):
        ref = max_product(self.table, self.D)
        self.assertEqual(int(ref.syndromes[0]), 0)
        self.assertFalse(bool(ref.logical[0]))


if __name__ == "__main__":
    unittest.main()
