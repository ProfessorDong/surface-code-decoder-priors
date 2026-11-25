import unittest

import numpy as np
import stim

from simulation.prx.dem.edges import EdgeTable
from simulation.prx.theory.exact import (ExactModel, enumerate_dem, mwpm_predictions, repetition_memory, risk)


class ExactTests(unittest.TestCase):
    def test_enumeration_is_a_probability_distribution(self):
        c = repetition_memory(3, 2, 0.02, 0.02)
        m = enumerate_dem(c.detector_error_model())
        self.assertAlmostEqual(m.total_probability, 1.0, places=12)
        self.assertLessEqual(m.ml_failure(), risk(m, EdgeTable.from_dem(c.detector_error_model())) + 1e-15)

    def test_matches_monte_carlo(self):
        c = repetition_memory(3, 3, 0.03, 0.03)
        dem = c.detector_error_model()
        m = enumerate_dem(dem)
        table = EdgeTable.from_dem(dem)
        exact = risk(m, table)
        d, o = c.compile_detector_sampler(seed=7).sample(400_000, separate_observables=True, bit_packed=True)
        pred = table.to_matching().decode_batch(d, bit_packed_shots=True, bit_packed_predictions=True)
        mc = float((((pred[:, 0] ^ o[:, 0]) & 1) != 0).mean())
        self.assertAlmostEqual(exact, mc, delta=4 * np.sqrt(mc * (1 - mc) / 400_000))


if __name__ == "__main__":
    unittest.main()


class TestConvolutionAgainstBruteForce(unittest.TestCase):
    """Proposition 1 checked against complete 2^M enumeration of mechanism subsets."""

    @staticmethod
    def brute(dem):
        dem = dem.flattened()
        probs, masks, logs = [], [], []
        for ins in dem:
            if ins.type != "error":
                continue
            m = l = 0
            for t in ins.targets_copy():
                if t.is_relative_detector_id():
                    m ^= 1 << t.val
                elif t.is_logical_observable_id():
                    l ^= 1
            probs.append(ins.args_copy()[0]); masks.append(m); logs.append(l)
        M = len(probs)
        assert M <= 20, M
        dist = {}
        for bits in range(1 << M):
            pr, s, l = 1.0, 0, 0
            for i in range(M):
                if (bits >> i) & 1:
                    pr *= probs[i]; s ^= masks[i]; l ^= logs[i]
                else:
                    pr *= 1 - probs[i]
            dist[(s, l)] = dist.get((s, l), 0.0) + pr
        return dist

    def test_two_small_models(self):
        for d, T, pd_, pm in [(3, 1, 0.05, 0.07), (3, 2, 0.03, 0.11)]:
            dem = repetition_memory(d, T, pd_, pm).detector_error_model()
            ref = self.brute(dem)
            em = enumerate_dem(dem)
            got = {}
            for s, a, b in zip(em.syndromes, em.p_logical0, em.p_logical1):
                got[(int(s), 0)] = a
                got[(int(s), 1)] = b
            for key in set(ref) | set(got):
                self.assertAlmostEqual(ref.get(key, 0.0), got.get(key, 0.0), delta=1e-15)
