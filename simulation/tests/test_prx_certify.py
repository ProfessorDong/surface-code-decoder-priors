"""Certified per-class minimum: the min-plus decision is matching's decision, the box
lower bound is sound, and the bound machinery reproduces the exact physical risk."""
import unittest

import numpy as np
import pandas as pd

from simulation.prx.config import RESULTS_ROOT
from simulation.prx.theory.certify_class_family import Problem, certify


class TestCertify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prob = Problem(2, 0.002)
        cls.rng = np.random.default_rng(7)

    def test_physical_risk_matches_exact_table(self):
        d = pd.read_csv(RESULTS_ROOT / "theory" / "exact_candidates_untruncated.csv")
        r = float(d[(d["T"] == 2) & np.isclose(d.p, 0.002)].R_phys.iloc[0])
        self.assertAlmostEqual(self.prob.r_phys / r, 1.0, places=12)
        self.assertAlmostEqual(self.prob.risk_matching(np.zeros(4)) / r, 1.0, places=12)

    def test_minplus_decision_is_matching_decision(self):
        for _ in range(5):
            u = self.rng.uniform(np.log(0.2), np.log(5), 4)
            self.assertAlmostEqual(self.prob.risk_upper(u), self.prob.risk_matching(u), places=15)

    def test_box_lower_bound_is_sound(self):
        for _ in range(20):
            c = self.rng.uniform(np.log(0.2), np.log(5), 4)
            h = self.rng.uniform(0.05, 1.0, 4)
            a, b = c - h, c + h
            lb, _ = self.prob.lower_bound(a, b, 1e-9)
            for _ in range(5):
                u = a + (b - a) * self.rng.uniform(0, 1, 4)
                self.assertLessEqual(lb, self.prob.risk_matching(u) + 1e-18)

    def test_certificate_brackets_a_grid(self):
        res = certify(self.prob, 0.5, 2.0, 1e-6, 1500, log=lambda *a: None)
        grid = np.log(np.geomspace(0.5, 2.0, 5))
        best = min(self.prob.risk_upper(np.array(u)) for u in np.stack(np.meshgrid(*[grid] * 4), -1).reshape(-1, 4))
        self.assertLessEqual(res["LB"], best + 1e-18)
        self.assertLessEqual(res["LB"], res["UB"])


if __name__ == "__main__":
    unittest.main()


class TestBranchRules(unittest.TestCase):
    """Both branching rules split inside the box and leave the certificate sound."""

    @classmethod
    def setUpClass(cls):
        from simulation.prx.theory.certify_class_family import Problem
        cls.prob = Problem(2, 0.002)

    def test_split_inside_box(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            a = rng.uniform(np.log(0.15), np.log(1.0), 4)
            b = a + rng.uniform(0.05, 1.5, 4)
            for rule in ("longest", "wspan"):
                k, mid = self.prob.split(a, b, rule)
                self.assertTrue(a[k] < mid < b[k], (rule, a[k], mid, b[k]))

    def test_rules_agree_on_the_certificate(self):
        from simulation.prx.theory.certify_class_family import certify
        out = {r: certify(self.prob, 0.3, 2.0, 1e-6, 400, rule=r, log=lambda *a, **k: None)
               for r in ("longest", "wspan")}
        for r, res in out.items():
            self.assertLessEqual(res["LB"], res["UB"] + 1e-18, r)
            self.assertLessEqual(res["LB"], self.prob.r_phys, r)
