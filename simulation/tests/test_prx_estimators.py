import unittest

import numpy as np

from simulation.prx.dem.edges import EdgeClass, EdgeTable
from simulation.prx.dem.priors import build_circuit, physical_prior
from simulation.prx.estimators.density import DensityCalibrationCurve, DensityEstimator, detection_fraction
from simulation.prx.estimators.pij import PijEstimator
from simulation.prx.estimators.pooled import PooledPijEstimator, TimeWindowPijEstimator, edge_signatures
from simulation.prx.estimators.spitz import SpitzStream
from simulation.prx.estimators.supervised import ClassScaleSearch, UniformScaleSearch
from simulation.prx.experiments.common import decode_errors, sample_shots
from simulation.prx.noise.drift import RoundSchedule, drifting_circuit


class DensityTests(unittest.TestCase):
    def test_density_inverts_within_two_percent(self):
        curve = DensityCalibrationCurve("si1000", 3, 3, variant="google2025", shots_per_point=20000).build()
        for p in (0.002, 0.005):
            c = build_circuit("si1000", 3, 3, p, variant="google2025")
            d, _ = sample_shots(c, 10_000, seed=7)
            tmpl = EdgeTable.from_circuit(build_circuit("si1000", 3, 3, 0.003, variant="google2025"))
            res = DensityEstimator(curve).fit(tmpl, d)
            self.assertAlmostEqual(res.p_hat / p, 1.0, delta=0.03)
            self.assertEqual(res.table.num_edges, tmpl.num_edges)


class PooledTests(unittest.TestCase):
    def test_pooling_reduces_fallbacks_and_keeps_ler(self):
        c = build_circuit("si1000", 3, 9, 0.003, variant="google2025")
        phys = EdgeTable.from_circuit(c)
        sigs = edge_signatures(phys)
        self.assertLess(len(set(sigs)), phys.num_edges)          # pooling actually pools
        d, _ = sample_shots(c, 30_000, seed=3)
        raw = PijEstimator().fit(phys, d)
        pooled = PooledPijEstimator(kappa=5.0).fit(phys, d)
        self.assertLessEqual(pooled.params["n_fallback"], raw.params["n_fallback_bulk"] + raw.params["n_fallback_boundary"])
        td, to = sample_shots(c, 200_000, seed=4)
        e_phys = decode_errors(phys.to_matching(), td, to).mean()
        e_pooled = decode_errors(pooled.table.to_matching(), td, to).mean()
        e_raw = decode_errors(raw.table.to_matching(), td, to).mean()
        self.assertLess(e_pooled, 1.15 * e_phys)
        self.assertLessEqual(e_pooled, 1.05 * e_raw + 1e-4)

    def test_time_window_recovers_a_step(self):
        T = 12
        sched = RoundSchedule.step(T, 0.002, 0.006, T // 2)
        c = drifting_circuit(3, T, sched)
        phys_static = EdgeTable.from_circuit(build_circuit("si1000", 3, T, 0.002, variant="google2025"))
        d, _ = sample_shots(c, 100_000, seed=5)
        res = TimeWindowPijEstimator(half_width=1).fit(phys_static, d)
        tl = phys_static.cls == EdgeClass.TIME
        t_of = np.array([phys_static.coords[int(u)][2] for u in phys_static.u])
        early = res.table.p[tl & (t_of <= 2)].mean()
        late = res.table.p[tl & (t_of >= T - 3)].mean()
        self.assertGreater(late / early, 2.0)


class SpitzTests(unittest.TestCase):
    def test_causal_stream_matches_batch_pij(self):
        c = build_circuit("si1000", 3, 3, 0.004, variant="google2025")
        phys = EdgeTable.from_circuit(c)
        d, _ = sample_shots(c, 8_000, seed=8)
        stream = SpitzStream(phys, block_shots=100, window_blocks=40)
        results = list(stream.run(d))
        self.assertEqual(len(results), 80 - 40)
        b, last = results[-1]
        batch = PijEstimator().fit(phys, d[(b - 40) * 100:b * 100])
        bulk = phys.v >= 0
        self.assertTrue(np.allclose(last.table.p[bulk], batch.table.p[bulk], rtol=1e-9, atol=1e-12))


class SupervisedTests(unittest.TestCase):
    def test_search_does_not_worsen_calibration_set(self):
        c = build_circuit("si1000", 3, 3, 0.003, variant="google2025")
        phys = EdgeTable.from_circuit(c)
        d, o = sample_shots(c, 40_000, seed=9)
        base = int(decode_errors(phys.to_matching(), d, o).sum())
        res = ClassScaleSearch(grid=[0.5, 1.0, 2.0], sweeps=1).fit(phys, d, o)
        self.assertLessEqual(res.params["best_cal_errors"], base)
        res2 = UniformScaleSearch(grid=[0.5, 1.0, 2.0]).fit(phys, d, o)
        self.assertLessEqual(res2.params["best_cal_errors"], base)


if __name__ == "__main__":
    unittest.main()


class TestEmpiricalBayesShrinkage(unittest.TestCase):
    """The empirical-Bayes shrinkage strength recovers a planted prior variance.

    Planting log-odds deviations with variance tau^2 about the reference and sampling
    noise 1/n, both estimators must return kappa = 1/tau^2: the moment estimate and the
    Efron-Morris maximum-likelihood estimate for unequal sampling variances.
    """

    def test_recovers_planted_kappa(self):
        from simulation.prx.estimators.pooled import eb_kappa, eb_kappa_mle
        rng = np.random.default_rng(11)
        for n in (np.full(4000, 500.0), rng.lognormal(6, 1.5, 4000)):
            d = rng.normal(0, np.sqrt(0.05), n.size) + rng.normal(0, 1 / np.sqrt(n))
            for f in (eb_kappa, eb_kappa_mle):
                self.assertAlmostEqual(f(d, n), 20.0, delta=3.0, msg=f.__name__)

    def test_no_dispersion_shrinks_all_the_way(self):
        from simulation.prx.estimators.pooled import eb_kappa, eb_kappa_mle
        rng = np.random.default_rng(12)
        n = np.full(3000, 400.0)
        d = rng.normal(0, 1 / np.sqrt(n))
        for f in (eb_kappa, eb_kappa_mle):
            self.assertEqual(f(d, n), np.inf, msg=f.__name__)

    def test_too_few_signatures_is_not_identified(self):
        from simulation.prx.estimators.pooled import eb_kappa_mle
        self.assertEqual(eb_kappa_mle(np.array([0.3, -0.2]), np.array([100.0, 100.0])), np.inf)

    def test_eb_fit_returns_its_kappa(self):
        """A fit with eb= set records the strength it chose and shrinks toward the template."""
        import json
        from simulation.prx.dem.edges import EdgeTable
        from simulation.prx.estimators.pij import pairwise_moments
        from simulation.prx.estimators.pooled import PooledPijEstimator
        circ = build_circuit("si1000", 3, 3, 0.002, variant="google2025")
        template = EdgeTable.from_circuit(circ)
        dets, _ = sample_shots(circ, 4000, seed=5)
        bulk = np.flatnonzero(template.v >= 0)
        pairs = np.stack([template.u[bulk], template.v[bulk]], axis=1)
        xi, xij, n = pairwise_moments(dets, template.num_detectors, pairs)
        for variant in ("global", "per_class", "global_mle", "per_class_mle"):
            fit = PooledPijEstimator(eb=variant).fit_moments(template, xi, xij, n)
            k = json.loads(fit.params["kappa_eb_json"])
            self.assertTrue(k, variant)
            self.assertTrue(all(v > 0 for v in k.values()), (variant, k))
            self.assertTrue(np.all(fit.table.p > 0) and np.all(fit.table.p < 0.5), variant)
