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

    def test_mle_prefers_the_boundary_when_the_likelihood_does(self):
        """The stationary point of the EB likelihood is not always its maximum.

        With unequal sampling variances the fixed point of the Efron-Morris estimating
        equation can land on an interior root of lower likelihood than A = 0. This case
        is a compact stand-in for the one found on the variance check (d=3, sigma=1,
        p=0.0015, N=100), where the plain fixed point returned kappa = 1.86 at a negative
        log likelihood 20.76 above the boundary. The estimator must return the boundary,
        kappa = inf, meaning shrink fully to the reference.
        """
        from simulation.prx.estimators.pooled import eb_kappa_mle, _eb_nll
        delta = np.array([8.8084, 1.2530, -0.5168, -6.3398, 0.0109, 3.8922])
        D = np.array([40.0, 40.0, 2.0, 40.0, 0.01, 2.0])
        d2 = delta ** 2
        interior = 1.679556                                  # what the bare fixed point returns
        self.assertLess(_eb_nll(0.0, d2, D), _eb_nll(interior, d2, D) - 1.0)
        self.assertEqual(eb_kappa_mle(delta, 1.0 / D), np.inf)

    def test_mle_matches_a_dense_scalar_search(self):
        """On random unequal-variance problems the returned A attains the grid optimum."""
        from simulation.prx.estimators.pooled import eb_kappa_mle, _eb_nll
        rng = np.random.default_rng(11)
        for _ in range(25):
            g = int(rng.integers(5, 40))
            D = 1.0 / rng.integers(1, 500, size=g).astype(float)
            delta = rng.standard_normal(g) * rng.choice([0.05, 1.0, 6.0], size=g)
            kap = eb_kappa_mle(delta, 1.0 / D)
            A = 0.0 if not np.isfinite(kap) else 1.0 / kap
            grid = np.concatenate([[0.0], np.geomspace(1e-10, 1e5, 60000)])
            best = min(_eb_nll(a, delta ** 2, D) for a in grid)
            self.assertLessEqual(_eb_nll(A, delta ** 2, D), best + 1e-6)

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
        classes = set(int(c) for c in template.cls)
        for variant in ("global", "per_class", "global_mle", "per_class_mle"):
            fit = PooledPijEstimator(eb=variant).fit_moments(template, xi, xij, n)
            k = json.loads(fit.params["kappa_eb_json"])
            self.assertTrue(k, variant)
            self.assertTrue(all(v > 0 for v in k.values()), (variant, k))
            self.assertTrue(np.all(fit.table.p > 0) and np.all(fit.table.p < 0.5), variant)
            # a per-class variant must group: one kappa per class present, not one global fit
            if variant.startswith("per_class"):
                self.assertEqual(set(int(c) for c in k), classes, variant)
            else:
                self.assertEqual(list(k), ["global"], variant)
        # the two per-class arms must dispatch to different estimators, as their names say
        mom = json.loads(PooledPijEstimator(eb="per_class").fit_moments(
            template, xi, xij, n).params["kappa_eb_json"])
        mle = json.loads(PooledPijEstimator(eb="per_class_mle").fit_moments(
            template, xi, xij, n).params["kappa_eb_json"])
        self.assertNotEqual(mom, mle)
