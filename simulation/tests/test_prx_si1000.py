import unittest
import zipfile

import numpy as np
import stim

from simulation.prx.config import WILLOW_ZIP
from simulation.prx.noise.si1000 import (SI1000Rates, inject_si1000, lognormal_qubit_scales,
                                         noise_op_histogram, noiseless_memory_circuit, strip_noise)
from simulation.prx.noise.drift import RoundSchedule, drifting_circuit, parse_round_rates, scale_noise_by_round
from simulation.prx.dem.priors import build_circuit


def _willow_circuits(leaf: str):
    with zipfile.ZipFile(WILLOW_ZIP) as z:
        base = f"google_105Q_surface_code_d3_d5_d7/{leaf}/"
        ideal = stim.Circuit(z.read(base + "circuit_ideal.stim").decode())
        noisy = stim.Circuit(z.read(base + "circuit_noisy_si1000.stim").decode())
    return ideal, noisy


class SI1000Tests(unittest.TestCase):
    @unittest.skipUnless(WILLOW_ZIP.exists(), "Willow archive absent")
    def test_reproduces_google_noisy_circuits(self):
        for leaf in ("d3_at_q6_7/Z/r13", "d5_at_q4_7/X/r10", "d7_at_q6_7/Z/r13"):
            ideal, noisy = _willow_circuits(leaf)
            ours = inject_si1000(ideal, SI1000Rates.from_variant("google2025", 0.001))
            self.assertEqual(noise_op_histogram(ours), noise_op_histogram(noisy), leaf)
            dem_ref = noisy.detector_error_model(decompose_errors=True)
            dem_ours = ours.detector_error_model(decompose_errors=True)
            self.assertEqual(dem_ours.num_detectors, dem_ref.num_detectors)
            self.assertTrue(dem_ours.approx_equals(dem_ref, atol=1e-9), leaf)

    def test_generated_circuit_rules_and_strip(self):
        c = noiseless_memory_circuit(3, 3)
        noisy = inject_si1000(c, SI1000Rates.from_variant("google2025", 0.002))
        hist = noise_op_histogram(noisy)
        # two-qubit channels on every CX pair: 4 CX layers x 12 targets per round-ish; just check presence
        self.assertIn(("DEPOLARIZE2", (0.002,)), hist)
        self.assertIn(("MR", (0.01,)), hist)          # 5p measurement flips
        self.assertIn(("X_ERROR", (0.004,)), hist)    # 2p reset flips
        self.assertIn(("DEPOLARIZE1", (0.0002,)), hist)
        self.assertIn(("DEPOLARIZE1", (0.004,)), hist)
        self.assertEqual(noisy.num_detectors, c.num_detectors)
        self.assertEqual(noisy.num_observables, c.num_observables)
        # stripping the noise gives back the ideal structure
        self.assertEqual(str(strip_noise(noisy)), str(strip_noise(c)))
        # registry family
        fam = build_circuit("si1000", 3, 3, 0.002, variant="google2025")
        self.assertTrue(fam.detector_error_model(decompose_errors=True).approx_equals(
            noisy.detector_error_model(decompose_errors=True), atol=1e-12))

    def test_heterogeneous_scales_change_only_touched_qubits(self):
        c = noiseless_memory_circuit(3, 3)
        base = SI1000Rates.from_variant("google2025", 0.002)
        scales = lognormal_qubit_scales(c.num_qubits, 0.5, seed=3)
        het = inject_si1000(c, base, qubit_scale=scales)
        hom = inject_si1000(c, base)
        self.assertEqual(het.num_detectors, hom.num_detectors)
        # single-qubit depolarization on qubit q carries factor scale[q]
        found = 0
        for ins in het.flattened():
            if ins.name == "DEPOLARIZE1":
                q = ins.targets_copy()[0].value
                a = ins.gate_args_copy()[0]
                ratio = a / (base.one_qubit * scales[q])
                if abs(ratio - 1) < 1e-9 or abs(a / (base.idle_during_mr * scales[q]) - 1) < 1e-9 or abs(a / (base.post_meas_depol * scales[q]) - 1) < 1e-9:
                    found += 1
                else:
                    self.fail(f"unexpected DEPOLARIZE1 arg {a} on qubit {q}")
        self.assertGreater(found, 0)


class DriftTests(unittest.TestCase):
    def test_round_indexed_schedule_round_trips(self):
        for d, T in ((3, 25), (5, 50)):
            for sched in (RoundSchedule.sinusoid(T, 0.005, 0.003, 20.0, onset=5),
                          RoundSchedule.step(T, 0.005, 0.01, T // 2),
                          RoundSchedule.burst(T, 0.005, 0.02, 7, 4),
                          RoundSchedule.ornstein_uhlenbeck(T, 0.005, 0.3, 10.0, seed=1)):
                c = drifting_circuit(d, T, sched)
                got = parse_round_rates(c, "DEPOLARIZE2")
                # Stim's generator places the final data measurement in the last ancilla-measurement layer.
                self.assertEqual(got.size, T)
                self.assertTrue(np.allclose(got, sched.values, rtol=1e-12))
                self.assertEqual(c.num_detectors, noiseless_memory_circuit(d, T).num_detectors)

    def test_scale_noise_by_round_any_family(self):
        T = 10
        noisy = build_circuit("legacy_from_physical_rate", 3, T, 0.005)
        factors = [1.0] * 5 + [2.0] * 5
        scaled = scale_noise_by_round(noisy, factors)
        got = parse_round_rates(scaled, "DEPOLARIZE2")
        self.assertTrue(np.allclose(got[:5], 0.005) and np.allclose(got[5:10], 0.01))
        # DEM of a step circuit: time-like edges late in the circuit are heavier than early ones
        dem = scaled.detector_error_model(decompose_errors=True)
        self.assertEqual(dem.num_detectors, noisy.num_detectors)

    def test_legacy_drift_function_is_not_used_here(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1] / "prx"
        for path in root.rglob("*.py"):
            txt = path.read_text()
            self.assertNotIn("apply_temporal_drift(", txt, path)
            self.assertNotIn("import apply_temporal_drift", txt, path)


if __name__ == "__main__":
    unittest.main()


class ChannelHeterogeneityTests(unittest.TestCase):
    def test_channel_restricted_scaling(self):
        from simulation.prx.dem.edges import EdgeClass, EdgeTable
        c = noiseless_memory_circuit(5, 5)
        base = SI1000Rates.from_variant("google2025", 0.002)
        scales = lognormal_qubit_scales(c.num_qubits, 0.8, seed=5)
        meas_only = inject_si1000(c, base, qubit_scale={"meas_flip": scales})
        flat = inject_si1000(c, base)
        # only measurement-flip arguments differ from the homogeneous circuit
        args_meas = sorted(round(i.gate_args_copy()[0], 12) for i in meas_only.flattened()
                           if i.name == "MR" and i.gate_args_copy())
        args_flat = sorted(round(i.gate_args_copy()[0], 12) for i in flat.flattened()
                           if i.name == "MR" and i.gate_args_copy())
        self.assertNotEqual(args_meas, args_flat)
        dep_meas = sorted(round(i.gate_args_copy()[0], 12) for i in meas_only.flattened() if i.name == "DEPOLARIZE2")
        dep_flat = sorted(round(i.gate_args_copy()[0], 12) for i in flat.flattened() if i.name == "DEPOLARIZE2")
        self.assertEqual(dep_meas, dep_flat)
        # the time-like edge class picks up the spread, as on hardware
        t_meas = EdgeTable.from_circuit(meas_only)
        t_flat = EdgeTable.from_circuit(flat)
        sd = lambda t: float(np.log(t.p[t.cls == EdgeClass.TIME]).std())
        self.assertGreater(sd(t_meas), 3 * sd(t_flat))
        with self.assertRaises(KeyError):
            inject_si1000(c, base, qubit_scale={"not_a_channel": scales})
