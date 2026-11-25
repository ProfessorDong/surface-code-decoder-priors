import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simulation.prx import results_io as R


class ResultsIOTests(unittest.TestCase):
    def test_run_dir_manifest_rows_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = R.new_run_dir("unit", root=Path(tmp), label="t")
            self.assertTrue((run / "errors").is_dir())
            man = R.write_manifest(run, {"grid": [1, 2, 3], "seed": np.int64(5)})
            data = json.loads(man.read_text())
            self.assertIn("code_hash", data)
            self.assertEqual(data["seed"], 5)
            w = R.RowWriter(run)
            row = {k: "" for k in R.SCHEMA}
            row.update(experiment="unit", distance=3, ler=0.01, estimator_params_json={"a": 1})
            w.write(row)
            with self.assertRaises(KeyError):
                w.write({"not_a_column": 1})
            df = R.read_rows(run / "rows.csv")
            self.assertEqual(len(df), 1)
            self.assertEqual(int(df.loc[0, "distance"]), 3)
            errs = np.random.default_rng(0).random(1001) < 0.3
            rel = R.save_error_vector(run, "x", errs)
            back = R.load_error_vector(run, rel)
            self.assertTrue(np.array_equal(errs, back))


if __name__ == "__main__":
    unittest.main()
