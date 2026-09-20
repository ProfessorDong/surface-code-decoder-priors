# When calibration helps a surface-code matching decoder

Code and result tables for a study of *decoder priors*: what a prior fitted from
syndromes is worth to a minimum-weight matching decoder, for whom, and at what data
cost. Three strands:

1. **Exact theory.** The risk of matching under any prior is computed exactly, by XOR
   convolution over the detector error model, with no sampling. This gives the gain of
   the best per-class prior, the gap to maximum-likelihood decoding, and a
   branch-and-bound *certificate* bounding what the whole per-class family can achieve.
2. **Simulation.** Circuit-level SI1000 noise with per-qubit heterogeneity, and the
   shot-budget frontier: how many calibration shots a syndrome-only fit needs before it
   helps rather than harms, with shrinkage toward the uncalibrated prior.
3. **Hardware.** The same estimators applied to Google's Willow surface-code release
   (Zenodo record 13273331, not redistributed here), compared with the priors shipped
   with that data.

Author: Liang Dong (Baylor University, Department of Electrical and Computer
Engineering).

## Layout

```
simulation/prx/            the package: dem/ noise/ estimators/ experiments/
                           analysis/ theory/ plotting/
simulation/tests/          unit tests (test_prx_*.py)
research/prx_quantum/
  results/<experiment>/<run_id>/   per-run manifests and result tables
  preregistration/                 the shrinkage-tuning protocol and its hash
```

Every run directory holds a `manifest.json` recording seeds, grids, code versions and
configuration, beside the CSVs the analyses read.

## Running

The environment is Python 3.12 with `stim`, `pymatching`, `numpy`, `scipy`, `pandas`
and `matplotlib`.

```bash
# unit tests (pytest is not required)
python -m unittest discover -s simulation/tests -t . -p 'test_prx_*.py'

# exact risk of every candidate prior, two and three rounds
python -m simulation.prx.theory.exact_candidates

# certified bound over the continuous per-class family (two rounds)
python -m simulation.prx.theory.certify_class_family --rounds 2 --workers 16

# shot-budget frontier under per-qubit heterogeneity
python -m simulation.prx.experiments.frontier --profile main --workers 16 --label main

# hardware analysis (expects the Willow archive locally; see config.py)
python -m simulation.prx.experiments.willow --workers 16 --label primary
```

Long runs hold a `flock` so that two copies cannot race on the same output files.

## Data

The result tables in `research/prx_quantum/results` are complete: every number in the
accompanying paper is re-derived from them by a checking script that fails loudly on
any mismatch.

Two things are deliberately not in this repository:

* **Per-shot error vectors.** Each run also writes bit-packed per-shot decoder outcomes
  (tens of thousands of files, several hundred megabytes). They are needed only to
  recompute the paired statistics from scratch, and are available on request.
* **The Willow hardware archive**, which belongs to Google and is public at Zenodo
  record 13273331.

## License

Apache License 2.0; see `LICENSE`. If you use this code or these tables, please cite
the accompanying paper.
