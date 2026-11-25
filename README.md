# When Calibration Helps Surface-Code Matching Decoders

Code, result tables and number-verification script for the manuscript
**"When Calibration Helps Surface-Code Matching Decoders"** (Liang Dong).

[![data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24545303.svg)](https://doi.org/10.5281/zenodo.24545303)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Code and result tables for a study of *decoder priors*: what a prior fitted from
syndromes is worth to a minimum-weight matching decoder, for whom, and at what data
cost. Three strands:

1. **Exact theory.** The risk of matching under any prior is computed exactly, by XOR
   convolution over the detector error model, with no sampling. This gives the gain of
   the best per-class prior, the gap to maximum-likelihood decoding, and a
   branch-and-bound bound on what the whole per-class family can achieve. The bounding
   argument is a proof in real arithmetic; the computation runs in float64 without a
   propagated error enclosure, so the numbers it reports are numerically validated
   upper bounds rather than interval-enclosed ones.
2. **Simulation.** Circuit-level SI1000 noise with per-qubit heterogeneity, and the
   shot-budget frontier: how many calibration shots a syndrome-only fit needs before it
   helps rather than harms, with shrinkage toward the uncalibrated prior.
3. **Hardware.** The same estimators applied to Google's Willow surface-code release
   (Zenodo record 13273331, not redistributed here), compared with the priors shipped
   with that data.

Author: Liang Dong (Baylor University, Department of Electrical and Computer
Engineering), <Liang_Dong@baylor.edu>, ORCID
[0000-0002-8585-1087](https://orcid.org/0000-0002-8585-1087).

## Layout

```
simulation/prx/            the package: dem/ noise/ estimators/ experiments/
                           analysis/ theory/ plotting/
simulation/tests/          unit tests (test_prx_*.py)
research/results/<experiment>/<run_id>/   per-run manifests and result tables
research/results/theory/                  exact-evaluation tables
research/preregistration/                 the shrinkage-tuning protocol and its hash
manuscript/verify_numbers.py              recomputes every number in the paper
manuscript/make_tables.py                 rebuilds the paper's table bodies
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

# numerical bound over the continuous per-class family (two rounds)
python -m simulation.prx.theory.certify_class_family --rounds 2 --workers 16

# shot-budget frontier under per-qubit heterogeneity
python -m simulation.prx.experiments.frontier --profile main --workers 16 --label main

# hardware analysis (expects the Willow archive locally; see config.py)
python -m simulation.prx.experiments.willow --workers 16 --label primary
```

Three smaller scripts establish the claims the analyses rest on, and write their
own tables under `research/results/theory`:

```bash
# what the graphlike projection discards, over every model the study uses
python -m simulation.prx.theory.decomposition_census

# the pairwise estimator's population target, against the estimator itself
python -m simulation.prx.theory.population_target

# what the best prior changes, decision by decision, on every syndrome
python -m simulation.prx.theory.reweight_mechanism
```

Long runs hold a `flock` so that two copies cannot race on the same output files.

## Data

The result tables in `research/results` are complete: every number in the
accompanying paper is re-derived from them by a checking script that fails loudly on
any mismatch. That script is included, and runs without the paper:

```bash
python manuscript/verify_numbers.py     # recomputes every quoted number from the tables
python manuscript/make_tables.py        # rebuilds the paper's table bodies from the same data
```

No number in the paper is typed by hand; each is either recomputed here or the check
fails. The manuscript source itself is not distributed, so the checks that also require
a quoted phrase to appear in the text report themselves as skipped.

### Where each result is archived

The same map is Table S1 of the paper's Supplementary Material. Paths are under
`research/results/`.

| Result in the paper | Files and run directories |
|---|---|
| Largest applied channel probability (Sec. II) | `theory/max_channel_probability.json` |
| Decomposition census (Sec. II, Sec. S4-A) | `theory/decomposition_census.csv` |
| Weight and probability rescaling, Fig. 2 (Sec. III-A) | `theory/offset_check.json`; `prior_sweep/20260917T175304Z_d2f74a_gate0_deep/`; `prior_sweep/20260917T142535Z_fbc712_gate0/` |
| Table 2, Figs. 3 and S1, Sec. S2 | `theory/exact_candidates_untruncated.csv`; `theory/certified_class_family_T2_wspan.csv`; `theory/certified_class_family_T3_gpu.csv`; `theory/moment_limit_exact.csv`; `theory/exact_at_mc_scales.csv`; `theory/rep_untruncated/` |
| Table 3 and the check of Lemma 1 | `theory/three_way_surface_d3_T2.csv`; `theory/three_way_surface_d3_T3.csv`; `theory/three_way_repetition_d5_T3.csv` |
| Fig. 5 | `theory/reweight_mechanism.csv` |
| Table 4 | `hetero_seeds/20260918T024551Z_b76ee0_main/` |
| Fig. 6, Table 5, population limits (Sec. IV) | `frontier/20260918T213127Z_f9ee6b_draws/`; `frontier/20260919T131450Z_4e7c86_draws_kappa20/`; `frontier/20260919T103005Z_de00e4_draws_dgr/`; `theory/dgr_limit_exact.csv`; `theory/limit_agreement.csv` |
| Measurement-only heterogeneity (Sec. S6) | `frontier/20260918T054857Z_6e4d15_device_like/` |
| Shrinkage strength (Sec. IV, Fig. S3) | `kappa_dev/20260919T114029Z_c1ecbd_dev/`; `kappa_dev/20260920T022238Z_d8a61a_dev/`; `eb_variance/20261024T021839Z_478095_check/`; protocol in `research/preregistration/` |
| Figs. 7 and 8 (Sec. V) | `willow/20260919T084410Z_c3ed43_primary_allvec/` (repeat of `willow/20260917T161601Z_62b495_primary/` with every per-shot vector saved); `willow_timelike_spread/` |
| Fig. 9 | `willow_factorial/20260917T231403Z_cfca17_primary/` |
| Population target, Corollary S1 (Sec. S5) | `theory/population_target_identity.csv` |

Two things are deliberately not in this repository:

* **Per-shot error vectors.** Each run also writes bit-packed per-shot decoder outcomes
  (40,437 files, 571 MB compressed). They are needed only to recompute the paired
  statistics from scratch, and are deposited separately at
  [10.5281/zenodo.24545303](https://doi.org/10.5281/zenodo.24545303), with the record
  format documented there. Unpacking that archive from the repository root,

  ```bash
  tar -xzf per_shot_records.tar.gz
  ```

  puts every record into its run directory under `research/results/`.
* **The Willow hardware archive**, which belongs to Google and is public at Zenodo
  record 13273331.

## License

Apache License 2.0 for this repository; see `LICENSE`. The separately deposited
per-shot records are CC-BY-4.0. If you use this code or these tables, please cite the
accompanying paper; see `CITATION.cff`.
