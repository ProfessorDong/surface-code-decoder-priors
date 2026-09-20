# Protocol: tuning the shrinkage strength kappa (written before any tuning run)

Written 2026-09-19. The published heterogeneity frontier (`frontier/20260918T213127Z_f9ee6b_draws`)
used kappa = 5, a default fixed before the first run and never tuned. This protocol
tunes kappa on development devices that no published result uses and evaluates the
frozen choice against the default on the published (test) devices.

## Development set (tuning only)
* Disorder seeds 301-306 at sigma = 0.6 and 401-406 at sigma = 1.0. These seeds are
  disjoint from every seed used in the paper (frontier main 1-3; draws 101-106, 201-206;
  ten-draw study 1-10; measurement-only 1-2).
* Geometries as in the draws frontier: d in {3, 5, 7}, T in {d, 3d} (T = d only at d = 7),
  p in {1.5e-3, 3e-3}, SI1000 (google2025 variant), reference prior homogeneous at p/1.5.
  120 development conditions.
* Per condition: a calibration pool of 3e5 shots and 2e5 test shots, seeded from base seed
  20260919 with tags distinct from the published runs. Budgets N in {1e2, 3e2, 1e3, 3e3,
  1e4, 3e4, 1e5, 3e5}; calibration replicates from disjoint pool chunks, 5 up to 3e4, 3 at
  1e5, 1 at 3e5 (as in the published frontier).
* Candidate grid: kappa in {0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}; kappa = 0 is
  the unshrunk pooled fit. For each budget and replicate the pooled moments are computed
  once and every kappa is applied to the same moments.

## Selection criterion (primary, fixed now)
For each kappa: the mean, over the 120 development conditions and the eight budgets, of
the relative excess over the oracle, R_fit / R_oracle - 1, where R_fit pools errors over
the calibration replicates at that budget. kappa* = argmin. If two candidates are within
1e-4 of each other, the smaller kappa is chosen. No other criterion may change kappa*.

Reported alongside (not used for selection): the fraction of conditions in which the fit
is worse than the uncalibrated prior, per budget; the median excess per budget.

## Evaluation (test set)
kappa* is frozen and evaluated once on the 120 published conditions with the published
protocol (seeds, 5e5 test shots, calibration pools), which reproduces the published shots
exactly. The comparison with kappa = 5 is paired condition by condition. Reported: the
primary criterion for both, its difference with a 95% interval from resampling the six
disorder seeds per sigma (whole seeds), the probability of harm per budget, and the
Table V budgets. If kappa* = 5, the evaluation reduces to the published result.

## Hardware
The Willow analysis used kappa = 5 as fixed in the pre-run plan and stays primary. If
kappa* differs from 5, the hardware fit is repeated with kappa* as a labelled post hoc
sensitivity analysis, not as a replacement.
