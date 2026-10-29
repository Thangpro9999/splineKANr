# Reproducing the results of splineKANr

Every number in the validation report, the benchmark tables and the
application can be regenerated with the scripts below. All scripts fix their
random seeds; none of them changes a threshold to make a check pass.
Locate the installed copies with
`system.file("validation", package = "splineKANr")` (likewise
`"benchmarks"` and `"applications"`).

## Environment used for the recorded results

| Component | Version |
|---|---|
| R | 4.5.3 (Linux), 4.6.0 (Windows 11, torch backend) |
| torch (R) | 0.17.0 (libtorch 2.8.0) |
| mgcv / nnet / ranger | 1.9.4 / 7.3.21 / 0.18.0 |
| plspm (cross-check only) | 0.6.0 (R 4.4) |
| Python / pykan / PyTorch / NumPy | 3.11 / 0.2.8 / 2.14.1 / 2.4.6 |

Record yours with `sessionInfo()` and `pip freeze`.

## 1. Numerical checks of the B-spline core (Phases 1-2)

```r
res <- splineKANr::kan_validate_all(backend = "both")   # "r" without libtorch
attr(res, "all_pass")
```

`inst/validation/run_validation.R` writes the same checks plus the
multi-seed convergence benchmark to `validation-logs/`.

## 2. Equivalence with pykan (Phase 6)

Two kinds of equivalence are tested and reported separately.

**Numerical equivalence**: the same knots, coefficients, weights, network and
inputs give the same numbers.

| Level | Generator (Python) | Test (R) |
|---|---|---|
| KANLayer basis, forward, gradients, grid update | `pykan_reference.py` | `tests/testthat/test-pykan-fixture.R` |
| MultKAN forward, gradients, 5 Adam steps, grid update | `pykan_multilayer_reference.py` | `test-pykan-multilayer.R` |
| `MultKAN.fit(opt = "Adam")`: 200-300 steps, with and without pykan's grid schedule | `pykan_training_reference.py` | `test-pykan-training.R` |

**Algorithmic equivalence**: each package's own initialisation, same data,
same procedure; the *distribution* of out-of-sample RMSE over 10 seeds is
compared (paired by seed), not the individual numbers.

```sh
python -I inst/validation/pykan_training_reference.py validation-logs/pykan
Rscript inst/validation/compare_pykan_training.R validation-logs/pykan
```

Known, documented algorithmic differences:

* the coefficient refit after a grid update uses ridge least squares with
  relative penalty `grid_ridge = 1e-8` (pykan: plain `lstsq`). With
  `kan_fit(grid_ridge = 0)` the trajectories agree to about 1e-13;
* initial coefficients are drawn directly as uniform noise; pykan draws the
  noise at the grid points and fits coefficients to it by least squares;
* default extrapolation is `"linear"` (pykan: B-splines vanish outside the
  grid, `extrapolate = "none"`);
* the LBFGS optimiser is torch's `optim_lbfgs` (pykan ships its own LBFGS
  variant); the LBFGS comparison requires the torch backend.

## 3. Benchmark against other models (synthetic data)

```sh
Rscript inst/benchmarks/run_benchmark.R inst/benchmarks validation-logs/benchmark 10 4
```

Six problems x 10 seeds, n_train = 500, n_test = 1000. Methods: training
mean, OLS, GAM (`mgcv`, REML), random forest (`ranger`), single-hidden-layer
ANN (`nnet`, 3 restarts), Fourier KAN (`inst/benchmarks/fourier_kan.R`,
gradients checked by finite differences) and B-spline KAN. Tuned methods
pick from grids of comparable size (4 configurations for each KAN, 9 for the
ANN) on the same inner hold-out, then are refitted on all training rows.

## 4. Application: PLS-SEM latent scores -> KAN

```r
write.csv(seminr::mobi, "mobi.csv", row.names = FALSE)
data(satisfaction, package = "plspm")
write.csv(satisfaction, "satisfaction.csv", row.names = FALSE)
write.csv(seminr::corp_rep_data, "corp_rep.csv", row.names = FALSE)
```

```sh
Rscript inst/applications/check_pls_pm.R inst/applications mobi.csv satisfaction.csv corp_rep.csv
Rscript inst/applications/run_application.R inst/applications inst/benchmarks \
  validation-logs/application mobi.csv satisfaction.csv corp_rep.csv 10 4
```

`pls_pm.R` is a base R PLS-PM (mode A or B, path weighting) and is checked
against `plspm::plspm()` (paths agree to < 1e-7 on all three data sets; on
`corp_rep_data` it reproduces the path coefficients and R^2 reported by
Hair et al. 2022). Missing values of `corp_rep_data` (code -99) are
mean-replaced, with training-fold means inside the cross-validation. In the 10 x 5-fold cross-validation the PLS weights are
re-estimated inside every training fold and the test-fold scores use the
training weights, means and standard deviations, so no information from
the test rows enters the scores. The data sets are not redistributed with
this package (seminr and plspm are GPL-3 licensed).

The recorded application results were produced in two runs (the ECSI data
sets, then `corp_rep.csv` alone with empty strings for the other two
paths); the seeds depend only on the repeat and fold, so a single run gives
the same numbers.
