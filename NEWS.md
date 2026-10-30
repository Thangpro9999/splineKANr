# splineKANr 0.5.2

* License changed from MIT to an academic and non-commercial license
  (`License: file LICENSE`): use, modification and redistribution are
  permitted for education and non-commercial research only, with
  attribution to Thang Le-Dinh; commercial use requires written
  permission. No change to code or results.

# splineKANr 0.5.1

* Documentation only: consistent US English spelling (the package declares
  `Language: en-US`), `inst/WORDLIST` for technical terms and names;
  `spelling::spell_check_package()` reports no errors. No change to code
  behavior.

# splineKANr 0.5.0

* Training-level comparison with pykan 0.2.8 (`MultKAN.fit(opt = "Adam")`,
  200-300 steps, fixture `pykan_trajectory.json`, test
  `test-pykan-training.R`). Without grid updates the loss trajectory and
  final parameters agree to 1e-14. With pykan's grid-update schedule the
  remaining difference came only from the ridge penalty of the coefficient
  refit; new argument `kan_fit(grid_ridge = )` (default `1e-8`, unchanged
  behavior), and `grid_ridge = 0` (plain least squares, as pykan) agrees
  to 1e-13. `kan_update_grid(ridge = 0)` now uses a QR least-squares
  solve.
* Algorithmic comparison (own initialization, same data and procedure,
  10 seeds): `inst/validation/compare_pykan_training.R`.
* Benchmark scripts (`inst/benchmarks/`): B-spline KAN against OLS, GAM,
  random forest, ANN and a Fourier KAN comparator, same data and
  out-of-sample protocol, 10 seeds.
* Application scripts (`inst/applications/`): base R PLS-PM (checked against
  plspm), fold-internal latent scores and repeated cross-validation of KAN
  vs. the PLS-SEM structural model on three public customer data sets
  (ECSI `mobi`, `plspm::satisfaction`, `corp_rep_data`). `pls_pm()` supports
  formative (mode B) blocks.
* New vignette "Method, validation and application"; `inst/REPRODUCE.md`
  lists every script, seed and software version behind the reported
  results.
* Fixed: `plot.kan_pd()` failed when `xlab` or `ylab` was supplied.

# splineKANr 0.4.0

* Phase 6: multi-layer cross-check against pykan 0.2.8 `MultKAN`
  (architectures 2-3-1, 3-4-2-1, 2-5-3-2): forward pass, loss and all
  layer gradients agree to < 1e-10, five full-batch Adam steps follow the
  same trajectory (< 1e-10), and the network-wide sample-based grid update
  places the same knots (layer 1 exactly, deeper layers < 4e-6, the
  difference coming from least squares vs. ridge refits). Generator script:
  `inst/validation/pykan_multilayer_reference.py`.
* CRAN preparation: `cran-comments.md`, MIT license files, `cph` role,
  `Language` field.

# splineKANr 0.3.2

* Torch backend verified on Windows 11 / R 4.6.0 / torch 0.17.0:
  187/187 tests pass, `R CMD check --no-manual` 1 NOTE (fixed here).
* Use `stats::predict()` and `utils::str()` explicitly (R CMD check NOTE);
  fixed an Rd link warning in an internal comment.

# splineKANr 0.3.1

* Fixed: torch regularization built R constants as float32 tensors, so
  float64 models lost precision (found by the torch test on Windows).
* `torch_basis_autograd()` (validation) now uses one backward pass instead
  of one per basis function; added a repeated-run stability test after a
  NaN result was observed once on Windows.
* `RUN_IN_RSTUDIO.R` skips installation when torch is ready and runs
  `R CMD check` directly, so Rtools is not required.

# splineKANr 0.3.0

Rewrite of the earlier prototype, following phases 1-5 of the release plan.

## Mathematical core
* Extended knot grid (`G + 2k + 1` knots, pykan convention) and Cox-de Boor
  basis in torch and in base R (`kan_bspline_basis()`,
  `kan_bspline_basis_r()`), analytic derivative (`kan_bspline_deriv_r()`).
* The old `torch_clamp()` domain restriction is replaced by an explicit
  policy `extrapolate = "linear"` (default, C1 Taylor extension, non-zero
  gradient), `"none"` (pykan behavior) or `"clamp"`.
* Right boundary handled as a left limit (`right_closed`).

## Validation
* `kan_validate_basis()`, `kan_validate_derivative()`,
  `kan_validate_gradient()`, `kan_validate_all()`: comparisons with
  `splines::splineDesign()`, finite differences and torch autograd.
* Base R reference forward pass and analytic backpropagation
  (`kan_forward_r()`, `kan_gradient_r()`, `kan_fd_gradient_r()`).
* Fixture-based cross-check against pykan 0.2.8 (forward, gradients, grid
  update); generator in `inst/validation/pykan_reference.py`.

## Training
* `kan_fit()`: Adam / L-BFGS, mini-batches, validation split or data, early
  stopping with best-epoch restore, train-only normalization of `x` and `y`,
  gradient clipping, weight decay, seeds, checkpoints, warm start, base R
  backend.
* `kan_benchmark()` and `kan_synthetic()` for multi-seed convergence studies.

## KAN features
* Adaptive sample-based grids (`kan_update_grid()`), grid refinement
  (`kan_refine()`), pykan-style regularization (`kan_reg_control()`),
  pruning (`kan_prune()`), edge importance, edge-function, network,
  partial-dependence and history plots.
* `kan_save()` / `kan_load()` store plain R data (no torch pointers).
