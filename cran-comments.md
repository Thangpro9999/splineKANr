## Submission

This is a new submission of splineKANr 0.5.1.

splineKANr implements B-spline Kolmogorov-Arnold Networks (Liu et al. 2024)
in native R on top of the 'torch' package. It contains no compiled code.

## Test environments

* Windows 11 x64, R 4.6.0 (ucrt), torch 0.17.0 with libtorch installed:
  `R CMD check --no-manual` of 0.5.0 -- Status: OK; all 251 tests run
  (none skipped), including the 12 torch-backend tests.
* Ubuntu 24.04 (x86_64-conda-linux-gnu), R 4.5.3, torch 0.17.0 without
  libtorch: `R CMD check --as-cran` of 0.5.0 -- 0 errors | 0 warnings |
  3 notes (new submission plus two environmental notes, see below).
* win-builder (R-devel): TODO before submission
* mac-builder (R-release): TODO before submission

## R CMD check results

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  New submission

Notes seen only in the local Linux environment, not caused by the package:
"unable to verify current time" and an unreachable DOI URL (no internet
access in that environment; the URL https://doi.org/10.48550/arXiv.2404.19756
is valid), and "Skipping checking math rendering: package 'V8' unavailable".

## Notes for CRAN

* libtorch is not available on CRAN machines. All tests, examples and the
  vignette that need it are skipped or switch automatically to the
  package's base R reference implementation (`backend = "r"`), which is the
  same model with analytic gradients. Examples that call torch directly are
  wrapped in `if (torch::torch_is_installed())`. Nothing is downloaded
  during checks.
* Long-running convergence tests use `skip_on_cran()`.
* `inst/validation/` contains Python scripts used only by the authors to
  produce reference numbers from 'pykan' (stored as JSON test fixtures).
  They are never run by the package or its tests; the package has no Python
  dependency.
* `inst/benchmarks/` and `inst/applications/` contain R scripts that
  reproduce the benchmark and application results reported in the
  vignette (see `inst/REPRODUCE.md`). They are not run during checks; the
  extra packages they use (mgcv, nnet, ranger) are not needed by the
  package. No third-party data sets are redistributed.

## Reverse dependencies

None (new package).
