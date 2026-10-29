# Convergence and accuracy benchmark (Phase 4) ----------------------------------

#' Synthetic regression problems
#'
#' @param name One of `"linear"`, `"sin"`, `"poly"`, `"interaction"`,
#'   `"noisy_sin"`, `"additive"`.
#' @param n Number of observations.
#' @param seed Optional seed.
#' @return A list with `x` (matrix), `y` (vector), `f` (noise-free target
#'   function) and `noise_sd`.
#' @examples
#' d <- kan_synthetic("sin", n = 50, seed = 1)
#' str(d)
#' @export
kan_synthetic <- function(name = c("linear", "sin", "poly", "interaction",
                                   "noisy_sin", "additive"),
                          n = 500L, seed = NULL) {
  name <- match.arg(name)
  n <- check_count(n, "n", 2L)
  if (!is.null(seed)) set.seed(seed)
  f <- switch(name,
    linear = function(x) 2 * x[, 1] - 0.5,
    sin = function(x) sin(pi * x[, 1]),
    poly = function(x) x[, 1]^3 - 0.5 * x[, 1],
    interaction = function(x) x[, 1] * x[, 2],
    noisy_sin = function(x) sin(pi * x[, 1]),
    additive = function(x) exp(sin(pi * x[, 1])) + x[, 2]^2
  )
  p <- if (name %in% c("interaction", "additive")) 2L else 1L
  noise_sd <- if (name == "noisy_sin") 0.2 else 0
  x <- matrix(stats::runif(n * p, -1, 1), n, p,
              dimnames = list(NULL, paste0("x", seq_len(p))))
  y <- f(x) + stats::rnorm(n, sd = noise_sd)
  list(x = x, y = y, f = f, noise_sd = noise_sd)
}

#' Multi-seed convergence and accuracy benchmark
#'
#' For each problem and seed: simulates training and an independent test
#' set, fits a KAN and reports out-of-sample RMSE, MAE and R^2 (against the
#' noisy test response) plus the RMSE against the noise-free function.
#'
#' @param problems Problem names, see [kan_synthetic()].
#' @param seeds Integer seeds (one replicate each).
#' @param n_train,n_test Sample sizes.
#' @param hidden Hidden widths; `NULL` chooses them per problem, see
#'   Details.
#' @param ... Further arguments for [kan_fit()].
#' @details With `hidden = NULL`, problems with one input use an additive
#'   single-layer KAN and two-input problems use one hidden layer of width
#'   4, which can approximate `x1 * x2` through
#'   (`((x1 + x2)^2 - (x1 - x2)^2) / 4`).
#' @return A data frame with one row per problem and seed.
#' @examples
#' \donttest{
#' kan_benchmark("linear", seeds = 1, epochs = 50, backend = "r")
#' }
#' @export
kan_benchmark <- function(problems = c("linear", "sin", "poly", "interaction",
                                       "noisy_sin"),
                          seeds = 1:5, n_train = 500L, n_test = 1000L,
                          hidden = NULL, ...) {
  rows <- list()
  for (pr in problems) for (s in seeds) {
    tr <- kan_synthetic(pr, n_train, seed = s)
    te <- kan_synthetic(pr, n_test, seed = s + 100000L)
    hid <- if (!is.null(hidden)) hidden else
      if (ncol(tr$x) == 1L) integer() else 4L
    t0 <- proc.time()[["elapsed"]]
    fit <- kan_fit(tr$x, tr$y, hidden = hid, seed = s, ...)
    el <- proc.time()[["elapsed"]] - t0
    be <- fit$settings$backend
    pred <- stats::predict(fit, te$x, backend = be)
    m <- kan_metrics(te$y, pred)
    rows[[length(rows) + 1L]] <- data.frame(
      problem = pr, seed = s, backend = be, epochs_run = nrow(fit$history),
      test_rmse = m[["rmse"]], test_mae = m[["mae"]], test_r2 = m[["r2"]],
      rmse_vs_truth = sqrt(mean((pred - te$f(te$x))^2)),
      noise_sd = tr$noise_sd, seconds = el
    )
  }
  do.call(rbind, rows)
}
