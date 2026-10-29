# Phase 6b: training-level equivalence with pykan. The fixture
# (inst/validation/pykan_training_reference.py) holds MultKAN.fit(opt =
# "Adam") runs of 200-300 steps from a recorded initial network.

tr_cases <- function() {
  testthat::skip_if_not_installed("jsonlite")
  f <- testthat::test_path("fixtures", "pykan_trajectory.json")
  testthat::skip_if_not(file.exists(f), "pykan trajectory fixture missing")
  jsonlite::fromJSON(f, simplifyVector = TRUE, simplifyDataFrame = FALSE,
                     simplifyMatrix = TRUE)$cases
}

tr_state <- function(cs, layers) {
  w <- cs$width
  spec <- kan_spec(w, grid_size = cs$grid, spline_order = cs$k,
                   extrapolate = "none", bias = FALSE)
  L <- lapply(seq_len(length(w) - 1L), function(l) {
    ly <- layers[[l]]
    d <- c(w[l], w[l + 1L])
    list(coef = array(ly$coef, c(d, cs$grid + cs$k)),
         scale_base = array(ly$scale_base, d),
         scale_sp = array(ly$scale_sp, d), mask = matrix(1, d[1L], d[2L]),
         bias = numeric(d[2L]),
         grid = array(ly$grid, c(w[l], cs$grid + 2 * cs$k + 1)))
  })
  splineKANr:::new_kan_state(spec, L)
}

tr_fit <- function(cs, grid_ridge) {
  n <- length(cs$y)
  x <- array(cs$x, c(n, cs$width[1L]))
  y <- array(cs$y, c(n, 1L))
  st <- tr_state(cs, cs$init)
  init <- structure(list(state = st, spec = st$spec,
                         scalers = list(x = kan_scaler(x, "none"),
                                        y = kan_scaler(y, "none"))),
                    class = "kan_model")
  kan_fit(x, y, init = init, epochs = cs$steps, lr = cs$lr, backend = "r",
          validation_split = 0, patience = NULL, restore_best = FALSE,
          grid_update_every = if (isTRUE(cs$update_grid)) 5L else 0L,
          grid_update_until = 50L, grid_ridge = grid_ridge, seed = 1)
}

param_diff <- function(a, b) {
  max(unlist(Map(function(la, lb) {
    vapply(c("coef", "scale_base", "scale_sp", "grid"),
           function(p) max(abs(la[[p]] - lb[[p]])), numeric(1))
  }, a$layers, b$layers)))
}

test_that("kan_fit follows pykan's Adam trajectory step by step", {
  testthat::skip_on_cran()
  for (cs in Filter(function(cs) !isTRUE(cs$update_grid), tr_cases())) {
    fit <- tr_fit(cs, 1e-8)
    py <- c(cs$rmse_before_step[-1L]^2, cs$final_mse)
    expect_lt(max(abs(fit$history$train_loss - py) / py), 1e-10)
    expect_lt(param_diff(fit$state, tr_state(cs, cs$final)), 1e-10)
  }
})

test_that("with grid updates and grid_ridge = 0, kan_fit matches pykan", {
  testthat::skip_on_cran()
  for (cs in Filter(function(cs) isTRUE(cs$update_grid), tr_cases())) {
    fit <- tr_fit(cs, 0)
    expect_equal(sum(fit$history$grid_updated), 10L)
    expect_lt(abs(fit$history$train_loss[cs$steps] - cs$final_mse) /
                cs$final_mse, 1e-9)
    expect_lt(param_diff(fit$state, tr_state(cs, cs$final)), 1e-8)
  }
})

test_that("the default ridge refit deviates from pykan only slightly", {
  testthat::skip_on_cran()
  # documented algorithmic difference: ridge 1e-8 vs plain lstsq
  for (cs in Filter(function(cs) isTRUE(cs$update_grid), tr_cases())) {
    fit <- tr_fit(cs, 1e-8)
    expect_lt(abs(fit$history$train_loss[cs$steps] - cs$final_mse) /
                cs$final_mse, 1e-3)
  }
})
