# Phase 6 (preliminary): reference values produced by pykan with
# inst/validation/pykan_reference.py, in float64. splineKANr with
# extrapolate = "none" and bias = FALSE follows pykan's KANLayer convention.

pykan_cases <- function() {
  testthat::skip_if_not_installed("jsonlite")
  f <- testthat::test_path("fixtures", "pykan_reference.json")
  testthat::skip_if_not(file.exists(f), "pykan fixture missing")
  jsonlite::fromJSON(f, simplifyVector = TRUE, simplifyDataFrame = FALSE,
                     simplifyMatrix = TRUE)$cases
}

as_arr <- function(v, d) {
  # jsonlite drops length-one dimensions; restore the documented shape
  array(v, dim = d)
}

case_state <- function(cs) {
  spec <- kan_spec(c(cs$in_dim, cs$out_dim), grid_size = cs$num,
                   spline_order = cs$k, extrapolate = "none", bias = FALSE)
  nb <- cs$num + cs$k
  L <- list(
    coef = as_arr(cs$coef, c(cs$in_dim, cs$out_dim, nb)),
    scale_base = as_arr(cs$scale_base, c(cs$in_dim, cs$out_dim)),
    scale_sp = as_arr(cs$scale_sp, c(cs$in_dim, cs$out_dim)),
    mask = matrix(1, cs$in_dim, cs$out_dim),
    bias = numeric(cs$out_dim),
    grid = as_arr(cs$grid, c(cs$in_dim, cs$num + 2 * cs$k + 1))
  )
  splineKANr:::new_kan_state(spec, list(L))
}

test_that("forward pass and edge activations match pykan", {
  for (cs in pykan_cases()) {
    st <- case_state(cs)
    x <- as_arr(cs$x, c(25, cs$in_dim))
    fw <- kan_forward_r(st, x, detail = TRUE)
    y_py <- as_arr(cs$y, c(25, cs$out_dim))
    expect_lt(max(abs(fw$output - y_py)), 1e-10)
    post <- as_arr(cs$postacts, c(25, cs$out_dim, cs$in_dim))
    expect_lt(max(abs(aperm(fw$layers[[1]]$phi, c(1, 3, 2)) - post)), 1e-10)
  }
})

test_that("loss and parameter gradients match pykan autograd", {
  for (cs in pykan_cases()) {
    st <- case_state(cs)
    x <- as_arr(cs$x, c(25, cs$in_dim))
    tg <- as_arr(cs$target, c(25, cs$out_dim))
    g <- kan_gradient_r(st, x, tg)
    expect_equal(g$loss, cs$loss, tolerance = 1e-12)
    nb <- cs$num + cs$k
    gc <- as_arr(cs$grad_coef, c(cs$in_dim, cs$out_dim, nb))
    expect_lt(max(abs(g$grads[[1]]$coef - gc)), 1e-10)
    expect_lt(max(abs(g$grads[[1]]$scale_base -
                        as_arr(cs$grad_scale_base, c(cs$in_dim, cs$out_dim)))),
              1e-10)
    expect_lt(max(abs(g$grads[[1]]$scale_sp -
                        as_arr(cs$grad_scale_sp, c(cs$in_dim, cs$out_dim)))),
              1e-10)
  }
})

test_that("sample-based grid update places the same knots as pykan", {
  for (cs in pykan_cases()) {
    st <- case_state(cs)
    xs <- as_arr(cs$update_x, c(200, cs$in_dim))
    st2 <- kan_update_grid(st, xs, grid_eps = 0.02)
    g_py <- as_arr(cs$update_grid, c(cs$in_dim, cs$num + 2 * cs$k + 1))
    expect_lt(max(abs(st2$layers[[1]]$grid - g_py)), 1e-10)
  }
})
