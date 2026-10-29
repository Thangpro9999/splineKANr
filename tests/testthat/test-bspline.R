test_that("grid has the documented shape and extension", {
  g <- kan_make_grid(2, grid_size = 5, spline_order = 3)
  expect_equal(dim(g), c(2L, 12L))
  expect_equal(g[1, 4:9], seq(-1, 1, length.out = 6))
  expect_equal(diff(g[1, ]), rep(0.4, 11))
  rng <- rbind(c(0, 1), c(-2, 2))
  g2 <- kan_make_grid(2, 4, 2, grid_range = rng)
  expect_equal(g2[2, 3], -2)
  expect_equal(g2[2, ncol(g2) - 2], 2)
})

test_that("grid arguments are validated", {
  expect_error(kan_make_grid(0, 5, 3), "n_in")
  expect_error(kan_make_grid(1, 5, 3, grid_range = c(1, -1)), "lower < upper")
  expect_error(kan_extend_grid(matrix(c(0, 0, 1), 1), 1), "increasing")
})

test_that("basis matches splines::splineDesign on the domain", {
  for (k in 1:3) for (G in c(3, 7)) {
    knots <- kan_make_grid(1, G, k)[1, ]
    x <- seq(-0.999, 0.999, length.out = 301)
    ref <- splines::splineDesign(knots, x, ord = k + 1, outer.ok = TRUE)
    expect_lt(max(abs(kan_bspline_basis_r(x, knots, k) - ref)), 1e-12)
  }
})

test_that("partition of unity and non-negativity hold on [a, b]", {
  knots <- kan_make_grid(1, 6, 3)[1, ]
  x <- c(-1, seq(-1, 1, length.out = 200), 1)
  B <- kan_bspline_basis_r(x, knots, 3, right_closed = TRUE)
  expect_lt(max(abs(rowSums(B) - 1)), 1e-12)
  expect_true(all(B >= 0))
})

test_that("right_closed gives left limits at the right bound", {
  knots <- kan_make_grid(1, 4, 3)[1, ]
  B_b <- kan_bspline_basis_r(1, knots, 3, right_closed = TRUE)
  B_left <- kan_bspline_basis_r(1 - 1e-10, knots, 3)
  expect_lt(max(abs(B_b - B_left)), 1e-8)
  D_b <- kan_bspline_deriv_r(1, knots, 3, right_closed = TRUE)
  D_left <- kan_bspline_deriv_r(1 - 1e-10, knots, 3)
  expect_lt(max(abs(D_b - D_left)), 1e-6)
})

test_that("basis is zero outside the extended knots", {
  knots <- kan_make_grid(1, 5, 3)[1, ]
  B <- kan_bspline_basis_r(c(-10, 10), knots, 3)
  expect_equal(sum(B), 0)
})

test_that("analytic derivative matches splineDesign(derivs = 1)", {
  knots <- kan_make_grid(1, 5, 3)[1, ]
  x <- seq(-0.95, 0.95, length.out = 97)
  ref <- splines::splineDesign(knots, x, ord = 4, derivs = 1, outer.ok = TRUE)
  expect_lt(max(abs(kan_bspline_deriv_r(x, knots, 3) - ref)), 1e-10)
})

test_that("validation helpers report passing results", {
  vb <- kan_validate_basis(backend = "r", grid_sizes = c(3, 5))
  expect_true(all(vb$pass))
  vd <- kan_validate_derivative(backend = "r", grid_sizes = c(3, 5))
  expect_true(all(vd$pass))
})

test_that("extrapolation policies behave as documented", {
  knots <- kan_make_grid(1, 5, 3)[1, ]
  coef <- c(0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.6, -0.1)
  s <- function(x, ex) {
    as.numeric(splineKANr:::edge_features_r(x, knots, 3, ex)$value %*% coef)
  }
  ds <- function(x, ex) {
    as.numeric(splineKANr:::edge_features_r(x, knots, 3, ex)$deriv %*% coef)
  }
  # linear: continuous with matching slope at the bounds
  expect_equal(s(1.5, "linear"), s(1, "linear") + 0.5 * ds(1, "linear"))
  expect_equal(s(-1.25, "linear"), s(-1, "linear") - 0.25 * ds(-1, "linear"))
  expect_equal(ds(2, "linear"), ds(1, "linear"))
  # clamp: flat outside
  expect_equal(s(3, "clamp"), s(1, "clamp"))
  expect_equal(ds(3, "clamp"), 0)
  # none: raw B-spline, zero far away
  expect_equal(s(10, "none"), 0)
  # all agree inside the domain
  x <- seq(-0.9, 0.9, length.out = 11)
  expect_equal(s(x, "linear"), s(x, "none"))
  expect_equal(s(x, "clamp"), s(x, "none"))
})
