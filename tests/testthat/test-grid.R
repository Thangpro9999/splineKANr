test_that("grid update moves knots to the data and preserves the function", {
  st <- kan_init_state(kan_spec(c(1, 2, 1)), seed = 1, noise_scale = 1)
  set.seed(2)
  x <- matrix(stats::rnorm(400, 0.2, 0.3))
  st2 <- kan_update_grid(st, x, grid_eps = 0.02)
  dom <- splineKANr:::grid_domain(st2$layers[[1]]$grid, 3)
  expect_equal(unname(dom[1, 1]), min(x))
  expect_equal(unname(dom[1, 2]), max(x))
  f1 <- kan_forward_r(st, x)
  f2 <- kan_forward_r(st2, x)
  expect_lt(max(abs(f1 - f2)), 0.05 * diff(range(f1)))
})

test_that("grid_eps = 1 gives a uniform grid", {
  st <- kan_init_state(kan_spec(c(1, 1)), seed = 1)
  x <- matrix(c(-0.5, -0.4, 0, 0.1, 0.9))
  g <- kan_update_grid(st, x, grid_eps = 1)$layers[[1]]$grid[1, ]
  expect_equal(diff(g), rep(diff(g)[1], length(g) - 1))
})

test_that("refinement increases the grid and keeps the fit", {
  d <- kan_synthetic("sin", 300, seed = 1)
  fit <- kan_fit(d$x, d$y, hidden = integer(), epochs = 80, backend = "r",
                 seed = 1, lr = 0.02, batch_size = 64)
  r <- kan_refine(fit, d$x, grid_size = 12)
  expect_equal(r$spec$grid_size, 12L)
  expect_equal(dim(r$state$layers[[1]]$coef)[3], 15L)
  expect_lt(max(abs(predict(r, d$x, backend = "r") -
                      predict(fit, d$x, backend = "r"))), 0.01)
  # continue training the refined model
  r2 <- kan_fit(d$x, d$y, init = r, epochs = 5, backend = "r", seed = 1)
  expect_equal(r2$spec$grid_size, 12L)
})

test_that("constant inputs do not break the grid update", {
  st <- kan_init_state(kan_spec(c(2, 1)), seed = 1)
  x <- cbind(rep(0.3, 10), seq(-1, 1, length.out = 10))
  st2 <- kan_update_grid(st, x)
  expect_true(all(is.finite(st2$layers[[1]]$coef)))
})

test_that("grid updates during training are recorded", {
  d <- kan_synthetic("sin", 200, seed = 1)
  fit <- kan_fit(d$x, d$y, hidden = 2, epochs = 12, backend = "r", seed = 1,
                 grid_update_every = 5, grid_update_until = 10)
  expect_equal(which(fit$history$grid_updated), c(1L, 6L))
})
