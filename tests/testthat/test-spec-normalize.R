test_that("kan_spec validates its input", {
  expect_s3_class(kan_spec(c(2, 3, 1)), "kan_spec")
  expect_error(kan_spec(3), "width")
  expect_error(kan_spec(c(2, 0, 1)), "width")
  expect_error(kan_spec(c(2, 1), grid_range = c(1, 0)), "grid_range")
  expect_error(kan_spec(c(2, 1), extrapolate = "foo"))
  expect_output(print(kan_spec(c(2, 1))), "kan_spec")
})

test_that("initial state is reproducible and well shaped", {
  s1 <- kan_init_state(kan_spec(c(3, 4, 2)), seed = 11)
  s2 <- kan_init_state(kan_spec(c(3, 4, 2)), seed = 11)
  expect_identical(s1, s2)
  expect_equal(dim(s1$layers[[1]]$coef), c(3L, 4L, 8L))
  expect_equal(dim(s1$layers[[2]]$grid), c(4L, 12L))
  bad <- s1
  bad$layers[[1]]$scale_sp <- matrix(1, 2, 2)
  expect_error(splineKANr:::validate_kan_state(bad), "inconsistent")
})

test_that("scalers round-trip and use training statistics", {
  x <- cbind(a = c(1, 2, 3, 4), b = c(10, 20, 30, 50))
  for (m in c("minmax", "standardize", "none")) {
    s <- kan_scaler(x, m)
    expect_equal(kan_unscale(kan_scale(x, s), s), unname(x),
                 ignore_attr = TRUE)
  }
  s <- kan_scaler(x, "minmax")
  z <- kan_scale(x, s)
  expect_equal(apply(z, 2, range), matrix(c(-1, 1, -1, 1), 2),
               ignore_attr = TRUE)
  expect_warning(kan_scaler(cbind(1:3, 5), "minmax"), "constant")
  expect_error(kan_scale(x[, 1, drop = FALSE], s), "Column count")
})

test_that("metrics are correct and handle edge cases", {
  m <- kan_metrics(c(1, 2, 3, 4), c(1, 2, 3, 5))
  expect_equal(m[["rmse"]], 0.5)
  expect_equal(m[["mae"]], 0.25)
  expect_equal(m[["r2"]], 1 - 1 / 5)
  expect_warning(m2 <- kan_metrics(c(1, 1), c(1, 2)), "constant")
  expect_true(is.na(m2[["r2"]]))
  expect_warning(kan_metrics(c(1, NA, 3), c(1, 2, 3)), "non-finite")
  expect_error(kan_metrics(1:3, 1:2), "same length")
})
