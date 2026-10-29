test_that("save/load reproduces predictions exactly", {
  d <- kan_synthetic("interaction", 150, seed = 4)
  fit <- kan_fit(d$x, d$y, hidden = 3, epochs = 5, backend = "r", seed = 4)
  f <- tempfile(fileext = ".rds")
  kan_save(fit, f)
  fit2 <- kan_load(f)
  expect_lt(max(abs(predict(fit2, d$x, backend = "r") -
                      predict(fit, d$x, backend = "r"))), 1e-12)
  expect_identical(fit2$state, fit$state)
  expect_identical(fit2$scalers, fit$scalers)
  expect_equal(fit2$history, fit$history)
  expect_false(is.null(fit2$meta$r_version))
})

test_that("kan_load rejects foreign files", {
  f <- tempfile(fileext = ".rds")
  saveRDS(list(a = 1), f)
  expect_error(kan_load(f), "Not a splineKANr")
  expect_error(kan_load(tempfile()), "not found")
  expect_error(kan_save(list(), f), "kan_model")
})
