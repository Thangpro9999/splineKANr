test_that("partial dependence plot accepts custom axis labels", {
  set.seed(1)
  x <- matrix(runif(200, -1, 1), ncol = 2, dimnames = list(NULL, c("a", "b")))
  fit <- kan_fit(x, x[, "a"]^2 + x[, "b"], hidden = integer(), epochs = 5,
                 backend = "r", seed = 1)
  pd <- kan_partial_dependence(fit, x, "a", n = 10)
  expect_s3_class(pd, "kan_pd")
  f <- tempfile(fileext = ".png")
  grDevices::png(f)
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_silent(plot(pd, ylab = "PD", main = "a"))
  expect_silent(plot(pd))
})
