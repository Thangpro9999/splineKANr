fit_small <- function(...) {
  d <- kan_synthetic("sin", 200, seed = 1)
  kan_fit(d$x, d$y, hidden = integer(), backend = "r", ...)
}

test_that("training reduces the loss (base R backend)", {
  fit <- fit_small(epochs = 60, batch_size = 50, seed = 1, lr = 0.02)
  h <- fit$history
  expect_s3_class(fit, "kan_model")
  expect_lt(min(h$train_loss), h$train_loss[1] / 10)
  expect_true(all(is.finite(h$val_loss)))
  expect_output(print(fit), "kan_model")
  expect_output(print(summary(fit)), "Parameters")
})

test_that("seed makes training reproducible", {
  f1 <- fit_small(epochs = 15, batch_size = 32, seed = 42)
  f2 <- fit_small(epochs = 15, batch_size = 32, seed = 42)
  expect_identical(f1$state, f2$state)
  expect_identical(f1$history, f2$history)
})

test_that("early stopping triggers and restores the best epoch", {
  fit <- fit_small(epochs = 400, seed = 3, patience = 5, lr = 0.05,
                   min_delta = 1e-3)
  expect_true(fit$stopped_early)
  expect_lt(nrow(fit$history), 400)
  # with min_delta > 0 the best epoch is within min_delta of the minimum
  h <- fit$history
  expect_lte(h$val_loss[fit$best_epoch], min(h$val_loss) + 1e-3)
  expect_equal(nrow(h) - fit$best_epoch, 5L)
  # with min_delta = 0 it is exactly the arg-min, and restore_best returns it
  fit0 <- fit_small(epochs = 60, seed = 3, patience = 5, lr = 0.05)
  h0 <- fit0$history
  expect_equal(fit0$best_epoch, which.min(h0$val_loss))
  d <- kan_synthetic("sin", 200, seed = 1)
  tr <- fit0$settings$train_index
  zx <- kan_scale(d$x[-tr, , drop = FALSE], fit0$scalers$x)
  zy <- kan_scale(matrix(d$y[-tr]), fit0$scalers$y)
  expect_equal(mean((kan_forward_r(fit0, zx) - zy)^2), min(h0$val_loss))
})

test_that("validation_data and no-validation modes work", {
  d <- kan_synthetic("linear", 100, seed = 1)
  v <- kan_synthetic("linear", 50, seed = 2)
  fit <- kan_fit(d$x, d$y, hidden = integer(), epochs = 5, backend = "r",
                 validation_data = list(x = v$x, y = v$y), seed = 1)
  expect_equal(fit$settings$n_val, 50L)
  fit0 <- kan_fit(d$x, d$y, hidden = integer(), epochs = 5, backend = "r",
                  validation_split = 0, seed = 1)
  expect_true(all(is.na(fit0$history$val_loss)))
  expect_equal(fit0$settings$n_train, 100L)
})

test_that("normalisation uses the training rows only", {
  d <- kan_synthetic("linear", 100, seed = 1)
  fit <- kan_fit(d$x, d$y, hidden = integer(), epochs = 2, backend = "r",
                 seed = 7, validation_split = 0.3)
  tr <- fit$settings$train_index
  expect_equal(fit$scalers$x$center, min(d$x[tr, 1]))
  expect_equal(fit$scalers$y$center, mean(d$y[tr]))
})

test_that("gradient clipping and weight decay run", {
  fit <- fit_small(epochs = 5, seed = 1, grad_clip = 0.1, weight_decay = 1e-3)
  expect_true(all(is.finite(fit$history$train_loss)))
})

test_that("divergence is reported instead of returning NaN", {
  d <- kan_synthetic("sin", 100, seed = 1)
  expect_error(
    kan_fit(d$x, d$y * 1e200, hidden = integer(), epochs = 3,
            backend = "r", normalize_y = FALSE, seed = 1),
    "non-finite"
  )
})

test_that("input validation catches common mistakes", {
  x <- matrix(runif(20), 10)
  expect_error(kan_fit(x, 1:9, backend = "r"), "same number of rows")
  xn <- x; xn[1] <- NA
  expect_error(kan_fit(xn, 1:10, backend = "r"), "missing")
  expect_error(kan_fit(x, 1:10, backend = "r", optimizer = "lbfgs"),
               "adam")
  expect_error(kan_fit(x, 1:10, backend = "r",
                       reg = kan_reg_control(lambda = 1)), "torch")
  expect_error(kan_fit(x, 1:10, backend = "r", validation_split = 1),
               "validation_split")
  expect_error(kan_fit(data.frame(a = letters[1:10]), 1:10, backend = "r"),
               "numeric")
})

test_that("multi-output models and data frames work", {
  set.seed(1)
  x <- data.frame(a = runif(80, -1, 1), b = runif(80, -1, 1))
  y <- cbind(u = x$a + x$b, v = x$a - x$b)
  fit <- kan_fit(x, y, hidden = integer(), epochs = 30, backend = "r",
                 seed = 1, lr = 0.05)
  p <- predict(fit, x[, c("b", "a")], backend = "r")  # columns by name
  expect_equal(dim(p), c(80L, 2L))
  expect_equal(colnames(p), c("u", "v"))
  expect_gt(kan_metrics(y, p)[["r2"]], 0.95)
  expect_error(predict(fit, matrix(0, 2, 3), backend = "r"), "2 columns")
})

test_that("checkpoint file holds a loadable model", {
  f <- tempfile(fileext = ".rds")
  fit <- fit_small(epochs = 5, seed = 1, checkpoint_path = f)
  expect_true(file.exists(f))
  ck <- kan_load(f)
  expect_s3_class(ck, "kan_model")
})

test_that("warm start continues from an existing model", {
  d <- kan_synthetic("sin", 200, seed = 1)
  f1 <- kan_fit(d$x, d$y, hidden = integer(), epochs = 10, backend = "r",
                seed = 1)
  f2 <- kan_fit(d$x, d$y, hidden = integer(), epochs = 10, backend = "r",
                seed = 1, init = f1)
  expect_identical(f2$scalers, f1$scalers)
  expect_lte(min(f2$history$train_loss), min(f1$history$train_loss))
})
