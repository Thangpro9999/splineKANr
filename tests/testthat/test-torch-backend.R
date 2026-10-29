# Everything in this file needs a working libtorch. The base R reference is
# the oracle: the torch backend must reproduce it to floating-point accuracy.

test_that("torch basis matches splineDesign and the R reference", {
  skip_if_no_torch()
  vb <- kan_validate_basis(backend = "torch")
  expect_true(all(vb$pass), info = paste(capture.output(print(vb)),
                                         collapse = "\n"))
})

test_that("torch right_closed basis matches the R reference", {
  skip_if_no_torch()
  knots <- kan_make_grid(1, 5, 3)[1, ]
  x <- c(-1, -0.5, 0.2, 1)
  ref <- kan_bspline_basis_r(x, knots, 3, right_closed = TRUE)
  tb <- splineKANr:::basis_backend(x, knots, 3, "torch", right_closed = TRUE)
  expect_lt(max(abs(tb - ref)), 1e-12)
})

test_that("torch autograd derivative of the basis is correct", {
  skip_if_no_torch()
  vd <- kan_validate_derivative(backend = "torch")
  expect_true(all(vd$pass), info = paste(capture.output(print(vd)),
                                         collapse = "\n"))
})

test_that("torch autograd derivative is stable across repeated runs", {
  skip_if_no_torch()
  knots <- kan_make_grid(1, 10, 3)[1, ]
  x <- seq(-0.95, 0.95, length.out = 300)
  ref <- kan_bspline_deriv_r(x, knots, 3)
  for (r in 1:5) {
    D <- splineKANr:::torch_basis_autograd(x, knots, 3)
    expect_true(all(is.finite(D)), label = paste("run", r))
    expect_lt(max(abs(D - ref)), 1e-10, label = paste("run", r))
  }
})

test_that("torch analytic basis derivative matches the R reference", {
  skip_if_no_torch()
  knots <- kan_make_grid(1, 5, 3)[1, ]
  x <- seq(-0.97, 0.97, length.out = 41)
  dt <- torch::torch_float64()
  D <- splineKANr:::kan_bspline_deriv(
    torch::torch_tensor(matrix(x), dtype = dt),
    torch::torch_tensor(matrix(knots, 1), dtype = dt), 3)
  expect_lt(max(abs(matrix(torch::as_array(D), nrow = length(x)) -
                      kan_bspline_deriv_r(x, knots, 3))), 1e-12)
})

test_that("torch forward equals the R reference for all policies", {
  skip_if_no_torch()
  for (ex in c("linear", "none", "clamp")) {
    st <- kan_init_state(kan_spec(c(3, 4, 2), extrapolate = ex), seed = 2,
                         noise_scale = 1)
    x <- matrix(stats::runif(60, -1.5, 1.5), ncol = 3)
    net <- kan_module_from_state(st, dtype = "float64")
    out <- torch::as_array(net(torch::torch_tensor(x,
                                 dtype = torch::torch_float64())))
    expect_lt(max(abs(out - kan_forward_r(st, x))), 1e-10, label = ex)
  }
})

test_that("torch autograd gradients equal analytic gradients", {
  skip_if_no_torch()
  res <- kan_validate_gradient(backend = "torch")
  expect_true(all(res$pass), info = paste(capture.output(print(res)),
                                          collapse = "\n"))
})

test_that("state round-trips through a torch module", {
  skip_if_no_torch()
  st <- kan_init_state(kan_spec(c(2, 3, 1)), seed = 9)
  net <- kan_module_from_state(st, dtype = "float64")
  st2 <- splineKANr:::kan_state_from_module(net, st$spec)
  expect_equal(st2$layers, st$layers, tolerance = 1e-14)
})

test_that("torch regularisation equals the R reference", {
  skip_if_no_torch()
  st <- kan_init_state(kan_spec(c(2, 3, 1)), seed = 1)
  x <- matrix(stats::runif(40, -1, 1), ncol = 2)
  reg <- kan_reg_control(lambda = 1, lambda_coef = 0.5, lambda_coefdiff = 0.3)
  net <- kan_module_from_state(st, dtype = "float64")
  out <- net(torch::torch_tensor(x, dtype = torch::torch_float64()),
             detail = TRUE)
  rt <- splineKANr:::kan_regularization_torch(out$phi, net, reg)$item()
  expect_equal(rt, kan_regularization_r(st, x, reg), tolerance = 1e-10)
})

test_that("torch training converges, is reproducible and saves/loads", {
  skip_if_no_torch()
  d <- kan_synthetic("sin", 300, seed = 1)
  f1 <- kan_fit(d$x, d$y, hidden = integer(), epochs = 150, batch_size = 64,
                seed = 1, lr = 0.02, backend = "torch")
  f2 <- kan_fit(d$x, d$y, hidden = integer(), epochs = 150, batch_size = 64,
                seed = 1, lr = 0.02, backend = "torch")
  expect_equal(f1$state, f2$state)
  te <- kan_synthetic("sin", 300, seed = 2)
  expect_gt(kan_metrics(te$y, predict(f1, te$x))[["r2"]], 0.99)
  # torch and R backends agree on predictions of the same model
  expect_lt(max(abs(predict(f1, te$x, backend = "torch") -
                      predict(f1, te$x, backend = "r"))), 1e-5)
  f <- tempfile(fileext = ".rds")
  kan_save(f1, f)
  expect_lt(max(abs(predict(kan_load(f), te$x) - predict(f1, te$x))), 1e-6)
})

test_that("torch training with grid updates, regularisation, L-BFGS", {
  skip_if_no_torch()
  d <- kan_synthetic("interaction", 300, seed = 3)
  fit <- kan_fit(d$x, d$y, hidden = 4, epochs = 60, batch_size = 64, seed = 3,
                 grid_update_every = 10, reg = kan_reg_control(lambda = 1e-4),
                 grad_clip = 5)
  expect_true(all(is.finite(fit$history$train_loss)))
  expect_lt(min(fit$history$train_loss), fit$history$train_loss[1])
  fl <- kan_fit(d$x, d$y, hidden = 4, epochs = 10, seed = 3,
                optimizer = "lbfgs")
  expect_true(all(is.finite(fl$history$train_loss)))
})

test_that("torch and R backends follow the same Adam trajectory", {
  skip_if_no_torch()
  d <- kan_synthetic("sin", 64, seed = 5)
  common <- list(x = d$x, y = d$y, hidden = 2, epochs = 3, seed = 5,
                 validation_split = 0, dtype = "float64")
  fr <- do.call(kan_fit, c(common, backend = "r"))
  ft <- do.call(kan_fit, c(common, backend = "torch"))
  expect_equal(ft$history$train_loss, fr$history$train_loss,
               tolerance = 1e-8)
})
