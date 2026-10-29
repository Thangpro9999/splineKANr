test_that("regularisation penalty is finite, positive and scales", {
  st <- kan_init_state(kan_spec(c(2, 3, 1)), seed = 1)
  x <- matrix(runif(60, -1, 1), ncol = 2)
  r1 <- kan_regularization_r(st, x, kan_reg_control(lambda = 1))
  r2 <- kan_regularization_r(st, x, kan_reg_control(lambda = 1,
                                                    lambda_l1 = 2))
  expect_true(is.finite(r1) && r1 > 0)
  expect_gt(r2, r1)
  expect_error(kan_reg_control(lambda = -1), "lambda")
})

test_that("edge importance is zero for masked edges", {
  st <- kan_init_state(kan_spec(c(2, 2, 1)), seed = 1)
  st$layers[[1]]$mask[2, 1] <- 0
  imp <- kan_edge_importance(st, matrix(runif(20, -1, 1), ncol = 2))
  expect_equal(imp[[1]][2, 1], 0)
  expect_true(all(imp[[2]] > 0))
})

test_that("pruning removes an irrelevant input's edges and dead nodes", {
  set.seed(1)
  x <- matrix(runif(600, -1, 1), ncol = 2)
  y <- sin(pi * x[, 1])
  fit <- kan_fit(x, y, hidden = 3, epochs = 40, backend = "r", seed = 1,
                 lr = 0.02, batch_size = 64)
  # zero the contribution of input 2 and of hidden node 3, then prune
  fit$state$layers[[1]]$scale_base[2, ] <- 0
  fit$state$layers[[1]]$scale_sp[2, ] <- 0
  fit$state$layers[[1]]$scale_base[, 3] <- 0
  fit$state$layers[[1]]$scale_sp[, 3] <- 0
  fit$state$layers[[2]]$scale_base[3, ] <- 0
  fit$state$layers[[2]]$scale_sp[3, ] <- 0
  p <- kan_prune(fit, x)
  expect_equal(p$spec$width, c(2L, 2L, 1L))
  expect_equal(p$pruning$removed_nodes$node, 3L)
  expect_true(all(p$state$layers[[1]]$mask[2, ] == 0))
  expect_length(predict(p, x, backend = "r"), 300)
})

test_that("plots run without error", {
  d <- kan_synthetic("interaction", 120, seed = 1)
  fit <- kan_fit(d$x, d$y, hidden = 2, epochs = 3, backend = "r", seed = 1)
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off())
  expect_invisible(plot(fit, layer = 1, data = d$x))
  expect_invisible(plot(fit, layer = 2))
  imp <- kan_plot_network(fit, d$x)
  expect_length(imp, 2)
  pd <- kan_partial_dependence(fit, d$x, "x1", n = 10)
  expect_s3_class(pd, "kan_pd")
  expect_equal(nrow(pd), 10L)
  expect_invisible(plot(pd))
  expect_invisible(kan_plot_history(fit))
  ef <- kan_edge_function(fit, 1, 1, 2, n = 5)
  expect_equal(ef$phi, ef$spline + ef$base)
  expect_error(kan_edge_function(fit, 1, 3, 1), "out of range")
})

test_that("activation check flags inputs outside the domain", {
  st <- kan_init_state(kan_spec(c(1, 1)), seed = 1)
  ac <- kan_activation_check(st, matrix(c(-2, 0, 0.5, 2)))
  expect_equal(ac$frac_outside, 0.5)
})
