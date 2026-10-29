test_that("analytic gradients match finite differences", {
  res <- kan_validate_gradient(widths = list(c(2, 3, 1), c(2, 2, 2)),
                               backend = "r", n = 25)
  expect_true(all(res$pass), info = paste(capture.output(print(res)),
                                          collapse = "\n"))
})

test_that("input gradient matches finite differences for every policy", {
  for (ex in c("linear", "none", "clamp")) {
    st <- kan_init_state(kan_spec(c(2, 3, 1), extrapolate = ex), seed = 3,
                         noise_scale = 1)
    x <- matrix(c(-1.4, -0.33, 0.12, 0.71, 1.2, -0.57), ncol = 2)
    y <- matrix(c(0.2, -0.1, 0.4))
    g <- kan_gradient_r(st, x, y)$input_grad
    h <- 1e-6
    fd <- x * 0
    for (id in seq_along(x)) {
      xp <- x; xp[id] <- xp[id] + h
      xm <- x; xm[id] <- xm[id] - h
      fd[id] <- (mean((kan_forward_r(st, xp) - y)^2) -
                   mean((kan_forward_r(st, xm) - y)^2)) / (2 * h)
    }
    expect_lt(max(abs(g - fd) / pmax(abs(fd), 1e-6)), 1e-4, label = ex)
  }
})

test_that("linear extrapolation keeps a non-zero input gradient outside", {
  st <- kan_init_state(kan_spec(c(1, 1), base_fun = "none"), seed = 1,
                       noise_scale = 1)
  g_out <- kan_gradient_r(st, matrix(3), matrix(0))$input_grad
  expect_true(abs(g_out) > 0)
  st$spec$extrapolate <- "clamp"
  g_clamp <- kan_gradient_r(st, matrix(3), matrix(0))$input_grad
  expect_equal(as.numeric(g_clamp), 0)
})

test_that("forward pass sums edges and adds bias", {
  st <- kan_init_state(kan_spec(c(2, 1)), seed = 5)
  st$layers[[1]]$bias <- 0.7
  x <- matrix(c(0.1, -0.2), 1)
  fw <- kan_forward_r(st, x, detail = TRUE)
  expect_equal(as.numeric(fw$output), sum(fw$layers[[1]]$phi) + 0.7)
  st$layers[[1]]$mask[1, 1] <- 0
  fw2 <- kan_forward_r(st, x, detail = TRUE)
  expect_equal(fw2$layers[[1]]$phi[1, 1, 1], 0)
})
