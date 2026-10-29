# Phase 4: convergence on synthetic functions across seeds (base R backend).
# Thresholds are acceptance criteria on the noise-free test RMSE.

test_that("KAN converges on standard synthetic problems over seeds", {
  skip_on_cran()
  thr <- c(linear = 0.02, sin = 0.03, poly = 0.02, interaction = 0.05,
           noisy_sin = 0.08)
  res <- kan_benchmark(names(thr), seeds = 1:3, n_train = 400, n_test = 500,
                       backend = "r", epochs = 300, batch_size = 64,
                       lr = 0.02, patience = 40)
  agg <- tapply(res$rmse_vs_truth, res$problem, max)
  for (p in names(thr)) {
    expect_lt(agg[[p]], thr[[p]], label = paste("max RMSE vs truth for", p))
  }
  noisy <- res[res$problem == "noisy_sin", ]
  # cannot beat the noise level on noisy data (sanity check, no overfit)
  expect_true(all(noisy$test_rmse > 0.8 * noisy$noise_sd))
})
