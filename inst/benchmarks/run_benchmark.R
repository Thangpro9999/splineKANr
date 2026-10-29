# Synthetic out-of-sample benchmark: B-spline KAN vs comparators.
#
#   Rscript run_benchmark.R BENCH_DIR OUT_DIR [n_seeds] [cores]
#
# For every problem and seed: an independent training set (n = 500) and test
# set (n = 1000) are simulated; all methods (see models.R) are trained on
# the same rows and scored on the same test rows, against the noisy response
# and against the noise-free function. Results are reported per method as
# mean, sd and the paired difference to bspline_kan over seeds.

args <- commandArgs(trailingOnly = TRUE)
bench_dir <- if (length(args) >= 1L) args[1L] else "inst/benchmarks"
out_dir <- if (length(args) >= 2L) args[2L] else "validation-logs/benchmark"
n_seeds <- if (length(args) >= 3L) as.integer(args[3L]) else 10L
cores <- if (length(args) >= 4L) as.integer(args[4L]) else 1L
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
source(file.path(bench_dir, "fourier_kan.R"))
source(file.path(bench_dir, "models.R"))

friedman1 <- function(n, seed) {
  set.seed(seed)
  x <- matrix(stats::runif(n * 10), n, 10,
              dimnames = list(NULL, paste0("x", 1:10)))
  f <- function(x) 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 +
    10 * x[, 4] + 5 * x[, 5]
  list(x = x, y = f(x) + stats::rnorm(n), f = f, noise_sd = 1)
}
make <- function(pr, n, seed) {
  if (pr == "friedman1") friedman1(n, seed) else
    splineKANr::kan_synthetic(pr, n, seed = seed)
}
problems <- c("sin", "poly", "interaction", "noisy_sin", "additive",
              "friedman1")
jobs <- expand.grid(problem = problems, seed = seq_len(n_seeds),
                    stringsAsFactors = FALSE)

run_job <- function(i) {
  pr <- jobs$problem[i]
  s <- jobs$seed[i]
  tr <- make(pr, 500L, s)
  te <- make(pr, 1000L, s + 100000L)
  do.call(rbind, lapply(bench_methods, function(m) {
    r <- fit_predict(m, tr$x, tr$y, te$x, seed = s)
    sc <- score(te$y, r$pred, mean(tr$y))
    data.frame(problem = pr, seed = s, method = m, config = r$config,
               test_rmse = sc[["rmse"]], test_mae = sc[["mae"]],
               q2_predict = sc[["q2_predict"]],
               rmse_vs_truth = sqrt(mean((r$pred - te$f(te$x))^2)),
               noise_sd = tr$noise_sd, seconds = r$seconds)
  }))
}
res <- if (cores > 1L) {
  do.call(rbind, parallel::mclapply(seq_len(nrow(jobs)), run_job,
                                    mc.cores = cores))
} else {
  do.call(rbind, lapply(seq_len(nrow(jobs)), run_job))
}
utils::write.csv(res, file.path(out_dir, "benchmark_results.csv"),
                 row.names = FALSE)

summarise <- function(res, metric) {
  do.call(rbind, lapply(split(res, list(res$problem, res$method),
                              drop = TRUE), function(d) {
    ref <- res[res$problem == d$problem[1L] & res$method == "bspline_kan", ]
    ref <- ref[match(d$seed, ref$seed), ]
    dd <- d[[metric]] - ref[[metric]]
    data.frame(problem = d$problem[1L], method = d$method[1L],
               mean = mean(d[[metric]]), sd = stats::sd(d[[metric]]),
               diff_vs_bspline_kan = mean(dd),
               p_paired = if (all(dd == 0)) NA_real_ else
                 suppressWarnings(stats::wilcox.test(dd)$p.value),
               wins_vs_bspline_kan = sum(dd < 0),
               mean_seconds = mean(d$seconds))
  }))
}
options(width = 160)
sink(file.path(out_dir, "benchmark_summary.txt"), split = TRUE)
cat("Seeds:", n_seeds, " n_train = 500, n_test = 1000\n")
cat("\n== RMSE against the noise-free function (lower is better) ==\n")
s1 <- summarise(res, "rmse_vs_truth")
print(s1[order(s1$problem, s1$mean), ], digits = 3, row.names = FALSE)
cat("\n== Test RMSE against the noisy response ==\n")
s2 <- summarise(res, "test_rmse")
print(s2[order(s2$problem, s2$mean), ], digits = 3, row.names = FALSE)
cat("\nChosen configurations (counts):\n")
print(table(paste(res$method, res$config))[table(paste(res$method,
                                                       res$config)) > 0])
sink()
