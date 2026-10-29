# Full validation runner (run in RStudio from the package root).
# Writes a timestamped log; does not modify any test to make it pass.
#
#   source(system.file("validation", "run_validation.R", package = "splineKANr"))
#
log_dir <- file.path(getwd(), "validation-logs")
dir.create(log_dir, showWarnings = FALSE)
stamp <- format(Sys.time(), "%Y%m%d-%H%M%S")
log_file <- file.path(log_dir, paste0("validation-", stamp, ".txt"))
sink(log_file, split = TRUE)
cat("splineKANr validation", stamp, "\n")
print(sessionInfo())
cat("\ntorch installed:", torch::torch_is_installed(), "\n")
res <- splineKANr::kan_validate_all(backend = "both")
print(res)
cat("\nALL NUMERICAL CHECKS PASS:", attr(res, "all_pass"), "\n")
bench <- splineKANr::kan_benchmark(
  c("linear", "sin", "poly", "interaction", "noisy_sin"), seeds = 1:5,
  epochs = 300, batch_size = 64, lr = 0.02, patience = 40)
print(bench)
print(aggregate(cbind(test_rmse, test_mae, test_r2, rmse_vs_truth) ~ problem,
                data = bench, FUN = function(v) c(mean = mean(v), sd = sd(v))))
sink()
message("Log written to ", log_file)
