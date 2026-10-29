# Phase 6b: training-level comparison with pykan.
#
#   Rscript compare_pykan_training.R JSON_DIR [OUT_DIR]
#
# JSON_DIR holds the output of pykan_training_reference.py. Two questions are
# answered separately:
#
# 1. Numerical equivalence of the training algorithm (pykan_trajectory.json):
#    from the *same* initial network and data, does kan_fit() follow the same
#    Adam trajectory as MultKAN.fit(opt = "Adam"), with and without pykan's
#    grid-update schedule?
# 2. Algorithmic equivalence of the procedure (pykan_outcomes.json): with
#    each package's *own* random initialisation, do the two implementations
#    of the same procedure reach the same distribution of out-of-sample
#    error over seeds? (Numbers are not expected to agree; distributions
#    are.)
#
# Nothing here changes a threshold to make a comparison pass: all
# differences are reported as measured.

args <- commandArgs(trailingOnly = TRUE)
json_dir <- if (length(args) >= 1L) args[1L] else "validation-logs/pykan"
out_dir <- if (length(args) >= 2L) args[2L] else json_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages(library(splineKANr))

read_json <- function(f) {
  jsonlite::fromJSON(file.path(json_dir, f), simplifyVector = TRUE,
                     simplifyDataFrame = FALSE, simplifyMatrix = TRUE)
}

# pykan layer dump -> kan_state (extrapolate = "none", no node bias)
py_state <- function(width, grid, k, layers) {
  spec <- kan_spec(width, grid_size = grid, spline_order = k,
                   extrapolate = "none", bias = FALSE)
  L <- lapply(seq_len(length(width) - 1L), function(l) {
    ly <- layers[[l]]
    d <- c(width[l], width[l + 1L])
    list(coef = array(ly$coef, c(d, grid + k)),
         scale_base = array(ly$scale_base, d),
         scale_sp = array(ly$scale_sp, d), mask = matrix(1, d[1L], d[2L]),
         bias = numeric(d[2L]),
         grid = array(ly$grid, c(width[l], grid + 2 * k + 1)))
  })
  splineKANr:::new_kan_state(spec, L)
}

as_init_model <- function(state, x, y) {
  structure(list(state = state, spec = state$spec,
                 scalers = list(x = kan_scaler(x, "none"),
                                y = kan_scaler(y, "none"))),
            class = "kan_model")
}

max_param_diff <- function(a, b) {
  max(unlist(Map(function(la, lb) {
    vapply(c("coef", "scale_base", "scale_sp", "grid"), function(p) {
      max(abs(la[[p]] - lb[[p]]))
    }, numeric(1))
  }, a$layers, b$layers)))
}

# ---- 1. same init, same data: trajectories ---------------------------------
traj <- read_json("pykan_trajectory.json")
rows <- list()
for (cs in traj$cases) {
  w <- cs$width
  n <- length(cs$y)
  x <- array(cs$x, c(n, w[1L]))
  y <- array(cs$y, c(n, 1L))
  st0 <- py_state(w, cs$grid, cs$k, cs$init)
  every <- if (isTRUE(cs$update_grid)) {
    as.integer(cs$stop_grid_update_step / cs$grid_update_num)
  } else 0L
  # grid_ridge = 1e-8 is the splineKANr default; 0 is pykan's plain lstsq
  for (gr in if (every > 0L) c(1e-8, 0) else 1e-8) {
    fit <- kan_fit(x, y, init = as_init_model(st0, x, y), epochs = cs$steps,
                   optimizer = "adam", lr = cs$lr, backend = "r",
                   validation_split = 0, patience = NULL, restore_best = FALSE,
                   grid_update_every = every,
                   grid_update_until = cs$stop_grid_update_step,
                   grid_eps = 0.02, grid_ridge = gr, seed = 1)
    # pykan logs the loss before step t; kan_fit logs it after step t, so
    # kan_fit's epoch t is pykan's step t + 1 (and its final MSE at the end).
    py_mse <- c(cs$rmse_before_step[-1L]^2, cs$final_mse)
    r_mse <- fit$history$train_loss
    rel <- abs(r_mse - py_mse) / py_mse
    # the pykan value at step t+1 is taken after a grid update, kan_fit's
    # value at epoch t before it: exclude those epochs from the pointwise check
    upd_next <- c(fit$history$grid_updated[-1L], FALSE)
    st_py_final <- py_state(w, cs$grid, cs$k, cs$final)
    rows[[length(rows) + 1L]] <- data.frame(
      problem = cs$problem, width = paste(w, collapse = "-"),
      grid_update = isTRUE(cs$update_grid),
      grid_ridge = if (every > 0L) gr else NA_real_, steps = cs$steps,
      loss_start = py_mse[1L], loss_end_pykan = cs$final_mse,
      loss_end_splineKANr = r_mse[length(r_mse)],
      max_rel_loss_diff = max(rel[!upd_next]),
      rel_loss_diff_end = rel[length(rel)],
      max_abs_param_diff_end = max_param_diff(fit$state, st_py_final))
  }
}
traj_res <- do.call(rbind, rows)

# ---- 2. own init: distributions over seeds --------------------------------
oc <- read_json("pykan_outcomes.json")
has_torch <- torch::torch_is_installed()
rows <- list()
for (cs in oc$cases) {
  w <- cs$width
  x <- matrix(cs$x, ncol = w[1L])
  y <- as.numeric(cs$y)
  xt <- matrix(cs$x_test, ncol = w[1L])
  yt <- as.numeric(cs$y_test)
  hid <- w[-c(1L, length(w))]
  base <- list(x = x, y = y, hidden = hid, grid_size = 5L, spline_order = 3L,
               extrapolate = "none", bias = FALSE, validation_split = 0,
               patience = NULL, restore_best = FALSE, normalize_x = "none",
               normalize_y = FALSE, grid_update_every = 5L,
               grid_update_until = 50L, seed = cs$seed)
  runs <- list(Adam = c(base, list(optimizer = "adam", lr = 0.01,
                                   epochs = 500L, backend = "r")))
  if (has_torch) {
    runs$LBFGS <- c(base, list(optimizer = "lbfgs", lr = 1, epochs = 50L,
                               backend = "torch", dtype = "float64"))
  }
  for (opt in names(runs)) {
    t0 <- proc.time()[["elapsed"]]
    fit <- do.call(kan_fit, runs[[opt]])
    el <- proc.time()[["elapsed"]] - t0
    pred <- stats::predict(fit, xt, backend = fit$settings$backend)
    py <- cs$fits[[opt]]
    rows[[length(rows) + 1L]] <- data.frame(
      problem = cs$problem, seed = cs$seed, optimizer = opt,
      rmse_splineKANr = sqrt(mean((pred - yt)^2)),
      rmse_pykan = py$test_rmse, sec_splineKANr = el,
      sec_pykan = py$seconds)
  }
}
oc_res <- do.call(rbind, rows)
oc_res$diff <- oc_res$rmse_splineKANr - oc_res$rmse_pykan
summ <- do.call(rbind, lapply(split(oc_res, list(oc_res$problem,
                                                 oc_res$optimizer),
                                    drop = TRUE), function(d) {
  # paired by seed (same data); Wilcoxon signed-rank and paired t
  wt <- suppressWarnings(stats::wilcox.test(d$rmse_splineKANr, d$rmse_pykan,
                                            paired = TRUE))
  tt <- stats::t.test(d$rmse_splineKANr, d$rmse_pykan, paired = TRUE)
  data.frame(problem = d$problem[1L], optimizer = d$optimizer[1L],
             n_seeds = nrow(d),
             mean_rmse_splineKANr = mean(d$rmse_splineKANr),
             sd_rmse_splineKANr = stats::sd(d$rmse_splineKANr),
             mean_rmse_pykan = mean(d$rmse_pykan),
             sd_rmse_pykan = stats::sd(d$rmse_pykan),
             mean_diff = mean(d$diff), ci_low = tt$conf.int[1L],
             ci_high = tt$conf.int[2L], p_wilcoxon = wt$p.value,
             noise_floor = 0.1)
}))

options(width = 160)
sink(file.path(out_dir, "pykan_training_comparison.txt"), split = TRUE)
cat("pykan", traj$pykan_version, "/ torch", traj$torch_version, "\n")
cat("\n== 1. Same init, same data: Adam trajectories (numerical) ==\n")
print(traj_res, digits = 4, row.names = FALSE)
cat("\n== 2. Own init, same data: out-of-sample RMSE over seeds",
    "(algorithmic) ==\n")
print(summ, digits = 4, row.names = FALSE)
if (!has_torch) cat("\n(LBFGS comparison needs the torch backend; skipped.)\n")
sink()
utils::write.csv(traj_res, file.path(out_dir, "pykan_trajectory_results.csv"),
                 row.names = FALSE)
utils::write.csv(oc_res, file.path(out_dir, "pykan_outcome_results.csv"),
                 row.names = FALSE)
