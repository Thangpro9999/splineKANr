# Business / behavioural application: PLS-SEM latent scores -> B-spline KAN.
#
#   Rscript run_application.R APP_DIR BENCH_DIR OUT_DIR mobi.csv
#     [satisfaction.csv] [corp_rep.csv] [repeats] [cores]
#
# mobi.csv: seminr's ECSI `mobi` data, exported with
#   write.csv(seminr::mobi, "mobi.csv", row.names = FALSE)
# satisfaction.csv: plspm's ECSI `satisfaction` data, exported with
#   data(satisfaction, package = "plspm"); write.csv(satisfaction, ...)
#   (pass "" to skip).
# corp_rep.csv: seminr's `corp_rep_data` (missing values coded -99).
#
# For every target construct:
#   1. full-sample PLS-SEM (pls_pm.R): loadings, CR, AVE, paths, R^2;
#   2. repeated K-fold cross-validation. Inside each fold the PLS weights are
#      re-estimated on the training rows only, and the latent scores of the
#      test rows are computed with the training weights, means and sds (no
#      leakage). All methods in BENCH_DIR/models.R, plus "pls_sem" (OLS on
#      the target's structural predictors, i.e. the PLS-SEM prediction), are
#      trained on the same training scores and scored on the same test rows:
#      RMSE, MAE and Q^2_predict (benchmark: training mean);
#   3. a B-spline KAN fitted on the full sample: partial dependence plots of
#      each predictor and edge importances, to inspect non-linearity.

args <- commandArgs(trailingOnly = TRUE)
app_dir <- args[1L]
bench_dir <- args[2L]
out_dir <- args[3L]
mobi_file <- args[4L]        # "" to skip
sat_file <- if (length(args) >= 5L && nzchar(args[5L])) args[5L] else NULL
cr_file <- if (length(args) >= 6L && nzchar(args[6L])) args[6L] else NULL
repeats <- if (length(args) >= 7L) as.integer(args[7L]) else 10L
cores <- if (length(args) >= 8L) as.integer(args[8L]) else 1L
folds <- 5L
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
source(file.path(app_dir, "pls_pm.R"))
source(file.path(app_dir, "models.R"))
source(file.path(bench_dir, "fourier_kan.R"))
source(file.path(bench_dir, "models.R"))

apps <- list()
if (nzchar(mobi_file)) {
  apps <- list(
    mobi_satisfaction = list(
      data = read.csv(mobi_file), model = mobi_model, target = "Satisfaction",
      predictors = c("Image", "Expectation", "Quality", "Value")),
    mobi_loyalty = list(
      data = read.csv(mobi_file), model = mobi_model, target = "Loyalty",
      predictors = c("Image", "Expectation", "Quality", "Value",
                     "Satisfaction", "Complaints"))
  )
}
if (!is.null(sat_file)) {
  sat <- read.csv(sat_file)
  apps$plspm_satisfaction <- list(
    data = sat, model = satisfaction_model, target = "SAT",
    predictors = c("IMAG", "EXPE", "QUAL", "VAL"))
  apps$plspm_loyalty <- list(
    data = sat, model = satisfaction_model, target = "LOY",
    predictors = c("IMAG", "EXPE", "QUAL", "VAL", "SAT"))
}
if (!is.null(cr_file)) {
  cr <- na_code(read.csv(cr_file))
  apps$corp_rep_satisfaction <- list(
    data = cr, model = corp_rep_model, target = "CUSA", impute = TRUE,
    predictors = c("QUAL", "PERF", "CSOR", "ATTR", "COMP", "LIKE"))
  apps$corp_rep_loyalty <- list(
    data = cr, model = corp_rep_model, target = "CUSL", impute = TRUE,
    predictors = c("QUAL", "PERF", "CSOR", "ATTR", "COMP", "LIKE", "CUSA"))
}
methods <- c("pls_sem", setdiff(bench_methods, "ols"))

options(width = 160)
log_con <- file(file.path(out_dir, "application_summary.txt"), "w")
say <- function(...) {
  txt <- utils::capture.output(...)
  writeLines(txt, log_con)
  writeLines(txt)
}

all_cv <- list()
for (nm in names(apps)) {
  a <- apps[[nm]]
  m <- a$model
  struct_pred <- m$paths[m$paths[, 2L] == a$target, 1L]

  # ---- 1. full-sample PLS-SEM ---------------------------------------------
  imp_on <- isTRUE(a$impute)
  full <- if (imp_on) mean_impute(a$data) else a$data
  fit <- pls_pm(full, m$blocks, m$paths, m$modes)
  say(cat(sprintf("\n######## %s (n = %d): target %s ########\n", nm,
                  nrow(a$data), a$target)))
  say(cat("PLS converged:", fit$converged, "in", fit$iterations,
          "iterations\n"))
  say(print(round(rbind(CR = fit$cr, AVE = fit$ave, rho_A = fit$rho_a), 3)))
  say(print(pls_path_table(fit, "pls"), digits = 3, row.names = FALSE))

  # ---- 2. repeated K-fold CV ------------------------------------------------
  n <- nrow(a$data)
  jobs <- expand.grid(fold = seq_len(folds), rep = seq_len(repeats))
  run_job <- function(i) {
    r <- jobs$rep[i]
    k <- jobs$fold[i]
    set.seed(1000L + r)
    fold_id <- sample(rep(seq_len(folds), length.out = n))
    tr <- a$data[fold_id != k, ]
    te <- a$data[fold_id == k, ]
    if (imp_on) {                      # training-fold means only
      te <- mean_impute(te, ref = tr)
      tr <- mean_impute(tr)
    }
    f_tr <- pls_pm(tr, m$blocks, m$paths, m$modes)
    s_tr <- pls_scores(f_tr, tr)
    s_te <- pls_scores(f_tr, te)
    ytr <- s_tr[, a$target]
    yte <- s_te[, a$target]
    seed <- 100L * r + k
    do.call(rbind, lapply(methods, function(meth) {
      if (meth == "pls_sem") {
        res <- fit_predict("ols", s_tr[, struct_pred, drop = FALSE], ytr,
                           s_te[, struct_pred, drop = FALSE], seed)
      } else {
        res <- fit_predict(meth, s_tr[, a$predictors, drop = FALSE], ytr,
                           s_te[, a$predictors, drop = FALSE], seed)
      }
      sc <- score(yte, res$pred, mean(ytr))
      data.frame(app = nm, rep = r, fold = k, method = meth,
                 config = res$config, rmse = sc[["rmse"]], mae = sc[["mae"]],
                 q2_predict = sc[["q2_predict"]], seconds = res$seconds)
    }))
  }
  cv <- if (cores > 1L) {
    do.call(rbind, parallel::mclapply(seq_len(nrow(jobs)), run_job,
                                      mc.cores = cores))
  } else {
    do.call(rbind, lapply(seq_len(nrow(jobs)), run_job))
  }
  all_cv[[nm]] <- cv
  # average over folds within a repeat, then compare across repeats
  per_rep <- stats::aggregate(cbind(rmse, mae, q2_predict) ~ method + rep,
                              data = cv, FUN = mean)
  ref <- per_rep[per_rep$method == "bspline_kan", ]
  summ <- do.call(rbind, lapply(split(per_rep, per_rep$method), function(d) {
    d <- d[order(d$rep), ]
    rr <- ref[match(d$rep, ref$rep), ]
    dd <- d$rmse - rr$rmse
    data.frame(method = d$method[1L], rmse = mean(d$rmse),
               sd_rmse = stats::sd(d$rmse), mae = mean(d$mae),
               q2_predict = mean(d$q2_predict),
               rmse_minus_bspline_kan = mean(dd),
               p_paired = if (all(dd == 0)) NA_real_ else
                 suppressWarnings(stats::wilcox.test(dd)$p.value))
  }))
  say(cat(sprintf("\n%d x %d-fold CV (fold-internal PLS weights):\n",
                  repeats, folds)))
  say(print(summ[order(summ$rmse), ], digits = 4, row.names = FALSE))
  kc <- cv[cv$method == "bspline_kan", "config"]
  say(cat("bspline_kan configurations chosen:\n"))
  say(print(table(kc)))

  # ---- 3. interpretation on the full sample ---------------------------------
  s_all <- fit$scores
  x_all <- s_all[, a$predictors, drop = FALSE]
  kfit <- splineKANr::kan_fit(x_all, s_all[, a$target], hidden = integer(),
                              grid_size = 5L, epochs = 400L, lr = 0.01,
                              patience = 30L, backend = "r", seed = 1L)
  imp <- splineKANr::kan_edge_importance(kfit, x_all)[[1L]][, 1L]
  lin <- stats::lm(s_all[, a$target] ~ x_all)
  sy <- stats::sd(s_all[, a$target])
  pd_stats <- t(vapply(a$predictors, function(v) {
    pd <- splineKANr::kan_partial_dependence(kfit, x_all, v, n = 50L)
    res <- stats::residuals(stats::lm(yhat ~ value, data = pd))
    # effect size and departure from a straight line, in sd(target) units
    c(pd_range = diff(range(pd$yhat)) / sy, pd_nonlin = stats::sd(res) / sy)
  }, numeric(2)))
  say(cat("\nAdditive KAN on full-sample scores (PD in sd(target) units):\n"))
  say(print(data.frame(predictor = a$predictors,
                       edge_importance = round(imp, 4),
                       ols_beta = round(stats::coef(lin)[-1L], 4),
                       pd_range = round(pd_stats[, "pd_range"], 3),
                       pd_nonlin = round(pd_stats[, "pd_nonlin"], 4)),
            row.names = FALSE))
  grDevices::png(file.path(out_dir, paste0(nm, "_partial_dependence.png")),
                 width = 1400, height = 900, res = 130)
  np <- length(a$predictors)
  graphics::par(mfrow = c(ceiling(np / 4), min(np, 4L)))
  for (v in a$predictors) {
    pd <- splineKANr::kan_partial_dependence(kfit, x_all, v, n = 50L)
    plot(pd, main = v, ylab = paste("PD of", a$target))
    graphics::rug(x_all[, v], col = "grey60")
  }
  grDevices::dev.off()
}
close(log_con)
utils::write.csv(do.call(rbind, all_cv),
                 file.path(out_dir, "application_cv_results.csv"),
                 row.names = FALSE)
