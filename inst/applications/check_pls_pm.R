# Cross-check of pls_pm() against plspm::plspm() (mode A, path scheme).
#   Rscript check_pls_pm.R APP_DIR mobi.csv [satisfaction.csv] [corp_rep.csv]
args <- commandArgs(trailingOnly = TRUE)
source(file.path(args[1L], "pls_pm.R"))
source(file.path(args[1L], "models.R"))
check <- function(name, data, m) {
  fit <- pls_pm(data, m$blocks, m$paths, m$modes, tol = 1e-10)
  lv <- names(m$blocks)
  inner <- matrix(0, length(lv), length(lv), dimnames = list(lv, lv))
  inner[m$paths[, c(2, 1)]] <- 1          # plspm: row = target
  ref <- plspm::plspm(data[, unlist(m$blocks)], inner,
                      lapply(m$blocks, function(b) match(b, unlist(m$blocks))),
                      modes = ifelse(lv %in% names(m$modes), "B", "A"), scheme = "path",
                      scaled = TRUE, tol = 1e-10, maxiter = 1000)
  sc_ref <- as.matrix(ref$scores)
  r_sc <- min(diag(cor(fit$scores, sc_ref[, lv])))
  ratio <- range(apply(sc_ref[, lv], 2, sd) / apply(fit$scores, 2, sd))
  pt <- pls_path_table(fit, "pls")
  b_ref <- ref$path_coefs[cbind(pt$to, pt$from)]
  cat(sprintf(paste("%s: converged=%s iter=%d | max|score diff|=%.2e",
                    "max|path diff|=%.2e max|loading diff|=%.2e | min cor(scores)=%.12f sd ratio %.6f-%.6f\n"),
              name, fit$converged, fit$iterations,
              max(abs(abs(fit$scores) - abs(sc_ref[, lv]))),
              max(abs(pt$beta - b_ref)),
              max(abs(unlist(fit$loadings) - ref$outer_model$loading)), r_sc, ratio[1], ratio[2]))
  invisible(fit)
}
mobi <- read.csv(args[2L])
check("mobi", mobi, mobi_model)
if (length(args) >= 3L) {
  check("satisfaction", read.csv(args[3L]), satisfaction_model)
}
if (length(args) >= 4L) {
  check("corp_rep", mean_impute(na_code(read.csv(args[4L]))), corp_rep_model)
}
