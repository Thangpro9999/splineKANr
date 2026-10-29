# Minimal PLS path modelling (PLS-PM / PLSc) in base R --------------------------
#
# Used by the business application scripts to obtain latent-variable scores
# without extra dependencies. Algorithm (Lohmoeller 1989; Hair et al. 2022):
#
#   * indicators standardised (mean 0, sd 1, n - 1 denominator);
#   * outer estimation mode A (reflective, default) or mode B (formative,
#     weights from the regression of the inner proxy on the block);
#   * inner weighting "path" (default of SmartPLS, seminr and cSEM);
#   * stop when max |w_new - w_old| < tol (weights scaled to unit-variance
#     composites);
#   * structural paths by OLS on the composite scores (PLS), or on the
#     construct correlations corrected with rho_A (consistent PLS, PLSc,
#     Dijkstra & Henseler 2015; the cSEM default `.disattenuate = TRUE`).
#
# The result is checked against plspm::plspm() in
# inst/applications/check_pls_pm.R.
#
# Scores for new observations (e.g. a test fold) are computed with the
# training means, standard deviations and weights: see pls_scores().

#' @param data data frame with the indicators.
#' @param blocks named list: construct -> character vector of indicators.
#' @param paths two-column matrix/data frame (from, to).
#' @param modes named character vector, "A" or "B" per construct (default A).
#' @noRd
pls_pm <- function(data, blocks, paths, modes = NULL, tol = 1e-5,
                   max_iter = 300L) {
  paths <- as.matrix(paths)
  lv <- names(blocks)
  J <- length(lv)
  mode <- stats::setNames(rep("A", J), lv)
  if (!is.null(modes)) mode[names(modes)] <- modes
  D <- matrix(0, J, J, dimnames = list(lv, lv))     # D[i, j] = 1 if i -> j
  D[paths] <- 1
  items <- unlist(blocks, use.names = FALSE)
  X <- as.matrix(data[, items])
  mu <- colMeans(X)
  sdv <- apply(X, 2, stats::sd)
  Z <- scale(X, mu, sdv)
  n <- nrow(Z)
  idx <- lapply(blocks, function(b) match(b, items))
  w <- lapply(idx, function(i) rep(1, length(i)))
  score <- function(w) {
    Y <- vapply(seq_len(J), function(j) Z[, idx[[j]], drop = FALSE] %*% w[[j]],
                numeric(n))
    s <- apply(Y, 2, stats::sd)
    list(Y = sweep(Y, 2, s, "/"), w = Map(`/`, w, s))
  }
  cur <- score(w)
  conv <- FALSE
  for (it in seq_len(max_iter)) {
    Y <- cur$Y
    R <- stats::cor(Y)
    E <- matrix(0, J, J)
    for (j in seq_len(J)) {
      pred <- which(D[, j] == 1)
      succ <- which(D[j, ] == 1)
      if (length(pred)) {
        E[pred, j] <- solve(R[pred, pred, drop = FALSE], R[pred, j])
      }
      if (length(succ)) E[succ, j] <- R[j, succ]
    }
    Zin <- Y %*% E                                   # inner proxies
    w_new <- lapply(seq_len(J), function(j) {
      Xb <- Z[, idx[[j]], drop = FALSE]
      if (mode[[j]] == "B" && ncol(Xb) > 1L) {
        as.numeric(solve(crossprod(Xb), crossprod(Xb, Zin[, j])))
      } else {
        as.numeric(crossprod(Xb, Zin[, j])) / (n - 1)
      }
    })
    nxt <- score(w_new)
    delta <- max(abs(unlist(nxt$w) - unlist(cur$w)))
    cur <- nxt
    if (delta < tol) {
      conv <- TRUE
      break
    }
  }
  Y <- cur$Y
  colnames(Y) <- lv
  w <- stats::setNames(cur$w, lv)
  for (j in seq_len(J)) names(w[[j]]) <- blocks[[j]]
  loadings <- stats::setNames(lapply(seq_len(J), function(j) {
    stats::setNames(as.numeric(stats::cor(Z[, idx[[j]], drop = FALSE],
                                          Y[, j])), blocks[[j]])
  }), lv)
  # rho_A (Dijkstra & Henseler 2015); 1 for single-indicator constructs
  rho_a <- vapply(seq_len(J), function(j) {
    k <- idx[[j]]
    if (length(k) == 1L) return(1)
    S <- stats::cor(Z[, k, drop = FALSE])
    ww <- w[[j]]
    off <- function(M) M - diag(diag(M))
    as.numeric((sum(ww^2))^2 * (t(ww) %*% off(S) %*% ww) /
                 (t(ww) %*% off(ww %*% t(ww)) %*% ww))
  }, numeric(1))
  names(rho_a) <- lv
  R <- stats::cor(Y)
  Rc <- R / sqrt(outer(rho_a, rho_a))
  diag(Rc) <- 1
  structural <- function(R) {
    out <- list()
    for (j in seq_len(J)) {
      pred <- which(D[, j] == 1)
      if (!length(pred)) next
      b <- solve(R[pred, pred, drop = FALSE], R[pred, j])
      out[[lv[j]]] <- list(beta = stats::setNames(b, lv[pred]),
                           r2 = sum(b * R[pred, j]))
    }
    out
  }
  ave <- vapply(loadings, function(l) mean(l^2), numeric(1))
  cr <- vapply(loadings, function(l) sum(l)^2 / (sum(l)^2 + sum(1 - l^2)),
               numeric(1))
  structure(list(
    scores = Y, weights = w, loadings = loadings, rho_a = rho_a,
    ave = ave, cr = cr, lv_cor = R, paths_pls = structural(R),
    paths_plsc = structural(Rc), iterations = it, converged = conv,
    center = mu, scale = sdv, blocks = blocks, D = D, modes = mode
  ), class = "pls_pm_fit")
}

#' Latent scores for (new) data with the weights of a fitted model
#' @noRd
pls_scores <- function(fit, newdata) {
  items <- unlist(fit$blocks, use.names = FALSE)
  Z <- scale(as.matrix(newdata[, items]), fit$center, fit$scale)
  out <- vapply(names(fit$blocks), function(b) {
    as.numeric(Z[, fit$blocks[[b]], drop = FALSE] %*% fit$weights[[b]])
  }, numeric(nrow(Z)))
  matrix(out, nrow = nrow(Z), dimnames = list(NULL, names(fit$blocks)))
}

#' Path table
#' @noRd
pls_path_table <- function(fit, type = c("plsc", "pls")) {
  type <- match.arg(type)
  p <- if (type == "plsc") fit$paths_plsc else fit$paths_pls
  do.call(rbind, lapply(names(p), function(to) {
    data.frame(from = names(p[[to]]$beta), to = to,
               beta = unname(p[[to]]$beta), r2 = p[[to]]$r2)
  }))
}
