# KAN-specific sparsity regularisation ----------------------------------------
#
# Follows the structure of pykan's `MultKAN.reg()`: for every layer the edge
# scale A_ij = mean_n |phi_ij(x_n)| enters through
#   lambda_l1 * sum(A) + lambda_entropy * (H_in + H_out),
# where H_in / H_out are entropies of A normalised over incoming / outgoing
# edges (with pykan's "+ 1" in the denominator and log2(p + 1e-4)). Optional
# coefficient penalties add lambda_coef * sum_ij mean_out|c| and
# lambda_coefdiff * the same for first differences of c.

#' Regularization settings
#'
#' @param lambda Overall strength; `0` disables regularization.
#' @param lambda_l1 Weight of the L1 norm of edge scales.
#' @param lambda_entropy Weight of the edge-scale entropy.
#' @param lambda_coef Weight of the L1 norm of spline coefficients.
#' @param lambda_coefdiff Weight of the L1 norm of coefficient differences
#'   (a smoothness penalty).
#' @return A list of class `kan_reg`.
#' @examples
#' kan_reg_control(lambda = 1e-3)
#' @export
kan_reg_control <- function(lambda = 0, lambda_l1 = 1, lambda_entropy = 2,
                            lambda_coef = 0, lambda_coefdiff = 0) {
  structure(list(
    lambda = check_number(lambda, "lambda", 0),
    lambda_l1 = check_number(lambda_l1, "lambda_l1", 0),
    lambda_entropy = check_number(lambda_entropy, "lambda_entropy", 0),
    lambda_coef = check_number(lambda_coef, "lambda_coef", 0),
    lambda_coefdiff = check_number(lambda_coefdiff, "lambda_coefdiff", 0)
  ), class = "kan_reg")
}

#' Regularization penalty in base R
#'
#' @param state A `kan_state` or `kan_model`.
#' @param x Normalized input matrix.
#' @param reg A [kan_reg_control()] object.
#' @return The penalty *before* multiplication by `reg$lambda`.
#' @examples
#' st <- kan_init_state(kan_spec(c(2, 3, 1)), seed = 1)
#' kan_regularization_r(st, matrix(runif(20, -1, 1), ncol = 2),
#'                      kan_reg_control(lambda = 1))
#' @export
kan_regularization_r <- function(state, x, reg = kan_reg_control()) {
  state <- as_kan_state(state)
  fw <- kan_forward_r(state, x, detail = TRUE)
  total <- 0
  for (l in seq_along(fw$layers)) {
    A <- apply(abs(fw$layers[[l]]$phi), c(2L, 3L), mean)
    A <- matrix(A, nrow = dim(fw$layers[[l]]$phi)[2L])
    p_in <- sweep(A, 2L, colSums(A) + 1, `/`)
    p_out <- sweep(A, 1L, rowSums(A) + 1, `/`)
    h_in <- -mean(colSums(p_in * log2(p_in + 1e-4)))
    h_out <- -mean(rowSums(p_out * log2(p_out + 1e-4)))
    total <- total + reg$lambda_l1 * sum(A) +
      reg$lambda_entropy * (h_in + h_out)
    cf <- state$layers[[l]]$coef
    total <- total + reg$lambda_coef * sum(apply(abs(cf), c(1L, 3L), mean))
    if (dim(cf)[3L] > 1L) {
      dcf <- cf[, , -1L, drop = FALSE] - cf[, , -dim(cf)[3L], drop = FALSE]
      total <- total + reg$lambda_coefdiff *
        sum(apply(abs(dcf), c(1L, 3L), mean))
    }
  }
  total
}

#' Regularization penalty in torch
#' @param phis List of per-layer `phi` tensors `(batch, n_in, n_out)`.
#' @param net The [kan_net()].
#' @param reg A [kan_reg_control()].
#' @return Scalar tensor (before multiplication by `reg$lambda`).
#' @noRd
kan_regularization_torch <- function(phis, net, reg) {
  total <- NULL
  add <- function(a, b) if (is.null(a)) b else a + b
  # R numbers become float32 tensors by default; build constants in the
  # model's dtype so float64 models are not rounded to float32 precision.
  dt <- phis[[1L]]$dtype
  sc <- function(v) torch::torch_tensor(v, dtype = dt)
  eps <- sc(1e-4)
  for (l in seq_along(phis)) {
    A <- phis[[l]]$abs()$mean(dim = 1)                # (n_in, n_out)
    p_in <- A / (A$sum(dim = 1, keepdim = TRUE) + 1)
    p_out <- A / (A$sum(dim = 2, keepdim = TRUE) + 1)
    h_in <- -(p_in * torch::torch_log2(p_in + eps))$sum(dim = 1)$mean()
    h_out <- -(p_out * torch::torch_log2(p_out + eps))$sum(dim = 2)$mean()
    total <- add(total, sc(reg$lambda_l1) * A$sum() +
                   sc(reg$lambda_entropy) * (h_in + h_out))
    cf <- net$layers[[l]]$coef
    if (reg$lambda_coef > 0) {
      total <- total + sc(reg$lambda_coef) * cf$abs()$mean(dim = 2)$sum()
    }
    if (reg$lambda_coefdiff > 0 && cf$size(3) > 1L) {
      total <- total + sc(reg$lambda_coefdiff) *
        torch::torch_diff(cf, dim = 3)$abs()$mean(dim = 2)$sum()
    }
  }
  total
}
