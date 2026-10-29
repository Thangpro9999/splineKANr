# Base R reference forward pass and analytic backpropagation ----------------
#
# Edge (i -> j) of a layer:
#   phi_ij(x) = mask_ij * (wb_ij * b(x) + ws_ij * s_ij(x)),
#   s_ij(x)   = sum_m coef_ijm * E_m(x),
# where E are the spline features after the extrapolation policy
# (see edge_features_r()). Node j sums its incoming edges and adds bias_j.

#' Forward pass of a KAN in base R
#'
#' Dependency-free reference implementation of the network defined by a
#' `kan_state` (see [kan_init_state()]). It is used as a numerical oracle
#' for the torch backend and
#' for analysis (edge plots, attribution, grid updates).
#'
#' @param state A `kan_state` (see [kan_init_state()]) or a fitted
#'   `kan_model`.
#' @param x Numeric matrix with `width[1]` columns (already normalized).
#' @param detail Logical; if `TRUE`, return per-layer intermediate values.
#' @return If `detail = FALSE`, the output matrix. Otherwise a list with
#'   `output` and `layers`; each layer entry has `input`, `phi` (array
#'   `n x n_in x n_out`), `spline`, `spline_deriv`, `base`, `base_deriv`
#'   and `features` (list of feature matrices per input).
#' @examples
#' st <- kan_init_state(kan_spec(c(2, 3, 1)), seed = 1)
#' kan_forward_r(st, matrix(c(0.1, -0.4, 0.5, 0.9), ncol = 2))
#' @export
kan_forward_r <- function(state, x, detail = FALSE) {
  state <- as_kan_state(state)
  spec <- state$spec
  x <- as_numeric_matrix(x)
  if (ncol(x) != spec$width[1L]) {
    stop(sprintf("`x` must have %d columns.", spec$width[1L]), call. = FALSE)
  }
  info <- vector("list", length(state$layers))
  h <- x
  for (l in seq_along(state$layers)) {
    res <- layer_forward_r(state$layers[[l]], h, spec)
    info[[l]] <- res
    h <- res$output
  }
  if (!detail) return(h)
  list(output = h, layers = info)
}

#' One layer forward in base R
#' @noRd
layer_forward_r <- function(L, x, spec) {
  n <- nrow(x)
  n_in <- ncol(x)
  n_out <- length(L$bias)
  k <- spec$spline_order
  spl <- array(0, c(n, n_in, n_out))
  dspl <- array(0, c(n, n_in, n_out))
  base <- matrix(0, n, n_in)
  dbase <- matrix(0, n, n_in)
  feats <- vector("list", n_in)
  for (i in seq_len(n_in)) {
    Fi <- edge_features_r(x[, i], L$grid[i, ], k, spec$extrapolate)
    feats[[i]] <- Fi
    Ci <- matrix(L$coef[i, , ], nrow = n_out)          # n_out x n_basis
    spl[, i, ] <- Fi$value %*% t(Ci)
    dspl[, i, ] <- Fi$deriv %*% t(Ci)
    bf <- base_fun_r(x[, i], spec$base_fun)
    base[, i] <- bf$value
    dbase[, i] <- bf$deriv
  }
  phi <- array(0, c(n, n_in, n_out))
  for (i in seq_len(n_in)) {
    for (j in seq_len(n_out)) {
      phi[, i, j] <- L$mask[i, j] *
        (L$scale_base[i, j] * base[, i] + L$scale_sp[i, j] * spl[, i, j])
    }
  }
  out <- apply(phi, c(1L, 3L), sum)
  out <- matrix(out, nrow = n, ncol = n_out)
  if (isTRUE(spec$bias)) out <- sweep(out, 2L, L$bias, `+`)
  list(input = x, phi = phi, spline = spl, spline_deriv = dspl,
       base = base, base_deriv = dbase, features = feats, output = out)
}

#' Mean squared error loss and analytic gradients in base R
#'
#' Computes `L = mean((f(x) - y)^2)` and its exact gradient with respect to
#' every trainable parameter by manual backpropagation. Used to validate
#' torch autograd and finite differences (Phase 2 of the validation plan).
#'
#' @param state A `kan_state`.
#' @param x Numeric input matrix.
#' @param y Numeric response (vector or matrix with `width[length(width)]`
#'   columns).
#' @return A list with `loss`, `grads` (per layer: `coef`, `scale_base`,
#'   `scale_sp`, `bias`) and `input_grad` (gradient with respect to `x`).
#' @examples
#' st <- kan_init_state(kan_spec(c(1, 2, 1)), seed = 2)
#' x <- matrix(seq(-0.9, 0.9, length.out = 7))
#' g <- kan_gradient_r(st, x, sin(3 * x))
#' g$loss
#' @export
kan_gradient_r <- function(state, x, y) {
  state <- as_kan_state(state)
  spec <- state$spec
  fw <- kan_forward_r(state, x, detail = TRUE)
  y <- as_numeric_matrix(y, "y")
  if (!identical(dim(y), dim(fw$output))) {
    stop("`y` does not match the network output shape.", call. = FALSE)
  }
  resid <- fw$output - y
  loss <- mean(resid^2)
  g_out <- 2 * resid / length(resid)            # dL / d output
  grads <- vector("list", length(state$layers))
  for (l in rev(seq_along(state$layers))) {
    L <- state$layers[[l]]
    I <- fw$layers[[l]]
    n_in <- ncol(I$input)
    n_out <- length(L$bias)
    g_coef <- array(0, dim(L$coef))
    g_sb <- matrix(0, n_in, n_out)
    g_ss <- matrix(0, n_in, n_out)
    g_x <- matrix(0, nrow(I$input), n_in)
    for (i in seq_len(n_in)) {
      FtG <- crossprod(I$features[[i]]$value, g_out)  # n_basis x n_out
      for (j in seq_len(n_out)) {
        m <- L$mask[i, j]
        g_coef[i, j, ] <- m * L$scale_sp[i, j] * FtG[, j]
        g_sb[i, j] <- m * sum(g_out[, j] * I$base[, i])
        g_ss[i, j] <- m * sum(g_out[, j] * I$spline[, i, j])
        g_x[, i] <- g_x[, i] + g_out[, j] * m *
          (L$scale_base[i, j] * I$base_deriv[, i] +
             L$scale_sp[i, j] * I$spline_deriv[, i, j])
      }
    }
    grads[[l]] <- list(
      coef = g_coef, scale_base = g_sb, scale_sp = g_ss,
      bias = if (isTRUE(spec$bias)) colSums(g_out) else numeric(n_out)
    )
    g_out <- g_x
  }
  list(loss = loss, grads = grads, input_grad = g_out)
}

#' Central finite-difference gradient of the MSE loss (base R)
#'
#' @inheritParams kan_gradient_r
#' @param layer Layer index.
#' @param param One of `"coef"`, `"scale_base"`, `"scale_sp"`, `"bias"`.
#' @param index Integer vector of linear indices into the parameter array.
#' @param h Step size.
#' @return Numeric vector of finite-difference derivatives
#'   `(L(p + h) - L(p - h)) / (2h)`.
#' @examples
#' st <- kan_init_state(kan_spec(c(1, 2, 1)), seed = 2)
#' x <- matrix(seq(-0.9, 0.9, length.out = 7))
#' kan_fd_gradient_r(st, x, sin(3 * x), layer = 1, param = "coef", index = 1:3)
#' @export
kan_fd_gradient_r <- function(state, x, y, layer, param, index, h = 1e-6) {
  state <- as_kan_state(state)
  param <- match.arg(param, c("coef", "scale_base", "scale_sp", "bias"))
  y <- as_numeric_matrix(y, "y")
  loss_at <- function(st) mean((kan_forward_r(st, x) - y)^2)
  vapply(index, function(id) {
    sp <- state
    sm <- state
    sp$layers[[layer]][[param]][id] <- sp$layers[[layer]][[param]][id] + h
    sm$layers[[layer]][[param]][id] <- sm$layers[[layer]][[param]][id] - h
    (loss_at(sp) - loss_at(sm)) / (2 * h)
  }, numeric(1))
}

#' Coerce to a kan_state
#' @noRd
as_kan_state <- function(x) {
  if (inherits(x, "kan_state")) return(x)
  if (inherits(x, "kan_model")) return(x$state)
  stop("Expected a `kan_state` or `kan_model` object.", call. = FALSE)
}
