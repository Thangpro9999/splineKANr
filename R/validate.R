# Numerical validation (Phases 1-2 of the release plan) -------------------------
#
# Every function returns a data frame with the measured error, the
# acceptance threshold and a logical `pass`, so results can be logged and
# reported. Thresholds are the internal acceptance criteria of the project.

#' Validate the B-spline basis
#'
#' Compares the basis with [splines::splineDesign()] on interior points and
#' checks non-negativity and the partition of unity on the domain.
#'
#' @param grid_sizes,spline_orders Settings to test.
#' @param n Number of evaluation points per setting.
#' @param backend `"r"`, `"torch"` or `"both"` (torch is skipped with a
#'   message when not installed).
#' @param tol Acceptance threshold for maximum absolute errors.
#' @return A data frame with one row per backend, grid size and order.
#' @examples
#' kan_validate_basis(backend = "r")
#' @export
kan_validate_basis <- function(grid_sizes = c(3L, 5L, 10L),
                               spline_orders = 1:3, n = 1000L,
                               backend = c("both", "r", "torch"),
                               tol = 1e-6) {
  backend <- match.arg(backend)
  backends <- resolve_backends(backend)
  rows <- list()
  for (G in grid_sizes) for (k in spline_orders) {
    knots <- kan_make_grid(1L, G, k)[1L, ]
    x <- interior_points(knots, k, n)
    ref <- splines::splineDesign(knots, x, ord = k + 1L, outer.ok = TRUE)
    for (be in backends) {
      B <- basis_backend(x, knots, k, be, right_closed = FALSE)
      rows[[length(rows) + 1L]] <- data.frame(
        backend = be, grid_size = G, spline_order = k,
        max_err_splineDesign = max(abs(B - ref)),
        max_err_partition = max(abs(rowSums(B) - 1)),
        min_value = min(B), tol = tol
      )
    }
  }
  out <- do.call(rbind, rows)
  out$pass <- out$max_err_splineDesign < tol & out$max_err_partition < tol &
    out$min_value > -tol
  out
}

#' Validate the first derivative of the B-spline basis
#'
#' Compares the analytic derivative (and, for torch, autograd) with
#' `splines::splineDesign(derivs = 1)` and with central finite differences at
#' points away from the knots. The relative error uses
#' `|a - b| / max(|b|, atol)` so values near zero are judged absolutely.
#'
#' @inheritParams kan_validate_basis
#' @param h Finite-difference step.
#' @param atol Absolute floor in the relative error.
#' @param tol Acceptance threshold for the relative error.
#' @return A data frame with one row per backend and setting.
#' @examples
#' kan_validate_derivative(backend = "r")
#' @export
kan_validate_derivative <- function(grid_sizes = c(3L, 5L, 10L),
                                    spline_orders = 1:3, n = 500L,
                                    backend = c("both", "r", "torch"),
                                    h = 1e-6, atol = 1e-6, tol = 1e-4) {
  backend <- match.arg(backend)
  backends <- resolve_backends(backend)
  rows <- list()
  for (G in grid_sizes) for (k in spline_orders) {
    knots <- kan_make_grid(1L, G, k)[1L, ]
    x <- interior_points(knots, k, n, avoid = 10 * h)
    ref <- splines::splineDesign(knots, x, ord = k + 1L, derivs = 1L,
                                 outer.ok = TRUE)
    fd <- (kan_bspline_basis_r(x + h, knots, k) -
             kan_bspline_basis_r(x - h, knots, k)) / (2 * h)
    rel <- function(a, b) max(abs(a - b) / pmax(abs(b), atol))
    for (be in backends) {
      D <- if (be == "r") kan_bspline_deriv_r(x, knots, k) else
        torch_basis_autograd(x, knots, k)
      rows[[length(rows) + 1L]] <- data.frame(
        backend = be, grid_size = G, spline_order = k,
        rel_err_splineDesign = rel(D, ref),
        rel_err_finite_diff = rel(D, fd), tol = tol
      )
    }
  }
  out <- do.call(rbind, rows)
  out$pass <- out$rel_err_splineDesign < tol & out$rel_err_finite_diff < tol
  out
}

#' Validate parameter gradients of the network
#'
#' For random networks, compares the analytic base R gradient of the MSE loss
#' with central finite differences, and (if available) with torch autograd in
#' double precision. Inputs avoid the knots so the loss is smooth.
#'
#' @param widths List of architectures to test.
#' @param extrapolate Extrapolation policies to test.
#' @param n Number of samples.
#' @param seed Seed.
#' @param backend `"r"`, `"torch"` or `"both"`.
#' @param h Finite-difference step.
#' @param atol Absolute floor in the relative error.
#' @param tol Acceptance threshold for the relative error.
#' @return A data frame with one row per architecture, policy and
#'   comparison.
#' @examples
#' kan_validate_gradient(widths = list(c(2, 3, 1)), backend = "r")
#' @export
kan_validate_gradient <- function(widths = list(c(1, 1), c(2, 3, 1),
                                                c(3, 4, 2, 1)),
                                  extrapolate = c("linear", "none", "clamp"),
                                  n = 40L, seed = 1L,
                                  backend = c("both", "r", "torch"),
                                  h = 1e-6, atol = 1e-6, tol = 1e-3) {
  backend <- match.arg(backend)
  backends <- resolve_backends(backend)
  rows <- list()
  for (w in widths) for (ex in extrapolate) {
    spec <- kan_spec(w, extrapolate = ex)
    st <- kan_init_state(spec, seed = seed, noise_scale = 1)
    # widen the input range so some points use the extrapolation branch
    x <- matrix(stats::runif(n * w[1L], -1.3, 1.3), n, w[1L])
    y <- matrix(stats::rnorm(n * w[length(w)]), n, w[length(w)])
    an <- kan_gradient_r(st, x, y)
    an_vec <- flatten_grads(an$grads)
    fd_vec <- unlist(lapply(seq_along(st$layers), function(l) {
      unlist(lapply(c("coef", "scale_base", "scale_sp", "bias"), function(p) {
        kan_fd_gradient_r(st, x, y, l, p,
                          seq_along(st$layers[[l]][[p]]), h = h)
      }))
    }))
    rel <- max(abs(an_vec - fd_vec) / pmax(abs(fd_vec), atol))
    rows[[length(rows) + 1L]] <- data.frame(
      width = paste(w, collapse = "-"), extrapolate = ex,
      comparison = "analytic_r vs finite_diff", n_params = length(an_vec),
      max_rel_err = rel, tol = tol
    )
    if ("torch" %in% backends) {
      tg <- torch_gradient(st, x, y)
      rel_t <- max(abs(tg - an_vec) / pmax(abs(an_vec), atol))
      rows[[length(rows) + 1L]] <- data.frame(
        width = paste(w, collapse = "-"), extrapolate = ex,
        comparison = "torch_autograd vs analytic_r",
        n_params = length(an_vec), max_rel_err = rel_t, tol = tol
      )
    }
  }
  out <- do.call(rbind, rows)
  out$pass <- out$max_rel_err < out$tol
  out
}

#' Run all numerical validations
#'
#' @param backend `"both"`, `"r"` or `"torch"`.
#' @return A list of data frames (`basis`, `derivative`, `gradient`) with an
#'   attribute `all_pass`.
#' @examples
#' \donttest{
#' res <- kan_validate_all(backend = "r")
#' attr(res, "all_pass")
#' }
#' @export
kan_validate_all <- function(backend = c("both", "r", "torch")) {
  backend <- match.arg(backend)
  res <- list(
    basis = kan_validate_basis(backend = backend),
    derivative = kan_validate_derivative(backend = backend),
    gradient = kan_validate_gradient(backend = backend)
  )
  attr(res, "all_pass") <- all(vapply(res, function(d) all(d$pass),
                                      logical(1)))
  res
}

# ---- helpers -----------------------------------------------------------------

#' @noRd
resolve_backends <- function(backend) {
  has_torch <- torch::torch_is_installed()
  if (backend == "torch" && !has_torch) {
    stop("torch backend requested but libtorch is not installed.",
         call. = FALSE)
  }
  if (backend == "both") {
    if (!has_torch) {
      message("torch is not installed: validating the base R backend only.")
      return("r")
    }
    return(c("r", "torch"))
  }
  backend
}

#' Points strictly inside the domain, away from knots
#' @noRd
interior_points <- function(knots, k, n, avoid = 1e-9) {
  a <- knots[k + 1L]
  b <- knots[length(knots) - k]
  x <- seq(a, b, length.out = n + 2L)[-c(1L, n + 2L)]
  inner <- knots[(k + 1L):(length(knots) - k)]
  near <- vapply(x, function(v) min(abs(v - inner)) < avoid, logical(1))
  x[!near]
}

#' Basis values from a backend
#' @noRd
basis_backend <- function(x, knots, k, backend, right_closed) {
  if (backend == "r") {
    return(kan_bspline_basis_r(x, knots, k, right_closed = right_closed))
  }
  dt <- torch::torch_float64()
  xt <- torch::torch_tensor(matrix(x), dtype = dt)
  gt <- torch::torch_tensor(matrix(knots, nrow = 1L), dtype = dt)
  B <- kan_bspline_basis(xt, gt, k, right_closed = right_closed)
  matrix(as_r_array(B), nrow = length(x))
}

#' Derivative of each basis function by torch autograd
#'
#' One backward pass: x is replicated into one column per basis function
#' (each column gets the same knots), and only the "diagonal" basis value
#' `B_m(x[, m])` of column m enters the summed objective, so the gradient of
#' column m is dB_m/dx.
#' @noRd
torch_basis_autograd <- function(x, knots, k) {
  dt <- torch::torch_float64()
  n_basis <- length(knots) - k - 1L
  xr <- matrix(x, nrow = length(x), ncol = n_basis)
  gr <- matrix(knots, nrow = n_basis, ncol = length(knots), byrow = TRUE)
  xt <- torch::torch_tensor(xr, dtype = dt, requires_grad = TRUE)
  gt <- torch::torch_tensor(gr, dtype = dt)
  B <- kan_bspline_basis(xt, gt, k)                  # (n, n_basis, n_basis)
  sel <- torch::torch_eye(n_basis, dtype = dt)$unsqueeze(1)
  (B * sel)$sum()$backward()
  out <- matrix(as_r_array(xt$grad), nrow = length(x))
  if (any(!is.finite(out))) {
    warning("torch autograd returned non-finite basis derivatives.",
            call. = FALSE)
  }
  out
}

#' Flatten gradients in the canonical order
#' @noRd
flatten_grads <- function(grads) {
  unlist(lapply(grads, function(g) {
    c(as.numeric(g$coef), as.numeric(g$scale_base), as.numeric(g$scale_sp),
      as.numeric(g$bias))
  }))
}

#' torch autograd gradient (float64) in the canonical order
#' @noRd
torch_gradient <- function(state, x, y) {
  net <- kan_module_from_state(state, dtype = "float64")
  dt <- torch::torch_float64()
  loss <- torch::nnf_mse_loss(net(torch::torch_tensor(x, dtype = dt)),
                              torch::torch_tensor(y, dtype = dt))
  loss$backward()
  unlist(lapply(seq_along(net$layers), function(l) {
    m <- net$layers[[l]]
    gb <- if (!m$use_bias) {
      numeric(m$n_out)
    } else {
      as.numeric(as_r_array(m$bias$grad))
    }
    c(as.numeric(as_r_array(m$coef$grad)),
      as.numeric(as_r_array(m$scale_base$grad)),
      as.numeric(as_r_array(m$scale_sp$grad)), gb)
  }))
}
