# B-spline knots and basis functions ----------------------------------------
#
# Conventions (shared by the base R reference and the torch backend):
#   * `G` = number of grid intervals on the domain [a, b], `k` = spline degree.
#   * The knot vector has G + 2k + 1 knots: the G + 1 domain knots plus k
#     extra uniformly spaced knots on each side ("extended grid", as in pykan).
#   * There are G + k basis functions. On [a, b] they form a partition of
#     unity; outside [a, b] they do not.
#   * Intervals are half-open [t_j, t_{j+1}). With `right_closed = TRUE` the
#     point x == b is assigned to the last interval inside the domain, so the
#     basis and its derivative at b equal their left limits.

#' Build an extended uniform knot grid
#'
#' Creates the knot matrix used by every KAN layer: `grid_size` uniform
#' intervals on `grid_range`, extended by `spline_order` knots on each side.
#'
#' @param n_in Number of input dimensions (rows of the result).
#' @param grid_size Number of intervals `G` on the domain (integer >= 1).
#' @param spline_order Spline degree `k` (integer >= 1; 3 = cubic).
#' @param grid_range Numeric vector `c(a, b)` with `a < b`, or a matrix with
#'   `n_in` rows and two columns giving one range per input.
#' @return A numeric matrix with `n_in` rows and `grid_size + 2 * spline_order
#'   + 1` columns. Row `i` holds the sorted knots for input `i`.
#' @examples
#' kan_make_grid(n_in = 2, grid_size = 5, spline_order = 3)
#' @export
kan_make_grid <- function(n_in, grid_size = 5L, spline_order = 3L,
                          grid_range = c(-1, 1)) {
  n_in <- check_count(n_in, "n_in")
  grid_size <- check_count(grid_size, "grid_size")
  spline_order <- check_count(spline_order, "spline_order")
  if (is.matrix(grid_range)) {
    if (nrow(grid_range) != n_in || ncol(grid_range) != 2L) {
      stop("`grid_range` matrix must have `n_in` rows and 2 columns.",
           call. = FALSE)
    }
    rng <- grid_range
  } else {
    if (!is.numeric(grid_range) || length(grid_range) != 2L) {
      stop("`grid_range` must be a numeric vector of length 2.", call. = FALSE)
    }
    rng <- matrix(grid_range, nrow = n_in, ncol = 2L, byrow = TRUE)
  }
  if (any(!is.finite(rng)) || any(rng[, 2] <= rng[, 1])) {
    stop("Each `grid_range` must be finite with lower < upper.", call. = FALSE)
  }
  inner <- t(apply(rng, 1L, function(r) {
    seq(r[1L], r[2L], length.out = grid_size + 1L)
  }))
  if (grid_size == 0L) inner <- matrix(inner, nrow = n_in)
  kan_extend_grid(matrix(inner, nrow = n_in), spline_order)
}

#' Extend a domain grid by uniform knots on both sides
#'
#' @param grid Numeric matrix of domain knots (one row per input), with at
#'   least two strictly increasing columns.
#' @param spline_order Number of knots `k` to add on each side.
#' @return A matrix with `2 * spline_order` more columns than `grid`. The
#'   spacing of the added knots is `(max - min) / (ncol(grid) - 1)`, as in
#'   pykan's `extend_grid()`.
#' @examples
#' kan_extend_grid(matrix(c(-1, 0, 1), nrow = 1), spline_order = 2)
#' @export
kan_extend_grid <- function(grid, spline_order) {
  if (!is.matrix(grid) || !is.numeric(grid) || ncol(grid) < 2L) {
    stop("`grid` must be a numeric matrix with at least 2 columns.",
         call. = FALSE)
  }
  spline_order <- check_count(spline_order, "spline_order", min = 0L)
  if (any(apply(grid, 1L, function(r) any(diff(r) <= 0)))) {
    stop("Each row of `grid` must be strictly increasing.", call. = FALSE)
  }
  h <- (grid[, ncol(grid)] - grid[, 1L]) / (ncol(grid) - 1L)
  if (spline_order == 0L) return(grid)
  left <- vapply(seq_len(spline_order), function(s) grid[, 1L] - s * h,
                 numeric(nrow(grid)))
  right <- vapply(seq_len(spline_order),
                  function(s) grid[, ncol(grid)] + s * h,
                  numeric(nrow(grid)))
  left <- matrix(left, nrow = nrow(grid))
  right <- matrix(right, nrow = nrow(grid))
  cbind(left[, rev(seq_len(spline_order)), drop = FALSE], grid, right)
}

#' Domain bounds of an extended grid
#' @param grid Knot matrix from [kan_make_grid()].
#' @param spline_order Spline degree.
#' @return Matrix with columns `lower` and `upper`, one row per input.
#' @noRd
grid_domain <- function(grid, spline_order) {
  cbind(lower = grid[, spline_order + 1L],
        upper = grid[, ncol(grid) - spline_order])
}

# Base R reference implementation ---------------------------------------------

#' B-spline basis by the Cox-de Boor recursion (base R reference)
#'
#' Evaluates all B-spline basis functions of degree `spline_order` defined by
#' the knot vector `knots` at the points `x`. This is the dependency-free
#' numerical oracle against which the torch backend is validated.
#'
#' @param x Numeric vector of evaluation points.
#' @param knots Strictly increasing numeric knot vector.
#' @param spline_order Spline degree `k` (>= 0).
#' @param right_closed Logical. If `TRUE`, points equal to the right end of
#'   the domain `knots[length(knots) - k]` are evaluated as left limits (see
#'   Details). Use `TRUE` to match [splines::splineDesign()] at the boundary.
#' @details Intervals are half-open `[t_j, t_{j+1})`. Points outside
#'   `range(knots)` get all-zero rows. Basis functions with zero-length
#'   support denominators are treated as zero (`0/0 := 0`).
#' @return A matrix with `length(x)` rows and `length(knots) - k - 1`
#'   columns.
#' @examples
#' g <- kan_make_grid(1, grid_size = 4, spline_order = 3)[1, ]
#' B <- kan_bspline_basis_r(seq(-1, 1, by = 0.25), g, 3)
#' rowSums(B) # partition of unity on [-1, 1]
#' @export
kan_bspline_basis_r <- function(x, knots, spline_order = 3L,
                                right_closed = FALSE) {
  check_knots(knots)
  k <- check_count(spline_order, "spline_order", min = 0L)
  if (!is.numeric(x)) stop("`x` must be numeric.", call. = FALSE)
  n_knots <- length(knots)
  if (n_knots < k + 2L) {
    stop("Need at least `spline_order + 2` knots.", call. = FALSE)
  }
  n_int <- n_knots - 1L
  B <- matrix(0, nrow = length(x), ncol = n_int)
  for (j in seq_len(n_int)) {
    B[, j] <- as.numeric(x >= knots[j] & x < knots[j + 1L])
  }
  if (right_closed) {
    b <- knots[n_knots - k]
    at_b <- !is.na(x) & x == b
    if (any(at_b)) {
      B[at_b, ] <- 0
      B[at_b, n_knots - k - 1L] <- 1
    }
  }
  if (k >= 1L) {
    for (p in seq_len(k)) {
      n_new <- n_int - p
      B_new <- matrix(0, nrow = length(x), ncol = n_new)
      for (i in seq_len(n_new)) {
        d1 <- knots[i + p] - knots[i]
        d2 <- knots[i + p + 1L] - knots[i + 1L]
        left <- if (d1 > 0) (x - knots[i]) / d1 * B[, i] else 0
        right <- if (d2 > 0) (knots[i + p + 1L] - x) / d2 * B[, i + 1L] else 0
        B_new[, i] <- left + right
      }
      B <- B_new
    }
  }
  B[is.na(B)] <- 0
  B
}

#' First derivative of the B-spline basis (base R reference)
#'
#' Uses the analytic identity
#' \deqn{\frac{d}{dx} N_{i,k}(x) = \frac{k}{t_{i+k} - t_i} N_{i,k-1}(x) -
#'   \frac{k}{t_{i+k+1} - t_{i+1}} N_{i+1,k-1}(x).}
#'
#' @inheritParams kan_bspline_basis_r
#' @return A matrix of the same shape as [kan_bspline_basis_r()].
#' @examples
#' g <- kan_make_grid(1, grid_size = 4, spline_order = 3)[1, ]
#' kan_bspline_deriv_r(c(-0.3, 0.2), g, 3)
#' @export
kan_bspline_deriv_r <- function(x, knots, spline_order = 3L,
                                right_closed = FALSE) {
  check_knots(knots)
  k <- check_count(spline_order, "spline_order", min = 0L)
  n_basis <- length(knots) - k - 1L
  if (k == 0L) return(matrix(0, nrow = length(x), ncol = n_basis))
  Bm1 <- kan_bspline_basis_r(x, knots, k - 1L, right_closed = right_closed)
  out <- matrix(0, nrow = length(x), ncol = n_basis)
  for (i in seq_len(n_basis)) {
    d1 <- knots[i + k] - knots[i]
    d2 <- knots[i + k + 1L] - knots[i + 1L]
    a1 <- if (d1 > 0) k / d1 * Bm1[, i] else 0
    a2 <- if (d2 > 0) k / d2 * Bm1[, i + 1L] else 0
    out[, i] <- a1 - a2
  }
  out
}

#' Edge spline features with an extrapolation policy (base R reference)
#'
#' Returns the design matrix `E` such that the spline part of an edge is
#' `E %*% coef`, together with its derivative with respect to `x`.
#'
#' @param x Numeric vector.
#' @param knots Extended knot vector of one input.
#' @param spline_order Spline degree.
#' @param extrapolate One of `"linear"`, `"none"`, `"clamp"`; see
#'   [kan_spec()].
#' @return List with matrices `value` and `deriv`.
#' @noRd
edge_features_r <- function(x, knots, spline_order, extrapolate) {
  k <- spline_order
  if (extrapolate == "none") {
    return(list(
      value = kan_bspline_basis_r(x, knots, k),
      deriv = kan_bspline_deriv_r(x, knots, k)
    ))
  }
  a <- knots[k + 1L]
  b <- knots[length(knots) - k]
  xc <- pmin(pmax(x, a), b)
  B <- kan_bspline_basis_r(xc, knots, k, right_closed = TRUE)
  dB <- kan_bspline_deriv_r(xc, knots, k, right_closed = TRUE)
  if (extrapolate == "clamp") {
    inside <- x >= a & x <= b
    return(list(value = B, deriv = dB * as.numeric(inside)))
  }
  # linear: first-order Taylor extension from the nearest domain bound
  list(value = B + dB * (x - xc), deriv = dB)
}

# Torch backend ---------------------------------------------------------------

#' B-spline basis in torch (Cox-de Boor, batched)
#'
#' Differentiable with respect to `x`; used inside [kan_layer()].
#'
#' @param x Tensor of shape `(batch, n_in)`.
#' @param grid Tensor of shape `(n_in, n_knots)` (extended knots).
#' @param spline_order Spline degree `k`.
#' @param right_closed Logical, see [kan_bspline_basis_r()].
#' @return Tensor of shape `(batch, n_in, n_knots - k - 1)`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   g <- torch::torch_tensor(kan_make_grid(2, 5, 3))
#'   x <- torch::torch_rand(4, 2) * 2 - 1
#'   kan_bspline_basis(x, g, 3)$sum(dim = 3) # ~1 on the domain
#' }
#' @export
kan_bspline_basis <- function(x, grid, spline_order = 3L,
                              right_closed = FALSE) {
  k <- check_count(spline_order, "spline_order", min = 0L)
  n_knots <- grid$size(2)
  xx <- x$unsqueeze(3)                     # (B, in, 1)
  g <- grid$unsqueeze(1)                   # (1, in, M)
  left <- g[, , 1:(n_knots - 1L), drop = FALSE]
  right <- g[, , 2:n_knots, drop = FALSE]
  B <- ((xx >= left) & (xx < right))$to(dtype = x$dtype)
  if (isTRUE(right_closed)) {
    b <- g[, , n_knots - k, drop = FALSE]  # (1, in, 1)
    onehot <- torch::torch_zeros(1L, 1L, n_knots - 1L, dtype = x$dtype,
                                 device = x$device)
    onehot[1L, 1L, n_knots - k - 1L] <- 1
    B <- torch::torch_where(xx == b, onehot, B)
  }
  if (k >= 1L) {
    for (p in seq_len(k)) {
      n_new <- n_knots - 1L - p
      t_i <- g[, , 1:n_new, drop = FALSE]
      t_ip <- g[, , (1L + p):(n_new + p), drop = FALSE]
      t_i1 <- g[, , 2:(n_new + 1L), drop = FALSE]
      t_ip1 <- g[, , (2L + p):(n_new + p + 1L), drop = FALSE]
      B_lo <- B[, , 1:n_new, drop = FALSE]
      B_hi <- B[, , 2:(n_new + 1L), drop = FALSE]
      B <- (xx - t_i) / (t_ip - t_i) * B_lo +
        (t_ip1 - xx) / (t_ip1 - t_i1) * B_hi
    }
  }
  zero_nonfinite(B)
}

#' Analytic derivative of the torch B-spline basis
#' @inheritParams kan_bspline_basis
#' @return Tensor of shape `(batch, n_in, n_knots - k - 1)`.
#' @noRd
kan_bspline_deriv <- function(x, grid, spline_order = 3L,
                              right_closed = FALSE) {
  k <- spline_order
  n_knots <- grid$size(2)
  n_basis <- n_knots - k - 1L
  Bm1 <- kan_bspline_basis(x, grid, k - 1L, right_closed = right_closed)
  g <- grid$unsqueeze(1)
  d1 <- g[, , (1L + k):(n_basis + k), drop = FALSE] -
    g[, , 1:n_basis, drop = FALSE]
  d2 <- g[, , (2L + k):(n_basis + k + 1L), drop = FALSE] -
    g[, , 2:(n_basis + 1L), drop = FALSE]
  out <- k / d1 * Bm1[, , 1:n_basis, drop = FALSE] -
    k / d2 * Bm1[, , 2:(n_basis + 1L), drop = FALSE]
  zero_nonfinite(out)
}

#' Replace non-finite entries by 0 (the 0/0 := 0 convention)
#' @noRd
zero_nonfinite <- function(t) {
  torch::torch_where(torch::torch_isfinite(t), t, torch::torch_zeros_like(t))
}
