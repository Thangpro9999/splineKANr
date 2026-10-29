# Adaptive grid update and grid refinement ------------------------------------

#' Update spline grids from data (adaptive grid)
#'
#' Re-places the knots of every edge spline at the empirical quantiles of the
#' current layer inputs, blended with a uniform grid:
#' `grid = grid_eps * uniform + (1 - grid_eps) * quantile`, as in pykan's
#' `update_grid_from_samples()`. The spline coefficients are then refitted
#' by (ridge-stabilized) least squares so that every edge keeps the same
#' spline values on the sample. Layers are processed in order, so each layer
#' sees the inputs produced by the already-updated layers before it.
#'
#' @param state A `kan_state` or `kan_model`. For a model, `x` is on the
#'   original scale and is normalized internally.
#' @param x Sample of network inputs (matrix with `width[1]` columns).
#' @param grid_size New number of intervals. `NULL` keeps the current size;
#'   a larger value refines the grid (pykan's grid extension).
#' @param grid_eps Weight of the uniform grid in `[0, 1]`; `1` gives a
#'   uniform grid over the data range, `0` a pure quantile grid.
#' @param ridge Relative ridge penalty for the coefficient refit (`0` for
#'   plain least squares, as pykan's `lstsq`).
#' @return An object of the same class as `state` with updated `grid` and
#'   `coef` arrays.
#' @examples
#' st <- kan_init_state(kan_spec(c(1, 2, 1)), seed = 1)
#' x <- matrix(rnorm(200, sd = 0.5))
#' st2 <- kan_update_grid(st, x)
#' range(st2$layers[[1]]$grid)
#' @export
kan_update_grid <- function(state, x, grid_size = NULL, grid_eps = 0.02,
                            ridge = 1e-8) {
  if (inherits(state, "kan_model")) {
    model <- state
    z <- kan_scale(as_numeric_matrix(x), model$scalers$x)
    model$state <- kan_update_grid(model$state, z, grid_size, grid_eps, ridge)
    model$spec <- model$state$spec
    return(model)
  }
  state <- as_kan_state(state)
  grid_eps <- check_number(grid_eps, "grid_eps", 0)
  if (grid_eps > 1) stop("`grid_eps` must be in [0, 1].", call. = FALSE)
  spec <- state$spec
  k <- spec$spline_order
  G_new <- if (is.null(grid_size)) spec$grid_size else
    check_count(grid_size, "grid_size")
  x <- as_numeric_matrix(x)
  if (nrow(x) < 2L) stop("Need at least two samples.", call. = FALSE)
  h <- x
  for (l in seq_along(state$layers)) {
    L <- state$layers[[l]]
    fw <- layer_forward_r(L, h, spec)
    n_in <- ncol(h)
    n_out <- length(L$bias)
    new_grid <- matrix(0, n_in, G_new + 2L * k + 1L)
    new_coef <- array(0, c(n_in, n_out, G_new + k))
    for (i in seq_len(n_in)) {
      g <- sample_grid(h[, i], G_new, grid_eps)
      if (is.null(g)) {
        # degenerate input (constant): keep a uniform grid on the old domain
        dom <- grid_domain(L$grid[i, , drop = FALSE], k)
        g <- seq(dom[1L], dom[2L], length.out = G_new + 1L)
      }
      knots <- kan_extend_grid(matrix(g, nrow = 1L), k)[1L, ]
      new_grid[i, ] <- knots
      E <- edge_features_r(h[, i], knots, k, spec$extrapolate)$value
      target <- matrix(fw$spline[, i, ], ncol = n_out)
      new_coef[i, , ] <- t(ridge_solve(E, target, ridge))
    }
    L$grid <- new_grid
    L$coef <- new_coef
    state$layers[[l]] <- L
    h <- layer_forward_r(L, h, spec)$output
  }
  state$spec$grid_size <- G_new
  validate_kan_state(state)
}

#' Refine the grid of a model (grid extension)
#'
#' Convenience wrapper around [kan_update_grid()] that increases the number
#' of intervals while preserving the learned functions on `x`. Continue
#' training with `kan_fit(..., init = refined_model)`.
#'
#' @param model A fitted `kan_model`.
#' @param x Input sample on the original scale.
#' @param grid_size New (usually larger) grid size.
#' @param grid_eps See [kan_update_grid()].
#' @return The refined `kan_model`.
#' @examples
#' \donttest{
#' set.seed(1)
#' x <- matrix(runif(200, -1, 1))
#' fit <- kan_fit(x, sin(3 * x), hidden = integer(), epochs = 20,
#'                backend = "r", verbose = FALSE)
#' fit10 <- kan_refine(fit, x, grid_size = 10)
#' }
#' @export
kan_refine <- function(model, x, grid_size, grid_eps = 0.02) {
  if (!inherits(model, "kan_model")) {
    stop("`model` must be a kan_model.", call. = FALSE)
  }
  kan_update_grid(model, x, grid_size = grid_size, grid_eps = grid_eps)
}

#' Knots from samples: quantile grid blended with a uniform grid
#' @return Increasing vector of length `G + 1`, or `NULL` if degenerate.
#' @noRd
sample_grid <- function(v, G, grid_eps) {
  v <- sort(v[is.finite(v)])
  n <- length(v)
  if (n < 2L || v[n] - v[1L] <= 1e-12 * max(1, abs(v[n]))) return(NULL)
  ids <- c(floor(n / G * (0:(G - 1L))) + 1L, n)
  adaptive <- v[ids]
  uniform <- seq(v[1L], v[n], length.out = G + 1L)
  g <- grid_eps * uniform + (1 - grid_eps) * adaptive
  if (any(diff(g) <= 1e-9 * (v[n] - v[1L]))) {
    # ties (discrete data): fall back to the uniform grid
    g <- uniform
  }
  g
}

#' Ridge-stabilized least squares
#' @noRd
ridge_solve <- function(A, B, ridge) {
  if (ridge == 0) {
    # plain least squares; rank-deficient columns get 0 (basic solution)
    co <- qr.coef(qr(A), B)
    co[is.na(co)] <- 0
    return(co)
  }
  AtA <- crossprod(A)
  lam <- ridge * max(mean(diag(AtA)), .Machine$double.eps)
  diag(AtA) <- diag(AtA) + lam
  solve(AtA, crossprod(A, B))
}
