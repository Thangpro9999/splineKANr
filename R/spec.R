# Model specification and parameter state -------------------------------------
#
# A KAN is fully described by a `kan_spec` (architecture) plus a `kan_state`
# (numbers). The state is a plain R list of arrays, which makes it backend
# independent: the torch module, the base R reference, persistence, grid
# updates and pruning all read and write the same object.

#' Specify a B-spline KAN architecture
#'
#' @param width Integer vector of layer widths, e.g. `c(2, 5, 1)` for two
#'   inputs, one hidden layer of five nodes and one output.
#' @param grid_size Number of grid intervals `G` per edge.
#' @param spline_order Spline degree `k` (3 = cubic).
#' @param grid_range Initial domain `c(a, b)` of every edge spline.
#' @param extrapolate Behavior of an edge spline outside its domain:
#'   * `"linear"` (default): first-order Taylor extension from the nearest
#'     bound. The edge stays continuous and differentiable and its gradient
#'     with respect to the input is not zero outside the domain.
#'   * `"none"`: raw B-spline on the extended knots, which decays to zero
#'     outside the domain. This matches pykan.
#'   * `"clamp"`: evaluate at the nearest bound. The spline part is then
#'     flat outside the domain (zero input gradient).
#' @param base_fun Residual base activation `b(x)` in
#'   `phi(x) = w_b * b(x) + w_s * spline(x)`: `"silu"` (pykan default),
#'   `"identity"` or `"none"`.
#' @param bias Logical; add a trainable bias to every node of each layer.
#' @return An object of class `kan_spec`.
#' @examples
#' kan_spec(c(2, 5, 1))
#' @export
kan_spec <- function(width, grid_size = 5L, spline_order = 3L,
                     grid_range = c(-1, 1),
                     extrapolate = c("linear", "none", "clamp"),
                     base_fun = c("silu", "identity", "none"),
                     bias = TRUE) {
  if (!is.numeric(width) || length(width) < 2L || anyNA(width) ||
      any(width < 1) || any(width != round(width))) {
    stop("`width` must be an integer vector of length >= 2 with entries >= 1.",
         call. = FALSE)
  }
  if (!is.numeric(grid_range) || length(grid_range) != 2L ||
      !all(is.finite(grid_range)) || grid_range[1L] >= grid_range[2L]) {
    stop("`grid_range` must be `c(lower, upper)` with lower < upper.",
         call. = FALSE)
  }
  if (!is.logical(bias) || length(bias) != 1L || is.na(bias)) {
    stop("`bias` must be TRUE or FALSE.", call. = FALSE)
  }
  structure(list(
    width = as.integer(width),
    grid_size = check_count(grid_size, "grid_size"),
    spline_order = check_count(spline_order, "spline_order"),
    grid_range = as.numeric(grid_range),
    extrapolate = match.arg(extrapolate),
    base_fun = match.arg(base_fun),
    bias = bias
  ), class = "kan_spec")
}

#' @export
print.kan_spec <- function(x, ...) {
  cat("<kan_spec> width = [", paste(x$width, collapse = ", "), "]",
      ", G = ", x$grid_size, ", k = ", x$spline_order,
      ", extrapolate = ", x$extrapolate, ", base = ", x$base_fun, "\n",
      sep = "")
  invisible(x)
}

#' Initialize KAN parameters
#'
#' Draws initial parameters with base R's random number generator, so the
#' same `seed` gives the same network for every backend. The scheme follows
#' pykan's `KANLayer`: small random spline coefficients of size
#' `noise_scale / G`, base weights
#' `scale_base_mu / sqrt(n_in) + scale_base_sigma * U(-1, 1) / sqrt(n_in)`
#' and spline weights `scale_sp / sqrt(n_in)`.
#'
#' @param spec A [kan_spec()].
#' @param seed Optional integer seed passed to [set.seed()].
#' @param noise_scale Size of the initial coefficient noise.
#' @param scale_base_mu,scale_base_sigma Mean and spread of the base weights.
#' @param scale_sp Initial spline weight.
#' @return An object of class `kan_state`: a list with `spec` and `layers`.
#'   Each layer holds `coef` (array `n_in x n_out x (G + k)`), `scale_base`,
#'   `scale_sp`, `mask` (matrices `n_in x n_out`), `bias` (length `n_out`)
#'   and `grid` (matrix `n_in x (G + 2k + 1)`).
#' @examples
#' st <- kan_init_state(kan_spec(c(2, 3, 1)), seed = 1)
#' str(st$layers[[1]], max.level = 1)
#' @export
kan_init_state <- function(spec, seed = NULL, noise_scale = 0.3,
                           scale_base_mu = 0, scale_base_sigma = 1,
                           scale_sp = 1) {
  if (!inherits(spec, "kan_spec")) {
    stop("`spec` must be created with kan_spec().", call. = FALSE)
  }
  if (!is.null(seed)) set.seed(check_count(seed, "seed", min = 0L))
  G <- spec$grid_size
  k <- spec$spline_order
  n_basis <- G + k
  layers <- lapply(seq_len(length(spec$width) - 1L), function(l) {
    n_in <- spec$width[l]
    n_out <- spec$width[l + 1L]
    sq <- sqrt(n_in)
    list(
      coef = array((stats::runif(n_in * n_out * n_basis) - 0.5) *
                     noise_scale / G, dim = c(n_in, n_out, n_basis)),
      scale_base = matrix(scale_base_mu / sq + scale_base_sigma *
                            (stats::runif(n_in * n_out) * 2 - 1) / sq,
                          n_in, n_out),
      scale_sp = matrix(scale_sp / sq, n_in, n_out),
      mask = matrix(1, n_in, n_out),
      bias = numeric(n_out),
      grid = kan_make_grid(n_in, G, k, spec$grid_range)
    )
  })
  new_kan_state(spec, layers)
}

#' Construct and validate a state object
#' @noRd
new_kan_state <- function(spec, layers) {
  st <- structure(list(spec = spec, layers = layers), class = "kan_state")
  validate_kan_state(st)
}

#' Validate dimensions of a state
#' @noRd
validate_kan_state <- function(state) {
  spec <- state$spec
  w <- spec$width
  if (length(state$layers) != length(w) - 1L) {
    stop("State has the wrong number of layers.", call. = FALSE)
  }
  for (l in seq_along(state$layers)) {
    L <- state$layers[[l]]
    n_in <- w[l]
    n_out <- w[l + 1L]
    n_basis <- ncol(L$grid) - spec$spline_order - 1L
    ok <- identical(dim(L$coef), c(n_in, n_out, n_basis)) &&
      identical(dim(L$scale_base), c(n_in, n_out)) &&
      identical(dim(L$scale_sp), c(n_in, n_out)) &&
      identical(dim(L$mask), c(n_in, n_out)) &&
      length(L$bias) == n_out && nrow(L$grid) == n_in
    if (!ok) stop(sprintf("Layer %d of the state has inconsistent shapes.", l),
                  call. = FALSE)
  }
  state
}

#' Number of trainable parameters of a state
#' @noRd
kan_n_params <- function(state) {
  sum(vapply(state$layers, function(L) {
    length(L$coef) + length(L$scale_base) + length(L$scale_sp) +
      if (isTRUE(state$spec$bias)) length(L$bias) else 0
  }, numeric(1)))
}
