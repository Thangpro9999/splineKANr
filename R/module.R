# torch modules ---------------------------------------------------------------

#' A single B-spline KAN layer (torch module)
#'
#' Maps `(batch, n_in)` to `(batch, n_out)`. Every edge `i -> j` carries
#' `phi_ij(x) = mask_ij * (scale_base_ij * b(x) + scale_sp_ij * s_ij(x))`
#' with `s_ij` a B-spline with coefficients `coef[i, j, ]`.
#'
#' @param n_in,n_out Input and output widths.
#' @param grid_size Number of grid intervals `G`.
#' @param spline_order Spline degree `k`.
#' @param extrapolate,base_fun,bias See [kan_spec()].
#' @param dtype `"float32"` or `"float64"`.
#' @return An `nn_module`. Parameters: `coef`, `scale_base`, `scale_sp`,
#'   `bias`; buffers: `grid`, `mask`. Its `forward(x, detail = FALSE)`
#'   returns the output, or with `detail = TRUE` a list with `output`, `phi`
#'   and `spline`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   layer <- kan_layer(2, 3)
#'   layer(torch::torch_randn(4, 2))
#' }
#' @export
kan_layer <- torch::nn_module(
  "kan_layer",
  initialize = function(n_in, n_out, grid_size = 5L, spline_order = 3L,
                        extrapolate = "linear", base_fun = "silu",
                        bias = TRUE, dtype = "float32") {
    dt <- torch_dtype_from(dtype)
    self$n_in <- as.integer(n_in)
    self$n_out <- as.integer(n_out)
    self$k <- as.integer(spline_order)
    self$extrapolate <- extrapolate
    self$base_fun <- base_fun
    self$use_bias <- isTRUE(bias)
    n_basis <- grid_size + spline_order
    self$coef <- torch::nn_parameter(
      torch::torch_zeros(n_in, n_out, n_basis, dtype = dt))
    self$scale_base <- torch::nn_parameter(
      torch::torch_zeros(n_in, n_out, dtype = dt))
    self$scale_sp <- torch::nn_parameter(
      torch::torch_ones(n_in, n_out, dtype = dt))
    # without bias the zero vector is a buffer, so optimisers never see it
    self$bias <- if (self$use_bias) {
      torch::nn_parameter(torch::torch_zeros(n_out, dtype = dt))
    } else {
      torch::nn_buffer(torch::torch_zeros(n_out, dtype = dt))
    }
    self$grid <- torch::nn_buffer(torch::torch_tensor(
      kan_make_grid(n_in, grid_size, spline_order), dtype = dt))
    self$mask <- torch::nn_buffer(torch::torch_ones(n_in, n_out, dtype = dt))
  },
  spline_part = function(x) {
    k <- self$k
    grid <- self$grid
    if (self$extrapolate == "none") {
      B <- kan_bspline_basis(x, grid, k)
      return(torch::torch_einsum("bim,iom->bio", list(B, self$coef)))
    }
    n_knots <- grid$size(2)
    lo <- grid[, k + 1L]$unsqueeze(1)             # (1, n_in)
    hi <- grid[, n_knots - k]$unsqueeze(1)
    xc <- torch::torch_maximum(torch::torch_minimum(x, hi), lo)
    B <- kan_bspline_basis(xc, grid, k, right_closed = TRUE)
    s <- torch::torch_einsum("bim,iom->bio", list(B, self$coef))
    if (self$extrapolate == "clamp") return(s)
    dB <- kan_bspline_deriv(xc, grid, k, right_closed = TRUE)
    ds <- torch::torch_einsum("bim,iom->bio", list(dB, self$coef))
    s + ds * (x - xc)$unsqueeze(3)
  },
  base_part = function(x) {
    switch(self$base_fun,
      silu = torch::nnf_silu(x),
      identity = x,
      none = torch::torch_zeros_like(x)
    )
  },
  forward = function(x, detail = FALSE) {
    s <- self$spline_part(x)                        # (B, in, out)
    b <- self$base_part(x)$unsqueeze(3)             # (B, in, 1)
    phi <- (self$scale_base$unsqueeze(1) * b +
              self$scale_sp$unsqueeze(1) * s) * self$mask$unsqueeze(1)
    out <- phi$sum(dim = 2)
    if (self$use_bias) out <- out + self$bias$unsqueeze(1)
    if (isTRUE(detail)) return(list(output = out, phi = phi, spline = s))
    out
  }
)

#' Multi-layer B-spline KAN (torch module)
#'
#' @param spec A [kan_spec()].
#' @param dtype `"float32"` or `"float64"`.
#' @return An `nn_module` whose `forward(x, detail = FALSE)` returns the
#'   output tensor, or with `detail = TRUE` a list with `output` and the
#'   per-layer `inputs`, `phi`.
#' @examples
#' if (torch::torch_is_installed()) {
#'   net <- kan_net(kan_spec(c(2, 4, 1)))
#'   net(torch::torch_randn(3, 2))
#' }
#' @export
kan_net <- torch::nn_module(
  "kan_net",
  initialize = function(spec, dtype = "float32") {
    self$spec <- spec
    w <- spec$width
    self$layers <- torch::nn_module_list(lapply(
      seq_len(length(w) - 1L),
      function(l) kan_layer(w[l], w[l + 1L], spec$grid_size,
                            spec$spline_order, spec$extrapolate,
                            spec$base_fun, spec$bias, dtype)
    ))
  },
  forward = function(x, detail = FALSE) {
    n_layers <- length(self$layers)
    if (!isTRUE(detail)) {
      for (l in seq_len(n_layers)) x <- self$layers[[l]](x)
      return(x)
    }
    inputs <- vector("list", n_layers)
    phis <- vector("list", n_layers)
    for (l in seq_len(n_layers)) {
      inputs[[l]] <- x
      res <- self$layers[[l]](x, detail = TRUE)
      phis[[l]] <- res$phi
      x <- res$output
    }
    list(output = x, inputs = inputs, phi = phis)
  }
)

#' Build a torch module from a state
#'
#' @param state A `kan_state`.
#' @param dtype `"float32"` or `"float64"`.
#' @return A [kan_net()] holding the state's numbers.
#' @examples
#' if (torch::torch_is_installed()) {
#'   st <- kan_init_state(kan_spec(c(2, 3, 1)), seed = 1)
#'   net <- kan_module_from_state(st)
#' }
#' @export
kan_module_from_state <- function(state, dtype = "float32") {
  require_torch()
  state <- as_kan_state(state)
  spec <- state$spec
  spec$grid_size <- ncol(state$layers[[1L]]$grid) - 2L * spec$spline_order - 1L
  net <- kan_net(spec, dtype = dtype)
  kan_load_state_into(net, state)
  net
}

#' Copy a state into an existing module (in place)
#' @noRd
kan_load_state_into <- function(net, state) {
  torch::with_no_grad({
    for (l in seq_along(state$layers)) {
      L <- state$layers[[l]]
      m <- net$layers[[l]]
      dt <- m$coef$dtype
      m$coef$copy_(torch::torch_tensor(L$coef, dtype = dt))
      m$scale_base$copy_(torch::torch_tensor(L$scale_base, dtype = dt))
      m$scale_sp$copy_(torch::torch_tensor(L$scale_sp, dtype = dt))
      m$bias$copy_(torch::torch_tensor(L$bias, dtype = dt))
      m$grid$copy_(torch::torch_tensor(L$grid, dtype = dt))
      m$mask$copy_(torch::torch_tensor(L$mask, dtype = dt))
    }
  })
  invisible(net)
}

#' Extract a state from a module
#'
#' @param net A [kan_net()].
#' @param spec The [kan_spec()] used to build `net`.
#' @return A `kan_state` (plain R arrays).
#' @noRd
kan_state_from_module <- function(net, spec) {
  layers <- lapply(seq_along(net$layers), function(l) {
    m <- net$layers[[l]]
    n_in <- m$n_in
    n_out <- m$n_out
    list(
      coef = array(as_r_array(m$coef), dim = m$coef$shape),
      scale_base = matrix(as_r_array(m$scale_base), n_in, n_out),
      scale_sp = matrix(as_r_array(m$scale_sp), n_in, n_out),
      mask = matrix(as_r_array(m$mask), n_in, n_out),
      bias = as.numeric(as_r_array(m$bias)),
      grid = matrix(as_r_array(m$grid), nrow = n_in)
    )
  })
  new_kan_state(spec, layers)
}
