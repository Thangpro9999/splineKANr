# Visualisation ------------------------------------------------------------------

#' Evaluate one edge function on a grid of inputs
#'
#' @param model A `kan_model` or `kan_state`.
#' @param layer,i,j Layer index and edge `i -> j`.
#' @param from,to Range of the (normalized) layer input. Defaults to the
#'   spline domain of the edge.
#' @param n Number of evaluation points.
#' @return A data frame with columns `x`, `phi` (full edge function),
#'   `spline` and `base` (the weighted components).
#' @examples
#' st <- kan_init_state(kan_spec(c(2, 3, 1)), seed = 1)
#' head(kan_edge_function(st, layer = 1, i = 1, j = 2))
#' @export
kan_edge_function <- function(model, layer, i, j, from = NULL, to = NULL,
                              n = 200L) {
  st <- as_kan_state(model)
  layer <- check_count(layer, "layer")
  if (layer > length(st$layers)) stop("`layer` out of range.", call. = FALSE)
  L <- st$layers[[layer]]
  i <- check_count(i, "i")
  j <- check_count(j, "j")
  if (i > nrow(L$mask) || j > ncol(L$mask)) {
    stop("Edge index out of range.", call. = FALSE)
  }
  dom <- grid_domain(L$grid[i, , drop = FALSE], st$spec$spline_order)
  if (is.null(from)) from <- dom[1L]
  if (is.null(to)) to <- dom[2L]
  xs <- seq(from, to, length.out = check_count(n, "n", 2L))
  Lij <- subset_layer(L, inp = seq_len(nrow(L$mask)) == i,
                      out = seq_len(ncol(L$mask)) == j)
  spec1 <- st$spec
  spec1$bias <- FALSE
  res <- layer_forward_r(Lij, matrix(xs), spec1)
  m <- Lij$mask[1L, 1L]
  data.frame(
    x = xs,
    phi = res$phi[, 1L, 1L],
    spline = m * Lij$scale_sp[1L, 1L] * res$spline[, 1L, 1L],
    base = m * Lij$scale_base[1L, 1L] * res$base[, 1L]
  )
}

#' Plot the edge functions of a KAN layer
#'
#' Draws a panel for every edge `i -> j` of `layer`, showing
#' `phi_ij` (solid) with its spline (dashed) and base (dotted) components,
#' on the normalized input scale of that layer. Masked edges are gray.
#'
#' @param x A `kan_model`.
#' @param layer Layer to draw.
#' @param data Optional input sample (original scale) used to set the
#'   plotted range to the observed layer inputs and to add a rug.
#' @param n Points per curve.
#' @param ... Further arguments passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @examples
#' st <- kan_init_state(kan_spec(c(2, 2, 1)), seed = 1)
#' m <- structure(list(state = st, spec = st$spec,
#'   scalers = list(x = kan_scaler(matrix(c(-1, 1, -1, 1), 2), "none"),
#'                  y = kan_scaler(matrix(c(0, 1)), "none")),
#'   history = data.frame(), feature_names = c("x1", "x2"),
#'   response_names = "y"), class = "kan_model")
#' plot(m, layer = 1)
#' @export
plot.kan_model <- function(x, layer = 1L, data = NULL, n = 200L, ...) {
  st <- x$state
  L <- st$layers[[layer]]
  n_in <- nrow(L$mask)
  n_out <- ncol(L$mask)
  inputs <- NULL
  if (!is.null(data)) {
    z <- kan_scale(prepare_newdata(x, data), x$scalers$x)
    inputs <- kan_forward_r(st, z, detail = TRUE)$layers[[layer]]$input
  }
  op <- graphics::par(mfrow = c(n_out, n_in), mar = c(3, 3, 2, 1),
                      mgp = c(1.8, 0.6, 0))
  on.exit(graphics::par(op), add = TRUE)
  in_names <- if (layer == 1L) x$feature_names else
    paste0("h", layer - 1L, ".", seq_len(n_in))
  for (j in seq_len(n_out)) {
    for (i in seq_len(n_in)) {
      rng <- if (is.null(inputs)) NULL else range(inputs[, i])
      ef <- kan_edge_function(st, layer, i, j, rng[1L], rng[2L], n)
      col <- if (L$mask[i, j] == 0) "grey70" else "black"
      graphics::plot(ef$x, ef$phi, type = "l", col = col, lwd = 2,
                     xlab = in_names[i], ylab = "phi",
                     main = sprintf("L%d: %d -> %d", layer, i, j),
                     ylim = range(c(ef$phi, ef$spline, ef$base)), ...)
      graphics::lines(ef$x, ef$spline, lty = 2, col = "steelblue")
      graphics::lines(ef$x, ef$base, lty = 3, col = "darkorange")
      if (!is.null(inputs)) graphics::rug(inputs[, i], col = "grey50")
    }
  }
  invisible(x)
}

#' Plot the network structure
#'
#' Nodes are drawn layer by layer; edge width and opacity are proportional
#' to edge importance on `data` (or 1 for active edges if `data` is
#' `NULL`).
#'
#' @param model A `kan_model`.
#' @param data Optional input sample on the original scale.
#' @return The list of importance matrices, invisibly.
#' @examples
#' st <- kan_init_state(kan_spec(c(2, 3, 1)), seed = 1)
#' m <- structure(list(state = st, spec = st$spec,
#'   scalers = list(x = kan_scaler(matrix(c(-1, 1, -1, 1), 2), "none"),
#'                  y = kan_scaler(matrix(c(0, 1)), "none")),
#'   feature_names = c("x1", "x2"), response_names = "y"),
#'   class = "kan_model")
#' kan_plot_network(m)
#' @export
kan_plot_network <- function(model, data = NULL) {
  st <- as_kan_state(model)
  w <- st$spec$width
  imp <- if (is.null(data)) lapply(st$layers, function(L) L$mask) else
    kan_edge_importance(model, data)
  top <- max(unlist(imp), 1e-12)
  pos <- lapply(seq_along(w), function(l) {
    cbind(x = (seq_len(w[l]) - 0.5) / w[l], y = (l - 1) / (length(w) - 1))
  })
  op <- graphics::par(mar = c(1, 1, 1, 1))
  on.exit(graphics::par(op), add = TRUE)
  graphics::plot(NA, xlim = c(0, 1), ylim = c(-0.08, 1.08), axes = FALSE,
                 xlab = "", ylab = "")
  for (l in seq_along(st$layers)) {
    A <- imp[[l]] * st$layers[[l]]$mask
    for (i in seq_len(w[l])) for (j in seq_len(w[l + 1L])) {
      a <- A[i, j] / top
      if (a <= 0) next
      graphics::segments(pos[[l]][i, 1L], pos[[l]][i, 2L],
                         pos[[l + 1L]][j, 1L], pos[[l + 1L]][j, 2L],
                         lwd = 0.5 + 4 * a,
                         col = grDevices::adjustcolor("black",
                                                      alpha.f = 0.15 + 0.85 * a))
    }
  }
  for (l in seq_along(w)) {
    graphics::points(pos[[l]], pch = 21, bg = "white", cex = 2)
  }
  if (inherits(model, "kan_model")) {
    graphics::text(pos[[1L]][, 1L], -0.06, model$feature_names, cex = 0.8)
    graphics::text(pos[[length(w)]][, 1L], 1.06, model$response_names,
                   cex = 0.8)
  }
  invisible(imp)
}

#' Partial dependence of the prediction on one predictor
#'
#' Averages predictions over `data` while `var` is set to each value of a
#' grid: the "effect plot" used to interpret a fitted model.
#'
#' @param model A `kan_model`.
#' @param data Reference data (original scale).
#' @param var Column name or index of the predictor.
#' @param n Number of grid values.
#' @param output Output column for multi-output models.
#' @return A data frame of class `kan_pd` with columns `value` and `yhat`;
#'   it has a `plot()` method.
#' @examples
#' \donttest{
#' set.seed(1)
#' x <- matrix(runif(400, -1, 1), ncol = 2, dimnames = list(NULL, c("a", "b")))
#' fit <- kan_fit(x, x[, "a"]^2 + x[, "b"], hidden = integer(), epochs = 20,
#'                backend = "r", seed = 1)
#' plot(kan_partial_dependence(fit, x, "a"))
#' }
#' @export
kan_partial_dependence <- function(model, data, var, n = 50L, output = 1L) {
  x <- prepare_newdata(model, data)
  if (is.character(var)) {
    vi <- match(var, model$feature_names)
    if (is.na(vi)) stop("Unknown variable: ", var, call. = FALSE)
  } else {
    vi <- check_count(var, "var")
  }
  vals <- seq(min(x[, vi]), max(x[, vi]), length.out = check_count(n, "n", 2L))
  yhat <- vapply(vals, function(v) {
    xx <- x
    xx[, vi] <- v
    p <- as.matrix(stats::predict(model, xx, backend = "r"))
    mean(p[, output])
  }, numeric(1))
  structure(data.frame(value = vals, yhat = yhat),
            var = model$feature_names[vi], class = c("kan_pd", "data.frame"))
}

#' @export
plot.kan_pd <- function(x, xlab = attr(x, "var"),
                        ylab = "partial dependence", ...) {
  graphics::plot(x$value, x$yhat, type = "l", lwd = 2, xlab = xlab,
                 ylab = ylab, ...)
  invisible(x)
}

#' Plot the training history
#' @param model A `kan_model`.
#' @param log Logical; log-scale the loss axis.
#' @return `model`, invisibly.
#' @examples
#' set.seed(1)
#' x <- matrix(runif(100, -1, 1))
#' fit <- kan_fit(x, x[, 1]^2, hidden = integer(), epochs = 10,
#'                backend = "r", seed = 1)
#' kan_plot_history(fit)
#' @export
kan_plot_history <- function(model, log = TRUE) {
  h <- model$history
  yl <- range(c(h$train_loss, h$val_loss), na.rm = TRUE)
  graphics::plot(h$epoch, h$train_loss, type = "l", lwd = 2,
                 log = if (log) "y" else "", ylim = yl, xlab = "epoch",
                 ylab = "MSE (normalized)")
  if (any(!is.na(h$val_loss))) {
    graphics::lines(h$epoch, h$val_loss, lwd = 2, col = "firebrick")
  }
  if (model$best_epoch > 0) graphics::abline(v = model$best_epoch, lty = 3)
  graphics::legend("topright", c("train", "validation"), lwd = 2,
                   col = c("black", "firebrick"), bty = "n")
  invisible(model)
}
