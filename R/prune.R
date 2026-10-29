# Attribution and pruning ------------------------------------------------------

#' Edge importance (mean absolute edge activation)
#'
#' @param model A `kan_model` (then `x` is on the original scale) or a
#'   `kan_state` (then `x` is already normalized).
#' @param x Input sample.
#' @return A list with one `n_in x n_out` matrix per layer holding
#'   `mean_n |phi_ij(x_n)|`.
#' @examples
#' st <- kan_init_state(kan_spec(c(2, 3, 1)), seed = 1)
#' kan_edge_importance(st, matrix(runif(40, -1, 1), ncol = 2))
#' @export
kan_edge_importance <- function(model, x) {
  if (inherits(model, "kan_model")) {
    x <- kan_scale(prepare_newdata(model, x), model$scalers$x)
  }
  state <- as_kan_state(model)
  fw <- kan_forward_r(state, x, detail = TRUE)
  lapply(fw$layers, function(I) {
    A <- apply(abs(I$phi), c(2L, 3L), mean)
    matrix(A, nrow = dim(I$phi)[2L])
  })
}

#' Prune edges and hidden nodes with small activations
#'
#' Edges with importance below `edge_threshold` are masked. A hidden node is
#' removed when both its largest incoming and its largest outgoing edge
#' importance fall below `node_threshold`, mirroring pykan's
#' `prune_node()` / `prune_edge()` criteria. Retrain briefly afterwards
#' (`kan_fit(..., init = pruned)`) to recover accuracy.
#'
#' @param model A `kan_model`.
#' @param x Input sample on the original scale.
#' @param node_threshold,edge_threshold Importance thresholds.
#' @return The pruned `kan_model` with element `pruning` describing the
#'   removed nodes and masked edges.
#' @examples
#' \donttest{
#' set.seed(1)
#' x <- matrix(runif(400, -1, 1), ncol = 2)
#' fit <- kan_fit(x, sin(pi * x[, 1]), hidden = 3, epochs = 20,
#'                backend = "r", seed = 1)
#' pruned <- kan_prune(fit, x, edge_threshold = 0.05)
#' pruned$pruning
#' }
#' @export
kan_prune <- function(model, x, node_threshold = 1e-2,
                      edge_threshold = 3e-2) {
  if (!inherits(model, "kan_model")) {
    stop("`model` must be a kan_model.", call. = FALSE)
  }
  node_threshold <- check_number(node_threshold, "node_threshold", 0)
  edge_threshold <- check_number(edge_threshold, "edge_threshold", 0)
  st <- model$state
  imp <- kan_edge_importance(model, x)
  n_layers <- length(st$layers)
  removed <- list()
  # nodes (hidden layers only)
  if (n_layers >= 2L) {
    for (l in seq_len(n_layers - 1L)) {
      inc <- apply(imp[[l]], 2L, max)
      outg <- apply(imp[[l + 1L]], 1L, max)
      keep <- inc > node_threshold | outg > node_threshold
      if (!any(keep)) {
        keep[which.max(inc + outg)] <- TRUE
        warning(sprintf("All nodes of hidden layer %d fall below the threshold; keeping the strongest one.", l),
                call. = FALSE)
      }
      if (!all(keep)) {
        removed[[length(removed) + 1L]] <-
          data.frame(hidden_layer = l, node = which(!keep))
        st$layers[[l]] <- subset_layer(st$layers[[l]], out = keep)
        st$layers[[l + 1L]] <- subset_layer(st$layers[[l + 1L]], inp = keep)
        imp[[l]] <- imp[[l]][, keep, drop = FALSE]
        imp[[l + 1L]] <- imp[[l + 1L]][keep, , drop = FALSE]
        st$spec$width[l + 1L] <- sum(keep)
      }
    }
  }
  # edges
  n_masked <- integer(n_layers)
  for (l in seq_len(n_layers)) {
    drop <- imp[[l]] < edge_threshold
    st$layers[[l]]$mask[drop] <- 0
    n_masked[l] <- sum(st$layers[[l]]$mask == 0)
  }
  model$state <- validate_kan_state(st)
  model$spec <- st$spec
  model$pruning <- list(
    removed_nodes = if (length(removed)) do.call(rbind, removed) else
      data.frame(hidden_layer = integer(), node = integer()),
    masked_edges = n_masked, node_threshold = node_threshold,
    edge_threshold = edge_threshold
  )
  model
}

#' Keep a subset of input rows / output columns of a layer
#' @noRd
subset_layer <- function(L, inp = NULL, out = NULL) {
  if (is.null(inp)) inp <- rep(TRUE, nrow(L$mask))
  if (is.null(out)) out <- rep(TRUE, ncol(L$mask))
  list(
    coef = L$coef[inp, out, , drop = FALSE],
    scale_base = L$scale_base[inp, out, drop = FALSE],
    scale_sp = L$scale_sp[inp, out, drop = FALSE],
    mask = L$mask[inp, out, drop = FALSE],
    bias = L$bias[out],
    grid = L$grid[inp, , drop = FALSE]
  )
}
