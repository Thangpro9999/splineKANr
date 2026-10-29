# Prediction, metrics and S3 methods ------------------------------------------

#' Predict from a fitted KAN
#'
#' @param object A `kan_model`.
#' @param newdata Numeric matrix or data frame with the training predictors.
#'   If it has column names, they are matched to the training names.
#' @param backend `"auto"` (torch if installed, otherwise base R), `"torch"`
#'   or `"r"`. Both backends evaluate the same model.
#' @param ... Unused.
#' @return A numeric vector (single output) or matrix of predictions on the
#'   original response scale.
#' @examples
#' set.seed(1)
#' x <- matrix(runif(100, -1, 1))
#' fit <- kan_fit(x, x[, 1]^2, hidden = integer(), epochs = 10,
#'                backend = "r", seed = 1)
#' head(predict(fit, x, backend = "r"))
#' @export
predict.kan_model <- function(object, newdata, backend = c("auto", "torch", "r"),
                              ...) {
  backend <- match.arg(backend)
  if (backend == "auto") {
    backend <- if (torch::torch_is_installed()) "torch" else "r"
  }
  x <- prepare_newdata(object, newdata)
  z <- kan_scale(x, object$scalers$x)
  dtype <- object$settings$dtype
  if (is.null(dtype)) dtype <- "float32"
  out <- if (backend == "torch") {
    require_torch()
    net <- kan_module_from_state(object$state, dtype = dtype)
    torch::with_no_grad({
      res <- net(torch::torch_tensor(z, dtype = torch_dtype_from(dtype)))
    })
    matrix(as_r_array(res), nrow = nrow(z))
  } else {
    kan_forward_r(object$state, z)
  }
  out <- kan_unscale(out, object$scalers$y)
  colnames(out) <- object$response_names
  if (ncol(out) == 1L) return(as.numeric(out))
  out
}

#' Check and align new data
#' @noRd
prepare_newdata <- function(object, newdata) {
  nm <- colnames(newdata)
  x <- as_numeric_matrix(newdata, "newdata")
  p <- object$spec$width[1L]
  if (!is.null(nm) && all(object$feature_names %in% nm)) {
    x <- x[, object$feature_names, drop = FALSE]
  } else if (ncol(x) != p) {
    stop(sprintf("`newdata` must have %d columns.", p), call. = FALSE)
  }
  if (anyNA(x)) stop("`newdata` contains missing values.", call. = FALSE)
  x
}

#' Regression accuracy metrics
#'
#' @param y Observed values (vector or matrix).
#' @param yhat Predicted values, same shape as `y`.
#' @return A named numeric vector with `rmse`, `mae`, `r2` (computed over
#'   all elements) and `n`. `r2` is `NA` (with a warning) when `y` is
#'   constant.
#' @examples
#' kan_metrics(c(1, 2, 3), c(1.1, 1.9, 3.2))
#' @export
kan_metrics <- function(y, yhat) {
  y <- as.numeric(as.matrix(y))
  yhat <- as.numeric(as.matrix(yhat))
  if (length(y) != length(yhat)) {
    stop("`y` and `yhat` must have the same length.", call. = FALSE)
  }
  ok <- is.finite(y) & is.finite(yhat)
  if (!all(ok)) {
    warning(sprintf("Dropping %d non-finite pair(s).", sum(!ok)),
            call. = FALSE)
    y <- y[ok]
    yhat <- yhat[ok]
  }
  if (length(y) == 0L) stop("No finite observations.", call. = FALSE)
  e <- y - yhat
  sst <- sum((y - mean(y))^2)
  r2 <- if (sst > 0) 1 - sum(e^2) / sst else {
    warning("`y` is constant; R^2 is undefined.", call. = FALSE)
    NA_real_
  }
  c(rmse = sqrt(mean(e^2)), mae = mean(abs(e)), r2 = r2, n = length(y))
}

#' @export
print.kan_model <- function(x, ...) {
  s <- x$spec
  cat("<kan_model> B-spline KAN\n")
  cat("  width      : [", paste(s$width, collapse = ", "), "]\n", sep = "")
  cat("  grid / k   : G = ", s$grid_size, ", k = ", s$spline_order,
      ", extrapolate = ", s$extrapolate, "\n", sep = "")
  cat("  parameters : ", kan_n_params(x$state), "\n", sep = "")
  if (nrow(x$history) > 0L) {
    h <- x$history
    cat("  epochs run : ", nrow(h), if (isTRUE(x$stopped_early))
      " (early stopped)" else "", "; best epoch ", x$best_epoch, "\n",
      sep = "")
    cat(sprintf("  best loss  : train %.4g%s (normalized scale)\n",
                h$train_loss[max(1L, x$best_epoch)],
                if (all(is.na(h$val_loss))) "" else
                  sprintf(", val %.4g", h$val_loss[max(1L, x$best_epoch)])))
  }
  invisible(x)
}

#' @export
summary.kan_model <- function(object, ...) {
  structure(list(
    spec = object$spec, n_params = kan_n_params(object$state),
    history = object$history, best_epoch = object$best_epoch,
    activation_check = object$activation_check,
    settings = object$settings[c("backend", "optimizer", "lr", "epochs",
                                 "batch_size", "n_train", "n_val", "seed")]
  ), class = "summary.kan_model")
}

#' @export
print.summary.kan_model <- function(x, ...) {
  print(x$spec)
  cat("Parameters:", x$n_params, "\n")
  utils::str(x$settings, give.head = FALSE, no.list = TRUE)
  if (!is.null(x$activation_check) && length(x$activation_check)) {
    cat("Activation range check (training data):\n")
    print(x$activation_check, row.names = FALSE, digits = 4)
  }
  invisible(x)
}
