# Normalisation fitted on training data only ---------------------------------

#' Fit a column scaler
#'
#' @param x Numeric matrix.
#' @param method `"minmax"` maps each column of the training data onto
#'   `target` (the spline domain), `"standardize"` centers and scales to unit
#'   variance, `"none"` leaves data unchanged.
#' @param target Target interval for `"minmax"`.
#' @return A list of class `kan_scaler` with `method`, `center`, `scale`
#'   and `offset`, so that `z = (x - center) / scale + offset`.
#' @examples
#' s <- kan_scaler(matrix(1:10, ncol = 2), "minmax")
#' kan_scale(matrix(1:10, ncol = 2), s)
#' @export
kan_scaler <- function(x, method = c("minmax", "standardize", "none"),
                       target = c(-1, 1)) {
  method <- match.arg(method)
  x <- as_numeric_matrix(x)
  p <- ncol(x)
  center <- rep(0, p)
  scale <- rep(1, p)
  offset <- rep(0, p)
  if (method == "minmax") {
    lo <- apply(x, 2L, min)
    hi <- apply(x, 2L, max)
    center <- lo
    scale <- (hi - lo) / (target[2L] - target[1L])
    offset <- rep(target[1L], p)
  } else if (method == "standardize") {
    center <- colMeans(x)
    scale <- apply(x, 2L, stats::sd)
  }
  const <- !is.finite(scale) | scale <= .Machine$double.eps
  if (any(const) && method != "none") {
    warning(sprintf("%d constant column(s) detected; they are only centred.",
                    sum(const)), call. = FALSE)
    scale[const] <- 1
    if (method == "minmax") {
      offset[const] <- mean(target)
    }
  }
  structure(list(method = method, center = unname(center),
                 scale = unname(scale), offset = unname(offset)),
            class = "kan_scaler")
}

#' Apply or invert a scaler
#'
#' @param x Numeric matrix.
#' @param scaler A `kan_scaler`.
#' @return The transformed matrix.
#' @examples
#' s <- kan_scaler(matrix(1:10, ncol = 2), "standardize")
#' z <- kan_scale(matrix(1:10, ncol = 2), s)
#' kan_unscale(z, s)
#' @export
kan_scale <- function(x, scaler) {
  x <- as_numeric_matrix(x)
  if (ncol(x) != length(scaler$center)) {
    stop("Column count does not match the scaler.", call. = FALSE)
  }
  z <- sweep(sweep(x, 2L, scaler$center, `-`), 2L, scaler$scale, `/`)
  sweep(z, 2L, scaler$offset, `+`)
}

#' @rdname kan_scale
#' @export
kan_unscale <- function(x, scaler) {
  x <- as_numeric_matrix(x)
  z <- sweep(sweep(x, 2L, scaler$offset, `-`), 2L, scaler$scale, `*`)
  sweep(z, 2L, scaler$center, `+`)
}
