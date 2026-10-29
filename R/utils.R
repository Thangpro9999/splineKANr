# Internal validation helpers ------------------------------------------------

#' Check a scalar count
#' @param x Value to check.
#' @param name Argument name used in the error message.
#' @param min Minimum allowed value.
#' @return `x` as integer.
#' @noRd
check_count <- function(x, name, min = 1L) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || x != round(x) ||
      x < min) {
    stop(sprintf("`%s` must be a single integer >= %d.", name, min),
         call. = FALSE)
  }
  as.integer(x)
}

#' Check a positive scalar number
#' @noRd
check_number <- function(x, name, min = -Inf, allow_null = FALSE,
                         strict = FALSE) {
  if (allow_null && is.null(x)) return(NULL)
  ok <- is.numeric(x) && length(x) == 1L && is.finite(x) &&
    (if (strict) x > min else x >= min)
  if (!ok) {
    stop(sprintf("`%s` must be a single finite number %s %s.", name,
                 if (strict) ">" else ">=", format(min)), call. = FALSE)
  }
  as.numeric(x)
}

#' Check a knot vector
#' @noRd
check_knots <- function(knots) {
  if (!is.numeric(knots) || anyNA(knots) || any(diff(knots) <= 0)) {
    stop("`knots` must be a strictly increasing numeric vector.",
         call. = FALSE)
  }
  invisible(knots)
}

#' Coerce predictors to a numeric matrix
#' @noRd
as_numeric_matrix <- function(x, name = "x") {
  if (is.data.frame(x)) {
    is_num <- vapply(x, is.numeric, logical(1))
    if (!all(is_num)) {
      stop(sprintf("All columns of `%s` must be numeric (offending: %s).",
                   name, paste(names(x)[!is_num], collapse = ", ")),
           call. = FALSE)
    }
    x <- as.matrix(x)
  }
  if (is.vector(x) && is.numeric(x)) x <- matrix(x, ncol = 1L)
  if (!is.matrix(x) || !is.numeric(x)) {
    stop(sprintf("`%s` must be a numeric matrix, data frame or vector.",
                 name), call. = FALSE)
  }
  storage.mode(x) <- "double"
  x
}

#' Silu activation and its derivative (base R)
#' @noRd
silu_r <- function(x) x / (1 + exp(-x))

#' @noRd
silu_deriv_r <- function(x) {
  s <- 1 / (1 + exp(-x))
  s * (1 + x * (1 - s))
}

#' Base activation (value and derivative) by name
#' @noRd
base_fun_r <- function(x, base_fun) {
  switch(base_fun,
    silu = list(value = silu_r(x), deriv = silu_deriv_r(x)),
    none = list(value = 0 * x, deriv = 0 * x),
    identity = list(value = x, deriv = 1 + 0 * x),
    stop("Unknown base function: ", base_fun, call. = FALSE)
  )
}

#' Fail clearly when the torch backend is unusable
#' @noRd
require_torch <- function() {
  if (!torch::torch_is_installed()) {
    stop("The torch backend is not installed. Run `torch::install_torch()`.",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' torch dtype from a string
#' @noRd
torch_dtype_from <- function(dtype) {
  switch(dtype,
    float32 = torch::torch_float32(),
    float64 = torch::torch_float64(),
    stop("`dtype` must be \"float32\" or \"float64\".", call. = FALSE)
  )
}

#' Convert a tensor to a plain R array (detached, on CPU)
#' @noRd
as_r_array <- function(t) {
  a <- torch::as_array(t$detach()$cpu())
  storage.mode(a) <- "double"
  a
}
