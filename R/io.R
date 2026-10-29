# Persistence -------------------------------------------------------------------
#
# A model is stored as plain R data (no torch external pointers), so it can
# be reloaded in any session, with or without libtorch, and gives the same
# predictions.

#' Save and load a fitted KAN
#'
#' `kan_save()` writes the architecture, all parameters, knots,
#' normalization constants, training history and metadata to an `.rds`
#' file. `kan_load()` restores a `kan_model` that predicts identically.
#'
#' @param model A `kan_model`.
#' @param file Path of the `.rds` file.
#' @return `kan_save()` returns `file` invisibly; `kan_load()` returns the
#'   `kan_model`.
#' @examples
#' set.seed(1)
#' x <- matrix(runif(60, -1, 1))
#' fit <- kan_fit(x, x[, 1]^2, hidden = integer(), epochs = 5,
#'                backend = "r", seed = 1)
#' f <- tempfile(fileext = ".rds")
#' kan_save(fit, f)
#' fit2 <- kan_load(f)
#' all.equal(predict(fit, x, backend = "r"), predict(fit2, x, backend = "r"))
#' @export
kan_save <- function(model, file) {
  if (!inherits(model, "kan_model")) {
    stop("`model` must be a kan_model.", call. = FALSE)
  }
  if (!is.character(file) || length(file) != 1L) {
    stop("`file` must be a single path.", call. = FALSE)
  }
  obj <- list(format = "splineKANr_model", format_version = 1L,
              model = unclass(model))
  obj$model$call <- NULL
  saveRDS(obj, file)
  invisible(file)
}

#' @rdname kan_save
#' @export
kan_load <- function(file) {
  if (!file.exists(file)) stop("File not found: ", file, call. = FALSE)
  obj <- readRDS(file)
  if (!is.list(obj) || !identical(obj$format, "splineKANr_model")) {
    stop("Not a splineKANr model file.", call. = FALSE)
  }
  if (obj$format_version > 1L) {
    stop("Model was saved by a newer splineKANr version.", call. = FALSE)
  }
  model <- structure(obj$model, class = "kan_model")
  model$state <- validate_kan_state(model$state)
  model
}
