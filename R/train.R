# Training -------------------------------------------------------------------

#' Fit a B-spline Kolmogorov-Arnold Network
#'
#' Trains a KAN of architecture `c(ncol(x), hidden, ncol(y))` by minimizing
#' the mean squared error (plus optional KAN regularization) on normalized
#' data. Normalization constants are estimated on the training part only.
#'
#' @param x Numeric matrix or data frame of predictors.
#' @param y Numeric response vector, or matrix for several outputs.
#' @param hidden Integer vector of hidden layer widths; `integer()` gives a
#'   single-layer (additive) KAN.
#' @param grid_size,spline_order,grid_range,extrapolate,base_fun,bias
#'   Architecture, see [kan_spec()].
#' @param epochs Maximum number of passes over the training data.
#' @param batch_size Mini-batch size; `NULL` uses full batches.
#' @param optimizer `"adam"` or `"lbfgs"` (full batch, torch backend only).
#' @param lr Learning rate; defaults to `0.01` for Adam and `1` for L-BFGS.
#' @param weight_decay L2 weight decay passed to Adam.
#' @param validation_split Fraction of rows held out for validation (and
#'   early stopping) when `validation_data` is `NULL`.
#' @param validation_data Optional `list(x = , y = )` used for validation.
#' @param patience Stop after this many epochs without an improvement of at
#'   least `min_delta` in the monitored loss (validation loss if available,
#'   otherwise training loss). `NULL` disables early stopping.
#' @param min_delta Minimum improvement counted by early stopping.
#' @param restore_best Return the parameters of the best epoch.
#' @param normalize_x `"minmax"` (map training range onto `grid_range`),
#'   `"standardize"` or `"none"`.
#' @param normalize_y Standardize the response(s) during training.
#' @param reg Regularization settings from [kan_reg_control()] (torch
#'   backend only when `reg$lambda > 0`).
#' @param grid_update_every Update the grids from the training inputs every
#'   this many epochs (`0` = never), see [kan_update_grid()].
#' @param grid_update_until Last epoch at which grids may be updated.
#' @param grid_eps Uniform/quantile blending for grid updates.
#' @param grid_ridge Relative ridge penalty of the coefficient refit after a
#'   grid update, see [kan_update_grid()]. `0` gives plain least squares, as
#'   in pykan.
#' @param grad_clip Maximum gradient norm (`NULL` = no clipping).
#' @param seed Integer seed for splitting, initialization and batching.
#' @param backend `"torch"` (autograd, the default) or `"r"` (the base R
#'   reference with analytic gradients; Adam only, no regularization,
#'   slower, but needs no libtorch).
#' @param dtype `"float32"` or `"float64"` for the torch backend.
#' @param init Optional `kan_model` to warm start from (its architecture and
#'   scalers are reused, e.g. after [kan_refine()]).
#' @param init_control List of initialization options passed to
#'   [kan_init_state()].
#' @param checkpoint_path Optional file; the best model so far is written
#'   there with [kan_save()] whenever the monitored loss improves.
#' @param verbose Print progress every `max(1, epochs %/% 10)` epochs.
#' @return An object of class `kan_model` with elements `state`, `spec`,
#'   `scalers`, `history` (data frame of losses per epoch), `best_epoch`,
#'   `stopped_early`, `activation_check`, `settings` and `meta`.
#' @examples
#' set.seed(1)
#' x <- matrix(runif(300, -1, 1))
#' y <- sin(pi * x[, 1]) + rnorm(300, sd = 0.05)
#' fit <- kan_fit(x, y, hidden = integer(), epochs = 30, backend = "r",
#'                seed = 1)
#' kan_metrics(y, predict(fit, x, backend = "r"))
#' @export
kan_fit <- function(x, y, hidden = 5L, grid_size = 5L, spline_order = 3L,
                    grid_range = c(-1, 1),
                    extrapolate = c("linear", "none", "clamp"),
                    base_fun = c("silu", "identity", "none"), bias = TRUE,
                    epochs = 200L, batch_size = NULL,
                    optimizer = c("adam", "lbfgs"), lr = NULL,
                    weight_decay = 0, validation_split = 0.2,
                    validation_data = NULL, patience = 30L, min_delta = 0,
                    restore_best = TRUE,
                    normalize_x = c("minmax", "standardize", "none"),
                    normalize_y = TRUE, reg = kan_reg_control(),
                    grid_update_every = 0L, grid_update_until = 50L,
                    grid_eps = 0.02, grid_ridge = 1e-8, grad_clip = NULL,
                    seed = NULL,
                    backend = c("torch", "r"), dtype = "float32",
                    init = NULL, init_control = list(),
                    checkpoint_path = NULL, verbose = FALSE) {
  cl <- match.call()
  extrapolate <- match.arg(extrapolate)
  base_fun <- match.arg(base_fun)
  optimizer <- match.arg(optimizer)
  normalize_x <- match.arg(normalize_x)
  backend <- match.arg(backend)
  if (!inherits(reg, "kan_reg")) {
    stop("`reg` must come from kan_reg_control().", call. = FALSE)
  }
  epochs <- check_count(epochs, "epochs")
  validation_split <- check_number(validation_split, "validation_split", 0)
  if (validation_split >= 1) {
    stop("`validation_split` must be in [0, 1).", call. = FALSE)
  }
  if (!is.null(patience)) patience <- check_count(patience, "patience")
  min_delta <- check_number(min_delta, "min_delta", 0)
  weight_decay <- check_number(weight_decay, "weight_decay", 0)
  grad_clip <- check_number(grad_clip, "grad_clip", 0, allow_null = TRUE,
                            strict = TRUE)
  grid_update_every <- check_count(grid_update_every, "grid_update_every",
                                   min = 0L)
  grid_ridge <- check_number(grid_ridge, "grid_ridge", 0)
  if (backend == "torch") require_torch()
  if (backend == "r") {
    if (optimizer != "adam") {
      stop("The base R backend supports only optimizer = \"adam\".",
           call. = FALSE)
    }
    if (reg$lambda > 0) {
      stop("Regularisation (reg$lambda > 0) requires the torch backend.",
           call. = FALSE)
    }
  }
  if (is.null(lr)) lr <- if (optimizer == "adam") 0.01 else 1
  lr <- check_number(lr, "lr", 0, strict = TRUE)

  # ---- data ----------------------------------------------------------------
  feature_names <- colnames(x)
  x <- as_numeric_matrix(x)
  y_names <- if (is.matrix(y) || is.data.frame(y)) colnames(y) else NULL
  y <- as_numeric_matrix(y, "y")
  if (nrow(x) != nrow(y)) {
    stop("`x` and `y` must have the same number of rows.", call. = FALSE)
  }
  if (anyNA(x) || anyNA(y) || any(!is.finite(x)) || any(!is.finite(y))) {
    stop("`x` and `y` must not contain missing or non-finite values.",
         call. = FALSE)
  }
  if (is.null(feature_names)) feature_names <- paste0("x", seq_len(ncol(x)))
  if (is.null(y_names)) {
    y_names <- if (ncol(y) == 1L) "y" else paste0("y", seq_len(ncol(y)))
  }
  if (!is.null(seed)) {
    seed <- check_count(seed, "seed", min = 0L)
    set.seed(seed)
    if (backend == "torch") torch::torch_manual_seed(seed)
  }
  n <- nrow(x)
  if (!is.null(validation_data)) {
    if (!is.list(validation_data) || is.null(validation_data$x) ||
        is.null(validation_data$y)) {
      stop("`validation_data` must be list(x = , y = ).", call. = FALSE)
    }
    x_val <- as_numeric_matrix(validation_data$x, "validation_data$x")
    y_val <- as_numeric_matrix(validation_data$y, "validation_data$y")
    train_idx <- seq_len(n)
  } else if (validation_split > 0) {
    n_val <- max(1L, floor(n * validation_split))
    if (n - n_val < 2L) stop("Too few rows for this validation split.",
                             call. = FALSE)
    perm <- sample.int(n)
    val_idx <- sort(perm[seq_len(n_val)])
    train_idx <- sort(perm[-seq_len(n_val)])
    x_val <- x[val_idx, , drop = FALSE]
    y_val <- y[val_idx, , drop = FALSE]
  } else {
    train_idx <- seq_len(n)
    x_val <- NULL
    y_val <- NULL
  }
  x_tr <- x[train_idx, , drop = FALSE]
  y_tr <- y[train_idx, , drop = FALSE]

  # ---- scalers, spec, initial state -----------------------------------------
  if (!is.null(init)) {
    if (!inherits(init, "kan_model")) {
      stop("`init` must be a kan_model.", call. = FALSE)
    }
    scalers <- init$scalers
    state <- init$state
    spec <- state$spec
    if (spec$width[1L] != ncol(x) ||
        spec$width[length(spec$width)] != ncol(y)) {
      stop("`init` does not match the dimensions of `x` / `y`.",
           call. = FALSE)
    }
  } else {
    scalers <- list(
      x = kan_scaler(x_tr, normalize_x, target = grid_range),
      y = kan_scaler(y_tr, if (isTRUE(normalize_y)) "standardize" else "none")
    )
    spec <- kan_spec(c(ncol(x), as.integer(hidden), ncol(y)), grid_size,
                     spline_order, grid_range, extrapolate, base_fun, bias)
    state <- do.call(kan_init_state, c(list(spec = spec), init_control))
  }
  zx_tr <- kan_scale(x_tr, scalers$x)
  zy_tr <- kan_scale(y_tr, scalers$y)
  has_val <- !is.null(x_val)
  if (has_val) {
    zx_val <- kan_scale(x_val, scalers$x)
    zy_val <- kan_scale(y_val, scalers$y)
  }

  engine <- if (backend == "torch") {
    torch_engine(state, optimizer, lr, weight_decay, grad_clip, reg, dtype)
  } else {
    r_engine(state, lr, weight_decay, grad_clip)
  }
  engine$set_data(zx_tr, zy_tr, if (has_val) zx_val, if (has_val) zy_val)

  # ---- loop -----------------------------------------------------------------
  n_tr <- nrow(zx_tr)
  bs <- if (is.null(batch_size)) n_tr else
    min(check_count(batch_size, "batch_size"), n_tr)
  hist <- data.frame(epoch = integer(epochs), train_loss = NA_real_,
                     val_loss = NA_real_, grid_updated = FALSE)
  best <- Inf
  best_epoch <- 0L
  best_state <- NULL
  wait <- 0L
  stopped_early <- FALSE
  every <- max(1L, epochs %/% 10L)
  last <- 0L
  for (ep in seq_len(epochs)) {
    updated <- FALSE
    if (grid_update_every > 0L && (ep - 1L) %% grid_update_every == 0L &&
        ep <= grid_update_until) {
      st <- engine$get_state()
      engine$set_state(kan_update_grid(st, zx_tr, grid_eps = grid_eps,
                                       ridge = grid_ridge))
      updated <- TRUE
    }
    if (optimizer == "lbfgs") {
      engine$step_full()
    } else {
      perm <- sample.int(n_tr)
      for (b0 in seq(1L, n_tr, by = bs)) {
        idx <- perm[b0:min(b0 + bs - 1L, n_tr)]
        engine$step(idx)
      }
    }
    losses <- engine$eval_losses()
    if (!is.finite(losses[1L])) {
      stop(sprintf(paste0("Training diverged at epoch %d (non-finite loss). ",
                          "Try a smaller `lr` or set `grad_clip`."), ep),
           call. = FALSE)
    }
    hist[ep, ] <- list(ep, losses[1L], losses[2L], updated)
    last <- ep
    monitor <- if (has_val) losses[2L] else losses[1L]
    if (monitor < best - min_delta) {
      best <- monitor
      best_epoch <- ep
      wait <- 0L
      if (restore_best || !is.null(checkpoint_path)) {
        best_state <- engine$get_state()
      }
      if (!is.null(checkpoint_path)) {
        kan_save(new_kan_model(best_state, scalers, hist[seq_len(ep), ],
                               best_epoch, FALSE, NULL, list(), list(),
                               feature_names, y_names, cl),
                 checkpoint_path)
      }
    } else {
      wait <- wait + 1L
    }
    if (verbose && (ep %% every == 0L || ep == 1L)) {
      message(sprintf("epoch %4d  train %.6g%s", ep, losses[1L],
                      if (has_val) sprintf("  val %.6g", losses[2L]) else ""))
    }
    if (!is.null(patience) && wait >= patience) {
      stopped_early <- TRUE
      break
    }
  }
  hist <- hist[seq_len(last), , drop = FALSE]
  final_state <- if (restore_best && !is.null(best_state)) best_state else
    engine$get_state()

  act <- kan_activation_check(final_state, zx_tr)
  if (any(act$frac_outside > 0.05) && final_state$spec$extrapolate == "clamp") {
    warning(paste0("More than 5% of the inputs of some layer fall outside ",
                   "the spline domain with extrapolate = \"clamp\"; consider ",
                   "grid updates or extrapolate = \"linear\"."), call. = FALSE)
  }
  settings <- list(
    backend = backend, optimizer = optimizer, lr = lr,
    weight_decay = weight_decay, epochs = epochs, batch_size = bs,
    validation_split = validation_split, patience = patience,
    min_delta = min_delta, reg = reg, grid_update_every = grid_update_every,
    grid_update_until = grid_update_until, grid_eps = grid_eps,
    grid_ridge = grid_ridge,
    grad_clip = grad_clip, seed = seed, dtype = dtype,
    n_train = n_tr, n_val = if (has_val) nrow(zx_val) else 0L,
    train_index = if (is.null(validation_data)) train_idx else NULL
  )
  new_kan_model(final_state, scalers, hist, best_epoch, stopped_early, act,
                settings, kan_meta(), feature_names, y_names, cl)
}

#' Assemble a kan_model object
#' @noRd
new_kan_model <- function(state, scalers, history, best_epoch, stopped_early,
                          activation_check, settings, meta, feature_names,
                          response_names, call) {
  structure(list(
    state = state, spec = state$spec, scalers = scalers, history = history,
    best_epoch = best_epoch, stopped_early = stopped_early,
    activation_check = activation_check, settings = settings, meta = meta,
    feature_names = feature_names, response_names = response_names,
    call = call
  ), class = "kan_model")
}

#' Session metadata stored with a model
#' @noRd
kan_meta <- function() {
  list(
    created = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    r_version = R.version.string,
    splineKANr_version = as.character(utils::packageVersion("splineKANr")),
    torch_version = tryCatch(as.character(utils::packageVersion("torch")),
                             error = function(e) NA_character_)
  )
}

#' Fraction of layer inputs outside the spline domain
#'
#' @param state A `kan_state` or `kan_model` (for a model, `x` is on the
#'   original scale).
#' @param x Input matrix.
#' @return A data frame with one row per layer: observed input range,
#'   spline domain and the fraction of input values outside the domain.
#' @examples
#' st <- kan_init_state(kan_spec(c(2, 3, 1)), seed = 1)
#' kan_activation_check(st, matrix(runif(40, -1.5, 1.5), ncol = 2))
#' @export
kan_activation_check <- function(state, x) {
  if (inherits(state, "kan_model")) {
    x <- kan_scale(as_numeric_matrix(x), state$scalers$x)
  }
  state <- as_kan_state(state)
  fw <- kan_forward_r(state, x, detail = TRUE)
  k <- state$spec$spline_order
  rows <- lapply(seq_along(fw$layers), function(l) {
    inp <- fw$layers[[l]]$input
    dom <- grid_domain(state$layers[[l]]$grid, k)
    outside <- sweep(inp, 2L, dom[, 1L], `<`) | sweep(inp, 2L, dom[, 2L], `>`)
    data.frame(layer = l, input_min = min(inp), input_max = max(inp),
               domain_min = min(dom[, 1L]), domain_max = max(dom[, 2L]),
               frac_outside = mean(outside))
  })
  do.call(rbind, rows)
}

# ---- engines -------------------------------------------------------------------
# An engine owns the parameters during training and exposes
# set_data(), step(idx), step_full(), eval_losses(), get_state(), set_state().

#' torch autograd engine
#' @noRd
torch_engine <- function(state, optimizer, lr, weight_decay, grad_clip, reg,
                         dtype) {
  dt <- torch_dtype_from(dtype)
  net <- kan_module_from_state(state, dtype = dtype)
  spec <- state$spec
  opt <- NULL
  make_opt <- function() {
    if (optimizer == "adam") {
      torch::optim_adam(net$parameters, lr = lr, weight_decay = weight_decay)
    } else {
      torch::optim_lbfgs(net$parameters, lr = lr, max_iter = 20L,
                         history_size = 10L, line_search_fn = "strong_wolfe")
    }
  }
  opt <- make_opt()
  d <- new.env(parent = emptyenv())
  objective <- function(xb, yb) {
    use_reg <- reg$lambda > 0
    out <- net(xb, detail = use_reg)
    if (use_reg) {
      loss <- torch::nnf_mse_loss(out$output, yb) +
        reg$lambda * kan_regularization_torch(out$phi, net, reg)
    } else {
      loss <- torch::nnf_mse_loss(out, yb)
    }
    loss
  }
  clip <- function() {
    if (!is.null(grad_clip)) {
      torch::nn_utils_clip_grad_norm_(net$parameters, max_norm = grad_clip)
    }
  }
  list(
    set_data = function(x, y, xv, yv) {
      d$x <- torch::torch_tensor(x, dtype = dt)
      d$y <- torch::torch_tensor(y, dtype = dt)
      d$xv <- if (!is.null(xv)) torch::torch_tensor(xv, dtype = dt)
      d$yv <- if (!is.null(yv)) torch::torch_tensor(yv, dtype = dt)
    },
    step = function(idx) {
      xb <- d$x[as.integer(idx), , drop = FALSE]
      yb <- d$y[as.integer(idx), , drop = FALSE]
      opt$zero_grad()
      loss <- objective(xb, yb)
      loss$backward()
      clip()
      opt$step()
      invisible(loss$item())
    },
    step_full = function() {
      closure <- function() {
        opt$zero_grad()
        loss <- objective(d$x, d$y)
        loss$backward()
        clip()
        loss
      }
      opt$step(closure)
    },
    eval_losses = function() {
      torch::with_no_grad({
        tr <- torch::nnf_mse_loss(net(d$x), d$y)$item()
        va <- if (!is.null(d$xv)) torch::nnf_mse_loss(net(d$xv), d$yv)$item()
              else NA_real_
      })
      c(tr, va)
    },
    get_state = function() kan_state_from_module(net, spec),
    set_state = function(st) {
      if (!identical(dim(st$layers[[1L]]$coef),
                     as.integer(net$layers[[1L]]$coef$shape))) {
        stop("Grid size changes are not supported during training.",
             call. = FALSE)
      }
      kan_load_state_into(net, st)
    }
  )
}

#' Base R engine: analytic gradients + Adam (same update rule as torch)
#' @noRd
r_engine <- function(state, lr, weight_decay, grad_clip) {
  e <- new.env(parent = emptyenv())
  e$state <- state
  e$t <- 0L
  pnames <- c("coef", "scale_base", "scale_sp", "bias")
  if (!isTRUE(state$spec$bias)) pnames <- setdiff(pnames, "bias")
  zero_like <- function(st) lapply(st$layers, function(L) {
    stats::setNames(lapply(pnames, function(p) L[[p]] * 0), pnames)
  })
  e$m <- zero_like(state)
  e$v <- zero_like(state)
  b1 <- 0.9
  b2 <- 0.999
  eps <- 1e-8
  list(
    set_data = function(x, y, xv, yv) {
      e$x <- x
      e$y <- y
      e$xv <- xv
      e$yv <- yv
    },
    step = function(idx) {
      g <- kan_gradient_r(e$state, e$x[idx, , drop = FALSE],
                          e$y[idx, , drop = FALSE])
      grads <- lapply(g$grads, function(gl) gl[pnames])
      if (weight_decay > 0) {
        for (l in seq_along(grads)) for (p in pnames) {
          grads[[l]][[p]] <- grads[[l]][[p]] +
            weight_decay * e$state$layers[[l]][[p]]
        }
      }
      if (!is.null(grad_clip)) {
        tot <- sqrt(sum(unlist(lapply(grads, function(gl) {
          vapply(gl, function(a) sum(a^2), numeric(1))
        }))))
        if (tot > grad_clip) {
          f <- grad_clip / (tot + 1e-6)
          grads <- lapply(grads, function(gl) lapply(gl, `*`, f))
        }
      }
      e$t <- e$t + 1L
      for (l in seq_along(grads)) for (p in pnames) {
        gr <- grads[[l]][[p]]
        e$m[[l]][[p]] <- b1 * e$m[[l]][[p]] + (1 - b1) * gr
        e$v[[l]][[p]] <- b2 * e$v[[l]][[p]] + (1 - b2) * gr^2
        mh <- e$m[[l]][[p]] / (1 - b1^e$t)
        vh <- e$v[[l]][[p]] / (1 - b2^e$t)
        e$state$layers[[l]][[p]] <- e$state$layers[[l]][[p]] -
          lr * mh / (sqrt(vh) + eps)
      }
      invisible(g$loss)
    },
    step_full = function() stop("Not available in the R backend.",
                                call. = FALSE),
    eval_losses = function() {
      tr <- mean((kan_forward_r(e$state, e$x) - e$y)^2)
      va <- if (!is.null(e$xv)) mean((kan_forward_r(e$state, e$xv) - e$yv)^2)
            else NA_real_
      c(tr, va)
    },
    get_state = function() e$state,
    set_state = function(st) e$state <- st
  )
}
