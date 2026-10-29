# Model zoo for the out-of-sample benchmarks -----------------------------------
#
# Every method receives exactly the same training rows and is evaluated on
# exactly the same test rows. Methods with tuning parameters choose them on
# the same inner hold-out (25 % of the training rows, fixed per replicate)
# from a small grid of comparable size, then are refitted on all training
# rows. Requires: splineKANr, mgcv, nnet, ranger; fourier_kan.R sourced.

bench_methods <- c("mean", "ols", "gam", "ranger", "ann", "fourier_kan",
                   "bspline_kan")

bench_grids <- list(
  ann = expand.grid(size = c(3L, 5L, 10L), decay = c(0.01, 0.1, 1)),
  fourier_kan = list(list(hidden = integer(), K = 3L),
                     list(hidden = integer(), K = 5L),
                     list(hidden = 3L, K = 3L), list(hidden = 3L, K = 5L)),
  bspline_kan = list(list(hidden = integer(), grid_size = 3L),
                     list(hidden = integer(), grid_size = 5L),
                     list(hidden = 3L, grid_size = 3L),
                     list(hidden = 3L, grid_size = 5L))
)

fit_one <- function(method, x, y, cfg = NULL, seed = 1L) {
  x <- as.matrix(x)
  df <- data.frame(x, check.names = FALSE)
  set.seed(seed)
  switch(method,
    mean = list(m = mean(y)),
    ols = stats::lm(y ~ ., data = cbind(df, y = y)),
    gam = {
      terms <- vapply(colnames(x), function(v) {
        u <- length(unique(x[, v]))
        if (u >= 6L) sprintf("s(`%s`, k = %d)", v, min(10L, u - 1L)) else
          sprintf("`%s`", v)
      }, character(1))
      mgcv::gam(stats::as.formula(paste("y ~", paste(terms, collapse = "+"))),
                data = cbind(df, y = y), method = "REML")
    },
    ranger = ranger::ranger(y ~ ., data = cbind(df, y = y), num.trees = 500L,
                            seed = seed),
    ann = {
      mx <- colMeans(x)
      sx <- apply(x, 2, stats::sd)
      sx[sx == 0] <- 1
      my <- mean(y)
      sy <- stats::sd(y)
      nets <- lapply(1:3, function(r) {           # 3 restarts, best kept
        nnet::nnet(scale(x, mx, sx), (y - my) / sy, size = cfg$size,
                   decay = cfg$decay, linout = TRUE, maxit = 2000L,
                   trace = FALSE)
      })
      net <- nets[[which.min(vapply(nets, `[[`, numeric(1), "value"))]]
      list(net = net, mx = mx, sx = sx, my = my, sy = sy)
    },
    fourier_kan = fkan_fit(x, y, hidden = cfg$hidden, K = cfg$K,
                           epochs = 400L, lr = 0.01, seed = seed),
    bspline_kan = splineKANr::kan_fit(
      x, y, hidden = cfg$hidden, grid_size = cfg$grid_size,
      spline_order = 3L, epochs = 400L, lr = 0.01, patience = 30L,
      validation_split = 0.2, backend = "r", seed = seed),
    stop("unknown method ", method)
  )
}

predict_one <- function(method, fit, x) {
  x <- as.matrix(x)
  df <- data.frame(x, check.names = FALSE)
  switch(method,
    mean = rep(fit$m, nrow(x)),
    ols = , gam = as.numeric(stats::predict(fit, newdata = df)),
    ranger = stats::predict(fit, data = df)$predictions,
    ann = as.numeric(stats::predict(fit$net, scale(x, fit$mx, fit$sx))) *
      fit$sy + fit$my,
    fourier_kan = stats::predict(fit, x),
    bspline_kan = stats::predict(fit, x, backend = "r")
  )
}

configs <- function(method) {
  g <- bench_grids[[method]]
  if (is.null(g)) return(list(NULL))
  if (is.data.frame(g)) return(lapply(seq_len(nrow(g)),
                                      function(i) as.list(g[i, ])))
  g
}

cfg_label <- function(cfg) {
  if (is.null(cfg)) return("")
  paste(names(cfg), vapply(cfg, function(v) {
    if (length(v)) paste(v, collapse = "-") else "none"
  }, character(1)), sep = "=", collapse = ",")
}

#' Tune on an inner hold-out, refit on all training rows, predict the test rows
fit_predict <- function(method, xtr, ytr, xte, seed) {
  cf <- configs(method)
  chosen <- cf[[1L]]
  if (length(cf) > 1L) {
    set.seed(seed + 7919L)
    hold <- sample.int(nrow(xtr), floor(0.25 * nrow(xtr)))
    err <- vapply(cf, function(cfg) {
      f <- fit_one(method, xtr[-hold, , drop = FALSE], ytr[-hold], cfg, seed)
      mean((predict_one(method, f, xtr[hold, , drop = FALSE]) -
              ytr[hold])^2)
    }, numeric(1))
    chosen <- cf[[which.min(err)]]
  }
  t0 <- proc.time()[["elapsed"]]
  f <- fit_one(method, xtr, ytr, chosen, seed)
  list(pred = predict_one(method, f, xte), config = cfg_label(chosen),
       seconds = proc.time()[["elapsed"]] - t0)
}

score <- function(y, pred, ybar_train) {
  c(rmse = sqrt(mean((y - pred)^2)), mae = mean(abs(y - pred)),
    q2_predict = 1 - sum((y - pred)^2) / sum((y - ybar_train)^2))
}
