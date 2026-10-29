# Fourier KAN comparator (base R) ----------------------------------------------
#
# Benchmark comparator only -- not part of the package API. Each edge is a
# truncated Fourier series (FourierKAN, Xu et al. 2024):
#
#   phi_ij(x) = sum_{k=1}^K  a_ijk cos(k x) + b_ijk sin(k x),
#
# each node sums its incoming edges and adds a bias. Gradients are analytic
# (checked against finite differences by fkan_check_gradient()), training is
# full-batch Adam with the same early-stopping protocol as kan_fit():
# inputs min-max scaled to [-1, 1] on the training rows, response
# standardised, 20 % validation split, patience 30, best epoch restored.

fkan_init <- function(width, K, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  lapply(seq_len(length(width) - 1L), function(l) {
    p <- width[l]
    q <- width[l + 1L]
    s <- 1 / sqrt(p * K)
    list(A = array(stats::rnorm(K * p * q, sd = s), c(K, p, q)),
         B = array(stats::rnorm(K * p * q, sd = s), c(K, p, q)),
         b = numeric(q))
  })
}

fkan_forward <- function(par, X, keep = FALSE) {
  cache <- vector("list", length(par))
  for (l in seq_along(par)) {
    L <- par[[l]]
    K <- dim(L$A)[1L]
    Y <- matrix(L$b, nrow(X), length(L$b), byrow = TRUE)
    for (k in seq_len(K)) {
      Y <- Y + cos(k * X) %*% matrix(L$A[k, , ], dim(L$A)[2L]) +
        sin(k * X) %*% matrix(L$B[k, , ], dim(L$B)[2L])
    }
    if (keep) cache[[l]] <- X
    X <- Y
  }
  if (keep) list(out = X, cache = cache) else X
}

fkan_grad <- function(par, X, y) {
  fw <- fkan_forward(par, X, keep = TRUE)
  n <- nrow(X)
  dY <- 2 * (fw$out - y) / length(y)
  g <- vector("list", length(par))
  for (l in rev(seq_along(par))) {
    L <- par[[l]]
    Xl <- fw$cache[[l]]
    K <- dim(L$A)[1L]
    p <- dim(L$A)[2L]
    gA <- gB <- array(0, dim(L$A))
    dX <- matrix(0, n, p)
    for (k in seq_len(K)) {
      Ck <- cos(k * Xl)
      Sk <- sin(k * Xl)
      Ak <- matrix(L$A[k, , ], p)
      Bk <- matrix(L$B[k, , ], p)
      gA[k, , ] <- crossprod(Ck, dY)
      gB[k, , ] <- crossprod(Sk, dY)
      dX <- dX - k * Sk * (dY %*% t(Ak)) + k * Ck * (dY %*% t(Bk))
    }
    g[[l]] <- list(A = gA, B = gB, b = colSums(dY))
    dY <- dX
  }
  list(loss = mean((fw$out - y)^2), grads = g)
}

fkan_check_gradient <- function(width = c(3, 4, 1), K = 4, n = 30, h = 1e-6,
                                seed = 1) {
  par <- fkan_init(width, K, seed)
  X <- matrix(stats::runif(n * width[1L], -1, 1), n)
  y <- matrix(stats::rnorm(n))
  an <- unlist(fkan_grad(par, X, y)$grads)
  flat <- unlist(par)
  rebuild <- function(v) {
    pos <- 0L
    lapply(par, function(L) lapply(L, function(a) {
      a[] <- v[pos + seq_along(a)]
      pos <<- pos + length(a)
      a
    }))
  }
  fd <- vapply(seq_along(flat), function(i) {
    up <- dn <- flat
    up[i] <- up[i] + h
    dn[i] <- dn[i] - h
    (mean((fkan_forward(rebuild(up), X) - y)^2) -
       mean((fkan_forward(rebuild(dn), X) - y)^2)) / (2 * h)
  }, numeric(1))
  max(abs(an - fd) / pmax(abs(fd), 1e-6))
}

fkan_fit <- function(x, y, hidden = integer(), K = 5L, epochs = 300L,
                     lr = 0.01, validation_split = 0.2, patience = 30L,
                     seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  x <- as.matrix(x)
  y <- as.matrix(y)
  lo <- apply(x, 2, min)
  hi <- apply(x, 2, max)
  rng <- ifelse(hi > lo, hi - lo, 1)
  sx <- function(z) sweep(sweep(as.matrix(z), 2, lo), 2, rng, "/") * 2 - 1
  my <- mean(y)
  sy <- stats::sd(y)
  if (!is.finite(sy) || sy == 0) sy <- 1
  Z <- sx(x)
  Y <- (y - my) / sy
  n <- nrow(Z)
  vi <- sample.int(n, max(1L, floor(n * validation_split)))
  par <- fkan_init(c(ncol(Z), hidden, 1L), K)
  m <- lapply(par, function(L) lapply(L, `*`, 0))
  v <- m
  best <- Inf
  best_par <- par
  wait <- 0L
  for (t in seq_len(epochs)) {
    g <- fkan_grad(par, Z[-vi, , drop = FALSE], Y[-vi, , drop = FALSE])$grads
    for (l in seq_along(par)) for (p in names(par[[l]])) {
      m[[l]][[p]] <- 0.9 * m[[l]][[p]] + 0.1 * g[[l]][[p]]
      v[[l]][[p]] <- 0.999 * v[[l]][[p]] + 0.001 * g[[l]][[p]]^2
      par[[l]][[p]] <- par[[l]][[p]] - lr * (m[[l]][[p]] / (1 - 0.9^t)) /
        (sqrt(v[[l]][[p]] / (1 - 0.999^t)) + 1e-8)
    }
    vl <- mean((fkan_forward(par, Z[vi, , drop = FALSE]) - Y[vi, ])^2)
    if (!is.finite(vl)) break
    if (vl < best) {
      best <- vl
      best_par <- par
      wait <- 0L
    } else {
      wait <- wait + 1L
      if (wait >= patience) break
    }
  }
  structure(list(par = best_par, sx = sx, my = my, sy = sy, epochs = t),
            class = "fkan")
}

predict.fkan <- function(object, newdata, ...) {
  as.numeric(fkan_forward(object$par, object$sx(newdata))) * object$sy +
    object$my
}
