# Phase 6: multi-layer networks against pykan's MultKAN (float64), generated
# by inst/validation/pykan_multilayer_reference.py. With default settings a
# MultKAN is a composition of KANLayers without node bias, i.e. splineKANr
# with extrapolate = "none" and bias = FALSE.

ml_cases <- function() {
  testthat::skip_if_not_installed("jsonlite")
  f <- testthat::test_path("fixtures", "pykan_multilayer.json")
  testthat::skip_if_not(file.exists(f), "pykan multilayer fixture missing")
  jsonlite::fromJSON(f, simplifyVector = TRUE, simplifyDataFrame = FALSE,
                     simplifyMatrix = TRUE)$cases
}

ml_state <- function(cs, layers = cs$layers) {
  w <- cs$width
  spec <- kan_spec(w, grid_size = cs$grid, spline_order = cs$k,
                   extrapolate = "none", bias = FALSE)
  nb <- cs$grid + cs$k
  L <- lapply(seq_len(length(w) - 1L), function(l) {
    ly <- layers[[l]]
    list(
      coef = array(ly$coef, c(w[l], w[l + 1L], nb)),
      scale_base = array(ly$scale_base, c(w[l], w[l + 1L])),
      scale_sp = array(ly$scale_sp, c(w[l], w[l + 1L])),
      mask = matrix(1, w[l], w[l + 1L]),
      bias = numeric(w[l + 1L]),
      grid = array(ly$grid, c(w[l], cs$grid + 2 * cs$k + 1))
    )
  })
  splineKANr:::new_kan_state(spec, L)
}

ml_xy <- function(cs) {
  w <- cs$width
  list(x = array(cs$x, c(30, w[1])),
       y = array(cs$target, c(30, w[length(w)])))
}

test_that("multi-layer forward pass matches pykan MultKAN", {
  for (cs in ml_cases()) {
    st <- ml_state(cs)
    d <- ml_xy(cs)
    y_py <- array(cs$y, dim(d$y))
    expect_lt(max(abs(kan_forward_r(st, d$x) - y_py)), 1e-10,
              label = paste(cs$width, collapse = "-"))
  }
})

test_that("multi-layer loss and gradients match pykan autograd", {
  for (cs in ml_cases()) {
    st <- ml_state(cs)
    d <- ml_xy(cs)
    g <- kan_gradient_r(st, d$x, d$y)
    expect_equal(g$loss, cs$loss, tolerance = 1e-12)
    w <- cs$width
    for (l in seq_along(g$grads)) {
      gp <- cs$grads[[l]]
      dims <- c(w[l], w[l + 1L])
      expect_lt(max(abs(g$grads[[l]]$coef -
                          array(gp$coef, c(dims, cs$grid + cs$k)))), 1e-10)
      expect_lt(max(abs(g$grads[[l]]$scale_base -
                          array(gp$scale_base, dims))), 1e-10)
      expect_lt(max(abs(g$grads[[l]]$scale_sp -
                          array(gp$scale_sp, dims))), 1e-10)
    }
  }
})

test_that("five Adam steps follow the same trajectory as pykan", {
  for (cs in ml_cases()) {
    st <- ml_state(cs)
    d <- ml_xy(cs)
    pn <- c("coef", "scale_base", "scale_sp")
    m <- lapply(st$layers, function(L) lapply(L[pn], `*`, 0))
    v <- m
    lr <- cs$adam_lr
    losses <- numeric(5)
    for (t in 1:5) {
      g <- kan_gradient_r(st, d$x, d$y)
      losses[t] <- g$loss
      for (l in seq_along(st$layers)) for (p in pn) {
        gr <- g$grads[[l]][[p]]
        m[[l]][[p]] <- 0.9 * m[[l]][[p]] + 0.1 * gr
        v[[l]][[p]] <- 0.999 * v[[l]][[p]] + 0.001 * gr^2
        st$layers[[l]][[p]] <- st$layers[[l]][[p]] - lr *
          (m[[l]][[p]] / (1 - 0.9^t)) /
          (sqrt(v[[l]][[p]] / (1 - 0.999^t)) + 1e-8)
      }
    }
    expect_equal(losses, cs$adam_losses, tolerance = 1e-10)
    ref <- ml_state(cs, cs$after_adam)
    for (l in seq_along(st$layers)) {
      expect_lt(max(abs(st$layers[[l]]$coef - ref$layers[[l]]$coef)), 1e-9)
    }
  }
})

test_that("network-wide grid update places the same knots as pykan", {
  for (cs in ml_cases()) {
    st <- ml_state(cs, cs$after_adam)   # pykan updated after the Adam steps
    xs <- array(cs$update_x, c(200, cs$width[1]))
    st2 <- kan_update_grid(st, xs, grid_eps = 0.02)
    for (l in seq_along(st2$layers)) {
      g_py <- array(cs$after_grid[[l]], dim(st2$layers[[l]]$grid))
      # layer 1 depends only on x; deeper layers also on the refitted
      # coefficients (least squares in pykan, ridge 1e-8 here)
      tol <- if (l == 1L) 1e-12 else 1e-5
      expect_lt(max(abs(st2$layers[[l]]$grid - g_py)), tol,
                label = sprintf("%s layer %d",
                                paste(cs$width, collapse = "-"), l))
    }
  }
})
