#!/usr/bin/env Rscript
# Panel probabilistic forecasts via iETS: smooth::adam() + forecast(..., interval="prediction").
# Maps level c(90,80,50) to quantiles 0.1, 0.25, 0.5, 0.75, 0.9 (mean = median point).
# Usage: Rscript iets_panel_forecast_prob.R <train.csv> <eval.csv> <out.csv> <seed> [occurrence] [per_series_timeout_seconds]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) {
  stop("Usage: Rscript iets_panel_forecast_prob.R <train.csv> <eval.csv> <out.csv> <seed> [occurrence]")
}

train_path <- args[[1L]]
eval_path <- args[[2L]]
out_path <- args[[3L]]
seed <- suppressWarnings(as.integer(args[[4L]]))
if (is.na(seed)) {
  seed <- 42L
}
occurrence <- if (length(args) >= 5L) args[[5L]] else "auto"
per_series_timeout <- if (length(args) >= 6L) suppressWarnings(as.numeric(args[[6L]])) else 10
if (!is.finite(per_series_timeout) || per_series_timeout <= 0) {
  per_series_timeout <- 10
}

suppressPackageStartupMessages({
  if (!requireNamespace("smooth", quietly = TRUE)) {
    stop("R package 'smooth' is required. Install with: install.packages(c('smooth','greybox'))")
  }
  library(smooth)
})

allowed_occ <- c(
  "auto", "none", "fixed", "odds-ratio", "inverse-odds-ratio",
  "direct", "general"
)
if (!(occurrence %in% allowed_occ)) {
  stop(sprintf("Invalid occurrence='%s'. Choose one of: %s", occurrence, paste(allowed_occ, collapse = ", ")))
}

set.seed(seed)
train <- read.csv(train_path, stringsAsFactors = FALSE)
eval <- read.csv(eval_path, stringsAsFactors = FALSE)

needed <- c("unique_id", "ds", "y")
if (!all(needed %in% names(train))) {
  stop("train.csv must contain columns: unique_id, ds, y")
}
if (!all(c("unique_id", "ds") %in% names(eval))) {
  stop("eval.csv must contain columns: unique_id, ds")
}

train <- train[, needed, drop = FALSE]
eval <- eval[, unique(c("unique_id", "ds", intersect(names(eval), "y"))), drop = FALSE]

train <- train[order(train$unique_id, train$ds), , drop = FALSE]
eval <- eval[order(eval$unique_id, eval$ds), , drop = FALSE]

uids <- unique(as.character(eval$unique_id))
rows <- vector("list", length(uids))

.fallback_quantiles <- function(h, y) {
  y <- as.numeric(y)
  y[!is.finite(y)] <- 0
  y <- pmax(y, 0)
  if (length(y) == 0L) {
    qs <- rep(0, 5L)
  } else {
    qs <- as.numeric(stats::quantile(y, probs = c(0.10, 0.25, 0.50, 0.75, 0.90), type = 8, na.rm = TRUE))
    qs[!is.finite(qs)] <- 0
    qs <- pmax(qs, 0)
    for (jj in 2L:length(qs)) {
      qs[[jj]] <- max(qs[[jj]], qs[[jj - 1L]])
    }
  }
  m <- matrix(qs, nrow = h, ncol = 5L, byrow = TRUE)
  colnames(m) <- c("q_0.1", "q_0.25", "q_0.5", "q_0.75", "q_0.9")
  as.data.frame(m, stringsAsFactors = FALSE)
}

for (i in seq_along(uids)) {
  uid <- uids[[i]]
  tr <- train[as.character(train$unique_id) == uid, "y", drop = TRUE]
  ev <- eval[as.character(eval$unique_id) == uid, , drop = FALSE]
  h <- nrow(ev)
  if (h == 0L) {
    next
  }

  y <- as.numeric(tr)
  y[is.na(y)] <- 0
  mu0 <- max(mean(y, na.rm = TRUE), 0)

  if (length(y) < 5L || sum(y > 0, na.rm = TRUE) < 1L) {
    qdf <- .fallback_quantiles(h, y)
  } else {
    qdf <- tryCatch(
      {
        setTimeLimit(elapsed = per_series_timeout, transient = TRUE)
        m <- adam(
          y,
          model = "YYN",
          occurrence = occurrence,
          h = h
        )
        fc <- forecast(m, h = h, interval = "prediction", level = c(90, 80, 50))
        mean_v <- as.numeric(fc$mean)
        if (length(mean_v) != h) {
          mean_v <- rep(mu0, h)
        }
        L <- as.matrix(fc$lower)
        U <- as.matrix(fc$upper)
        if (nrow(L) != h || ncol(L) < 3L || nrow(U) != h || ncol(U) < 3L) {
          stop("unexpected forecast lower/upper dimensions")
        }
        # level = c(90, 80, 50): column 2 is 80% PI, column 3 is 50% PI.
        q10 <- as.numeric(L[, 2L])
        q25 <- as.numeric(L[, 3L])
        q50 <- as.numeric(mean_v)
        q75 <- as.numeric(U[, 3L])
        q90 <- as.numeric(U[, 2L])
        qmat <- cbind(q10, q25, q50, q75, q90)
        colnames(qmat) <- c("q_0.1", "q_0.25", "q_0.5", "q_0.75", "q_0.9")
        qmat[!is.finite(qmat)] <- NA_real_
        qmat <- pmax(qmat, 0)
        # Intervals can underflow to ~0 while the mean stays O(1); impute from central path.
        med <- suppressWarnings(apply(qmat, 1L, stats::median, na.rm = TRUE))
        med[!is.finite(med)] <- mu0
        for (cc in seq_len(ncol(qmat))) {
          bad <- !is.finite(qmat[, cc])
          if (cc <= 2L) {
            bad <- bad | (qmat[, cc] < 1e-8 & med > 1e-6)
          }
          qmat[bad, cc] <- med[bad]
        }
        for (jj in 2L:ncol(qmat)) {
          qmat[, jj] <- pmax(qmat[, jj], qmat[, jj - 1L])
        }
        y_pos <- y[y > 0 & is.finite(y)]
        y_scale <- if (length(y_pos) > 0L) {
          max(stats::quantile(y_pos, 0.95, na.rm = TRUE), mu0, 1, na.rm = TRUE)
        } else {
          max(mu0, 1, na.rm = TRUE)
        }
        cap <- y_scale * 20
        qmat <- pmin(qmat, cap)
        for (jj in 2L:ncol(qmat)) {
          qmat[, jj] <- pmax(qmat[, jj], qmat[, jj - 1L])
        }
        as.data.frame(qmat, stringsAsFactors = FALSE)
      },
      error = function(e) {
        .fallback_quantiles(h, y)
      },
      finally = {
        setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
      }
    )
  }

  rows[[i]] <- data.frame(
    unique_id = uid,
    ds = ev$ds,
    qdf,
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, rows)
if (is.null(out) || nrow(out) == 0L) {
  out <- data.frame(
    unique_id = character(),
    ds = character(),
    q_0.1 = numeric(),
    q_0.25 = numeric(),
    q_0.5 = numeric(),
    q_0.75 = numeric(),
    q_0.9 = numeric()
  )
}

write.csv(out, out_path, row.names = FALSE)
