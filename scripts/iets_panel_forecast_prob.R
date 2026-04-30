#!/usr/bin/env Rscript
# Panel probabilistic forecasts via iETS: smooth::adam() + forecast(..., interval="prediction").
# Maps central prediction intervals to requested quantiles.
# Usage: Rscript iets_panel_forecast_prob.R <train.csv> <eval.csv> <out.csv> <seed> [occurrence] [per_series_timeout_seconds] [quantiles]

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
quantile_arg <- if (length(args) >= 7L) args[[7L]] else "0.1,0.25,0.5,0.75,0.9"

.parse_quantiles <- function(x) {
  vals <- suppressWarnings(as.numeric(strsplit(gsub(";", ",", x), ",", fixed = FALSE)[[1L]]))
  vals <- vals[is.finite(vals)]
  vals[vals > 1] <- vals[vals > 1] / 100
  vals <- sort(unique(vals[vals > 0 & vals < 1]))
  if (length(vals) == 0L) {
    stop("No valid quantiles supplied.")
  }
  vals
}

requested_probs <- .parse_quantiles(quantile_arg)
requested_cols <- paste0("q_", requested_probs)
needed_levels <- c()
if (any(requested_probs %in% c(0.1, 0.9))) {
  needed_levels <- c(needed_levels, 80)
}
if (any(requested_probs %in% c(0.25, 0.5, 0.75))) {
  needed_levels <- c(needed_levels, 50)
}
needed_levels <- sort(unique(needed_levels), decreasing = TRUE)
if (length(needed_levels) == 0L) {
  stop("iETS R script supports quantiles among 0.1, 0.25, 0.5, 0.75, 0.9.")
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
    qs <- rep(0, length(requested_probs))
  } else {
    qs <- as.numeric(stats::quantile(y, probs = requested_probs, type = 8, na.rm = TRUE))
    qs[!is.finite(qs)] <- 0
    qs <- pmax(qs, 0)
    for (jj in 2L:length(qs)) {
      qs[[jj]] <- max(qs[[jj]], qs[[jj - 1L]])
    }
  }
  m <- matrix(qs, nrow = h, ncol = length(requested_probs), byrow = TRUE)
  colnames(m) <- requested_cols
  as.data.frame(m, stringsAsFactors = FALSE)
}

.interval_col <- function(levels, target) {
  idx <- which(as.numeric(levels) == as.numeric(target))
  if (length(idx) == 0L) {
    return(NA_integer_)
  }
  idx[[1L]]
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
        fc <- forecast(m, h = h, interval = "prediction", level = needed_levels)
        mean_v <- as.numeric(fc$mean)
        if (length(mean_v) != h) {
          mean_v <- rep(mu0, h)
        }
        L <- as.matrix(fc$lower)
        U <- as.matrix(fc$upper)
        if (nrow(L) != h || nrow(U) != h) {
          stop("unexpected forecast lower/upper dimensions")
        }
        col80 <- .interval_col(needed_levels, 80)
        col50 <- .interval_col(needed_levels, 50)
        q10 <- if (is.finite(col80)) as.numeric(L[, col80]) else rep(NA_real_, h)
        q25 <- if (is.finite(col50)) as.numeric(L[, col50]) else rep(NA_real_, h)
        q75 <- if (is.finite(col50)) as.numeric(U[, col50]) else rep(NA_real_, h)
        q90 <- if (is.finite(col80)) as.numeric(U[, col80]) else rep(NA_real_, h)
        q50 <- 0.5 * (q25 + q75)
        bad_q50 <- !is.finite(q50)
        q50[bad_q50] <- mean_v[bad_q50]
        qmat_all <- cbind(q10, q25, q50, q75, q90)
        colnames(qmat_all) <- c("q_0.1", "q_0.25", "q_0.5", "q_0.75", "q_0.9")
        qmat <- qmat_all[, requested_cols, drop = FALSE]
        qmat[!is.finite(qmat)] <- NA_real_
        qmat <- pmax(qmat, 0)
        med <- suppressWarnings(apply(qmat, 1L, stats::median, na.rm = TRUE))
        med[!is.finite(med)] <- mu0
        for (cc in seq_len(ncol(qmat))) {
          bad <- !is.finite(qmat[, cc])
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
    qdf[, requested_cols, drop = FALSE],
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, rows)
if (is.null(out) || nrow(out) == 0L) {
  out <- data.frame(
    unique_id = character(),
    ds = character()
  )
  for (cc in requested_cols) {
    out[[cc]] <- numeric()
  }
}

write.csv(out, out_path, row.names = FALSE)
