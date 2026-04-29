#!/usr/bin/env Rscript
# Panel point forecasts via iETS (intermittent ETS) using smooth::adam().
# Usage: Rscript iets_panel_forecast.R <train.csv> <eval.csv> <out.csv> <seed> [occurrence]
#   train.csv / eval.csv: columns unique_id, ds, y
#   occurrence: passed to adam(..., occurrence=) — default "auto"

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) {
  stop("Usage: Rscript iets_panel_forecast.R <train.csv> <eval.csv> <out.csv> <seed> [occurrence]")
}

train_path <- args[[1L]]
eval_path <- args[[2L]]
out_path <- args[[3L]]
seed <- suppressWarnings(as.integer(args[[4L]]))
if (is.na(seed)) {
  seed <- 42L
}
occurrence <- if (length(args) >= 5L) args[[5L]] else "auto"

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

.safe_mean_fc <- function(fc, h) {
  if (!is.null(fc$mean)) {
    v <- as.numeric(fc$mean)
    if (length(v) == h) {
      return(v)
    }
  }
  if (!is.null(fc$forecast)) {
    v <- as.numeric(fc$forecast)
    if (length(v) == h) {
      return(v)
    }
  }
  rep(NA_real_, h)
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

  if (length(y) < 5L) {
    yhat <- rep(0.0, h)
  } else if (sum(y > 0, na.rm = TRUE) < 1L) {
    yhat <- rep(0.0, h)
  } else {
    yhat <- tryCatch(
      {
        m <- adam(
          y,
          model = "YYN",
          occurrence = occurrence,
          h = h
        )
        fc <- forecast(m, h = h)
        v <- .safe_mean_fc(fc, h)
        if (any(is.na(v))) {
          v <- rep(mean(y, na.rm = TRUE), h)
        } else {
          v <- pmax(v, 0)
        }
        v[!is.finite(v)] <- mean(y, na.rm = TRUE)
        y_pos <- y[y > 0 & is.finite(y)]
        y_scale <- if (length(y_pos) > 0L) {
          max(stats::quantile(y_pos, 0.95, na.rm = TRUE), mean(y, na.rm = TRUE), 1, na.rm = TRUE)
        } else {
          max(mean(y, na.rm = TRUE), 1, na.rm = TRUE)
        }
        v <- pmin(v, y_scale * 200)
        v[!is.finite(v)] <- 0
        v
      },
      error = function(e) {
        rep(max(mean(y, na.rm = TRUE), 0), h)
      }
    )
  }

  if (length(yhat) != h) {
    yhat <- rep(max(mean(y, na.rm = TRUE), 0), h)[seq_len(h)]
  }

  rows[[i]] <- data.frame(
    unique_id = uid,
    ds = ev$ds,
    iETS = as.numeric(yhat),
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, rows)
if (is.null(out) || nrow(out) == 0L) {
  out <- data.frame(unique_id = character(), ds = character(), iETS = numeric())
}

write.csv(out, out_path, row.names = FALSE)
