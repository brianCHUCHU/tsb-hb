from __future__ import annotations

from statistics import NormalDist
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import TweedieRegressor

TWEEDIE_POINT_COL = "Tweedie"
TWEEDIE_PROB_MODEL = "Tweedie"


def _build_lag_design(y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= lags:
        return np.empty((0, lags), dtype=float), np.empty((0,), dtype=float)
    rows = [y[i - lags:i] for i in range(lags, len(y))]
    X = np.asarray(rows, dtype=float)
    target = np.asarray(y[lags:], dtype=float)
    return X, target


def _recursive_forecast(model: TweedieRegressor, history: np.ndarray, horizon: int, lags: int) -> np.ndarray:
    hist_arr = np.asarray(history, dtype=float)
    hist_arr = np.where(np.isfinite(hist_arr), hist_arr, 0.0)
    cap = float(max(np.nanquantile(hist_arr, 0.99) * 20.0, np.nanmean(hist_arr) * 20.0, 1e3))
    hist = np.clip(hist_arr, 0.0, cap).tolist()
    preds: list[float] = []
    for _ in range(horizon):
        x = np.asarray(hist[-lags:], dtype=float)
        x = np.where(np.isfinite(x), x, 0.0)
        x = np.clip(x, 0.0, cap).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            yhat = float(model.predict(x)[0])
        if not np.isfinite(yhat):
            yhat = float(np.nanmean(x))
        yhat = float(np.clip(yhat, 0.0, cap))
        preds.append(yhat)
        hist.append(yhat)
    return np.asarray(preds, dtype=float)


def _fit_uid_tweedie(
    y: np.ndarray,
    lags: int,
    power: float,
    alpha: float,
    max_iter: int,
) -> tuple[TweedieRegressor | None, float]:
    y = np.asarray(y, dtype=float)
    y = np.where(np.isfinite(y), y, 0.0)
    cap = float(max(np.nanquantile(y, 0.99) * 20.0, np.nanmean(y) * 20.0, 1e3))
    y = np.clip(y, 0.0, cap)
    X, target = _build_lag_design(y, lags)
    if X.shape[0] < 8 or np.allclose(target, target[0]):
        return None, float(np.std(target)) if target.size else 0.0
    model = TweedieRegressor(
        power=float(power),
        alpha=float(max(alpha, 0.0)),
        link="log",
        max_iter=int(max(max_iter, 100)),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model.fit(X, np.maximum(target, 0.0))
        in_sample = np.maximum(model.predict(X), 0.0)
    sigma = float(np.std(target - in_sample))
    return model, sigma


def fit_predict_tweedie_panel(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    lags: int = 14,
    power: float = 1.5,
    alpha: float = 0.1,
    max_iter: int = 1000,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    train_sorted = train_df.sort_values(["unique_id", "ds"])
    eval_sorted = eval_df.sort_values(["unique_id", "ds"])
    for uid, grp_eval in eval_sorted.groupby("unique_id", sort=False):
        y_hist = train_sorted.loc[train_sorted["unique_id"] == uid, "y"].to_numpy(dtype=float)
        h = int(len(grp_eval))
        if h <= 0:
            continue
        if y_hist.size < max(int(lags) + 2, 6):
            fallback = float(max(np.nanmean(y_hist) if y_hist.size else 0.0, 0.0))
            yhat = np.repeat(fallback, h)
        else:
            model, _ = _fit_uid_tweedie(y_hist, int(lags), power=float(power), alpha=float(alpha), max_iter=int(max_iter))
            if model is None:
                fallback = float(max(np.nanmean(y_hist), 0.0))
                yhat = np.repeat(fallback, h)
            else:
                yhat = _recursive_forecast(model, y_hist, horizon=h, lags=int(lags))
        rows.append(
            pd.DataFrame(
                {
                    "unique_id": grp_eval["unique_id"].astype(str).to_numpy(),
                    "ds": pd.to_datetime(grp_eval["ds"]).to_numpy(),
                    TWEEDIE_POINT_COL: np.maximum(yhat, 0.0),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["unique_id", "ds", TWEEDIE_POINT_COL])
    return pd.concat(rows, ignore_index=True)


def fit_predict_tweedie_prob_panel(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    quantiles: list[float],
    lags: int = 14,
    power: float = 1.5,
    alpha: float = 0.1,
    max_iter: int = 1000,
    model_name: str = TWEEDIE_PROB_MODEL,
) -> pd.DataFrame:
    qcols = [f"q_{q}" for q in quantiles]
    rows: list[pd.DataFrame] = []
    nd = NormalDist()
    train_sorted = train_df.sort_values(["unique_id", "ds"])
    eval_sorted = eval_df.sort_values(["unique_id", "ds"])
    for uid, grp_eval in eval_sorted.groupby("unique_id", sort=False):
        y_hist = train_sorted.loc[train_sorted["unique_id"] == uid, "y"].to_numpy(dtype=float)
        h = int(len(grp_eval))
        if h <= 0:
            continue
        sigma = float(np.std(y_hist[y_hist > 0])) if np.any(y_hist > 0) else 1.0
        if y_hist.size < max(int(lags) + 2, 6):
            mu = np.repeat(float(max(np.nanmean(y_hist) if y_hist.size else 0.0, 0.0)), h)
        else:
            model, fit_sigma = _fit_uid_tweedie(y_hist, int(lags), power=float(power), alpha=float(alpha), max_iter=int(max_iter))
            if model is None:
                mu = np.repeat(float(max(np.nanmean(y_hist), 0.0)), h)
            else:
                mu = _recursive_forecast(model, y_hist, horizon=h, lags=int(lags))
            sigma = max(float(fit_sigma), sigma * 0.2, 1e-3)
        out = pd.DataFrame(
            {
                "unique_id": grp_eval["unique_id"].astype(str).to_numpy(),
                "ds": pd.to_datetime(grp_eval["ds"]).to_numpy(),
            }
        )
        for q in quantiles:
            z = nd.inv_cdf(float(np.clip(q, 1e-6, 1 - 1e-6)))
            out[f"q_{q}"] = np.maximum(mu + z * sigma, 0.0)
        arr = out[qcols].to_numpy(dtype=float)
        for j in range(1, arr.shape[1]):
            arr[:, j] = np.maximum(arr[:, j], arr[:, j - 1])
        out[qcols] = arr
        out["model"] = model_name
        rows.append(out[["model", "unique_id", "ds"] + qcols])
    if not rows:
        return pd.DataFrame(columns=["model", "unique_id", "ds"] + qcols)
    return pd.concat(rows, ignore_index=True)
