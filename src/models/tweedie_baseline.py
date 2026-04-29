from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TWEEDIE_POINT_COL = "Tweedie"
TWEEDIE_PROB_MODEL = "Tweedie"


def _clean_nonnegative(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.maximum(arr, 0.0)


def _history_cap(y: np.ndarray) -> float:
    arr = _clean_nonnegative(y)
    if arr.size == 0:
        return 1.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    return float(max(np.nanmax(finite), np.nanquantile(finite, 0.995) * 3.0, np.nanmean(finite) * 8.0, 1.0))


def _build_lag_design(y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= lags:
        return np.empty((0, lags), dtype=float), np.empty((0,), dtype=float)
    rows = [y[i - lags:i] for i in range(lags, len(y))]
    X = np.log1p(np.asarray(rows, dtype=float))
    target = np.asarray(y[lags:], dtype=float)
    return X, target


def _recursive_forecast(model: Pipeline, history: np.ndarray, horizon: int, lags: int) -> np.ndarray:
    hist_arr = _clean_nonnegative(history)
    cap = _history_cap(hist_arr)
    hist = np.clip(hist_arr, 0.0, cap).tolist()
    preds: list[float] = []
    for _ in range(horizon):
        x = np.asarray(hist[-lags:], dtype=float)
        x = np.where(np.isfinite(x), x, 0.0)
        x = np.log1p(np.clip(x, 0.0, cap)).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            yhat = float(model.predict(x)[0])
        if not np.isfinite(yhat):
            yhat = float(np.nanmean(hist_arr))
        yhat = float(np.clip(yhat, 0.0, cap))
        preds.append(yhat)
        hist.append(yhat)
    return np.asarray(preds, dtype=float)


def _moment_dispersion(y: np.ndarray, power: float) -> float:
    y = _clean_nonnegative(y)
    if y.size == 0:
        return 1.0
    mu = float(np.nanmean(y))
    if not np.isfinite(mu) or mu <= 1e-9:
        return 1.0
    var = float(np.nanvar(y))
    phi = var / max(mu ** float(power), 1e-9)
    return float(np.clip(phi if np.isfinite(phi) else 1.0, 1e-6, 1e6))


def _pearson_dispersion(y: np.ndarray, mu: np.ndarray, power: float, n_params: int) -> float:
    y = _clean_nonnegative(y)
    mu = np.maximum(_clean_nonnegative(mu), 1e-9)
    if y.size == 0 or mu.size == 0:
        return 1.0
    denom = np.maximum(mu ** float(power), 1e-9)
    resid = ((y - mu) ** 2) / denom
    dof = max(int(y.size) - int(n_params), 1)
    phi = float(np.nansum(resid) / dof)
    if not np.isfinite(phi) or phi <= 0:
        phi = _moment_dispersion(y, power)
    return float(np.clip(phi, 1e-6, 1e6))


def _tweedie_quantile_approx(mu: np.ndarray, phi: float, power: float, quantiles: list[float]) -> np.ndarray:
    """Approximate 1<p<2 Tweedie quantiles with zero mass plus gamma-matched positives."""
    mu_arr = np.maximum(_clean_nonnegative(mu), 0.0)
    out = np.zeros((mu_arr.size, len(quantiles)), dtype=float)
    if mu_arr.size == 0:
        return out

    p = float(power)
    phi = float(np.clip(phi if np.isfinite(phi) else 1.0, 1e-6, 1e6))
    active = mu_arr > 1e-12
    if not np.any(active):
        return out

    mu_pos = mu_arr[active]
    if not 1.0 < p < 2.0:
        sd = np.sqrt(np.maximum(phi * np.power(mu_pos, p), 1e-9))
        for j, q in enumerate(quantiles):
            out[active, j] = np.maximum(mu_pos + norm.ppf(float(np.clip(q, 1e-6, 1 - 1e-6))) * sd, 0.0)
        return out

    lam = np.maximum(np.power(mu_pos, 2.0 - p) / (phi * (2.0 - p)), 1e-12)
    p0 = np.exp(-np.minimum(lam, 745.0))
    p_pos = np.maximum(1.0 - p0, 1e-12)
    var = np.maximum(phi * np.power(mu_pos, p), 1e-9)
    cond_mean = mu_pos / p_pos
    cond_second = (var + mu_pos**2) / p_pos
    cond_var = np.maximum(cond_second - cond_mean**2, 1e-9)
    shape = np.maximum(cond_mean**2 / cond_var, 1e-6)
    scale = np.maximum(cond_var / cond_mean, 1e-9)

    for j, q_raw in enumerate(quantiles):
        q = float(np.clip(q_raw, 1e-6, 1 - 1e-6))
        vals = np.zeros_like(mu_pos)
        pos_mask = q > p0
        if np.any(pos_mask):
            q_pos = np.clip((q - p0[pos_mask]) / p_pos[pos_mask], 1e-9, 1 - 1e-9)
            vals[pos_mask] = gamma.ppf(q_pos, a=shape[pos_mask], scale=scale[pos_mask])
            bad = ~np.isfinite(vals)
            if np.any(bad):
                sd = np.sqrt(var[bad])
                vals[bad] = np.maximum(mu_pos[bad] + norm.ppf(q) * sd, 0.0)
        out[active, j] = vals
    return out


def _fit_uid_tweedie(
    y: np.ndarray,
    lags: int,
    power: float,
    alpha: float,
    max_iter: int,
) -> tuple[Pipeline | None, float]:
    y = _clean_nonnegative(y)
    cap = _history_cap(y)
    y = np.clip(y, 0.0, cap)
    X, target = _build_lag_design(y, lags)
    if X.shape[0] < 8 or np.allclose(target, target[0]):
        return None, _moment_dispersion(target if target.size else y, power)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "tweedie",
                TweedieRegressor(
                    power=float(power),
                    alpha=float(max(alpha, 0.0)),
                    link="log",
                    max_iter=int(max(max_iter, 100)),
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model.fit(X, np.maximum(target, 0.0))
        in_sample = np.maximum(model.predict(X), 0.0)
    phi = _pearson_dispersion(target, in_sample, power=power, n_params=X.shape[1] + 1)
    return model, phi


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
    train_sorted = train_df.sort_values(["unique_id", "ds"])
    eval_sorted = eval_df.sort_values(["unique_id", "ds"])
    for uid, grp_eval in eval_sorted.groupby("unique_id", sort=False):
        y_hist = train_sorted.loc[train_sorted["unique_id"] == uid, "y"].to_numpy(dtype=float)
        h = int(len(grp_eval))
        if h <= 0:
            continue
        phi = _moment_dispersion(y_hist, power=float(power))
        if y_hist.size < max(int(lags) + 2, 6):
            mu = np.repeat(float(max(np.nanmean(y_hist) if y_hist.size else 0.0, 0.0)), h)
        else:
            model, fit_phi = _fit_uid_tweedie(y_hist, int(lags), power=float(power), alpha=float(alpha), max_iter=int(max_iter))
            if model is None:
                mu = np.repeat(float(max(np.nanmean(y_hist), 0.0)), h)
            else:
                mu = _recursive_forecast(model, y_hist, horizon=h, lags=int(lags))
            phi = float(fit_phi)
        out = pd.DataFrame(
            {
                "unique_id": grp_eval["unique_id"].astype(str).to_numpy(),
                "ds": pd.to_datetime(grp_eval["ds"]).to_numpy(),
            }
        )
        q_arr = _tweedie_quantile_approx(mu, phi=phi, power=float(power), quantiles=quantiles)
        for j, q in enumerate(quantiles):
            out[f"q_{q}"] = q_arr[:, j]
        arr = out[qcols].to_numpy(dtype=float)
        for j in range(1, arr.shape[1]):
            arr[:, j] = np.maximum(arr[:, j], arr[:, j - 1])
        out[qcols] = arr
        out["model"] = model_name
        rows.append(out[["model", "unique_id", "ds"] + qcols])
    if not rows:
        return pd.DataFrame(columns=["model", "unique_id", "ds"] + qcols)
    return pd.concat(rows, ignore_index=True)
