from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def me(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.nanmean(yhat - y))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.nanmean(np.abs(yhat - y)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((yhat - y) ** 2)))


def _per_series_naive_mse(init_set: pd.DataFrame, y_col: str = "y") -> pd.Series:
    init_sorted = init_set.sort_values(["unique_id", "ds"]).copy()
    init_sorted["y_lag1"] = init_sorted.groupby("unique_id")[y_col].shift(1)
    init_sorted["naive_sq_err"] = (init_sorted[y_col] - init_sorted["y_lag1"]) ** 2
    epsilon = 1e-9
    return init_sorted.groupby("unique_id")["naive_sq_err"].mean().where(lambda s: s > 0, epsilon)


def _per_series_model_mse(
    eval_df: pd.DataFrame,
    y_col: str = "y",
    yhat_col: str = "y_pred",
) -> pd.Series:
    tmp = eval_df[["unique_id", y_col, yhat_col]].dropna().copy()
    tmp["model_sq_err"] = (tmp[y_col] - tmp[yhat_col]) ** 2
    return tmp.groupby("unique_id")["model_sq_err"].mean()


def _prepare_rmsse_frame(
    init_set: pd.DataFrame,
    eval_df: pd.DataFrame,
    y_col: str = "y",
    yhat_col: str = "y_pred",
) -> pd.DataFrame:
    denom = _per_series_naive_mse(init_set, y_col=y_col)
    mse_per_series = _per_series_model_mse(eval_df, y_col=y_col, yhat_col=yhat_col)
    series_eval = pd.concat([
        mse_per_series.rename("model_mse"),
        denom.rename("rmsse_denom"),
    ], axis=1)
    series_eval = series_eval.dropna(subset=["model_mse", "rmsse_denom"])
    if series_eval.empty:
        return series_eval
    series_eval["scaled_err_sq"] = series_eval["model_mse"] / series_eval["rmsse_denom"]
    return series_eval


def rmsse(
    init_set: pd.DataFrame,
    eval_df: pd.DataFrame,
    y_col: str = "y",
    yhat_col: str = "y_pred",
) -> float:
    series_eval = _prepare_rmsse_frame(init_set, eval_df, y_col=y_col, yhat_col=yhat_col)
    if series_eval.empty:
        return float("nan")
    return float(np.sqrt(np.nanmean(series_eval["scaled_err_sq"])) )


def wrmsse(
    init_set: pd.DataFrame,
    eval_df: pd.DataFrame,
    weights: pd.Series | None = None,
    y_col: str = "y",
    yhat_col: str = "y_pred",
) -> float:
    """Weighted RMSSE using per-series demand weights.

    Weights default to the share of total demand in ``init_set``.
    """

    series_eval = _prepare_rmsse_frame(init_set, eval_df, y_col=y_col, yhat_col=yhat_col)
    if series_eval.empty:
        return float("nan")

    if weights is None:
        weights = init_set.groupby("unique_id")[y_col].sum()
    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights)

    weights = weights.reindex(series_eval.index).fillna(0.0)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        weights = pd.Series(1.0, index=series_eval.index)
        total_weight = float(weights.sum())
    norm_weights = weights / total_weight

    weighted_scaled_err = series_eval["scaled_err_sq"] * norm_weights
    return float(np.sqrt(np.nansum(weighted_scaled_err)))


def coverage_rate(df: pd.DataFrame, lower_q: float, upper_q: float, alpha: float) -> Dict[str, float]:
    cover = ((df["y"] >= df[f"q_{lower_q}"]) & (df["y"] <= df[f"q_{upper_q}"])).mean()
    width = (df[f"q_{upper_q}"] - df[f"q_{lower_q}"]).mean()
    return {f"Coverage@{int(alpha*100)}": float(cover), f"AIW@{int(alpha*100)}": float(width)}


def pit_values(df: pd.DataFrame, quantiles: Iterable[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> np.ndarray:
    qs = list(quantiles)
    pits = []
    for _, row in df.iterrows():
        y = row["y"]
        qvals = [row[f"q_{q}"] for q in qs]
        if y <= qvals[0]:
            pit = 0.0
        elif y >= qvals[-1]:
            pit = 1.0
        else:
            pit = 0.0
            for i in range(len(qs) - 1):
                if qvals[i] <= y <= qvals[i + 1]:
                    q_low, q_high = qs[i], qs[i + 1]
                    pit = q_low + (q_high - q_low) * (y - qvals[i]) / max(qvals[i + 1] - qvals[i], 1e-12)
                    break
        pits.append(pit)
    return np.asarray(pits)


def compute_adi_cv2(init_set: pd.DataFrame) -> pd.DataFrame:
    # CV^2 only on positive demand
    demand_events = init_set[init_set["y"] > 0]
    series_stats = demand_events.groupby("unique_id")["y"].agg(["mean", "std"]).reset_index()
    series_stats["cv_sq"] = (series_stats["std"] / series_stats["mean"]) ** 2
    series_stats["cv_sq"] = series_stats["cv_sq"].fillna(0)

    # ADI from counts on init set
    g_init = init_set.copy()
    g_init["occ"] = (g_init["y"] > 0).astype(int)
    s = g_init.groupby("unique_id")["occ"].sum()
    n = g_init.groupby("unique_id")["ds"].nunique()
    adi_df = pd.DataFrame({"unique_id": s.index, "s": s.values, "n": n.values})
    adi_df["adi"] = np.where(adi_df["s"] > 0, adi_df["n"] / adi_df["s"], adi_df["n"])

    out = pd.merge(series_stats[["unique_id", "cv_sq"]], adi_df[["unique_id", "adi"]], on="unique_id")
    return out


def classify_adi_cv2(row: pd.Series, adi_threshold: float = 1.32, cv2_threshold: float = 0.49) -> str:
    if row["adi"] <= adi_threshold and row["cv_sq"] <= cv2_threshold:
        return "Smooth"
    elif row["adi"] > adi_threshold and row["cv_sq"] <= cv2_threshold:
        return "Intermittent"
    elif row["adi"] <= adi_threshold and row["cv_sq"] > cv2_threshold:
        return "Erratic"
    else:
        return "Lumpy"

