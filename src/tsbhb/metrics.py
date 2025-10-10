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


def rmsse(
    init_set: pd.DataFrame,
    eval_df: pd.DataFrame,
    y_col: str = "y",
    yhat_col: str = "y_pred",
) -> float:
    # Denominator: mean squared naive error on training (lag-1)
    init_sorted = init_set.sort_values(["unique_id", "ds"]).copy()
    init_sorted["y_lag1"] = init_sorted.groupby("unique_id")[y_col].shift(1)
    init_sorted["naive_sq_err"] = (init_sorted[y_col] - init_sorted["y_lag1"]) ** 2
    epsilon = 1e-9
    denom = init_sorted.groupby("unique_id")["naive_sq_err"].mean().where(lambda s: s > 0, epsilon)

    # Numerator per-series MSE on evaluation
    tmp = eval_df[["unique_id", y_col, yhat_col]].dropna().copy()
    tmp["model_sq_err"] = (tmp[y_col] - tmp[yhat_col]) ** 2
    mse_per_series = tmp.groupby("unique_id")["model_sq_err"].mean()
    series_eval = pd.concat([
        mse_per_series.rename("model_mse"),
        denom.rename("rmsse_denom"),
    ], axis=1)
    series_eval["scaled_err_sq"] = series_eval["model_mse"] / series_eval["rmsse_denom"]
    return float(np.sqrt(np.nanmean(series_eval["scaled_err_sq"])) )


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
    series_stats["cv_sq"].fillna(0, inplace=True)

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

