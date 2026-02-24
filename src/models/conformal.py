from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _split_calibration_chronological(
    train_df: pd.DataFrame,
    cal_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-series chronological split into fit and calibration portions."""
    ratio = float(np.clip(cal_ratio, 0.05, 0.5))
    fit_parts: list[pd.DataFrame] = []
    cal_parts: list[pd.DataFrame] = []
    for _, grp in train_df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
        n = len(grp)
        if n <= 2:
            fit_parts.append(grp)
            continue
        cut = max(int(np.floor(n * (1.0 - ratio))), 1)
        cut = min(cut, n - 1)
        fit_parts.append(grp.iloc[:cut])
        cal_parts.append(grp.iloc[cut:])
    fit_df = pd.concat(fit_parts, ignore_index=True) if fit_parts else train_df.iloc[0:0].copy()
    cal_df = pd.concat(cal_parts, ignore_index=True) if cal_parts else train_df.iloc[0:0].copy()
    return fit_df, cal_df


POINT_MODEL_NAMES = ["CrostonClassic", "CrostonSBA", "TSB", "ADIDA", "IMAPA"]


def _make_sf_point_models():
    from statsforecast.models import CrostonClassic, CrostonSBA, TSB, ADIDA, IMAPA

    return [
        CrostonClassic(),
        CrostonSBA(),
        TSB(alpha_d=0.5, alpha_p=0.45),
        ADIDA(),
        IMAPA(),
    ]


def _fit_predict_sf_point(
    train_panel: pd.DataFrame,
    target_df: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """Fit StatsForecast point models and predict, trimming to target timestamps."""
    from statsforecast import StatsForecast

    valid_uids = set(target_df["unique_id"].unique())
    panel = train_panel.loc[train_panel["unique_id"].isin(valid_uids), ["unique_id", "ds", "y"]]
    if panel.empty:
        return pd.DataFrame()

    h_max = int(target_df.groupby("unique_id").size().max())
    sf = StatsForecast(models=_make_sf_point_models(), freq=freq, n_jobs=-1)
    sf.fit(df=panel)
    pred = sf.predict(h=h_max).reset_index()
    pred = target_df[["unique_id", "ds"]].merge(pred, on=["unique_id", "ds"], how="inner")
    return pred


def fit_predict_conformal_baselines(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    quantiles: List[float],
    cal_ratio: float = 0.2,
    freq: str = "D",
) -> pd.DataFrame:
    """Wrap point-only StatsForecast baselines with split conformal prediction intervals.

    Pipeline:
      1. Chronologically split train_df into fit / calibration.
      2. Fit point baselines on fit portion, predict on calibration set.
      3. Compute per-model residual quantiles (conformal scores).
      4. Refit on full train_df, predict on eval_df.
      5. Shift point forecasts by residual quantiles → prediction intervals.

    Returns long-format DataFrame: [model, unique_id, ds, q_0.1, q_0.25, ...].
    Model names are prefixed with "CP-" (e.g. CP-CrostonClassic).
    """
    qcols = [f"q_{q}" for q in quantiles]
    empty = pd.DataFrame(columns=["model", "unique_id", "ds"] + qcols)

    if eval_df.empty or train_df.empty:
        return empty

    fit_df, cal_df = _split_calibration_chronological(train_df, cal_ratio=cal_ratio)
    if cal_df.empty or fit_df.empty:
        return empty

    cal_pred = _fit_predict_sf_point(fit_df, cal_df, freq=freq)
    if cal_pred.empty:
        return empty

    cal_merged = cal_df[["unique_id", "ds", "y"]].merge(
        cal_pred, on=["unique_id", "ds"], how="inner",
    )
    if cal_merged.empty:
        return empty

    model_cols = [c for c in POINT_MODEL_NAMES if c in cal_merged.columns]
    adjustments: dict[str, dict[float, float]] = {}
    for mcol in model_cols:
        valid = cal_merged.dropna(subset=[mcol])
        if len(valid) < 10:
            continue
        residuals = valid["y"].to_numpy(dtype=float) - valid[mcol].to_numpy(dtype=float)
        adjustments[mcol] = {q: float(np.quantile(residuals, q)) for q in quantiles}

    if not adjustments:
        return empty

    eval_pred = _fit_predict_sf_point(train_df, eval_df, freq=freq)
    if eval_pred.empty:
        return empty

    frames: list[pd.DataFrame] = []
    for mcol, adj in adjustments.items():
        if mcol not in eval_pred.columns:
            continue
        out = eval_pred[["unique_id", "ds"]].copy()
        point = eval_pred[mcol].to_numpy(dtype=float)
        for q in quantiles:
            out[f"q_{q}"] = np.maximum(point + adj[q], 0.0)
        arr = out[qcols].to_numpy(dtype=float)
        for j in range(1, arr.shape[1]):
            arr[:, j] = np.maximum(arr[:, j], arr[:, j - 1])
        out[qcols] = arr
        out["model"] = f"CP-{mcol}"
        frames.append(out[["model", "unique_id", "ds"] + qcols])

    if not frames:
        return empty
    return pd.concat(frames, ignore_index=True)
