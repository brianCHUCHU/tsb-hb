from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    set_seed,
    default_data_file,
    default_out_dir,
    default_m5_sales_file,
    default_m5_calendar_file,
)
from data_loading import (
    load_online_retail,
    preprocess_online_retail,
    train_eval_split_fixed_origin,
    load_m5_long,
    preprocess_m5,
)
from experiments.protocols import evaluate_prob_models, iter_walk_forward_frames
from models.tsb_hb import (
    fit_tsb_hb,
    initialize_online_tsb_hb,
    predict_online_tsb_hb,
    predict_tsb_hb,
    update_online_tsb_hb,
)
from models.baselines import fit_predict_baselines
from models.hurdle_baselines import (
    fit_hurdle_global_lognormal,
    fit_hurdle_local_lognormal,
    predict_hurdle_global_lognormal,
    predict_hurdle_local_lognormal,
)
from models.conformal import fit_predict_conformal_baselines
from models.tweedie_baseline import fit_predict_tweedie_prob_panel
from metrics import coverage_rate, pit_values, compute_adi_cv2, classify_adi_cv2

# Optional neural baseline
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import DeepAR
    from neuralforecast.losses.pytorch import DistributionLoss
except (ImportError, AttributeError):
    NeuralForecast = None


QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
PROB_TSBHB_MODEL = "TSB-HB"


def _normalize_prob_baseline_mode(baseline_mode: str | None) -> str:
    mode = str(baseline_mode or "paper").lower()
    if mode == "full":
        mode = "extended"
    valid = {"paper", "extended", "hurdle_only", "hb_only", "fast_classic"}
    if mode not in valid:
        raise ValueError("baseline_mode must be one of: paper, extended, full, hurdle_only, hb_only, fast_classic.")
    return mode


def _include_parametric_prob_baselines(baseline_mode: str) -> bool:
    return baseline_mode in {"paper", "extended"}


def _include_hurdle_prob_baselines(baseline_mode: str) -> bool:
    return baseline_mode in {"extended", "hurdle_only"}


def _include_conformal_prob_baselines(baseline_mode: str) -> bool:
    return baseline_mode in {"paper", "extended", "fast_classic"}


def _qcols(quantiles: list[float]) -> list[str]:
    return [f"q_{q}" for q in quantiles]


def _build_regime_group_labels(train_df: pd.DataFrame) -> pd.Series | None:
    feats = compute_adi_cv2(train_df)
    if feats.empty:
        return None
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)
    return feats.set_index("unique_id")["category"].astype(str)


def _enforce_monotonic_quantiles(df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    out = df.copy()
    qcols = _qcols(quantiles)
    for c in qcols:
        if c not in out.columns:
            out[c] = np.nan
    arr = out[qcols].to_numpy(dtype=float)
    arr = np.where(np.isnan(arr), np.nan, np.maximum(arr, 0.0))
    for j in range(1, arr.shape[1]):
        prev = arr[:, j - 1]
        cur = arr[:, j]
        cur = np.where(np.isnan(cur), prev, cur)
        prev_filled = np.where(np.isnan(prev), cur, prev)
        arr[:, j] = np.maximum(cur, prev_filled)
    out[qcols] = arr
    return out


def _split_hb_calibration_train(
    init_set: pd.DataFrame,
    calibration_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratio = float(np.clip(calibration_ratio, 0.0, 0.8))
    if ratio <= 0.0:
        return init_set.copy(), pd.DataFrame(columns=init_set.columns)

    pieces_train: list[pd.DataFrame] = []
    pieces_calib: list[pd.DataFrame] = []
    for _, grp in init_set.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
        n = len(grp)
        if n <= 1:
            pieces_train.append(grp)
            continue
        cut = int(np.floor(n * (1.0 - ratio)))
        cut = max(min(cut, n - 1), 1)
        pieces_train.append(grp.iloc[:cut])
        pieces_calib.append(grp.iloc[cut:])

    train_df = pd.concat(pieces_train, ignore_index=True) if pieces_train else init_set.iloc[0:0].copy()
    calib_df = pd.concat(pieces_calib, ignore_index=True) if pieces_calib else init_set.iloc[0:0].copy()
    return train_df, calib_df


def _pinball_mean(eval_merged: pd.DataFrame, quantiles: list[float]) -> float:
    losses: list[float] = []
    for q in quantiles:
        col = f"q_{q}"
        if col not in eval_merged.columns:
            continue
        dfx = eval_merged.dropna(subset=[col])
        if dfx.empty:
            continue
        err = dfx["y"] - dfx[col]
        loss = np.maximum(q * err, (q - 1) * err).mean()
        losses.append(float(loss))
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def _series_naive_scale(init_set: pd.DataFrame) -> pd.Series:
    """Per-series scale for normalized pinball (MASE-like denominator)."""
    tmp = init_set[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).copy()
    tmp["lag1"] = tmp.groupby("unique_id")["y"].shift(1)
    tmp["abs_diff"] = (tmp["y"] - tmp["lag1"]).abs()
    scale = tmp.groupby("unique_id")["abs_diff"].mean()
    return scale


def _scaled_pinball_table(
    eval_merged: pd.DataFrame,
    init_set: pd.DataFrame,
    quantiles: list[float],
) -> pd.DataFrame:
    scales = _series_naive_scale(init_set)
    rows: list[dict[str, float | str | int]] = []
    for model, dfm in eval_merged.groupby("model"):
        for q in quantiles:
            col = f"q_{q}"
            if col not in dfm.columns:
                continue
            dfx = dfm.dropna(subset=[col]).copy()
            if dfx.empty:
                continue
            err = dfx["y"] - dfx[col]
            dfx["pinball"] = np.maximum(q * err, (q - 1) * err)
            by_uid = dfx.groupby("unique_id", as_index=False)["pinball"].mean()
            by_uid["scale"] = by_uid["unique_id"].map(scales)
            valid = by_uid[(by_uid["scale"] > 0) & np.isfinite(by_uid["scale"])].copy()
            if valid.empty:
                rows.append(
                    {
                        "model": model,
                        "quantile": q,
                        "scaled_pinball": float("nan"),
                        "n_series_scaled": 0,
                    }
                )
                continue
            valid["scaled_pinball_uid"] = valid["pinball"] / valid["scale"]
            rows.append(
                {
                    "model": model,
                    "quantile": q,
                    "scaled_pinball": float(valid["scaled_pinball_uid"].mean()),
                    "n_series_scaled": int(valid.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _fit_hb_location_scale(
    init_set: pd.DataFrame,
    quantiles: list[float],
    hb_regime_aware: bool,
    hb_group_shrink_strength: float,
    hb_item_variance_mode: str,
    hb_variance_prior_df: float,
    hb_bootstrap_draws: int,
    hb_bootstrap_seed: Optional[int],
    hb_use_hyper_uncertainty: bool,
    calibration_ratio: float,
    calibration_samples: int,
    lambda_min: float,
    lambda_max: float,
    lambda_steps: int,
    coverage_weight: float,
) -> dict[str, float | str]:
    default: dict[str, float | str] = {"mode": "location_scale", "delta": 0.0, "lambda": 1.0}
    train_fit, calib_set = _split_hb_calibration_train(init_set, calibration_ratio=calibration_ratio)
    if calib_set.empty or train_fit.empty:
        return default

    group_labels = _build_regime_group_labels(train_fit) if hb_regime_aware else None
    params = fit_tsb_hb(
        train_fit,
        group_labels=group_labels,
        bootstrap_draws=max(int(hb_bootstrap_draws), 0),
        bootstrap_seed=hb_bootstrap_seed,
        group_shrink_strength=hb_group_shrink_strength,
        item_variance_mode=hb_item_variance_mode,
        item_variance_shrink_strength=hb_variance_prior_df,
    )
    calib_q = predict_tsb_hb(
        params,
        calib_set,
        quantiles=quantiles,
        n_samples=max(int(calibration_samples), 200),
        include_hyper_uncertainty=hb_use_hyper_uncertainty,
    )
    merged_raw = calib_set[["unique_id", "ds", "y"]].merge(calib_q, on=["unique_id", "ds"], how="inner")
    if merged_raw.empty or "q_0.5" not in merged_raw.columns:
        return default

    delta = float(np.nanmedian((merged_raw["y"] - merged_raw["q_0.5"]).to_numpy(dtype=float)))
    if not np.isfinite(delta):
        delta = 0.0

    lam_lo = float(lambda_min)
    lam_hi = float(lambda_max)
    if lam_lo > lam_hi:
        lam_lo, lam_hi = lam_hi, lam_lo
    n_steps = max(int(lambda_steps), 2)
    lambdas = np.linspace(lam_lo, lam_hi, n_steps)
    w_cov = float(max(coverage_weight, 0.0))

    best_obj = float("inf")
    best_lam = 1.0
    best_pin = float("nan")
    best_gap50 = float("nan")
    best_gap80 = float("nan")

    for lam in lambdas:
        adj = _apply_hb_location_scale(
            calib_q,
            quantiles=quantiles,
            delta=delta,
            lam=float(lam),
        )
        merged = calib_set[["unique_id", "ds", "y"]].merge(adj, on=["unique_id", "ds"], how="inner")
        if merged.empty:
            continue
        pin = _pinball_mean(merged, quantiles=quantiles)
        if not np.isfinite(pin):
            continue
        cov50 = coverage_rate(merged, 0.25, 0.75, 0.5)["Coverage@50"]
        cov80 = coverage_rate(merged, 0.1, 0.9, 0.8)["Coverage@80"]
        gap50 = abs(float(cov50) - 0.5)
        gap80 = abs(float(cov80) - 0.8)
        obj = float(pin + w_cov * (gap50 + gap80))
        if obj < best_obj:
            best_obj = obj
            best_lam = float(lam)
            best_pin = float(pin)
            best_gap50 = float(gap50)
            best_gap80 = float(gap80)

    if not np.isfinite(best_obj):
        return default
    return {
        "mode": "location_scale",
        "delta": float(delta),
        "lambda": float(best_lam),
        "objective": float(best_obj),
        "pinball_cal": float(best_pin),
        "cov_gap50_cal": float(best_gap50),
        "cov_gap80_cal": float(best_gap80),
        "coverage_weight": float(w_cov),
    }


def _apply_hb_location_scale(
    tsbhb_q: pd.DataFrame,
    quantiles: list[float],
    delta: float,
    lam: float,
) -> pd.DataFrame:
    qcols = _qcols(quantiles)
    if "q_0.5" not in tsbhb_q.columns:
        return _enforce_monotonic_quantiles(tsbhb_q, quantiles)

    out = tsbhb_q.copy()
    q50 = out["q_0.5"].astype(float)
    center = q50 + float(delta)
    lam = float(max(lam, 1e-6))
    for col in qcols:
        if col not in out.columns:
            continue
        out[col] = center + lam * (out[col].astype(float) - q50)
        out[col] = out[col].clip(lower=0.0)
    return _enforce_monotonic_quantiles(out, quantiles)


def _apply_hb_calibration(
    tsbhb_q: pd.DataFrame,
    quantiles: list[float],
    calibration: Optional[dict[str, float | str]],
) -> pd.DataFrame:
    if not calibration:
        return _enforce_monotonic_quantiles(tsbhb_q, quantiles)

    mode = str(calibration.get("mode", "none")).lower()
    if mode == "location_scale":
        delta = float(calibration.get("delta", 0.0))
        lam = float(calibration.get("lambda", 1.0))
        return _apply_hb_location_scale(tsbhb_q, quantiles=quantiles, delta=delta, lam=lam)
    return _enforce_monotonic_quantiles(tsbhb_q, quantiles)


def _extract_sf_quantiles(sf_q: pd.DataFrame, model: str, quantiles: list[float]) -> pd.DataFrame:
    rename_map = {
        f"{model}": "q_0.5",
        f"{model}-lo-80": "q_0.1",
        f"{model}-hi-80": "q_0.9",
        f"{model}-lo-50": "q_0.25",
        f"{model}-hi-50": "q_0.75",
    }
    cols = [model, f"{model}-lo-80", f"{model}-hi-80", f"{model}-lo-50", f"{model}-hi-50", "unique_id", "ds"]
    cols = [c for c in cols if c in sf_q.columns]
    qcols = _qcols(quantiles)
    if not cols:
        out = pd.DataFrame(columns=["model", "unique_id", "ds"] + qcols)
        return out

    out = sf_q[cols].copy().rename(columns=rename_map)
    for c in qcols:
        if c not in out.columns:
            out[c] = np.nan
    out = _enforce_monotonic_quantiles(out, quantiles=quantiles)
    out["model"] = model
    return out[["model", "unique_id", "ds"] + qcols]


def _predict_non_hb_prob_models_once(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    quantiles: list[float],
    baseline_mode: str = "paper",
    with_tweedie: bool = False,
    tweedie_lags: int = 14,
    tweedie_power: float = 1.5,
    tweedie_alpha: float = 0.1,
    tweedie_max_iter: int = 1000,
) -> pd.DataFrame:
    qcols = _qcols(quantiles)
    if eval_df.empty:
        return pd.DataFrame(columns=["model", "unique_id", "ds"] + qcols)
    baseline_mode = _normalize_prob_baseline_mode(baseline_mode)
    if baseline_mode == "hb_only" and not with_tweedie:
        return pd.DataFrame(columns=["model", "unique_id", "ds"] + qcols)

    frames: list[pd.DataFrame] = []
    if _include_parametric_prob_baselines(baseline_mode):
        eval_h = eval_df["unique_id"].value_counts()
        sf_q = fit_predict_baselines(train_df, eval_h, freq="D", probabilistic=True, levels=[80, 50])
        arima = _extract_sf_quantiles(sf_q, "AutoARIMA", quantiles)
        theta = _extract_sf_quantiles(sf_q, "AutoTheta", quantiles)
        frames.extend([arima, theta])

    if _include_hurdle_prob_baselines(baseline_mode):
        local_params = fit_hurdle_local_lognormal(train_df)
        local_q = predict_hurdle_local_lognormal(local_params, eval_df, quantiles=quantiles)
        for c in qcols:
            if c not in local_q.columns:
                local_q[c] = np.nan
        local_q = _enforce_monotonic_quantiles(local_q, quantiles=quantiles)
        local_q["model"] = "Hurdle-Local-LogNormal"
        local_q = local_q[["model", "unique_id", "ds"] + qcols]

        global_params = fit_hurdle_global_lognormal(train_df)
        global_q = predict_hurdle_global_lognormal(global_params, eval_df, quantiles=quantiles)
        for c in qcols:
            if c not in global_q.columns:
                global_q[c] = np.nan
        global_q = _enforce_monotonic_quantiles(global_q, quantiles=quantiles)
        global_q["model"] = "Hurdle-Global-LogNormal"
        global_q = global_q[["model", "unique_id", "ds"] + qcols]

        frames.extend([local_q, global_q])
    if with_tweedie:
        tw_q = fit_predict_tweedie_prob_panel(
            train_df,
            eval_df,
            quantiles=quantiles,
            lags=tweedie_lags,
            power=tweedie_power,
            alpha=tweedie_alpha,
            max_iter=tweedie_max_iter,
        )
        frames.append(tw_q)

    if not frames:
        return pd.DataFrame(columns=["model", "unique_id", "ds"] + qcols)
    return pd.concat(frames, ignore_index=True)


def _predict_prob_models_once(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    quantiles: list[float],
    hb_group_labels: pd.Series | None = None,
    hb_bootstrap_draws: int = 0,
    hb_bootstrap_seed: int | None = None,
    hb_use_hyper_uncertainty: bool = True,
    hb_group_shrink_strength: float = 0.0,
    hb_item_variance_mode: str = "group",
    hb_variance_prior_df: float = 20.0,
    hb_calibration: Optional[dict[str, float | str]] = None,
    baseline_mode: str = "paper",
    include_conformal: bool = False,
    conformal_cal_ratio: float = 0.2,
    with_tweedie: bool = False,
    tweedie_lags: int = 14,
    tweedie_power: float = 1.5,
    tweedie_alpha: float = 0.1,
    tweedie_max_iter: int = 1000,
) -> pd.DataFrame:
    qcols = _qcols(quantiles)
    if eval_df.empty:
        return pd.DataFrame(columns=["model", "unique_id", "ds"] + qcols)

    params = fit_tsb_hb(
        train_df,
        group_labels=hb_group_labels,
        bootstrap_draws=hb_bootstrap_draws,
        bootstrap_seed=hb_bootstrap_seed,
        group_shrink_strength=hb_group_shrink_strength,
        item_variance_mode=hb_item_variance_mode,
        item_variance_shrink_strength=hb_variance_prior_df,
    )
    tsbhb_q = predict_tsb_hb(
        params,
        eval_df,
        quantiles=quantiles,
        n_samples=2000,
        include_hyper_uncertainty=hb_use_hyper_uncertainty,
    )
    for c in qcols:
        if c not in tsbhb_q.columns:
            tsbhb_q[c] = np.nan
    tsbhb_q = _apply_hb_calibration(tsbhb_q, quantiles=quantiles, calibration=hb_calibration)
    tsbhb_q["model"] = PROB_TSBHB_MODEL
    tsbhb_q = tsbhb_q[["model", "unique_id", "ds"] + qcols]

    non_hb = _predict_non_hb_prob_models_once(
        train_df,
        eval_df,
        quantiles,
        baseline_mode=baseline_mode,
        with_tweedie=with_tweedie,
        tweedie_lags=tweedie_lags,
        tweedie_power=tweedie_power,
        tweedie_alpha=tweedie_alpha,
        tweedie_max_iter=tweedie_max_iter,
    )

    all_frames = [tsbhb_q, non_hb]

    baseline_mode = _normalize_prob_baseline_mode(baseline_mode)
    if include_conformal and _include_conformal_prob_baselines(baseline_mode):
        conformal_q = fit_predict_conformal_baselines(
            train_df, eval_df,
            quantiles=quantiles,
            cal_ratio=conformal_cal_ratio,
            freq="D",
        )
        if not conformal_q.empty:
            all_frames.append(conformal_q)

    return pd.concat(all_frames, ignore_index=True)


def _rolling_forecast_over_eval_probabilistic(
    nf,
    base_model_col: str,
    quantile_cols: list[str],
    init_hist: pd.DataFrame,
    eval_set: pd.DataFrame,
    h: int,
) -> pd.DataFrame:
    if h <= 0:
        raise ValueError("h must be positive.")

    hist = init_hist[["unique_id", "ds", "y"]].copy()
    eval_idx = eval_set[["unique_id", "ds", "y"]].copy()
    eval_idx["k"] = eval_idx.groupby("unique_id").cumcount()
    k_max = int(eval_idx["k"].max()) if not eval_idx.empty else -1

    preds: list[pd.DataFrame] = []
    n_blocks = int(np.ceil((k_max + 1) / h)) if k_max >= 0 else 0
    all_pred_cols = [base_model_col] + quantile_cols

    for _ in range(n_blocks):
        block_fcst = nf.predict(df=hist).reset_index()
        for col in all_pred_cols:
            if col not in block_fcst.columns:
                raise KeyError(f"Forecast output missing expected column '{col}'. Got: {list(block_fcst.columns)}")

        block_out = block_fcst[["unique_id", "ds"] + all_pred_cols].copy()
        preds.append(block_out)

        advance = block_out.rename(columns={base_model_col: "y"})[["unique_id", "ds", "y"]].copy()
        advance["y"] = advance["y"].fillna(0.0)
        hist = pd.concat([hist, advance], ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    if not preds:
        return pd.DataFrame(columns=["unique_id", "ds"] + all_pred_cols)
    all_preds = pd.concat(preds, ignore_index=True)
    return eval_set[["unique_id", "ds"]].merge(all_preds, on=["unique_id", "ds"], how="left")


def _predict_deepar_fixed(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    quantiles: list[float],
    horizon: int,
    input_size: int,
    start_padding_enabled: bool,
    max_steps: int,
) -> pd.DataFrame:
    if NeuralForecast is None:
        raise ImportError("DeepAR requested but neuralforecast is not available in this environment.")

    loss = DistributionLoss(distribution="NegativeBinomial", level=[50, 80])
    models = [
        DeepAR(
            h=horizon,
            input_size=max(int(input_size), 1),
            loss=loss,
            scaler_type="robust",
            start_padding_enabled=start_padding_enabled,
            max_steps=max_steps,
        )
    ]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df=init_set[["unique_id", "ds", "y"]])

    base_col = "DeepAR"
    q_cols = ["DeepAR-lo-80", "DeepAR-hi-80", "DeepAR-lo-50", "DeepAR-hi-50"]
    nf_preds = _rolling_forecast_over_eval_probabilistic(
        nf=nf,
        base_model_col=base_col,
        quantile_cols=q_cols,
        init_hist=init_set,
        eval_set=eval_set,
        h=horizon,
    )

    rename_nf = {
        "DeepAR": "q_0.5",
        "DeepAR-lo-80": "q_0.1",
        "DeepAR-hi-80": "q_0.9",
        "DeepAR-lo-50": "q_0.25",
        "DeepAR-hi-50": "q_0.75",
    }
    qcols = _qcols(quantiles)
    out = nf_preds.rename(columns=rename_nf)
    for c in qcols:
        if c not in out.columns:
            out[c] = np.nan
    out = _enforce_monotonic_quantiles(out, quantiles=quantiles)
    out["model"] = "DeepAR"
    return out[["model", "unique_id", "ds"] + qcols]


def _predict_iets_prob_fixed(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    quantiles: list[float],
    *,
    rscript: str,
    script_path: Path | None,
    seed: int,
    occurrence: str,
) -> pd.DataFrame:
    from models.iets_baseline import fit_predict_iets_prob_panel

    return fit_predict_iets_prob_panel(
        init_set,
        eval_set,
        quantiles=quantiles,
        rscript=rscript,
        script_path=script_path,
        seed=seed,
        occurrence=occurrence,
    )


def _predict_deep_renewal_fixed(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    quantiles: list[float],
    horizon: int,
    context_length: int,
    num_layers: int,
    num_cells: int,
    dropout_rate: float,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    num_samples: int,
    learning_rate: float,
) -> pd.DataFrame:
    try:
        from gluonts.dataset.common import ListDataset
        from gluonts.evaluation.backtest import make_evaluation_predictions
        from gluonts.mx.model.renewal import DeepRenewalProcessEstimator
        from gluonts.mx.trainer import Trainer
    except ImportError as exc:
        raise ImportError(
            "Deep Renewal Process requested but GluonTS MXNet components are unavailable. "
            "Install gluonts and mxnet."
        ) from exc

    if horizon <= 0:
        raise ValueError("DeepRenewal horizon must be positive.")
    qcols = _qcols(quantiles)
    eval_idx = eval_set[["unique_id", "ds"]].copy()
    eval_idx["k"] = eval_idx.groupby("unique_id").cumcount()
    k_max = int(eval_idx["k"].max()) if not eval_idx.empty else -1
    n_blocks = int(np.ceil((k_max + 1) / horizon)) if k_max >= 0 else 0

    hist = init_set[["unique_id", "ds", "y"]].copy().sort_values(["unique_id", "ds"]).reset_index(drop=True)
    block_rows: list[pd.DataFrame] = []
    q_to_name = {0.1: "q_0.1", 0.25: "q_0.25", 0.5: "q_0.5", 0.75: "q_0.75", 0.9: "q_0.9"}

    def _build_list_dataset(source_df: pd.DataFrame) -> "ListDataset":
        entries = []
        for uid, grp in source_df.groupby("unique_id", sort=False):
            g = grp.sort_values("ds")
            entries.append(
                {
                    "item_id": uid,
                    "start": pd.Timestamp(g["ds"].iloc[0]),
                    "target": g["y"].to_numpy(dtype=float),
                }
            )
        return ListDataset(entries, freq="D")

    train_ds = _build_list_dataset(hist)
    estimator = DeepRenewalProcessEstimator(
        prediction_length=horizon,
        context_length=max(int(context_length), 1),
        num_layers=max(int(num_layers), 1),
        num_cells=max(int(num_cells), 1),
        dropout_rate=float(max(dropout_rate, 0.0)),
        batch_size=max(int(batch_size), 1),
        trainer=Trainer(
            epochs=max(int(epochs), 1),
            num_batches_per_epoch=max(int(batches_per_epoch), 1),
            learning_rate=float(max(learning_rate, 1e-6)),
        ),
    )
    predictor = estimator.train(train_ds)

    for b in range(n_blocks):
        block_start = b * horizon
        block_end = block_start + horizon - 1
        block_eval = eval_idx[(eval_idx["k"] >= block_start) & (eval_idx["k"] <= block_end)].copy()
        if block_eval.empty:
            continue
        infer_ds = _build_list_dataset(hist)
        fcst_iter, _ = make_evaluation_predictions(
            dataset=infer_ds,
            predictor=predictor,
            num_samples=max(int(num_samples), 100),
        )
        forecasts = list(fcst_iter)
        infer_ids = [entry["item_id"] for entry in infer_ds]
        if len(forecasts) != len(infer_ids):
            raise RuntimeError("DeepRenewal forecast count mismatch.")

        rows = []
        for uid, fcst in zip(infer_ids, forecasts):
            uid_block = block_eval[block_eval["unique_id"] == uid].sort_values("k")
            if uid_block.empty:
                continue
            n_take = len(uid_block)
            row = uid_block[["unique_id", "ds"]].copy()
            for q in quantiles:
                qf = float(np.clip(q, 1e-6, 1 - 1e-6))
                vals = np.asarray(fcst.quantile(qf), dtype=float)[:n_take]
                row[q_to_name.get(q, f"q_{q}")] = vals
            rows.append(row)
        if rows:
            block_out = pd.concat(rows, ignore_index=True)
            block_rows.append(block_out)
            advance = block_out[["unique_id", "ds", "q_0.5"]].rename(columns={"q_0.5": "y"})
            advance["y"] = advance["y"].fillna(0.0)
            hist = (
                pd.concat([hist, advance], ignore_index=True)
                .sort_values(["unique_id", "ds"])
                .reset_index(drop=True)
            )

    if not block_rows:
        return pd.DataFrame(columns=["model", "unique_id", "ds"] + qcols)

    out = pd.concat(block_rows, ignore_index=True)
    for c in qcols:
        if c not in out.columns:
            out[c] = np.nan
    out = _enforce_monotonic_quantiles(out, quantiles=quantiles)
    out["model"] = "DeepRenewal"
    return out[["model", "unique_id", "ds"] + qcols]


def plot_calibration_curve(
    eval_merged: pd.DataFrame,
    quantiles: list[float],
    out_dir: Path,
    models_to_plot: list[str] | None = None,
) -> None:
    if models_to_plot is not None:
        eval_merged = eval_merged[eval_merged["model"].isin(models_to_plot)]
        if eval_merged.empty:
            plt.figure(figsize=(8, 8))
            plt.savefig(out_dir / "calibration_curve.png", dpi=300, bbox_inches="tight")
            plt.close()
            return
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    for model, dfm in eval_merged.groupby("model"):
        emp_coverages = []
        for q in quantiles:
            col = f"q_{q}"
            if col not in dfm.columns:
                emp_coverages.append(np.nan)
                continue
            emp_cov = (dfm["y"] <= dfm[col]).mean()
            emp_coverages.append(emp_cov)
        plt.plot(quantiles, emp_coverages, marker="o", label=model)

    plt.xlabel("Nominal Coverage (Quantile)")
    plt.ylabel("Empirical Coverage")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.6, linewidth=0.8, color="gray")
    plt.savefig(out_dir / "calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pit_histogram(eval_merged: pd.DataFrame, quantiles: list[float], out_dir: Path) -> None:
    models = eval_merged["model"].unique()
    if len(models) == 0:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    bins = [0.0] + quantiles + [1.0]
    for ax, model in zip(axes, models):
        dfm = eval_merged[eval_merged["model"] == model].copy()
        pits = pit_values(dfm, quantiles=quantiles)
        ax.hist(pits, bins=bins, density=True, color="skyblue", alpha=0.85)
        ax.axhline(1.0, color="r", linestyle="--", label="Uniform (Ideal)")
        ax.set_title(f"PIT Histogram: {model}")
        ax.set_xlabel("PIT Value")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "pit_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()


def _coverage_summary(eval_merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, dfm in eval_merged.groupby("model"):
        cov50 = coverage_rate(dfm, 0.25, 0.75, 0.5)
        cov80 = coverage_rate(dfm, 0.1, 0.9, 0.8)
        row = {"model": model, "n_obs": int(len(dfm))}
        row.update(cov50)
        row.update(cov80)
        row["CoverageGap@50"] = float(abs(cov50["Coverage@50"] - 0.50))
        row["CoverageGap@80"] = float(abs(cov80["Coverage@80"] - 0.80))
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "n_obs",
                "Coverage@50",
                "AIW@50",
                "Coverage@80",
                "AIW@80",
                "CoverageGap@50",
                "CoverageGap@80",
            ]
        )
    return pd.DataFrame(rows)


def _cold_start_bucket(init_pos: int) -> str:
    v = int(init_pos)
    if v <= 1:
        return "CS_0_1"
    if v <= 3:
        return "CS_2_3"
    if v <= 5:
        return "CS_4_5"
    if v <= 10:
        return "CS_6_10"
    return "CS_11_plus"


def _build_slice_features(init_set: pd.DataFrame) -> pd.DataFrame:
    tmp = init_set.copy()
    tmp["occ"] = (tmp["y"] > 0).astype(int)
    base = (
        tmp.groupby("unique_id", as_index=False)
        .agg(init_len=("ds", "nunique"), init_pos=("occ", "sum"))
    )
    base["init_pos"] = base["init_pos"].astype(int)
    base["cold_start_bin"] = base["init_pos"].map(_cold_start_bucket)

    feats = compute_adi_cv2(init_set)
    if not feats.empty:
        feats["category"] = feats.apply(classify_adi_cv2, axis=1)
        base = base.merge(feats[["unique_id", "adi", "cv_sq", "category"]], on="unique_id", how="left")
    else:
        base["adi"] = np.nan
        base["cv_sq"] = np.nan
        base["category"] = np.nan

    base["category"] = base["category"].fillna("NoPositiveInit")
    base["sparse_regime"] = np.where(
        base["category"].isin(["Intermittent", "Lumpy"]),
        "Sparse(Intermittent|Lumpy)",
        "NonSparse(Smooth|Erratic)",
    )
    return base


def _ordered_slice_values(slice_type: str, values: pd.Series) -> list[str]:
    observed = [str(v) for v in values.dropna().unique().tolist()]
    if slice_type == "regime":
        order = ["Smooth", "Erratic", "Intermittent", "Lumpy", "NoPositiveInit"]
        return [x for x in order if x in observed] + sorted([x for x in observed if x not in order])
    if slice_type == "cold_start":
        order = ["CS_0_1", "CS_2_3", "CS_4_5", "CS_6_10", "CS_11_plus"]
        return [x for x in order if x in observed] + sorted([x for x in observed if x not in order])
    if slice_type == "sparse":
        order = ["NonSparse(Smooth|Erratic)", "Sparse(Intermittent|Lumpy)"]
        return [x for x in order if x in observed] + sorted([x for x in observed if x not in order])
    return sorted(observed)


def _prob_metric_row(dfm: pd.DataFrame, quantiles: list[float]) -> dict[str, float]:
    cov50 = coverage_rate(dfm, 0.25, 0.75, 0.5)
    cov80 = coverage_rate(dfm, 0.1, 0.9, 0.8)
    pin = evaluate_prob_models(dfm, quantiles=quantiles)
    pin_mean = float(pin["pinball"].mean()) if not pin.empty else float("nan")
    row = {
        "Coverage@50": float(cov50["Coverage@50"]),
        "AIW@50": float(cov50["AIW@50"]),
        "Coverage@80": float(cov80["Coverage@80"]),
        "AIW@80": float(cov80["AIW@80"]),
        "CoverageGap@50": float(abs(float(cov50["Coverage@50"]) - 0.50)),
        "CoverageGap@80": float(abs(float(cov80["Coverage@80"]) - 0.80)),
        "pinball_mean": pin_mean,
    }
    row["GoalScore@80"] = row["AIW@80"] * (1.0 + row["CoverageGap@80"])

    dfp = dfm[dfm["y"] > 0].copy()
    row["n_obs_pos"] = int(len(dfp))
    if dfp.empty:
        row.update(
            {
                "Coverage@50_pos": np.nan,
                "AIW@50_pos": np.nan,
                "Coverage@80_pos": np.nan,
                "AIW@80_pos": np.nan,
                "CoverageGap@50_pos": np.nan,
                "CoverageGap@80_pos": np.nan,
                "pinball_mean_pos": np.nan,
                "GoalScore@80_pos": np.nan,
            }
        )
        return row

    cov50_pos = coverage_rate(dfp, 0.25, 0.75, 0.5)
    cov80_pos = coverage_rate(dfp, 0.1, 0.9, 0.8)
    pin_pos = evaluate_prob_models(dfp, quantiles=quantiles)
    pin_pos_mean = float(pin_pos["pinball"].mean()) if not pin_pos.empty else float("nan")

    row.update(
        {
            "Coverage@50_pos": float(cov50_pos["Coverage@50"]),
            "AIW@50_pos": float(cov50_pos["AIW@50"]),
            "Coverage@80_pos": float(cov80_pos["Coverage@80"]),
            "AIW@80_pos": float(cov80_pos["AIW@80"]),
            "CoverageGap@50_pos": float(abs(float(cov50_pos["Coverage@50"]) - 0.50)),
            "CoverageGap@80_pos": float(abs(float(cov80_pos["Coverage@80"]) - 0.80)),
            "pinball_mean_pos": pin_pos_mean,
        }
    )
    row["GoalScore@80_pos"] = row["AIW@80_pos"] * (1.0 + row["CoverageGap@80_pos"])
    return row


def _write_prob_slice_metrics(
    init_set: pd.DataFrame,
    eval_merged: pd.DataFrame,
    quantiles: list[float],
    out_dir: Path,
    protocol: str,
) -> None:
    slice_feats = _build_slice_features(init_set)
    eval_with_slices = eval_merged.merge(
        slice_feats[["unique_id", "category", "sparse_regime", "cold_start_bin", "init_len", "init_pos"]],
        on="unique_id",
        how="left",
    )
    spec = [
        ("regime", "category"),
        ("sparse", "sparse_regime"),
        ("cold_start", "cold_start_bin"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for slice_type, col in spec:
        for slice_value in _ordered_slice_values(slice_type, eval_with_slices[col]):
            block = eval_with_slices[eval_with_slices[col] == slice_value]
            if block.empty:
                continue
            for model, dfm in block.groupby("model"):
                if dfm.empty:
                    continue
                metric_row = _prob_metric_row(dfm, quantiles=quantiles)
                rows.append(
                    {
                        "protocol": protocol,
                        "slice_type": slice_type,
                        "slice": slice_value,
                        "model": model,
                        "n_obs": int(len(dfm)),
                        "n_series": int(dfm["unique_id"].nunique()),
                        **metric_row,
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "prob_slice_metrics.csv", index=False)


def _pit_long(eval_merged: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    for model, dfm in eval_merged.groupby("model"):
        pits = pit_values(dfm, quantiles=quantiles)
        if pits.size == 0:
            continue
        rows.append(pd.DataFrame({"model": model, "pit": pits}))
    if not rows:
        return pd.DataFrame(columns=["model", "pit"])
    return pd.concat(rows, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=default_data_file())
    ap.add_argument(
        "--dataset",
        choices=["online", "m5"],
        default="online",
        help="Dataset for probabilistic experiment: 'online' (Online Retail) or 'm5' (M5 competition).",
    )
    ap.add_argument(
        "--m5-sales",
        type=Path,
        default=default_m5_sales_file(),
        help="Path to M5 sales file (wide or long). Used when --dataset m5.",
    )
    ap.add_argument(
        "--m5-calendar",
        type=Path,
        default=default_m5_calendar_file(),
        help="Path to M5 calendar.csv. Used when --dataset m5.",
    )
    ap.add_argument(
        "--m5-sample-size",
        type=int,
        default=5000,
        help="Number of M5 series to subsample for experiments (None = use all). Used when --dataset m5.",
    )
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-series", type=int, default=None, help="Optional cap on number of series for faster end-to-end runs.")
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--init-ratio", type=float, default=1.0 / 3.0)
    ap.add_argument("--protocol", choices=["fixed", "walk_forward"], default="fixed")
    ap.add_argument("--walk-step", type=int, default=1, help="Block size (steps) for walk-forward protocol.")
    ap.add_argument("--baseline-mode", choices=["paper", "extended", "full", "hurdle_only", "hb_only", "fast_classic"], default=None, help="Baseline set for both fixed and walk-forward: paper=TSB-HB + AutoARIMA/AutoTheta + conformal wrappers for point baselines, extended=paper + hurdle baselines, full=legacy alias for extended, hurdle_only=TSB-HB + hurdle baselines, hb_only=TSB-HB only, fast_classic=TSB-HB + conformal wrappers only (no AutoARIMA/AutoTheta/hurdle).")
    ap.add_argument("--hb-regime-aware", dest="hb_regime_aware", action="store_true", default=True, help="Use ADI/CV^2 regime-aware HB priors.")
    ap.add_argument("--no-hb-regime-aware", dest="hb_regime_aware", action="store_false", help="Disable regime-aware priors and use global HB priors.")
    ap.add_argument("--hb-group-shrink-strength", type=float, default=0.0, help="Extra shrink from group-level hyperparameters back to global hyperparameters (0 disables).")
    ap.add_argument("--hb-online-update", dest="hb_online_update", action="store_true", default=True, help="Use online sufficient-statistics updates for TSB-HB in walk-forward.")
    ap.add_argument("--no-hb-online-update", dest="hb_online_update", action="store_false", help="Disable online update and re-fit TSB-HB each walk-forward step.")
    ap.add_argument("--hb-dynamic-occurrence", dest="hb_dynamic_occurrence", action="store_true", default=False, help="Enable discounted dynamic occurrence updates for TSB-HB in walk-forward.")
    ap.add_argument("--no-hb-dynamic-occurrence", dest="hb_dynamic_occurrence", action="store_false", help="Disable dynamic occurrence updates.")
    ap.add_argument("--hb-occ-discount", type=float, default=1.0, help="Discount factor for dynamic occurrence update (0<d<=1).")
    ap.add_argument("--hb-item-variance-mode", choices=["group", "conjugate"], default="conjugate", help="Process variance mode for size: group or conjugate.")
    ap.add_argument("--hb-variance-prior-df", type=float, default=20.0, help="Prior degrees of freedom for conjugate variance model (larger = stronger shrinkage).")
    ap.add_argument("--hb-bootstrap-draws", type=int, default=20, help="Bootstrap draws for TSB-HB hyperparameter uncertainty (fixed protocol).")
    ap.add_argument("--hb-disable-hyper-uncertainty", action="store_true", help="Disable bootstrap hyperparameter uncertainty even when draws > 0.")
    ap.add_argument("--hb-calibration-mode", choices=["none", "location_scale"], default="location_scale", help="Optional post-hoc calibration for TSB-HB quantiles. The paper release uses location_scale.")
    ap.add_argument("--hb-calibration-ratio", type=float, default=0.20, help="Tail fraction of init_set reserved to fit calibration shifts.")
    ap.add_argument("--hb-calibration-samples", type=int, default=1000, help="Monte Carlo samples when fitting calibration shifts.")
    ap.add_argument("--hb-calibration-lambda-min", type=float, default=0.60, help="Lower bound for location-scale calibration lambda.")
    ap.add_argument("--hb-calibration-lambda-max", type=float, default=1.20, help="Upper bound for location-scale calibration lambda.")
    ap.add_argument("--hb-calibration-lambda-steps", type=int, default=13, help="Number of lambda grid points for location-scale calibration.")
    ap.add_argument("--hb-calibration-coverage-weight", type=float, default=0.50, help="Penalty weight on coverage gaps when fitting location-scale calibration.")
    ap.add_argument("--include-conformal-baselines", dest="include_conformal", action="store_true", default=True, help="Include conformal prediction intervals for point baselines (CP-Croston, CP-SBA, etc.).")
    ap.add_argument("--no-conformal-baselines", dest="include_conformal", action="store_false", help="Exclude conformal baselines.")
    ap.add_argument("--conformal-cal-ratio", type=float, default=0.2, help="Calibration ratio for conformal prediction split (fraction of training data held out).")
    ap.add_argument("--with-deepar", action="store_true", help="Include DeepAR baseline (fixed protocol only).")
    ap.add_argument("--with-deep-renewal", action="store_true", help="Include GluonTS Deep Renewal Process baseline (fixed protocol only).")
    ap.add_argument("--with-tweedie", action="store_true", help="Include Tweedie autoregressive probabilistic baseline.")
    ap.add_argument("--tweedie-lags", type=int, default=14, help="Number of lag features for Tweedie baseline.")
    ap.add_argument("--tweedie-power", type=float, default=1.5, help="Tweedie variance power (1<p<2 typical for intermittent demand).")
    ap.add_argument("--tweedie-alpha", type=float, default=0.1, help="L2 regularization strength for Tweedie baseline.")
    ap.add_argument("--tweedie-max-iter", type=int, default=1000, help="Max optimizer iterations for Tweedie baseline.")
    ap.add_argument(
        "--with-iets",
        action="store_true",
        help="Include iETS probabilistic forecasts via R smooth (fixed protocol only; uses prediction intervals).",
    )
    ap.add_argument("--iets-rscript", type=str, default="Rscript", help="Rscript executable for iETS.")
    ap.add_argument(
        "--iets-script",
        type=Path,
        default=None,
        help="Path to iets_panel_forecast_prob.R (default: <repo>/scripts/iets_panel_forecast_prob.R).",
    )
    ap.add_argument(
        "--iets-occurrence",
        type=str,
        default="auto",
        help="smooth::adam occurrence argument for iETS (e.g. auto, none, fixed).",
    )
    ap.add_argument("--horizon", type=int, default=10, help="Block size for DeepAR rolling forecast.")
    ap.add_argument("--input-size", type=int, default=14)
    ap.add_argument("--start-padding-enabled", action="store_true", default=True)
    ap.add_argument("--max-steps", type=int, default=500, help="Max training steps for DeepAR.")
    ap.add_argument("--drp-context-length", type=int, default=30, help="Context length for DeepRenewal.")
    ap.add_argument("--drp-num-layers", type=int, default=2, help="RNN layers for DeepRenewal.")
    ap.add_argument("--drp-num-cells", type=int, default=40, help="RNN hidden units for DeepRenewal.")
    ap.add_argument("--drp-dropout", type=float, default=0.1, help="Dropout for DeepRenewal.")
    ap.add_argument("--drp-epochs", type=int, default=20, help="Training epochs for DeepRenewal.")
    ap.add_argument("--drp-batches-per-epoch", type=int, default=50, help="Number of batches per epoch for DeepRenewal.")
    ap.add_argument("--drp-batch-size", type=int, default=32, help="Batch size for DeepRenewal.")
    ap.add_argument("--drp-num-samples", type=int, default=400, help="Prediction samples for DeepRenewal.")
    ap.add_argument("--drp-learning-rate", type=float, default=1e-3, help="Learning rate for DeepRenewal trainer.")
    return ap


def run(args: argparse.Namespace) -> None:
    if args.protocol == "walk_forward" and args.with_deepar:
        raise ValueError("--with-deepar is currently supported only for --protocol fixed.")
    if args.protocol == "walk_forward" and args.with_deep_renewal:
        raise ValueError("--with-deep-renewal is currently supported only for --protocol fixed.")
    if args.protocol == "walk_forward" and args.with_iets:
        raise ValueError("--with-iets is currently supported only for --protocol fixed.")

    set_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "online":
        df_raw = load_online_retail(args.data)
        df = preprocess_online_retail(df_raw)
    else:
        sales_df, calendar_df = load_m5_long(args.m5_sales, args.m5_calendar)
        df = preprocess_m5(sales_df, calendar_df, sample_size=args.m5_sample_size)
    if args.max_series is not None:
        max_series = max(int(args.max_series), 1)
        uids = df["unique_id"].drop_duplicates()
        if len(uids) > max_series:
            keep = np.random.choice(uids.to_numpy(), size=max_series, replace=False)
            df = df[df["unique_id"].isin(keep)].copy()
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=args.init_ratio, min_len=args.min_len)
    if eval_set.empty:
        raise ValueError("Evaluation set is empty; verify split parameters and input data.")

    qcols = _qcols(QUANTILES)
    hb_group_labels = _build_regime_group_labels(init_set) if args.hb_regime_aware else None
    hb_use_hyper_uncertainty = (not args.hb_disable_hyper_uncertainty) and (args.hb_bootstrap_draws > 0)
    baseline_mode = _normalize_prob_baseline_mode(args.baseline_mode)
    hb_variance_prior_df = float(max(args.hb_variance_prior_df, 2.1))
    hb_calibration: Optional[dict[str, float | str]] = None
    if args.hb_calibration_mode == "location_scale":
        hb_calibration = _fit_hb_location_scale(
            init_set=init_set,
            quantiles=QUANTILES,
            hb_regime_aware=bool(args.hb_regime_aware),
            hb_group_shrink_strength=float(max(args.hb_group_shrink_strength, 0.0)),
            hb_item_variance_mode=str(args.hb_item_variance_mode),
            hb_variance_prior_df=hb_variance_prior_df,
            hb_bootstrap_draws=max(int(args.hb_bootstrap_draws), 0),
            hb_bootstrap_seed=args.seed,
            hb_use_hyper_uncertainty=hb_use_hyper_uncertainty,
            calibration_ratio=float(args.hb_calibration_ratio),
            calibration_samples=max(int(args.hb_calibration_samples), 200),
            lambda_min=float(args.hb_calibration_lambda_min),
            lambda_max=float(args.hb_calibration_lambda_max),
            lambda_steps=max(int(args.hb_calibration_lambda_steps), 2),
            coverage_weight=float(max(args.hb_calibration_coverage_weight, 0.0)),
        )
    if hb_calibration is not None:
        pd.DataFrame(
            [
                {
                    "model": PROB_TSBHB_MODEL,
                    "calibration_mode": str(hb_calibration.get("mode", args.hb_calibration_mode)),
                    **hb_calibration,
                }
            ]
        ).to_csv(out_dir / "hb_calibration_params.csv", index=False)

    if args.protocol == "fixed":
        all_q = _predict_prob_models_once(
            init_set,
            eval_set,
            quantiles=QUANTILES,
            hb_group_labels=hb_group_labels,
            hb_bootstrap_draws=max(int(args.hb_bootstrap_draws), 0),
            hb_bootstrap_seed=args.seed,
            hb_use_hyper_uncertainty=hb_use_hyper_uncertainty,
            hb_group_shrink_strength=float(max(args.hb_group_shrink_strength, 0.0)),
            hb_item_variance_mode=str(args.hb_item_variance_mode),
            hb_variance_prior_df=hb_variance_prior_df,
            hb_calibration=hb_calibration,
            baseline_mode=baseline_mode,
            include_conformal=bool(args.include_conformal),
            conformal_cal_ratio=float(args.conformal_cal_ratio),
            with_tweedie=bool(args.with_tweedie),
            tweedie_lags=int(max(args.tweedie_lags, 1)),
            tweedie_power=float(args.tweedie_power),
            tweedie_alpha=float(max(args.tweedie_alpha, 0.0)),
            tweedie_max_iter=int(max(args.tweedie_max_iter, 100)),
        )
        if args.with_deepar:
            deepar_q = _predict_deepar_fixed(
                init_set=init_set,
                eval_set=eval_set,
                quantiles=QUANTILES,
                horizon=args.horizon,
                input_size=args.input_size,
                start_padding_enabled=args.start_padding_enabled,
                max_steps=args.max_steps,
            )
            all_q = pd.concat([all_q, deepar_q], ignore_index=True)
        if args.with_deep_renewal:
            drp_q = _predict_deep_renewal_fixed(
                init_set=init_set,
                eval_set=eval_set,
                quantiles=QUANTILES,
                horizon=int(args.horizon),
                context_length=int(args.drp_context_length),
                num_layers=int(args.drp_num_layers),
                num_cells=int(args.drp_num_cells),
                dropout_rate=float(args.drp_dropout),
                epochs=int(args.drp_epochs),
                batches_per_epoch=int(args.drp_batches_per_epoch),
                batch_size=int(args.drp_batch_size),
                num_samples=int(args.drp_num_samples),
                learning_rate=float(args.drp_learning_rate),
            )
            all_q = pd.concat([all_q, drp_q], ignore_index=True)
        if args.with_iets:
            iets_q = _predict_iets_prob_fixed(
                init_set=init_set,
                eval_set=eval_set,
                quantiles=QUANTILES,
                rscript=str(args.iets_rscript),
                script_path=args.iets_script,
                seed=int(args.seed),
                occurrence=str(args.iets_occurrence),
            )
            all_q = pd.concat([all_q, iets_q], ignore_index=True)
    else:
        step_outputs = []
        hb_state = None
        if args.hb_online_update:
            hb_state = initialize_online_tsb_hb(
                init_set,
                group_labels=hb_group_labels,
                bootstrap_draws=0,
                bootstrap_seed=args.seed,
                group_shrink_strength=float(max(args.hb_group_shrink_strength, 0.0)),
                dynamic_occurrence=bool(args.hb_dynamic_occurrence),
                occurrence_discount=float(args.hb_occ_discount),
                item_variance_mode=str(args.hb_item_variance_mode),
                item_variance_shrink_strength=hb_variance_prior_df,
            )
        for frame in iter_walk_forward_frames(init_set, eval_set, step_size=args.walk_step):
            if hb_state is not None:
                tsbhb_q = predict_online_tsb_hb(
                    hb_state,
                    frame.target,
                    quantiles=QUANTILES,
                    n_samples=2000,
                    include_hyper_uncertainty=False,
                )
                for c in qcols:
                    if c not in tsbhb_q.columns:
                        tsbhb_q[c] = np.nan
                tsbhb_q = _apply_hb_calibration(
                    tsbhb_q,
                    quantiles=QUANTILES,
                    calibration=hb_calibration,
                )
                tsbhb_q["model"] = PROB_TSBHB_MODEL
                tsbhb_q = tsbhb_q[["model", "unique_id", "ds"] + qcols]

                non_hb_q = _predict_non_hb_prob_models_once(
                    frame.history,
                    frame.target,
                    quantiles=QUANTILES,
                    baseline_mode=baseline_mode,
                    with_tweedie=bool(args.with_tweedie),
                    tweedie_lags=int(max(args.tweedie_lags, 1)),
                    tweedie_power=float(args.tweedie_power),
                    tweedie_alpha=float(max(args.tweedie_alpha, 0.0)),
                    tweedie_max_iter=int(max(args.tweedie_max_iter, 100)),
                )
                step_frames = [tsbhb_q, non_hb_q]
                if args.include_conformal and _include_conformal_prob_baselines(baseline_mode):
                    cp_q = fit_predict_conformal_baselines(
                        frame.history, frame.target,
                        quantiles=QUANTILES,
                        cal_ratio=float(args.conformal_cal_ratio),
                        freq="D",
                    )
                    if not cp_q.empty:
                        step_frames.append(cp_q)
                step_q = pd.concat(step_frames, ignore_index=True)
                hb_state = update_online_tsb_hb(hb_state, frame.target)
            else:
                step_q = _predict_prob_models_once(
                    frame.history,
                    frame.target,
                    quantiles=QUANTILES,
                    hb_group_labels=hb_group_labels,
                    hb_bootstrap_draws=0,
                    hb_bootstrap_seed=args.seed,
                    hb_use_hyper_uncertainty=False,
                    hb_group_shrink_strength=float(max(args.hb_group_shrink_strength, 0.0)),
                    hb_item_variance_mode=str(args.hb_item_variance_mode),
                    hb_variance_prior_df=hb_variance_prior_df,
                    hb_calibration=hb_calibration,
                    baseline_mode=baseline_mode,
                    include_conformal=bool(args.include_conformal),
                    conformal_cal_ratio=float(args.conformal_cal_ratio),
                    with_tweedie=bool(args.with_tweedie),
                    tweedie_lags=int(max(args.tweedie_lags, 1)),
                    tweedie_power=float(args.tweedie_power),
                    tweedie_alpha=float(max(args.tweedie_alpha, 0.0)),
                    tweedie_max_iter=int(max(args.tweedie_max_iter, 100)),
                )
            step_outputs.append(step_q)
        if not step_outputs:
            all_q = pd.DataFrame(columns=["model", "unique_id", "ds"] + qcols)
        else:
            all_q = pd.concat(step_outputs, ignore_index=True)

    all_q = all_q.sort_values(["model", "unique_id", "ds"]).reset_index(drop=True)
    all_q_out = all_q.copy()
    all_q_out.insert(0, "protocol", args.protocol)
    all_q_out.to_csv(out_dir / "prob_quantiles.csv", index=False)

    eval_merged = eval_set[["unique_id", "ds", "y"]].merge(all_q, on=["unique_id", "ds"], how="inner")
    if eval_merged.empty:
        raise ValueError("Merged probabilistic evaluation frame is empty; no predictions matched evaluation timestamps.")

    plot_calibration_curve(eval_merged, QUANTILES, out_dir)
    plot_pit_histogram(eval_merged, QUANTILES, out_dir)

    pinball_df = evaluate_prob_models(eval_merged, QUANTILES)
    pinball_df.insert(0, "protocol", args.protocol)
    pinball_df.to_csv(out_dir / "prob_pinball.csv", index=False)
    pinball_df.to_csv(out_dir / "probabilistic_forecast_pinball_results.csv", index=False)
    scaled_pinball_df = _scaled_pinball_table(eval_merged, init_set=init_set, quantiles=QUANTILES)
    if not scaled_pinball_df.empty:
        scaled_pinball_df.insert(0, "protocol", args.protocol)
    scaled_pinball_df.to_csv(out_dir / "prob_pinball_scaled.csv", index=False)

    coverage_df = _coverage_summary(eval_merged)
    coverage_df.insert(0, "protocol", args.protocol)
    coverage_df.to_csv(out_dir / "coverage_summary.csv", index=False)

    eval_pos = eval_merged[eval_merged["y"] > 0].copy()
    coverage_pos_df = _coverage_summary(eval_pos)
    if not coverage_pos_df.empty:
        coverage_pos_df = coverage_pos_df.rename(
            columns={
                "n_obs": "n_obs_pos",
                "Coverage@50": "Coverage@50_pos",
                "AIW@50": "AIW@50_pos",
                "Coverage@80": "Coverage@80_pos",
                "AIW@80": "AIW@80_pos",
                "CoverageGap@50": "CoverageGap@50_pos",
                "CoverageGap@80": "CoverageGap@80_pos",
            }
        )
    coverage_pos_df.insert(0, "protocol", args.protocol)
    coverage_pos_df.to_csv(out_dir / "coverage_summary_positive.csv", index=False)

    pit_df = _pit_long(eval_merged, QUANTILES)
    pit_df.insert(0, "protocol", args.protocol)
    pit_df.to_csv(out_dir / "pit_values.csv", index=False)

    pinball_mean = pinball_df.groupby(["protocol", "model"], as_index=False)["pinball"].mean().rename(columns={"pinball": "pinball_mean"})
    scaled_pinball_mean = pd.DataFrame(columns=["protocol", "model", "scaled_pinball_mean"])
    if not scaled_pinball_df.empty:
        scaled_pinball_mean = (
            scaled_pinball_df.groupby(["protocol", "model"], as_index=False)["scaled_pinball"]
            .mean()
            .rename(columns={"scaled_pinball": "scaled_pinball_mean"})
        )
    pinball_pos_mean = pd.DataFrame(columns=["protocol", "model", "pinball_mean_pos"])
    if not eval_pos.empty:
        pinball_pos_df = evaluate_prob_models(eval_pos, QUANTILES)
        if not pinball_pos_df.empty:
            pinball_pos_df.insert(0, "protocol", args.protocol)
            pinball_pos_mean = pinball_pos_df.groupby(["protocol", "model"], as_index=False)["pinball"].mean().rename(columns={"pinball": "pinball_mean_pos"})

    prob_metrics = coverage_df.merge(pinball_mean, on=["protocol", "model"], how="left")
    prob_metrics = prob_metrics.merge(scaled_pinball_mean, on=["protocol", "model"], how="left")
    prob_metrics = prob_metrics.merge(coverage_pos_df, on=["protocol", "model"], how="left")
    prob_metrics = prob_metrics.merge(pinball_pos_mean, on=["protocol", "model"], how="left")
    prob_metrics["GoalScore@80"] = prob_metrics["AIW@80"] * (1.0 + prob_metrics["CoverageGap@80"])
    if "AIW@80_pos" in prob_metrics.columns and "CoverageGap@80_pos" in prob_metrics.columns:
        prob_metrics["GoalScore@80_pos"] = prob_metrics["AIW@80_pos"] * (1.0 + prob_metrics["CoverageGap@80_pos"])
    prob_metrics.to_csv(out_dir / "prob_metrics.csv", index=False)
    _write_prob_slice_metrics(
        init_set=init_set,
        eval_merged=eval_merged,
        quantiles=QUANTILES,
        out_dir=out_dir,
        protocol=args.protocol,
    )

    print("Evaluation complete. Results and plots saved to:", out_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
