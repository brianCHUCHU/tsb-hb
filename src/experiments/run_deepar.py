from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from utils import set_seed, default_data_file, default_out_dir
from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from metrics import me, mae, rmse, rmsse, wrmsse
from models.tsb_hb import fit_tsb_hb, predict_tsb_hb


def _compute_rmsse_denominator(init_set: pd.DataFrame) -> pd.Series:
    """Per-series naive (lag-1) MSE for RMSSE denominator."""
    init_sorted = init_set.sort_values(["unique_id", "ds"]).copy()
    init_sorted["y_lag1"] = init_sorted.groupby("unique_id")["y"].shift(1)
    init_sorted["naive_sq_err"] = (init_sorted["y"] - init_sorted["y_lag1"]) ** 2
    epsilon = 1e-9
    denom = init_sorted.groupby("unique_id")["naive_sq_err"].mean().where(lambda s: s > 0, epsilon)
    return denom


def _parse_horizons(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x.strip()]


def _rolling_forecast_over_eval(
    nf,
    model_col: str,
    init_hist: pd.DataFrame,
    eval_set: pd.DataFrame,
    h: int,
) -> pd.DataFrame:
    """Forecast the full eval segment by chaining blocks of size h.

    Pred-only chaining: we never refit and we never peek at realized evaluation targets.
    To move beyond the first h steps, we append the model's own predictions as context.
    """
    if h <= 0:
        raise ValueError("h must be positive.")

    hist = init_hist[["unique_id", "ds", "y"]].copy()

    eval_idx = eval_set[["unique_id", "ds", "y"]].copy()
    eval_idx["k"] = eval_idx.groupby("unique_id").cumcount()
    k_max = int(eval_idx["k"].max()) if not eval_idx.empty else -1

    preds: list[pd.DataFrame] = []
    n_blocks = int(np.ceil((k_max + 1) / h)) if k_max >= 0 else 0

    for _ in range(n_blocks):
        # Predict the next h steps for ALL series (complete id×time grid expected by neuralforecast).
        block_fcst = nf.predict(df=hist).reset_index()
        if model_col not in block_fcst.columns:
            raise KeyError(f"Forecast output missing expected column '{model_col}'. Got: {list(block_fcst.columns)}")

        block_out = block_fcst[["unique_id", "ds", model_col]].copy().rename(columns={model_col: "y_pred"})
        preds.append(block_out)

        # Advance the context to support the next block.
        advance = block_out.rename(columns={"y_pred": "y"})[["unique_id", "ds", "y"]].copy()
        advance["y"] = advance["y"].fillna(0.0)

        hist = pd.concat([hist, advance], ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    if not preds:
        return pd.DataFrame(columns=["unique_id", "ds", "y_pred"])
    all_preds = pd.concat(preds, ignore_index=True)
    # Keep only the timestamps that exist in the evaluation set.
    return eval_set[["unique_id", "ds"]].merge(all_preds, on=["unique_id", "ds"], how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=default_data_file())
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--init-ratio", type=float, default=1.0 / 3.0)
    ap.add_argument(
        "--horizons",
        type=str,
        default="10",
        help=(
            "Comma-separated forecast horizons (block sizes) for DeepAR. "
            "For each h, we train a DeepAR(h) model on the first init-ratio portion "
            "and then score over the FULL out-of-sample segment by chaining h-step forecasts "
            "(pred-only, no peeking). Example: --horizons \"10,20,30\"."
        ),
    )
    ap.add_argument(
        "--use-auto",
        action="store_true",
        help="Use AutoDeepAR (Ray Tune). Default uses fixed DeepAR to avoid tuning failures on short series.",
    )
    ap.add_argument(
        "--input-size",
        type=int,
        default=14,
        help="DeepAR input window length. Smaller values help when the training history is short.",
    )
    ap.add_argument(
        "--start-padding-enabled",
        action="store_true",
        default=True,
        help="Enable start padding so short series can still train (neuralforecast).",
    )
    ap.add_argument("--max-series", type=int, default=100)
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--cpus", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    df_raw = load_online_retail(args.data)
    df = preprocess_online_retail(df_raw)

    # Ensure minimum history and sufficient eval length for max horizon
    horizons = _parse_horizons(args.horizons)
    max_h = max(horizons) if horizons else 10
    df["t"] = df.groupby("unique_id").cumcount()
    df["L"] = df.groupby("unique_id")["t"].transform("max") + 1
    df = df[(df["L"] >= args.min_len) & (df["L"] * (1 - args.init_ratio) >= max_h)].copy()

    # Optional downsampling of series for quicker runs
    if args.max_series is not None:
        uids = df["unique_id"].unique()
        if len(uids) > args.max_series:
            keep = np.random.choice(uids, args.max_series, replace=False)
            df = df[df["unique_id"].isin(keep)].copy()

    # Fixed-origin split
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=args.init_ratio, min_len=args.min_len)

    # TSB-HB point predictions (constant mean per series)
    params = fit_tsb_hb(init_set)
    tsbhb_point = predict_tsb_hb(params, eval_set, quantiles=None).rename(columns={"yhat": "TSB-HB"})

    # Lazy import heavy deps
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import DistributionLoss
    try:
        # Preferred (newer) location
        from neuralforecast.models import DeepAR  # type: ignore
    except Exception:
        # Fallback for some installations
        from neuralforecast.auto import DeepAR  # type: ignore
    try:
        from neuralforecast.auto import AutoDeepAR  # type: ignore
    except Exception:
        AutoDeepAR = None  # type: ignore

    results: list[dict] = []

    horizons = _parse_horizons(args.horizons)
    if not horizons:
        horizons = [10]

    for h in horizons:
        loss = DistributionLoss(distribution="Normal", level=[90])
        if args.use_auto:
            if AutoDeepAR is None:
                raise ImportError("AutoDeepAR is not available in this neuralforecast installation.")
            models = [
                AutoDeepAR(
                    h=h,
                    loss=loss,
                    valid_loss=loss,
                    num_samples=args.num_samples,
                    cpus=args.cpus,
                    start_padding_enabled=args.start_padding_enabled,
                )
            ]
            deepar_col = "AutoDeepAR"
        else:
            models = [
                DeepAR(
                    h=h,
                    input_size=max(int(args.input_size), 1),
                    loss=loss,
                    valid_loss=loss,
                    start_padding_enabled=args.start_padding_enabled,
                )
            ]
            deepar_col = "DeepAR"

        nf = NeuralForecast(models=models, freq="D")
        nf.fit(df=init_set[["unique_id", "ds", "y"]])

        # Full eval segment via chained blocks of size h (pred-only, no peeking)
        deepar_pred = _rolling_forecast_over_eval(
            nf=nf,
            model_col=deepar_col,
            init_hist=init_set,
            eval_set=eval_set,
            h=h,
        )
        eva_h = eval_set[["unique_id", "ds", "y"]].merge(deepar_pred, on=["unique_id", "ds"], how="left")

        # Add TSB-HB baseline to the same evaluation frame
        eva_h = eva_h.merge(tsbhb_point, on=["unique_id", "ds"], how="left")

        # Evaluate
        for model_col in ["TSB-HB", deepar_col]:
            if model_col not in eva_h.columns:
                continue
            tmp = eva_h[["unique_id", "ds", "y", model_col]].dropna(subset=[model_col]).copy()
            tmp = tmp.rename(columns={model_col: "y_pred"})

            results.append({
                "Horizon": h,
                "Model": model_col,
                "ME": me(tmp["y"].to_numpy(), tmp["y_pred"].to_numpy()),
                "MAE": mae(tmp["y"].to_numpy(), tmp["y_pred"].to_numpy()),
                "RMSE": rmse(tmp["y"].to_numpy(), tmp["y_pred"].to_numpy()),
                # Global RMSSE across series using shared denominator definition
                "RMSSE": rmsse(init_set, tmp),
                "WRMSSE": wrmsse(init_set, tmp),
            })

    if results:
        out = pd.DataFrame(results)
        out.to_csv(out_dir / "multi_horizon_comparison_results.csv", index=False)


if __name__ == "__main__":
    main()


