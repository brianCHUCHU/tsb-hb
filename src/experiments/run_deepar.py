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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=default_data_file())
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--init-ratio", type=float, default=1.0 / 3.0)
    ap.add_argument("--horizons", type=str, default="10")
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

    # Precompute RMSSE denominator for optional per-series summaries
    denom = _compute_rmsse_denominator(init_set)

    # TSB-HB point predictions (constant mean per series)
    params = fit_tsb_hb(init_set)
    tsbhb_point = predict_tsb_hb(params, eval_set, quantiles=None).rename(columns={"yhat": "TSB-HB"})

    # Lazy import heavy deps
    from neuralforecast import NeuralForecast
    from neuralforecast.auto import AutoDeepAR, DeepAR
    from neuralforecast.losses.pytorch import DistributionLoss

    results: list[dict] = []

    for h in horizons:
        # Train AutoDeepAR for this horizon
        models = [
            AutoDeepAR(
                h=h,
                loss=DistributionLoss(distribution="Normal", level=[90]),
                valid_loss=DistributionLoss(distribution="Normal", level=[90]),
                num_samples=args.num_samples,
                cpus=args.cpus,
            )
        ]


        nf = NeuralForecast(models=models, freq="D")
        nf.fit(df=init_set[["unique_id", "ds", "y"]])

        # Predict on first h steps per series
        nf_pred = nf.predict().reset_index()  # contains [unique_id, ds, AutoDeepAR, ...]
        eva_h = eval_set.groupby("unique_id").head(h).copy()
        eva_h = eva_h.merge(nf_pred, on=["unique_id", "ds"], how="left")
        eva_h = eva_h.merge(tsbhb_point, on=["unique_id", "ds"], how="left")

        # Evaluate
        for model_col in ["TSB-HB", "AutoDeepAR"]:
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


