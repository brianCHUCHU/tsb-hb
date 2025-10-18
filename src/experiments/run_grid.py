from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import set_seed, default_data_file, default_out_dir
from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from models.baselines import fit_predict_baselines
from models.tsb_hb import fit_tsb_hb
from metrics import me, mae, rmse, rmsse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=default_data_file())
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_online_retail(args.data)
    df = preprocess_online_retail(df_raw)
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)

    # Reference performance for TSB-HB-LogNormal
    params = fit_tsb_hb(init_set)
    size_post_mean = np.exp(params.shrunk_mean_log + params.sigma_sq_process / 2.0)
    forecast_hb = (params.p_posterior * size_post_mean).fillna(0)
    eva_ref = eval_set[["unique_id", "ds", "y"]].merge(forecast_hb.rename("TSB-HB-LogNormal"), on="unique_id", how="left")
    eva_ref["TSB-HB-LogNormal"].fillna(0, inplace=True)
    ref_mae = mae(eva_ref["y"].values, eva_ref["TSB-HB-LogNormal"].values)
    ref_rmse = rmse(eva_ref["y"].values, eva_ref["TSB-HB-LogNormal"].values)

    # Grid for TSB
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    betas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    grid = [(a, b) for a in alphas for b in betas]

    eval_h = eval_set["unique_id"].value_counts()
    preds = fit_predict_baselines(init_set, horizons=eval_h, freq="D", tsb_grid=grid, probabilistic=False)

    # Evaluate each (alpha, beta)
    rows = []
    for (a, b) in grid:
        sub = preds[(preds.get("alpha_d") == a) & (preds.get("alpha_p") == b)]
        if sub.empty:
            continue
        eva = eval_set[["unique_id", "ds", "y"]].merge(sub, on=["unique_id", "ds"], how="left")
        eva = eva.rename(columns={"TSB": "y_pred"})
        eva["y_pred"].fillna(0, inplace=True)
        rows.append({
            "model": "TSB",
            "alpha": a,
            "beta": b,
            "ME": me(eva["y"].values, eva["y_pred"].values),
            "MAE": mae(eva["y"].values, eva["y_pred"].values),
            "RMSE": rmse(eva["y"].values, eva["y_pred"].values),
            "RMSSE": rmsse(init_set, eva[["unique_id", "y", "y_pred"]]),
            "HB_MAE": ref_mae,
            "HB_RMSE": ref_rmse,
        })

    pd.DataFrame(rows).to_csv(out_dir / "grid_summary.csv", index=False)


if __name__ == "__main__":
    main()

