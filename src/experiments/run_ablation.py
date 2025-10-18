from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import set_seed, default_data_file, default_out_dir
from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
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

    # Common stats for ablations
    init_local = init_set.copy()
    init_local["occ"] = (init_local["y"] > 0).astype(int)
    init_local["size"] = np.where(init_local["occ"] == 1, init_local["y"].astype(float), np.nan)
    init_local["log_size"] = np.log(init_local["size"])  # NaN for zeroes
    g_init = init_local.groupby("unique_id")
    s = g_init["occ"].sum()
    n = g_init["ds"].nunique()

    # TSB-MLE-LogNormal
    p_mle = (s / n)
    item_stats_for_mle = init_local.groupby("unique_id")["log_size"].agg(mean_log="mean", var_log="var").fillna(0)
    size_mle_ln = np.exp(item_stats_for_mle["mean_log"] + (item_stats_for_mle["var_log"] / 2.0))
    forecast_mle = (p_mle * size_mle_ln).fillna(0)

    # TSB-HB-LogNormal (our main)
    params = fit_tsb_hb(init_set)
    size_post_mean = np.exp(params.shrunk_mean_log + params.sigma_sq_process / 2.0)
    forecast_hb_logn = (params.p_posterior * size_post_mean).fillna(0)

    # TSB-HB-Gamma
    item_stats_gamma = g_init.agg({"size": "sum", "occ": "sum"})
    lambda_mle = (item_stats_gamma["size"] / item_stats_gamma["occ"]).replace([np.inf, -np.inf], np.nan)
    lam = lambda_mle.dropna()
    if len(lam) >= 2:
        m, v = float(lam.mean()), float(lam.var())
        v = max(v, 1e-6)
        a_hat, b_hat = max((m * m) / v, 1e-3), max(m / v, 1e-6)
    else:
        a_hat, b_hat = 1.0, 1.0
    alpha_post = a_hat + item_stats_gamma["size"]
    beta_post = b_hat + item_stats_gamma["occ"]
    size_post_gamma = (alpha_post / beta_post).replace([np.inf, -np.inf], 0).fillna(0)
    forecast_hb_gamma = (params.p_posterior * size_post_gamma).fillna(0)

    # Merge for evaluation
    eva = eval_set[["unique_id", "ds", "y"]].copy()
    eva = eva.merge(forecast_hb_logn.rename("TSB-HB-LogNormal"), on="unique_id", how="left")
    eva = eva.merge(forecast_mle.rename("TSB-MLE-LogNormal"), on="unique_id", how="left")
    eva = eva.merge(forecast_hb_gamma.rename("TSB-HB-Gamma"), on="unique_id", how="left")
    for c in ["TSB-HB-LogNormal", "TSB-MLE-LogNormal", "TSB-HB-Gamma"]:
        eva[c] = eva[c].fillna(0)

    results = []
    for c in ["TSB-HB-LogNormal", "TSB-MLE-LogNormal", "TSB-HB-Gamma"]:
        tmp = eva[["unique_id", "y", c]].rename(columns={c: "y_pred"}).copy()
        results.append({
            "model": c,
            "ME": me(tmp["y"].values, tmp["y_pred"].values),
            "MAE": mae(tmp["y"].values, tmp["y_pred"].values),
            "RMSE": rmse(tmp["y"].values, tmp["y_pred"].values),
            "RMSSE": rmsse(init_set, tmp),
        })

    pd.DataFrame(results).to_csv(out_dir / "ablation_metrics.csv", index=False)


if __name__ == "__main__":
    main()

