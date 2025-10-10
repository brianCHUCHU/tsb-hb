from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tsbhb.utils import set_seed, default_data_file, default_out_dir
from tsbhb.data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from tsbhb.models.tsb_hb import fit_tsb_hb, predict_tsb_hb
from tsbhb.models.baselines import fit_predict_baselines
from tsbhb.metrics import me, mae, rmse, rmsse, compute_adi_cv2, classify_adi_cv2
from tsbhb.plotting import plot_shrinkage_scatter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=default_data_file())
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & preprocess
    df_raw = load_online_retail(args.data)
    df = preprocess_online_retail(df_raw)

    # Split fixed origin
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)

    # Fit TSB-HB and predict point forecasts
    params = fit_tsb_hb(init_set)
    tsbhb_point = predict_tsb_hb(params, eval_set, quantiles=None)
    tsbhb_point = tsbhb_point.rename(columns={"yhat": "TSB-HB-LogNormal"})

    # Baselines: StatsForecast models (per-series horizon)
    eval_horizons = eval_set["unique_id"].value_counts()
    base_preds = fit_predict_baselines(init_set, horizons=eval_horizons, freq="D", probabilistic=False)

    # Merge all predictions
    merged = eval_set[["unique_id", "ds", "y"]].merge(base_preds, on=["unique_id", "ds"], how="left")
    merged = merged.merge(tsbhb_point, on=["unique_id", "ds"], how="left")

    # Metrics per model
    results = []
    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y"}]
    for model in model_cols:
        tmp = merged[["unique_id", "ds", "y", model]].dropna(subset=[model]).copy()
        tmp = tmp.rename(columns={model: "y_pred"})
        results.append({
            "model": model,
            "ME": me(tmp["y"].values, tmp["y_pred"].values),
            "MAE": mae(tmp["y"].values, tmp["y_pred"].values),
            "RMSE": rmse(tmp["y"].values, tmp["y_pred"].values),
            "RMSSE": rmsse(init_set, tmp),
        })
    point_df = pd.DataFrame(results)
    point_df.to_csv(out_dir / "point_metrics.csv", index=False)

    # Shrinkage plots
    # p MLE vs posterior
    init_set_local = init_set.copy()
    init_set_local["occ"] = (init_set_local["y"] > 0).astype(int)
    g_init = init_set_local.groupby("unique_id")
    s = g_init["occ"].sum()
    n = g_init["ds"].nunique()
    p_mle = (s / n).reindex(params.p_posterior.index).dropna()
    p_post = params.p_posterior.reindex(p_mle.index)
    plot_shrinkage_scatter(
        x=p_mle.values,
        y=p_post.values,
        xlabel="p MLE (per-item)",
        ylabel="p posterior mean (HB)",
        title="Shrinkage on Demand Probability (p)",
        out_path=out_dir / "fig_shrink_p.png",
    )

    # size MLE vs posterior (lognormal)
    init_set_local["size"] = np.where(init_set_local["occ"] == 1, init_set_local["y"].astype(float), np.nan)
    init_set_local["log_size"] = np.log(init_set_local["size"])
    size_mle = init_set_local.groupby("unique_id")["size"].mean()
    size_post = np.exp(params.shrunk_mean_log + params.sigma_sq_process / 2.0)
    idx = size_mle.index.intersection(size_post.index)
    plot_shrinkage_scatter(
        x=size_mle.reindex(idx).fillna(0).values,
        y=size_post.reindex(idx).fillna(0).values,
        xlabel="Size MLE (per-item avg size)",
        ylabel="Size posterior mean (HB)",
        title="Shrinkage on Demand Size",
        out_path=out_dir / "fig_shrink_size.png",
    )

    # Segmentation (ADI & CV^2)
    feats = compute_adi_cv2(init_set)
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)
    # Optionally, segmentation-specific metrics can be written separately if needed.


if __name__ == "__main__":
    main()

