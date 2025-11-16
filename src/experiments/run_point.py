from __future__ import annotations

import argparse
from pathlib import Path

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
from models.tsb_hb import fit_tsb_hb, predict_tsb_hb
from models.baselines import fit_predict_baselines
from metrics import me, mae, rmse, rmsse, wrmsse, compute_adi_cv2, classify_adi_cv2
from plotting import plot_shrinkage_scatter


def _run_m5_point(args: argparse.Namespace, out_dir: Path) -> None:
    from statsforecast import StatsForecast
    from statsforecast.models import CrostonClassic, CrostonSBA, TSB, ADIDA, IMAPA

    sales_df, calendar_df = load_m5_long(args.m5_sales, args.m5_calendar)
    df = preprocess_m5(sales_df, calendar_df, sample_size=args.m5_sample_size)

    if df.empty:
        raise ValueError("Preprocessed M5 dataframe is empty; check input files and sample size.")

    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=2.0 / 3.0, min_len=1)
    if eval_set.empty:
        raise ValueError("Evaluation set for M5 is empty; verify dataset contents and split parameters.")

    params = fit_tsb_hb(init_set)
    tsbhb_point = predict_tsb_hb(params, eval_set, quantiles=None)
    tsbhb_point = tsbhb_point.rename(columns={"yhat": "TSB-HB"})

    sf = StatsForecast(
        models=[
            CrostonClassic(),
            CrostonSBA(),
            TSB(alpha_d=0.5, alpha_p=0.45),
            ADIDA(),
            IMAPA(),
        ],
        freq="D",
        n_jobs=-1,
    )
    sf.fit(init_set[["unique_id", "ds", "y"]])
    h_max = int(eval_set.groupby("unique_id").size().max())
    fcst_df = sf.predict(h=h_max).reset_index()

    eval_horizon = eval_set[["unique_id", "ds"]]
    fcst_df = fcst_df.merge(eval_horizon, on=["unique_id", "ds"], how="inner")

    merged = eval_set[["unique_id", "ds", "y"]].merge(fcst_df, on=["unique_id", "ds"], how="left")
    merged = merged.merge(tsbhb_point, on=["unique_id", "ds"], how="left")

    results = []
    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y"}]
    for model in model_cols:
        tmp = merged[["unique_id", "ds", "y", model]].dropna(subset=[model]).copy()
        if tmp.empty:
            continue
        tmp = tmp.rename(columns={model: "y_pred"})
        results.append({
            "model": model,
            "ME": me(tmp["y"].to_numpy(), tmp["y_pred"].to_numpy()),
            "MAE": mae(tmp["y"].to_numpy(), tmp["y_pred"].to_numpy()),
            "RMSE": rmse(tmp["y"].to_numpy(), tmp["y_pred"].to_numpy()),
            "RMSSE": rmsse(init_set, tmp),
            "WRMSSE": wrmsse(init_set, tmp),
        })

    metrics_df = pd.DataFrame(results)
    metrics_path = out_dir / "point_metrics_m5.csv"
    metrics_df.to_csv(metrics_path, index=False)

    if not metrics_df.empty:
        print("=== M5 Point Forecast Metrics ===")
        print(metrics_df.sort_values("WRMSSE"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["online_retail", "m5"], default="online_retail")
    ap.add_argument("--data", type=Path, default=default_data_file())
    ap.add_argument("--m5-sales", type=Path, default=default_m5_sales_file())
    ap.add_argument("--m5-calendar", type=Path, default=default_m5_calendar_file())
    ap.add_argument("--m5-sample-size", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "m5":
        _run_m5_point(args, out_dir)
        return

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
            "WRMSSE": wrmsse(init_set, tmp),
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

    # Compute RMSSE by category per model (pivot like the notebook)
    eval_with_cats = merged.merge(feats[["unique_id", "adi", "cv_sq", "category"]], on="unique_id", how="left")
    categories = [c for c in ["Smooth", "Erratic", "Intermittent", "Lumpy"] if (eval_with_cats["category"] == c).any()]
    seg_rows = []
    for model in model_cols:
        for cat in categories:
            subset = eval_with_cats[eval_with_cats["category"] == cat][["unique_id", "ds", "y", model]].dropna(subset=[model]).copy()
            if subset.empty:
                val_r = np.nan
                val_wr = np.nan
            else:
                tmp = subset.rename(columns={model: "y_pred"})
                val_r = rmsse(init_set, tmp)
                val_wr = wrmsse(init_set, tmp)
            seg_rows.append({"model": model, "category": cat, "Metric": "RMSSE", "Value": val_r})
            seg_rows.append({"model": model, "category": cat, "Metric": "WRMSSE", "Value": val_wr})
    seg_df = pd.DataFrame(seg_rows)
    if not seg_df.empty:
        pivot = seg_df.pivot_table(index="model", columns=["Metric", "category"], values="Value")
        pivot = pivot.reindex(model_cols)
        # Flatten columns with Metric prefix
        pivot.columns = [f"{cat}_{metric}" for metric, cat in pivot.columns]
        pivot = pivot.reindex(columns=[f"{cat}_RMSSE" for cat in categories] + [f"{cat}_WRMSSE" for cat in categories if f"{cat}_WRMSSE" in pivot.columns])
        pivot.reset_index().to_csv(out_dir / "segmentation_rmsse.csv", index=False)


if __name__ == "__main__":
    main()
