# run_point.py
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
from models.hurdle import fit_predict_hurdle
from metrics import me, mae, rmse, rmsse, wrmsse, compute_adi_cv2, classify_adi_cv2
from plotting import plot_shrinkage_scatter


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure ds is datetime for safe merging/alignment."""
    out = df.copy()
    if "ds" in out.columns:
        out["ds"] = pd.to_datetime(out["ds"])
    return out


def _assert_full_coverage(merged: pd.DataFrame, model_cols: list[str]) -> None:
    """Hard-stop if any model is missing any eval rows (prevents unfair dropna-based eval)."""
    if merged.empty:
        raise ValueError("[Coverage] merged dataframe is empty.")

    # Ensure unique_id/ds/y exist
    for col in ("unique_id", "ds", "y"):
        if col not in merged.columns:
            raise ValueError(f"[Coverage] merged missing required column: {col}")

    # Make sure we are evaluating exactly on eval_set rows (no accidental duplicates)
    dup = merged.duplicated(subset=["unique_id", "ds"]).mean()
    if dup > 0:
        examples = merged.loc[merged.duplicated(subset=["unique_id", "ds"]), ["unique_id", "ds"]].head(10)
        raise ValueError(f"[Coverage] merged has duplicated (unique_id, ds) rows. Examples:\n{examples}")

    for m in model_cols:
        if m not in merged.columns:
            raise ValueError(f"[Coverage] model column not found: {m}")
        cov = float(merged[m].notna().mean())
        if cov < 1.0:
            miss = merged.loc[merged[m].isna(), ["unique_id", "ds"]].head(10)
            raise ValueError(
                f"[Coverage] {m} coverage={cov:.4f} < 1.0. "
                f"Fix forecasting alignment before scoring.\nMissing examples:\n{miss}"
            )

    # Also ensure every model covers the same set of series
    all_uids = set(merged["unique_id"].unique())
    for m in model_cols:
        u_m = set(merged.loc[merged[m].notna(), "unique_id"].unique())
        if u_m != all_uids:
            missing = list(all_uids - u_m)[:10]
            raise ValueError(f"[Coverage] {m} missing {len(all_uids - u_m)} series. Examples: {missing}")


def _score_point_forecasts(init_set: pd.DataFrame, merged: pd.DataFrame, out_path: Path) -> None:
    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y"}]
    _assert_full_coverage(merged, model_cols)

    results = []
    for model in model_cols:
        tmp = merged[["unique_id", "ds", "y", model]].rename(columns={model: "y_pred"}).copy()
        # No dropna: coverage already guaranteed
        results.append(
            {
                "model": model,
                "ME": me(tmp["y"].values, tmp["y_pred"].values),
                "MAE": mae(tmp["y"].values, tmp["y_pred"].values),
                "RMSE": rmse(tmp["y"].values, tmp["y_pred"].values),
                "RMSSE": rmsse(init_set, tmp),
                "WRMSSE": wrmsse(init_set, tmp),
            }
        )
    pd.DataFrame(results).to_csv(out_path, index=False)


def _run_m5_point(args: argparse.Namespace, out_dir: Path) -> None:
    from statsforecast import StatsForecast
    from statsforecast.models import CrostonClassic, CrostonSBA, TSB, ADIDA, IMAPA

    sales_df, calendar_df = load_m5_long(args.m5_sales, args.m5_calendar)
    df = preprocess_m5(sales_df, calendar_df, sample_size=args.m5_sample_size)
    df = _ensure_datetime(df)

    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=2.0 / 3.0, min_len=1)
    init_set = _ensure_datetime(init_set)
    eval_set = _ensure_datetime(eval_set)

    # -------- TSB-HB-RA --------
    params = fit_tsb_hb(init_set)  # regime-aware fit inside
    tsbhb_point = predict_tsb_hb(params, eval_set, quantiles=None)
    tsbhb_point = _ensure_datetime(tsbhb_point).rename(columns={"yhat": "TSB-HB-RA"})

    # -------- Hurdle baseline --------
    hurdle_point = fit_predict_hurdle(init_set, eval_set, quantiles=None)
    hurdle_point = _ensure_datetime(hurdle_point).rename(columns={"yhat": "Hurdle"})

    # -------- StatsForecast baselines --------
    sf = StatsForecast(
        models=[CrostonClassic(), CrostonSBA(), TSB(alpha_d=0.5, alpha_p=0.45), ADIDA(), IMAPA()],
        freq="D",
        n_jobs=-1,
    )
    sf.fit(init_set[["unique_id", "ds", "y"]])

    # Predict to max horizon once, then align strictly to eval_set via LEFT merge (never inner)
    max_h = int(eval_set.groupby("unique_id").size().max())
    fcst_df = sf.predict(h=max_h).reset_index()
    fcst_df = _ensure_datetime(fcst_df)

    # Align to the exact eval index
    fcst_df = eval_set[["unique_id", "ds"]].merge(fcst_df, on=["unique_id", "ds"], how="left")

    # Merge all predictions onto eval_set index
    merged = eval_set[["unique_id", "ds", "y"]].merge(fcst_df, on=["unique_id", "ds"], how="left")
    merged = merged.merge(tsbhb_point, on=["unique_id", "ds"], how="left")
    merged = merged.merge(hurdle_point, on=["unique_id", "ds"], how="left")

    _score_point_forecasts(init_set, merged, out_dir / "point_metrics_m5.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["online_retail", "m5"], default="online_retail")
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m5-sales", type=Path, default=default_m5_sales_file())
    ap.add_argument("--m5-calendar", type=Path, default=default_m5_calendar_file())
    ap.add_argument("--m5-sample-size", type=int, default=None, help="Sample size for M5 dataset (default: all)")
    args = ap.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "m5":
        _run_m5_point(args, args.out)
        return

    # ---------------- Online Retail ----------------
    df = preprocess_online_retail(load_online_retail(default_data_file()))
    df = _ensure_datetime(df)

    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    init_set = _ensure_datetime(init_set)
    eval_set = _ensure_datetime(eval_set)

    # TSB-HB-RA
    params = fit_tsb_hb(init_set)
    tsbhb_point = predict_tsb_hb(params, eval_set, quantiles=None)
    tsbhb_point = _ensure_datetime(tsbhb_point).rename(columns={"yhat": "TSB-HB-RA"})

    # Hurdle baseline
    hurdle_point = fit_predict_hurdle(init_set, eval_set, quantiles=None)
    hurdle_point = _ensure_datetime(hurdle_point).rename(columns={"yhat": "Hurdle"})

    # Other baselines
    horizons = eval_set["unique_id"].value_counts()
    base_preds = fit_predict_baselines(init_set, horizons=horizons, freq="D")
    base_preds = _ensure_datetime(base_preds)

    merged = (
        eval_set[["unique_id", "ds", "y"]]
        .merge(base_preds, on=["unique_id", "ds"], how="left")
        .merge(tsbhb_point, on=["unique_id", "ds"], how="left")
        .merge(hurdle_point, on=["unique_id", "ds"], how="left")
    )

    _score_point_forecasts(init_set, merged, args.out / "point_metrics.csv")
    print(f"[Done] Metrics saved to {args.out}")

    # ---------------- Optional Diagnostics (unchanged logic) ----------------
    # Shrinkage plots
    init_set_local = init_set.copy()
    init_set_local["occ"] = (init_set_local["y"] > 0).astype(int)
    g_init = init_set_local.groupby("unique_id")
    s = g_init["occ"].sum()
    n = g_init["ds"].nunique()
    p_mle = (s / n).reindex(params.p_mean.index).dropna()
    p_post = params.p_mean.reindex(p_mle.index)

    plot_shrinkage_scatter(
        x=p_mle.values,
        y=p_post.values,
        xlabel="p MLE (per-item)",
        ylabel="p posterior mean (HB)",
        title="Shrinkage on Demand Probability (p)",
        out_path=args.out / "fig_shrink_p.png",
    )

    # size MLE vs posterior (lognormal)
    init_set_local["size"] = np.where(init_set_local["occ"] == 1, init_set_local["y"].astype(float), np.nan)
    init_set_local["log_size"] = np.log(init_set_local["size"])
    size_mle = init_set_local.groupby("unique_id")["size"].mean()
    size_post = params.size_mean
    idx = size_mle.index.intersection(size_post.index)

    plot_shrinkage_scatter(
        x=size_mle.reindex(idx).fillna(0).values,
        y=size_post.reindex(idx).fillna(0).values,
        xlabel="Size MLE (per-item avg size)",
        ylabel="Size posterior mean (HB)",
        title="Shrinkage on Demand Size",
        out_path=args.out / "fig_shrink_size.png",
    )

    # Segmentation (ADI & CV^2)
    feats = compute_adi_cv2(init_set)
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)

    # Compute RMSSE/WRMSSE by category per model
    eval_with_cats = merged.merge(feats[["unique_id", "adi", "cv_sq", "category"]], on="unique_id", how="left")
    categories = [c for c in ["Smooth", "Erratic", "Intermittent", "Lumpy"] if (eval_with_cats["category"] == c).any()]
    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y"}]

    seg_rows = []
    for model in model_cols:
        for cat in categories:
            subset = (
                eval_with_cats[eval_with_cats["category"] == cat][["unique_id", "ds", "y", model]]
                .copy()
            )
            # If coverage assert passes globally, subsets shouldn't have NaNs, but keep safe:
            subset = subset.dropna(subset=[model])
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
        pivot.columns = [f"{cat}_{metric}" for metric, cat in pivot.columns]
        pivot = pivot.reindex(
            columns=[f"{cat}_RMSSE" for cat in categories]
            + [f"{cat}_WRMSSE" for cat in categories if f"{cat}_WRMSSE" in pivot.columns]
        )
        pivot.reset_index().to_csv(args.out / "segmentation_rmsse.csv", index=False)


if __name__ == "__main__":
    main()