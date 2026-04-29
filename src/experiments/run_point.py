from __future__ import annotations

import argparse
import time
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
from experiments.protocols import evaluate_point_models, iter_walk_forward_frames
from models.tsb_hb import (
    TSBHBOnlineState,
    TSBHBParams,
    fit_tsb_hb,
    initialize_online_tsb_hb,
    predict_online_tsb_hb,
    predict_tsb_hb,
    update_online_tsb_hb,
)
from models.baselines import fit_predict_baselines, fit_predict_single_baseline, POINT_BASELINE_KEYS
from models.hurdle_baselines import (
    fit_hurdle_global_lognormal,
    fit_hurdle_local_lognormal,
    predict_hurdle_global_lognormal,
    predict_hurdle_local_lognormal,
)
from models.tweedie_baseline import TWEEDIE_POINT_COL, fit_predict_tweedie_panel
from metrics import rmsse, wrmsse, compute_adi_cv2, classify_adi_cv2
from plotting import plot_shrinkage_scatter


POINT_TSBHB_MODEL = "TSB-HB"


def _normalize_point_baseline_mode(baseline_mode: str | None) -> str:
    mode = str(baseline_mode or "paper").lower()
    if mode == "full":
        mode = "extended"
    valid = {"paper", "extended", "hurdle_only", "hb_only"}
    if mode not in valid:
        raise ValueError("baseline_mode must be one of: paper, extended, full, hurdle_only, hb_only.")
    return mode


def _include_statistical_point_baselines(baseline_mode: str) -> bool:
    return baseline_mode in {"paper", "extended"}


def _include_hurdle_point_baselines(baseline_mode: str) -> bool:
    return baseline_mode in {"extended", "hurdle_only"}


def _build_regime_group_labels(train_df: pd.DataFrame) -> pd.Series | None:
    feats = compute_adi_cv2(train_df)
    if feats.empty:
        return None
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)
    return feats.set_index("unique_id")["category"].astype(str)


def _build_m5_hierarchy_group_labels(df: pd.DataFrame) -> pd.Series | None:
    cols = set(df.columns)
    if {"dept_id", "store_id", "unique_id"}.issubset(cols):
        tmp = df[["unique_id", "dept_id", "store_id"]].drop_duplicates(subset=["unique_id"]).copy()
        tmp["group"] = tmp["dept_id"].astype(str) + "|" + tmp["store_id"].astype(str)
        return tmp.set_index("unique_id")["group"]
    if {"cat_id", "store_id", "unique_id"}.issubset(cols):
        tmp = df[["unique_id", "cat_id", "store_id"]].drop_duplicates(subset=["unique_id"]).copy()
        tmp["group"] = tmp["cat_id"].astype(str) + "|" + tmp["store_id"].astype(str)
        return tmp.set_index("unique_id")["group"]
    if {"store_id", "unique_id"}.issubset(cols):
        tmp = df[["unique_id", "store_id"]].drop_duplicates(subset=["unique_id"]).copy()
        tmp["group"] = tmp["store_id"].astype(str)
        return tmp.set_index("unique_id")["group"]
    return None


def _build_m5_taxonomy4_group_labels(df: pd.DataFrame) -> pd.Series | None:
    """Build 4-group taxonomy labels on M5 via ADI/CV^2 (Smooth/Erratic/Intermittent/Lumpy)."""
    return _build_regime_group_labels(df)


def _run_m5_point(args: argparse.Namespace, out_dir: Path) -> None:
    sales_df, calendar_df = load_m5_long(args.m5_sales, args.m5_calendar)
    df = preprocess_m5(sales_df, calendar_df, sample_size=args.m5_sample_size)

    if df.empty:
        raise ValueError("Preprocessed M5 dataframe is empty; check input files and sample size.")

    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=2.0 / 3.0, min_len=1)
    if eval_set.empty:
        raise ValueError("Evaluation set for M5 is empty; verify dataset contents and split parameters.")

    eval_horizons = eval_set.groupby("unique_id").size()
    eval_horizon = eval_set[["unique_id", "ds"]]
    n_series = int(init_set["unique_id"].nunique())
    timings: dict[str, float] = {}
    baseline_mode = _normalize_point_baseline_mode(args.baseline_mode)

    hierarchy_labels = _build_m5_hierarchy_group_labels(init_set)
    taxonomy4_labels = _build_m5_taxonomy4_group_labels(init_set)
    mode = args.m5_hierarchy_mode
    if mode in {"on", "ablation"} and hierarchy_labels is None:
        print("Warning: hierarchy labels unavailable in M5 frame; falling back to non-hierarchical TSB-HB.")
        mode = "off"
    if mode in {"taxonomy4"} and taxonomy4_labels is None:
        print("Warning: taxonomy4 labels unavailable in M5 frame; falling back to non-hierarchical TSB-HB.")
        mode = "off"

    merged = eval_set[["unique_id", "ds", "y"]].copy()
    tsbhb_frames: list[pd.DataFrame] = []

    if mode in {"off", "ablation"}:
        t0 = time.perf_counter()
        params_plain = fit_tsb_hb(
            init_set,
            group_labels=None,
            group_shrink_strength=args.hb_group_shrink_strength,
            item_variance_mode=args.hb_item_variance_mode,
            item_variance_shrink_strength=args.hb_variance_prior_df,
        )
        tsbhb_plain = predict_tsb_hb(params_plain, eval_set, quantiles=None)
        model_name = POINT_TSBHB_MODEL if mode == "off" else "TSB-HB-NoHierarchy"
        timings[model_name] = time.perf_counter() - t0
        tsbhb_frames.append(tsbhb_plain.rename(columns={"yhat": model_name}))
    if mode in {"on", "ablation"} and hierarchy_labels is not None:
        t0 = time.perf_counter()
        params_hier = fit_tsb_hb(
            init_set,
            group_labels=hierarchy_labels,
            group_shrink_strength=args.hb_group_shrink_strength,
            item_variance_mode=args.hb_item_variance_mode,
            item_variance_shrink_strength=args.hb_variance_prior_df,
        )
        tsbhb_hier = predict_tsb_hb(params_hier, eval_set, quantiles=None)
        model_name = POINT_TSBHB_MODEL if mode == "on" else "TSB-HB-Hierarchy"
        timings[model_name] = time.perf_counter() - t0
        tsbhb_frames.append(tsbhb_hier.rename(columns={"yhat": model_name}))
    if mode in {"taxonomy4"} and taxonomy4_labels is not None:
        t0 = time.perf_counter()
        params_tax4 = fit_tsb_hb(
            init_set,
            group_labels=taxonomy4_labels,
            group_shrink_strength=args.hb_group_shrink_strength,
            item_variance_mode=args.hb_item_variance_mode,
            item_variance_shrink_strength=args.hb_variance_prior_df,
        )
        tsbhb_tax4 = predict_tsb_hb(params_tax4, eval_set, quantiles=None)
        model_name = "TSB-HB-Taxonomy4"
        timings[model_name] = time.perf_counter() - t0
        tsbhb_frames.append(tsbhb_tax4.rename(columns={"yhat": model_name}))

    for f in tsbhb_frames:
        merged = merged.merge(f, on=["unique_id", "ds"], how="left")

    if _include_statistical_point_baselines(baseline_mode):
        for model_key in POINT_BASELINE_KEYS:
            t0 = time.perf_counter()
            pred = fit_predict_single_baseline(init_set, eval_horizons, model_key=model_key, freq="D")
            timings[model_key] = time.perf_counter() - t0
            if not pred.empty:
                pred = pred.drop(columns=["index"], errors="ignore")
                pred = pred.merge(eval_horizon, on=["unique_id", "ds"], how="inner")
                merged = merged.merge(pred, on=["unique_id", "ds"], how="left")

    if _include_hurdle_point_baselines(baseline_mode):
        t0 = time.perf_counter()
        local_params = fit_hurdle_local_lognormal(init_set)
        local_preds = predict_hurdle_local_lognormal(local_params, eval_set)[
            ["unique_id", "ds", "Hurdle-Local-LogNormal"]
        ]
        timings["Hurdle-Local-LogNormal"] = time.perf_counter() - t0
        merged = merged.merge(local_preds, on=["unique_id", "ds"], how="left")

        t0 = time.perf_counter()
        global_params = fit_hurdle_global_lognormal(init_set)
        global_preds = predict_hurdle_global_lognormal(global_params, eval_set)[
            ["unique_id", "ds", "Hurdle-Global-LogNormal"]
        ]
        timings["Hurdle-Global-LogNormal"] = time.perf_counter() - t0
        merged = merged.merge(global_preds, on=["unique_id", "ds"], how="left")

    if getattr(args, "with_iets", False):
        from models.iets_baseline import IETS_POINT_COL, fit_predict_iets_panel

        t0 = time.perf_counter()
        iets_pred = fit_predict_iets_panel(
            init_set,
            eval_set,
            rscript=str(args.iets_rscript),
            script_path=args.iets_script,
            seed=int(args.seed),
            occurrence=str(args.iets_occurrence),
        )
        timings[IETS_POINT_COL] = time.perf_counter() - t0
        merged = merged.merge(iets_pred, on=["unique_id", "ds"], how="left")

    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y", "index"}]
    metrics_df = evaluate_point_models(init_set, merged, model_cols=model_cols)
    if n_series > 0:
        metrics_df["Efficiency"] = metrics_df["model"].map(
            lambda m: timings.get(m, 0.0) / n_series
        )
    metrics_path = out_dir / "point_metrics_m5.csv"
    metrics_df.to_csv(metrics_path, index=False)
    if mode == "ablation":
        ablation = metrics_df[metrics_df["model"].isin(["TSB-HB-NoHierarchy", "TSB-HB-Hierarchy"])].copy()
        ablation.to_csv(out_dir / "m5_hierarchy_ablation.csv", index=False)

    if not metrics_df.empty:
        print("=== M5 Point Forecast Metrics ===")
        print(metrics_df.sort_values("WRMSSE"))


def _predict_online_models_once(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    hb_group_labels: pd.Series | None = None,
    hb_group_shrink_strength: float = 0.0,
    hb_item_variance_mode: str = "group",
    hb_variance_prior_df: float = 20.0,
    tsbhb_override: pd.DataFrame | None = None,
    baseline_mode: str = "paper",
    with_tweedie: bool = False,
    tweedie_lags: int = 14,
    tweedie_power: float = 1.5,
    tweedie_alpha: float = 0.1,
    tweedie_max_iter: int = 1000,
) -> pd.DataFrame:
    baseline_mode = _normalize_point_baseline_mode(baseline_mode)

    if tsbhb_override is None:
        params = fit_tsb_hb(
            train_df,
            group_labels=hb_group_labels,
            group_shrink_strength=hb_group_shrink_strength,
            item_variance_mode=hb_item_variance_mode,
            item_variance_shrink_strength=hb_variance_prior_df,
        )
        tsbhb_point = predict_tsb_hb(params, eval_df, quantiles=None)
        tsbhb_point = tsbhb_point.rename(columns={"yhat": POINT_TSBHB_MODEL})
    else:
        tsbhb_point = tsbhb_override.copy()
        if POINT_TSBHB_MODEL not in tsbhb_point.columns and "yhat" in tsbhb_point.columns:
            tsbhb_point = tsbhb_point.rename(columns={"yhat": POINT_TSBHB_MODEL})

    merged = eval_df[["unique_id", "ds", "y"]].copy()
    merged = merged.merge(tsbhb_point, on=["unique_id", "ds"], how="left")

    if _include_statistical_point_baselines(baseline_mode):
        eval_horizons = eval_df["unique_id"].value_counts()
        base_preds = fit_predict_baselines(train_df, horizons=eval_horizons, freq="D", probabilistic=False)
        if not base_preds.empty:
            merged = merged.merge(base_preds, on=["unique_id", "ds"], how="left")

    if _include_hurdle_point_baselines(baseline_mode):
        local_params = fit_hurdle_local_lognormal(train_df)
        local_preds = predict_hurdle_local_lognormal(local_params, eval_df)
        local_preds = local_preds[["unique_id", "ds", "Hurdle-Local-LogNormal"]]

        global_params = fit_hurdle_global_lognormal(train_df)
        global_preds = predict_hurdle_global_lognormal(global_params, eval_df)
        global_preds = global_preds[["unique_id", "ds", "Hurdle-Global-LogNormal"]]

        merged = merged.merge(local_preds, on=["unique_id", "ds"], how="left")
        merged = merged.merge(global_preds, on=["unique_id", "ds"], how="left")
    if with_tweedie:
        tw_pred = fit_predict_tweedie_panel(
            train_df,
            eval_df,
            lags=tweedie_lags,
            power=tweedie_power,
            alpha=tweedie_alpha,
            max_iter=tweedie_max_iter,
        )
        merged = merged.merge(tw_pred, on=["unique_id", "ds"], how="left")
    return merged


def _run_online_point_fixed(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    hb_group_labels: pd.Series | None = None,
    hb_group_shrink_strength: float = 0.0,
    hb_item_variance_mode: str = "group",
    hb_variance_prior_df: float = 20.0,
    baseline_mode: str = "paper",
) -> pd.DataFrame:
    return _predict_online_models_once(
        init_set,
        eval_set,
        hb_group_labels=hb_group_labels,
        hb_group_shrink_strength=hb_group_shrink_strength,
        hb_item_variance_mode=hb_item_variance_mode,
        hb_variance_prior_df=hb_variance_prior_df,
        baseline_mode=baseline_mode,
    )


def _run_online_point_fixed_with_timing(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    hb_group_labels: pd.Series | None = None,
    hb_group_shrink_strength: float = 0.0,
    hb_item_variance_mode: str = "group",
    hb_variance_prior_df: float = 20.0,
    baseline_mode: str = "paper",
    *,
    seed: int = 42,
    with_iets: bool = False,
    iets_rscript: str = "Rscript",
    iets_script: Path | None = None,
    iets_occurrence: str = "auto",
    with_tweedie: bool = False,
    tweedie_lags: int = 14,
    tweedie_power: float = 1.5,
    tweedie_alpha: float = 0.1,
    tweedie_max_iter: int = 1000,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run fixed-origin point forecasting and record per-model elapsed time (fit+predict) in seconds."""
    timings: dict[str, float] = {}
    merged = eval_set[["unique_id", "ds", "y"]].copy()

    # TSB-HB
    t0 = time.perf_counter()
    params = fit_tsb_hb(
        init_set,
        group_labels=hb_group_labels,
        group_shrink_strength=hb_group_shrink_strength,
        item_variance_mode=hb_item_variance_mode,
        item_variance_shrink_strength=hb_variance_prior_df,
    )
    tsbhb_point = predict_tsb_hb(params, eval_set, quantiles=None)
    tsbhb_point = tsbhb_point.rename(columns={"yhat": POINT_TSBHB_MODEL})
    timings[POINT_TSBHB_MODEL] = time.perf_counter() - t0
    merged = merged.merge(tsbhb_point, on=["unique_id", "ds"], how="left")

    baseline_mode = _normalize_point_baseline_mode(baseline_mode)

    if _include_statistical_point_baselines(baseline_mode):
        eval_horizons = eval_set["unique_id"].value_counts()
        for model_key in POINT_BASELINE_KEYS:
            t0 = time.perf_counter()
            pred = fit_predict_single_baseline(
                init_set, eval_horizons, model_key=model_key, freq="D"
            )
            timings[model_key] = time.perf_counter() - t0
            if not pred.empty:
                pred = pred.drop(columns=["index"], errors="ignore")
                merged = merged.merge(pred, on=["unique_id", "ds"], how="left")

    if _include_hurdle_point_baselines(baseline_mode):
        t0 = time.perf_counter()
        local_params = fit_hurdle_local_lognormal(init_set)
        local_preds = predict_hurdle_local_lognormal(local_params, eval_set)
        local_preds = local_preds[["unique_id", "ds", "Hurdle-Local-LogNormal"]]
        timings["Hurdle-Local-LogNormal"] = time.perf_counter() - t0
        merged = merged.merge(local_preds, on=["unique_id", "ds"], how="left")

        t0 = time.perf_counter()
        global_params = fit_hurdle_global_lognormal(init_set)
        global_preds = predict_hurdle_global_lognormal(global_params, eval_set)
        global_preds = global_preds[["unique_id", "ds", "Hurdle-Global-LogNormal"]]
        timings["Hurdle-Global-LogNormal"] = time.perf_counter() - t0
        merged = merged.merge(global_preds, on=["unique_id", "ds"], how="left")

    if with_iets:
        from models.iets_baseline import IETS_POINT_COL, fit_predict_iets_panel

        t0 = time.perf_counter()
        iets_pred = fit_predict_iets_panel(
            init_set,
            eval_set,
            rscript=iets_rscript,
            script_path=iets_script,
            seed=seed,
            occurrence=iets_occurrence,
        )
        timings[IETS_POINT_COL] = time.perf_counter() - t0
        merged = merged.merge(iets_pred, on=["unique_id", "ds"], how="left")
    if with_tweedie:
        t0 = time.perf_counter()
        tw_pred = fit_predict_tweedie_panel(
            init_set,
            eval_set,
            lags=tweedie_lags,
            power=tweedie_power,
            alpha=tweedie_alpha,
            max_iter=tweedie_max_iter,
        )
        timings[TWEEDIE_POINT_COL] = time.perf_counter() - t0
        merged = merged.merge(tw_pred, on=["unique_id", "ds"], how="left")

    return merged, timings


def _run_online_point_walk_forward(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    walk_step: int,
    hb_group_labels: pd.Series | None = None,
    hb_group_shrink_strength: float = 0.0,
    hb_online_update: bool = True,
    hb_dynamic_occurrence: bool = False,
    hb_occ_discount: float = 1.0,
    hb_item_variance_mode: str = "group",
    hb_variance_prior_df: float = 20.0,
    baseline_mode: str = "paper",
    with_iets: bool = False,
    iets_rscript: str = "Rscript",
    iets_script: Path | None = None,
    iets_occurrence: str = "auto",
    seed: int = 42,
    with_tweedie: bool = False,
    tweedie_lags: int = 14,
    tweedie_power: float = 1.5,
    tweedie_alpha: float = 0.1,
    tweedie_max_iter: int = 1000,
) -> pd.DataFrame:
    step_outputs: list[pd.DataFrame] = []
    total_frames = max(int(np.ceil(eval_set.groupby("unique_id").size().max() / walk_step)), 1)
    hb_state: TSBHBOnlineState | None = None
    fit_predict_iets_panel = None
    iets_col: str | None = None
    if with_iets:
        from models.iets_baseline import IETS_POINT_COL, fit_predict_iets_panel as _fit_predict_iets_panel
        fit_predict_iets_panel = _fit_predict_iets_panel
        iets_col = IETS_POINT_COL
    if hb_online_update:
        hb_state = initialize_online_tsb_hb(
            init_set,
            group_labels=hb_group_labels,
            group_shrink_strength=hb_group_shrink_strength,
            dynamic_occurrence=hb_dynamic_occurrence,
            occurrence_discount=hb_occ_discount,
            item_variance_mode=hb_item_variance_mode,
            item_variance_shrink_strength=hb_variance_prior_df,
        )

    for frame_idx, frame in enumerate(iter_walk_forward_frames(init_set, eval_set, step_size=walk_step), start=1):
        if frame_idx == 1 or frame_idx % 10 == 0 or frame_idx == total_frames:
            print(
                f"[walk_forward] frame {frame_idx}/{total_frames} "
                f"(step={frame.step}, n_target={len(frame.target)})"
            )
        if hb_state is not None:
            tsbhb_step = predict_online_tsb_hb(
                hb_state,
                frame.target,
                quantiles=None,
                include_hyper_uncertainty=False,
            ).rename(columns={"yhat": POINT_TSBHB_MODEL})
            merged_step = _predict_online_models_once(
                frame.history,
                frame.target,
                hb_group_labels=hb_group_labels,
                hb_group_shrink_strength=hb_group_shrink_strength,
                hb_item_variance_mode=hb_item_variance_mode,
                hb_variance_prior_df=hb_variance_prior_df,
                tsbhb_override=tsbhb_step,
                baseline_mode=baseline_mode,
                with_tweedie=with_tweedie,
                tweedie_lags=tweedie_lags,
                tweedie_power=tweedie_power,
                tweedie_alpha=tweedie_alpha,
                tweedie_max_iter=tweedie_max_iter,
            )
            hb_state = update_online_tsb_hb(hb_state, frame.target)
        else:
            merged_step = _predict_online_models_once(
                frame.history,
                frame.target,
                hb_group_labels=hb_group_labels,
                hb_group_shrink_strength=hb_group_shrink_strength,
                hb_item_variance_mode=hb_item_variance_mode,
                hb_variance_prior_df=hb_variance_prior_df,
                baseline_mode=baseline_mode,
                with_tweedie=with_tweedie,
                tweedie_lags=tweedie_lags,
                tweedie_power=tweedie_power,
                tweedie_alpha=tweedie_alpha,
                tweedie_max_iter=tweedie_max_iter,
            )
        if with_iets and fit_predict_iets_panel is not None and iets_col is not None:
            iets_step = fit_predict_iets_panel(
                frame.history,
                frame.target,
                rscript=iets_rscript,
                script_path=iets_script,
                seed=seed,
                occurrence=iets_occurrence,
            )
            # R output can cast ids/dates to different dtypes; align keys before merging.
            merged_step["unique_id"] = merged_step["unique_id"].astype(str)
            merged_step["ds"] = pd.to_datetime(merged_step["ds"])
            iets_step["unique_id"] = iets_step["unique_id"].astype(str)
            iets_step["ds"] = pd.to_datetime(iets_step["ds"])
            merged_step = merged_step.merge(
                iets_step[["unique_id", "ds", iets_col]],
                on=["unique_id", "ds"],
                how="left",
            )
        step_outputs.append(merged_step)

    if not step_outputs:
        empty = eval_set[["unique_id", "ds", "y"]].copy()
        return empty

    merged = pd.concat(step_outputs, ignore_index=True)
    merged = merged.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return merged


def _write_shrinkage_plots(init_set: pd.DataFrame, params: TSBHBParams, out_dir: Path) -> None:
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


def _write_segmentation_metrics(
    init_set: pd.DataFrame,
    merged_eval: pd.DataFrame,
    model_cols: list[str],
    out_dir: Path,
) -> None:
    feats = compute_adi_cv2(init_set)
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)

    eval_with_cats = merged_eval.merge(
        feats[["unique_id", "adi", "cv_sq", "category"]],
        on="unique_id",
        how="left",
    )
    categories = [
        c for c in ["Smooth", "Erratic", "Intermittent", "Lumpy"]
        if (eval_with_cats["category"] == c).any()
    ]

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
    if seg_df.empty:
        return

    pivot = seg_df.pivot_table(index="model", columns=["Metric", "category"], values="Value")
    pivot = pivot.reindex(model_cols)
    pivot.columns = [f"{cat}_{metric}" for metric, cat in pivot.columns]
    pivot = pivot.reindex(
        columns=[f"{cat}_RMSSE" for cat in categories]
        + [f"{cat}_WRMSSE" for cat in categories if f"{cat}_WRMSSE" in pivot.columns]
    )
    pivot.reset_index().to_csv(out_dir / "segmentation_rmsse.csv", index=False)


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


def _write_point_slice_metrics(
    init_set: pd.DataFrame,
    merged_eval: pd.DataFrame,
    model_cols: list[str],
    out_dir: Path,
) -> None:
    slice_feats = _build_slice_features(init_set)
    eval_with_slices = merged_eval.merge(
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
            ids = block["unique_id"].dropna().unique()
            init_sub = init_set[init_set["unique_id"].isin(ids)]
            for model in model_cols:
                subset = block[["unique_id", "ds", "y", model]].dropna(subset=[model]).copy()
                if subset.empty:
                    continue
                metric_df = evaluate_point_models(init_sub, subset, model_cols=[model])
                if metric_df.empty:
                    continue
                metric_row = metric_df.iloc[0].to_dict()
                rows.append(
                    {
                        "slice_type": slice_type,
                        "slice": slice_value,
                        "model": model,
                        "n_obs": int(len(subset)),
                        "n_series": int(subset["unique_id"].nunique()),
                        **metric_row,
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "point_slice_metrics.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["online_retail", "m5"], default="online_retail")
    ap.add_argument("--protocol", choices=["fixed", "walk_forward"], default="fixed")
    ap.add_argument("--walk-step", type=int, default=7, help="Block size (steps) for walk-forward protocol. Larger values are faster and reduce repeated re-fitting cost.")
    ap.add_argument("--baseline-mode", choices=["paper", "extended", "full", "hurdle_only", "hb_only"], default=None, help="Baseline set for both fixed and walk-forward: paper=TSB-HB + classical baselines from the paper, extended=paper + hurdle baselines, full=legacy alias for extended, hurdle_only=TSB-HB + hurdle baselines, hb_only=TSB-HB only.")
    ap.add_argument("--hb-regime-aware", dest="hb_regime_aware", action="store_true", default=True, help="Use ADI/CV^2 regime-aware HB priors for Online Retail.")
    ap.add_argument("--no-hb-regime-aware", dest="hb_regime_aware", action="store_false", help="Disable regime-aware priors and use global HB priors.")
    ap.add_argument("--hb-group-shrink-strength", type=float, default=0.0, help="Extra shrink strength from group-level hyperparameters back to global hyperparameters (0 disables).")
    ap.add_argument("--hb-online-update", dest="hb_online_update", action="store_true", default=True, help="Use sufficient-statistics online update for TSB-HB in walk-forward.")
    ap.add_argument("--no-hb-online-update", dest="hb_online_update", action="store_false", help="Disable online update and re-fit TSB-HB each walk-forward step.")
    ap.add_argument("--hb-dynamic-occurrence", dest="hb_dynamic_occurrence", action="store_true", default=False, help="Enable dynamic discounted occurrence update for TSB-HB in online walk-forward.")
    ap.add_argument("--no-hb-dynamic-occurrence", dest="hb_dynamic_occurrence", action="store_false", help="Disable dynamic discounted occurrence update.")
    ap.add_argument("--hb-occ-discount", type=float, default=1.0, help="Discount factor for dynamic occurrence update (0<d<=1). Smaller means faster adaptation.")
    ap.add_argument("--hb-item-variance-mode", choices=["group", "conjugate"], default="conjugate", help="Process variance mode for size: group or conjugate.")
    ap.add_argument("--hb-variance-prior-df", type=float, default=20.0, help="Prior degrees of freedom for conjugate variance model (larger = stronger shrinkage).")
    ap.add_argument("--m5-hierarchy-mode", choices=["off", "on", "ablation", "taxonomy4"], default="off", help="M5 grouping for TSB-HB: off=single global pool (paper default), on=hier priors, ablation=off vs hierarchy, taxonomy4=ADI/CV^2 4-group pooling.")
    ap.add_argument("--data", type=Path, default=default_data_file())
    ap.add_argument("--m5-sales", type=Path, default=default_m5_sales_file())
    ap.add_argument("--m5-calendar", type=Path, default=default_m5_calendar_file())
    ap.add_argument("--m5-sample-size", type=int, default=5000)
    ap.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Optional cap on number of Online Retail series (after preprocess) for faster runs; omit for all series.",
    )
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--with-iets",
        action="store_true",
        help="Include iETS point forecasts via R smooth::adam (requires R on PATH and CRAN packages smooth, greybox). Supported for fixed and walk_forward on online retail, and fixed runs on M5.",
    )
    ap.add_argument("--with-tweedie", action="store_true", help="Include Tweedie autoregressive baseline.")
    ap.add_argument("--tweedie-lags", type=int, default=14, help="Number of lag features for Tweedie baseline.")
    ap.add_argument("--tweedie-power", type=float, default=1.5, help="Tweedie variance power (1<p<2 typical for intermittent demand).")
    ap.add_argument("--tweedie-alpha", type=float, default=0.1, help="L2 regularization strength for Tweedie baseline.")
    ap.add_argument("--tweedie-max-iter", type=int, default=1000, help="Max optimizer iterations for Tweedie baseline.")
    ap.add_argument("--iets-rscript", type=str, default="Rscript", help="Rscript executable for iETS.")
    ap.add_argument(
        "--iets-script",
        type=Path,
        default=None,
        help="Path to iets_panel_forecast.R (default: <repo>/scripts/iets_panel_forecast.R).",
    )
    ap.add_argument(
        "--iets-occurrence",
        type=str,
        default="auto",
        help="adam() occurrence= argument (e.g. auto, direct, fixed, odds-ratio).",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "m5":
        if args.protocol != "fixed":
            raise ValueError("M5 currently supports only --protocol fixed in run_point.")
        _run_m5_point(args, out_dir)
        return

    # Load & preprocess
    df_raw = load_online_retail(args.data)
    df = preprocess_online_retail(df_raw)
    if args.max_series is not None:
        max_series = max(int(args.max_series), 1)
        uids = df["unique_id"].drop_duplicates()
        if len(uids) > max_series:
            keep = np.random.choice(uids.to_numpy(), size=max_series, replace=False)
            df = df[df["unique_id"].isin(keep)].copy()

    # Split fixed origin
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    hb_group_labels = _build_regime_group_labels(init_set) if args.hb_regime_aware else None
    baseline_mode = _normalize_point_baseline_mode(args.baseline_mode)
    hb_variance_prior_df = float(max(args.hb_variance_prior_df, 2.1))

    n_skus = int(init_set["unique_id"].nunique())
    if args.protocol == "fixed":
        merged, model_timings = _run_online_point_fixed_with_timing(
            init_set,
            eval_set,
            hb_group_labels=hb_group_labels,
            hb_group_shrink_strength=args.hb_group_shrink_strength,
            hb_item_variance_mode=args.hb_item_variance_mode,
            hb_variance_prior_df=hb_variance_prior_df,
            baseline_mode=baseline_mode,
            seed=int(args.seed),
            with_iets=bool(args.with_iets),
            iets_rscript=str(args.iets_rscript),
            iets_script=args.iets_script,
            iets_occurrence=str(args.iets_occurrence),
            with_tweedie=bool(args.with_tweedie),
            tweedie_lags=int(max(args.tweedie_lags, 1)),
            tweedie_power=float(args.tweedie_power),
            tweedie_alpha=float(max(args.tweedie_alpha, 0.0)),
            tweedie_max_iter=int(max(args.tweedie_max_iter, 100)),
        )
    else:
        model_timings = {}
        merged = _run_online_point_walk_forward(
            init_set,
            eval_set,
            walk_step=args.walk_step,
            hb_group_labels=hb_group_labels,
            hb_group_shrink_strength=args.hb_group_shrink_strength,
            hb_online_update=args.hb_online_update,
            hb_dynamic_occurrence=args.hb_dynamic_occurrence,
            hb_occ_discount=args.hb_occ_discount,
            hb_item_variance_mode=args.hb_item_variance_mode,
            hb_variance_prior_df=hb_variance_prior_df,
            baseline_mode=baseline_mode,
            with_iets=bool(args.with_iets),
            iets_rscript=str(args.iets_rscript),
            iets_script=args.iets_script,
            iets_occurrence=str(args.iets_occurrence),
            seed=int(args.seed),
            with_tweedie=bool(args.with_tweedie),
            tweedie_lags=int(max(args.tweedie_lags, 1)),
            tweedie_power=float(args.tweedie_power),
            tweedie_alpha=float(max(args.tweedie_alpha, 0.0)),
            tweedie_max_iter=int(max(args.tweedie_max_iter, 100)),
        )

    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y"}]
    point_df = evaluate_point_models(init_set, merged, model_cols=model_cols)
    point_df.insert(0, "protocol", args.protocol)
    if model_timings and n_skus > 0:
        point_df["Efficiency"] = point_df["model"].map(
            lambda m: model_timings.get(m, 0.0) / n_skus
        )
    point_df.to_csv(out_dir / "point_metrics.csv", index=False)

    _write_shrinkage_plots(
        init_set,
        fit_tsb_hb(
            init_set,
            group_labels=hb_group_labels,
            group_shrink_strength=args.hb_group_shrink_strength,
            item_variance_mode=args.hb_item_variance_mode,
            item_variance_shrink_strength=hb_variance_prior_df,
        ),
        out_dir,
    )
    _write_segmentation_metrics(init_set, merged, model_cols, out_dir)
    _write_point_slice_metrics(init_set, merged, model_cols, out_dir)


if __name__ == "__main__":
    main()
