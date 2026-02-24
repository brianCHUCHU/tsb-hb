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
from models.baselines import fit_predict_baselines
from models.hurdle_baselines import (
    fit_hurdle_global_lognormal,
    fit_hurdle_local_lognormal,
    predict_hurdle_global_lognormal,
    predict_hurdle_local_lognormal,
)
from metrics import rmsse, wrmsse, compute_adi_cv2, classify_adi_cv2
from plotting import plot_shrinkage_scatter


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

    hierarchy_labels = _build_m5_hierarchy_group_labels(init_set)
    mode = args.m5_hierarchy_mode
    if mode in {"on", "ablation"} and hierarchy_labels is None:
        print("Warning: hierarchy labels unavailable in M5 frame; falling back to non-hierarchical TSB-HB.")
        mode = "off"

    tsbhb_frames: list[pd.DataFrame] = []
    if mode in {"off", "ablation"}:
        params_plain = fit_tsb_hb(
            init_set,
            group_labels=None,
            group_shrink_strength=args.hb_group_shrink_strength,
        )
        tsbhb_plain = predict_tsb_hb(params_plain, eval_set, quantiles=None)
        model_name = "TSB-HB" if mode == "off" else "TSB-HB-NoHierarchy"
        tsbhb_frames.append(tsbhb_plain.rename(columns={"yhat": model_name}))
    if mode in {"on", "ablation"} and hierarchy_labels is not None:
        params_hier = fit_tsb_hb(
            init_set,
            group_labels=hierarchy_labels,
            group_shrink_strength=args.hb_group_shrink_strength,
        )
        tsbhb_hier = predict_tsb_hb(params_hier, eval_set, quantiles=None)
        model_name = "TSB-HB" if mode == "on" else "TSB-HB-Hierarchy"
        tsbhb_frames.append(tsbhb_hier.rename(columns={"yhat": model_name}))

    local_params = fit_hurdle_local_lognormal(init_set)
    local_preds = predict_hurdle_local_lognormal(local_params, eval_set)[
        ["unique_id", "ds", "Hurdle-Local-LogNormal"]
    ]
    global_params = fit_hurdle_global_lognormal(init_set)
    global_preds = predict_hurdle_global_lognormal(global_params, eval_set)[
        ["unique_id", "ds", "Hurdle-Global-LogNormal"]
    ]

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
    for f in tsbhb_frames:
        merged = merged.merge(f, on=["unique_id", "ds"], how="left")
    merged = merged.merge(local_preds, on=["unique_id", "ds"], how="left")
    merged = merged.merge(global_preds, on=["unique_id", "ds"], how="left")

    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y", "index"}]
    metrics_df = evaluate_point_models(init_set, merged, model_cols=model_cols)
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
    baseline_mode: str = "full",
) -> pd.DataFrame:
    if baseline_mode not in {"full", "hurdle_only", "hb_only"}:
        raise ValueError("baseline_mode must be one of: full, hurdle_only, hb_only.")

    if tsbhb_override is None:
        params = fit_tsb_hb(
            train_df,
            group_labels=hb_group_labels,
            group_shrink_strength=hb_group_shrink_strength,
            item_variance_mode=hb_item_variance_mode,
            item_variance_shrink_strength=hb_variance_prior_df,
        )
        tsbhb_point = predict_tsb_hb(params, eval_df, quantiles=None)
        tsbhb_point = tsbhb_point.rename(columns={"yhat": "TSB-HB-LogNormal"})
    else:
        tsbhb_point = tsbhb_override.copy()
        if "TSB-HB-LogNormal" not in tsbhb_point.columns and "yhat" in tsbhb_point.columns:
            tsbhb_point = tsbhb_point.rename(columns={"yhat": "TSB-HB-LogNormal"})

    merged = eval_df[["unique_id", "ds", "y"]].copy()
    merged = merged.merge(tsbhb_point, on=["unique_id", "ds"], how="left")

    if baseline_mode == "full":
        eval_horizons = eval_df["unique_id"].value_counts()
        base_preds = fit_predict_baselines(train_df, horizons=eval_horizons, freq="D", probabilistic=False)
        if not base_preds.empty:
            merged = merged.merge(base_preds, on=["unique_id", "ds"], how="left")

    if baseline_mode in {"full", "hurdle_only"}:
        local_params = fit_hurdle_local_lognormal(train_df)
        local_preds = predict_hurdle_local_lognormal(local_params, eval_df)
        local_preds = local_preds[["unique_id", "ds", "Hurdle-Local-LogNormal"]]

        global_params = fit_hurdle_global_lognormal(train_df)
        global_preds = predict_hurdle_global_lognormal(global_params, eval_df)
        global_preds = global_preds[["unique_id", "ds", "Hurdle-Global-LogNormal"]]

        merged = merged.merge(local_preds, on=["unique_id", "ds"], how="left")
        merged = merged.merge(global_preds, on=["unique_id", "ds"], how="left")
    return merged


def _run_online_point_fixed(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    hb_group_labels: pd.Series | None = None,
    hb_group_shrink_strength: float = 0.0,
    hb_item_variance_mode: str = "group",
    hb_variance_prior_df: float = 20.0,
    baseline_mode: str = "full",
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
    baseline_mode: str = "full",
) -> pd.DataFrame:
    step_outputs: list[pd.DataFrame] = []
    hb_state: TSBHBOnlineState | None = None
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

    for frame in iter_walk_forward_frames(init_set, eval_set, step_size=walk_step):
        if hb_state is not None:
            tsbhb_step = predict_online_tsb_hb(
                hb_state,
                frame.target,
                quantiles=None,
                include_hyper_uncertainty=False,
            ).rename(columns={"yhat": "TSB-HB-LogNormal"})
            merged_step = _predict_online_models_once(
                frame.history,
                frame.target,
                hb_group_labels=hb_group_labels,
                hb_group_shrink_strength=hb_group_shrink_strength,
                hb_item_variance_mode=hb_item_variance_mode,
                hb_variance_prior_df=hb_variance_prior_df,
                tsbhb_override=tsbhb_step,
                baseline_mode=baseline_mode,
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
    ap.add_argument("--walk-step", type=int, default=1, help="Block size (steps) for walk-forward protocol.")
    ap.add_argument("--baseline-mode", choices=["full", "hurdle_only", "hb_only"], default=None, help="Baseline set for both fixed and walk-forward: full=StatsForecast+hurdle, hurdle_only=hurdle+TSB-HB, hb_only=TSB-HB only.")
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
    ap.add_argument("--m5-hierarchy-mode", choices=["off", "on", "ablation"], default="ablation", help="M5 hierarchy usage for TSB-HB: off=global priors, on=hier priors, ablation=report both.")
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
        if args.protocol != "fixed":
            raise ValueError("M5 currently supports only --protocol fixed in run_point.")
        _run_m5_point(args, out_dir)
        return

    # Load & preprocess
    df_raw = load_online_retail(args.data)
    df = preprocess_online_retail(df_raw)

    # Split fixed origin
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    hb_group_labels = _build_regime_group_labels(init_set) if args.hb_regime_aware else None
    baseline_mode = args.baseline_mode or "full"
    hb_variance_prior_df = float(max(args.hb_variance_prior_df, 2.1))

    if args.protocol == "fixed":
        merged = _run_online_point_fixed(
            init_set,
            eval_set,
            hb_group_labels=hb_group_labels,
            hb_group_shrink_strength=args.hb_group_shrink_strength,
            hb_item_variance_mode=args.hb_item_variance_mode,
            hb_variance_prior_df=hb_variance_prior_df,
            baseline_mode=baseline_mode,
        )
    else:
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
        )

    model_cols = [c for c in merged.columns if c not in {"unique_id", "ds", "y"}]
    point_df = evaluate_point_models(init_set, merged, model_cols=model_cols)
    point_df.insert(0, "protocol", args.protocol)
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
