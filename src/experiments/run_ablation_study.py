from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from utils import set_seed, default_data_file, default_out_dir
from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from models.tsb_hb import TSBHBParams, fit_tsb_hb, _analytic_mixture_quantile
from models.hurdle_baselines import (
    fit_hurdle_local_lognormal,
    predict_hurdle_local_lognormal,
)
from metrics import (
    mae,
    rmse,
    rmsse,
    compute_adi_cv2,
    classify_adi_cv2,
    coverage_rate,
)
from experiments.run_prob import (
    QUANTILES,
    _fit_hb_location_scale,
    _apply_hb_calibration,
    _enforce_monotonic_quantiles,
)


def _build_regime_group_labels(train_df: pd.DataFrame) -> pd.Series | None:
    """ADI/CV²-based demand regime labels, matching other experiments."""
    feats = compute_adi_cv2(train_df)
    if feats.empty:
        return None
    feats["category"] = feats.apply(classify_adi_cv2, axis=1)
    return feats.set_index("unique_id")["category"].astype(str)


def _compute_local_occ_and_size_stats(init_set: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per-series local MLE for occurrence p and LogNormal size."""
    init_local = init_set.copy()
    init_local["occ"] = (init_local["y"] > 0).astype(int)
    init_local["size"] = np.where(init_local["occ"] == 1, init_local["y"].astype(float), np.nan)
    init_local["log_size"] = np.log(init_local["size"])

    g = init_local.groupby("unique_id")
    s = g["occ"].sum()
    n = g["ds"].nunique()

    # Local p MLE
    p_mle = (s / n).astype(float)

    # Local LogNormal mean
    item_stats = g["log_size"].agg(mean_log="mean", var_log="var").fillna(0.0)
    size_mle_ln = np.exp(item_stats["mean_log"] + (item_stats["var_log"] / 2.0))

    # Replace any pathological values with 0
    p_mle = p_mle.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    size_mle_ln = size_mle_ln.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return p_mle, size_mle_ln


def _point_ablation_table(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    params: TSBHBParams,
) -> pd.DataFrame:
    """Build 'Contribution to Point Forecasting' table."""
    p_mle, size_mle_ln = _compute_local_occ_and_size_stats(init_set)

    # Map series-level parameters to evaluation frame
    eva = eval_set[["unique_id", "ds", "y"]].copy()

    # Local + Local
    local_local = (p_mle * size_mle_ln).rename("Local+Local")
    eva = eva.merge(local_local, on="unique_id", how="left")

    # HB(p) + Local(size)
    size_local = size_mle_ln
    p_hb = params.p_posterior
    hb_p_local_size = (p_hb * size_local).rename("HB(p)+Local(size)")
    eva = eva.merge(hb_p_local_size, on="unique_id", how="left")

    # Local(p) + HB(size)
    # HB size mean under LogNormal: exp(mu + sigma^2 / 2)
    size_hb_mean = np.exp(params.shrunk_mean_log + params.sigma_sq_process / 2.0)
    local_p_hb_size = (p_mle * size_hb_mean).rename("Local(p)+HB(size)")
    eva = eva.merge(local_p_hb_size, on="unique_id", how="left")

    # Fill any missing predictions with 0
    for col in ["Local+Local", "HB(p)+Local(size)", "Local(p)+HB(size)"]:
        eva[col] = eva[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    results: list[dict] = []
    for col in ["Local+Local", "HB(p)+Local(size)", "Local(p)+HB(size)"]:
        tmp = eva[["unique_id", "y", col]].rename(columns={col: "y_pred"}).copy()
        y = tmp["y"].to_numpy(dtype=float)
        y_pred = tmp["y_pred"].to_numpy(dtype=float)
        results.append(
            {
                "Variant": col,
                "MAE": mae(y, y_pred),
                "RMSE": rmse(y, y_pred),
                "RMSSE": rmsse(init_set, tmp),
            }
        )
    return pd.DataFrame(results)


def _pinball_mean(eval_df: pd.DataFrame, quantiles: List[float]) -> float:
    """Mean pinball loss across requested quantiles."""
    losses: list[float] = []
    for q in quantiles:
        col = f"q_{q}"
        if col not in eval_df.columns:
            continue
        dfx = eval_df.dropna(subset=[col])
        if dfx.empty:
            continue
        err = dfx["y"] - dfx[col]
        loss = np.maximum(q * err, (q - 1) * err).mean()
        losses.append(float(loss))
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def _tsbhb_quantiles_from_params(
    params: TSBHBParams,
    eval_set: pd.DataFrame,
    quantiles: List[float],
    use_process_variance: bool,
) -> pd.DataFrame:
    """Construct TSB-HB quantiles with or without process variance term."""
    out = eval_set[["unique_id", "ds"]].copy()
    uids = out["unique_id"]

    p_arr = pd.Series(uids).map(params.p_posterior).fillna(params.p_posterior.mean()).to_numpy(dtype=float)
    p_arr = np.clip(p_arr, 0.0, 1.0)

    mu_arr = pd.Series(uids).map(params.shrunk_mean_log).fillna(params.shrunk_mean_log.mean()).to_numpy(dtype=float)
    sigma_arr = pd.Series(uids).map(params.sigma_sq_process).fillna(params.sigma_sq_process.mean()).to_numpy(
        dtype=float
    )
    var_mu_arr = pd.Series(uids).map(params.posterior_var_mu).fillna(params.posterior_var_mu.mean()).to_numpy(
        dtype=float
    )

    if use_process_variance:
        pred_var_arr = np.maximum(sigma_arr + var_mu_arr, 1e-9)
    else:
        pred_var_arr = np.maximum(var_mu_arr, 1e-9)

    for q in quantiles:
        out[f"q_{q}"] = _analytic_mixture_quantile(p_arr, mu_arr, pred_var_arr, q)
    return out


def _prob_ablation_table(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    params: TSBHBParams,
    hb_calibration: dict[str, float | str] | None,
) -> pd.DataFrame:
    """Build 'Contribution to Probabilistic Quality' table."""
    eva_base = eval_set[["unique_id", "ds", "y"]].copy()

    # Full: HB(p)+HB(size) with process variance + calibration
    tsbhb_full_raw = _tsbhb_quantiles_from_params(params, eval_set, QUANTILES, use_process_variance=True)
    tsbhb_full = _apply_hb_calibration(
        tsbhb_full_raw,
        quantiles=QUANTILES,
        calibration=hb_calibration,
    )

    # Full w/o process variance: predictive variance uses only posterior mean variance, same calibration
    tsbhb_noproc_raw = _tsbhb_quantiles_from_params(params, eval_set, QUANTILES, use_process_variance=False)
    tsbhb_noproc = _apply_hb_calibration(
        tsbhb_noproc_raw,
        quantiles=QUANTILES,
        calibration=hb_calibration,
    )

    # Full w/o calibration: keep process variance, no location-scale calibration
    tsbhb_nocal = _enforce_monotonic_quantiles(tsbhb_full_raw, quantiles=QUANTILES)

    # Local-Hurdle baseline
    local_params = fit_hurdle_local_lognormal(init_set)
    hurdle_local = predict_hurdle_local_lognormal(local_params, eval_set, quantiles=QUANTILES).copy()
    hurdle_local = _enforce_monotonic_quantiles(hurdle_local, quantiles=QUANTILES)

    variants: list[tuple[str, pd.DataFrame]] = [
        ("Full", tsbhb_full),
        ("Full w/o process variance", tsbhb_noproc),
        ("Full w/o calibration", tsbhb_nocal),
        ("Local-Hurdle", hurdle_local),
    ]

    rows: list[dict] = []
    for name, qdf in variants:
        merged = eva_base.merge(qdf, on=["unique_id", "ds"], how="inner")
        if merged.empty:
            rows.append(
                {
                    "Variant": name,
                    "MeanPinball": float("nan"),
                    "Coverage@80": float("nan"),
                    "AIW@80": float("nan"),
                }
            )
            continue

        pin = _pinball_mean(merged, quantiles=QUANTILES)
        cov80 = coverage_rate(merged, 0.1, 0.9, 0.8)
        rows.append(
            {
                "Variant": name,
                "MeanPinball": pin,
                "Coverage@80": float(cov80["Coverage@80"]),
                "AIW@80": float(cov80["AIW@80"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=default_data_file())
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--init-ratio", type=float, default=1.0 / 3.0)
    ap.add_argument(
        "--hb-regime-aware",
        dest="hb_regime_aware",
        action="store_true",
        default=True,
        help="Use ADI/CV^2 regime-aware HB priors.",
    )
    ap.add_argument(
        "--no-hb-regime-aware",
        dest="hb_regime_aware",
        action="store_false",
        help="Disable regime-aware priors and use global HB priors.",
    )
    ap.add_argument(
        "--hb-group-shrink-strength",
        type=float,
        default=0.0,
        help="Extra shrink from group-level hyperparameters back to global hyperparameters (0 disables).",
    )
    ap.add_argument(
        "--hb-item-variance-mode",
        choices=["group", "conjugate"],
        default="conjugate",
        help="Process variance mode for size: group or conjugate.",
    )
    ap.add_argument(
        "--hb-variance-prior-df",
        type=float,
        default=20.0,
        help="Prior degrees of freedom for conjugate variance model (larger = stronger shrinkage).",
    )
    ap.add_argument(
        "--hb-calibration-ratio",
        type=float,
        default=0.20,
        help="Tail fraction of init_set reserved to fit calibration shifts.",
    )
    ap.add_argument(
        "--hb-calibration-samples",
        type=int,
        default=1000,
        help="Monte Carlo samples when fitting calibration shifts.",
    )
    ap.add_argument(
        "--hb-calibration-lambda-min",
        type=float,
        default=0.60,
        help="Lower bound for location-scale calibration lambda.",
    )
    ap.add_argument(
        "--hb-calibration-lambda-max",
        type=float,
        default=1.20,
        help="Upper bound for location-scale calibration lambda.",
    )
    ap.add_argument(
        "--hb-calibration-lambda-steps",
        type=int,
        default=13,
        help="Number of lambda grid points for location-scale calibration.",
    )
    ap.add_argument(
        "--hb-calibration-coverage-weight",
        type=float,
        default=0.50,
        help="Penalty weight on coverage gaps when fitting location-scale calibration.",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Online Retail dataset and split
    df_raw = load_online_retail(args.data)
    df = preprocess_online_retail(df_raw)
    init_set, eval_set = train_eval_split_fixed_origin(
        df,
        init_ratio=args.init_ratio,
        min_len=args.min_len,
    )
    if eval_set.empty:
        raise ValueError("Evaluation set is empty; verify split parameters and input data.")

    # Fit TSB-HB once (no bootstrap hyper-uncertainty; we ablate process variance explicitly)
    hb_group_labels = _build_regime_group_labels(init_set) if args.hb_regime_aware else None
    hb_variance_prior_df = float(max(args.hb_variance_prior_df, 2.1))
    params = fit_tsb_hb(
        init_set,
        group_labels=hb_group_labels,
        bootstrap_draws=0,
        bootstrap_seed=args.seed,
        group_shrink_strength=float(max(args.hb_group_shrink_strength, 0.0)),
        item_variance_mode=str(args.hb_item_variance_mode),
        item_variance_shrink_strength=hb_variance_prior_df,
    )

    # Calibration parameters for "Full" TSB-HB
    hb_calibration = _fit_hb_location_scale(
        init_set=init_set,
        quantiles=QUANTILES,
        hb_regime_aware=bool(args.hb_regime_aware),
        hb_group_shrink_strength=float(max(args.hb_group_shrink_strength, 0.0)),
        hb_item_variance_mode=str(args.hb_item_variance_mode),
        hb_variance_prior_df=hb_variance_prior_df,
        hb_bootstrap_draws=0,
        hb_bootstrap_seed=args.seed,
        hb_use_hyper_uncertainty=False,
        calibration_ratio=float(args.hb_calibration_ratio),
        calibration_samples=max(int(args.hb_calibration_samples), 200),
        lambda_min=float(args.hb_calibration_lambda_min),
        lambda_max=float(args.hb_calibration_lambda_max),
        lambda_steps=max(int(args.hb_calibration_lambda_steps), 2),
        coverage_weight=float(max(args.hb_calibration_coverage_weight, 0.0)),
    )

    # Point forecasting ablation table
    point_table = _point_ablation_table(init_set, eval_set, params)
    point_table.to_csv(out_dir / "ablation_point_contributions.csv", index=False)

    # Probabilistic quality ablation table
    prob_table = _prob_ablation_table(init_set, eval_set, params, hb_calibration=hb_calibration)
    prob_table.to_csv(out_dir / "ablation_prob_contributions.csv", index=False)

    print("Ablation study complete. Results saved to:", out_dir)


if __name__ == "__main__":
    main()

