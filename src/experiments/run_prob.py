# run_prob.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

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
from models.hurdle import fit_predict_hurdle
from metrics import coverage_rate, pit_values


# -----------------------
# Helpers
# -----------------------
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ds" in out.columns:
        out["ds"] = pd.to_datetime(out["ds"])
    return out


def _pinball_loss(y: np.ndarray, qhat: np.ndarray, q: float) -> float:
    # pinball(q) = mean( (q - 1_{y<qhat}) * (y - qhat) )
    y = np.asarray(y, dtype=float)
    qhat = np.asarray(qhat, dtype=float)
    diff = y - qhat
    return float(np.nanmean(np.maximum(q * diff, (q - 1.0) * diff)))


def _assert_prob_coverage(eval_merged: pd.DataFrame, quantiles: List[float]) -> None:
    """Hard-stop if any model is missing any quantile on any eval row (prevents unfair dropna eval)."""
    if eval_merged.empty:
        raise ValueError("[Prob Coverage] eval_merged is empty.")

    required_base = {"unique_id", "ds", "y", "model"}
    missing_base = required_base - set(eval_merged.columns)
    if missing_base:
        raise ValueError(f"[Prob Coverage] eval_merged missing columns: {missing_base}")

    qcols = [f"q_{q}" for q in quantiles] + ["prob_zero_predicted"]
    for qc in qcols:
        if qc not in eval_merged.columns:
            raise ValueError(f"[Prob Coverage] eval_merged missing quantile column: {qc}")

    # Ensure no duplicate (uid, ds, model)
    dup = eval_merged.duplicated(subset=["unique_id", "ds", "model"]).mean()
    if dup > 0:
        ex = eval_merged.loc[
            eval_merged.duplicated(subset=["unique_id", "ds", "model"]),
            ["unique_id", "ds", "model"],
        ].head(10)
        raise ValueError(f"[Prob Coverage] duplicated (unique_id, ds, model) rows found. Examples:\n{ex}")

    for model, dfm in eval_merged.groupby("model"):
        for qc in qcols:
            cov = float(dfm[qc].notna().mean())
            if cov < 1.0:
                miss = dfm.loc[dfm[qc].isna(), ["unique_id", "ds"]].head(10)
                raise ValueError(
                    f"[Prob Coverage] model={model}, {qc} coverage={cov:.4f} < 1.0.\n"
                    f"Missing examples:\n{miss}"
                )


def _evaluate_prob(eval_merged: pd.DataFrame, quantiles: List[float], out_dir: Path) -> None:
    """Compute pinball (per quantile), average pinball, coverage + interval width, PIT stats."""
    rows = []
    for model, dfm in eval_merged.groupby("model"):
        y = dfm["y"].values

        # Pinball per quantile
        pinballs = {}
        for q in quantiles:
            qhat = dfm[f"q_{q}"].values
            pinballs[q] = _pinball_loss(y, qhat, q)

        avg_pinball = float(np.mean(list(pinballs.values())))

        # Coverage / interval width for 80% and 50% central intervals
        cov_80 = coverage_rate(dfm, lower_q=0.1, upper_q=0.9, alpha=0.8)
        cov_50 = coverage_rate(dfm, lower_q=0.25, upper_q=0.75, alpha=0.5)

        # PIT summary
        pits = pit_values(dfm, quantiles=tuple(quantiles))
        pit_mean = float(np.mean(pits)) if len(pits) else float("nan")
        pit_var = float(np.var(pits)) if len(pits) else float("nan")

        row = {
            "model": model,
            "avg_pinball": avg_pinball,
            "pit_mean": pit_mean,
            "pit_var": pit_var,
            **{f"pinball_q{q}": v for q, v in pinballs.items()},
            **cov_80,
            **cov_50,
        }
        rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values("avg_pinball")
    metrics_df.to_csv(out_dir / "prob_metrics.csv", index=False)

    # Save per-row PIT values (optional but useful for debugging plots)
    pit_dump = []
    for model, dfm in eval_merged.groupby("model"):
        pits = pit_values(dfm, quantiles=tuple(quantiles))
        pit_dump.append(pd.DataFrame({"model": model, "pit": pits}))
    pd.concat(pit_dump, ignore_index=True).to_csv(out_dir / "pit_values.csv", index=False)


def _select_n_series(df: pd.DataFrame, n_series: int, seed: int = 42) -> pd.DataFrame:
    """Deterministic selection of series for prob experiments."""
    uids = sorted(df["unique_id"].unique())
    if len(uids) <= n_series:
        return df
    # deterministic sampling based on seed
    rng = np.random.default_rng(seed)
    chosen = rng.choice(uids, size=n_series, replace=False)
    chosen = set(chosen.tolist())
    return df[df["unique_id"].isin(chosen)].copy()


# -----------------------
# Prob Baselines
# -----------------------
def _predict_statsforecast_intervals_as_quantiles(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    freq: str,
    quantiles: List[float],
) -> pd.DataFrame:
    """
    Prob baselines via StatsForecast prediction intervals.
    We map central intervals to approximate quantiles:
      level=80 -> q0.1/q0.9, level=50 -> q0.25/q0.75
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoTheta

    init_set = init_set[["unique_id", "ds", "y"]].copy()
    eval_set = eval_set[["unique_id", "ds"]].copy()

    sf = StatsForecast(models=[AutoARIMA(), AutoTheta()], freq=freq, n_jobs=-1)
    sf.fit(init_set)

    max_h = int(eval_set.groupby("unique_id").size().max())

    # levels is in percentages
    fcst = sf.predict(h=max_h, level=[80, 50]).reset_index()
    fcst = _ensure_datetime(fcst)
    eval_set = _ensure_datetime(eval_set)

    # Align to eval index (LEFT merge, never inner)
    fcst = eval_set.merge(fcst, on=["unique_id", "ds"], how="left")

    # Expected columns (StatsForecast naming):
    # AutoARIMA, AutoARIMA-lo-80, AutoARIMA-hi-80, AutoARIMA-lo-50, AutoARIMA-hi-50
    # AutoTheta, AutoTheta-lo-80, ...
    out_parts = []
    for model in ["AutoARIMA", "AutoTheta"]:
        cols_needed = [
            model,
            f"{model}-lo-80",
            f"{model}-hi-80",
            f"{model}-lo-50",
            f"{model}-hi-50",
        ]
        missing = [c for c in cols_needed if c not in fcst.columns]
        if missing:
            raise ValueError(f"[StatsForecast] Missing columns for {model}: {missing}")

        dfm = fcst[["unique_id", "ds"] + cols_needed].copy()
        # Approx quantiles
        dfm[f"q_{0.1}"] = dfm[f"{model}-lo-80"]
        dfm[f"q_{0.9}"] = dfm[f"{model}-hi-80"]
        dfm[f"q_{0.25}"] = dfm[f"{model}-lo-50"]
        dfm[f"q_{0.75}"] = dfm[f"{model}-hi-50"]
        dfm[f"q_{0.5}"] = dfm[model]  # median approx = point forecast

        # prob_zero_predicted is not provided by these baselines; set to NaN then fill later?
        # To keep coverage strict, we set a defined value. Use empirical from init_set.
        occ = (init_set["y"] > 0).astype(int)
        p_hat = occ.groupby(init_set["unique_id"]).mean()
        dfm["prob_zero_predicted"] = (1.0 - p_hat.reindex(dfm["unique_id"]).values)

        keep = ["unique_id", "ds"] + [f"q_{q}" for q in quantiles] + ["prob_zero_predicted"]
        dfm = dfm[keep]
        dfm["model"] = model
        out_parts.append(dfm)

    out = pd.concat(out_parts, ignore_index=True)
    return out


def _predict_deepar_quantiles(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    freq: str,
    quantiles: List[float],
    horizon: int,
    seed: int,
) -> pd.DataFrame:
    """
    DeepAR probabilistic baseline using NeuralForecast.
    This version is deliberately simple: train once, predict for max_h, align to eval via left merge.
    """
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import DeepAR
    except Exception as e:
        raise RuntimeError(
            "NeuralForecast is required for DeepAR baseline. "
            "Install neuralforecast or disable DeepAR in the script."
        ) from e

    init_set = init_set[["unique_id", "ds", "y"]].copy()
    eval_set = eval_set[["unique_id", "ds"]].copy()
    init_set = _ensure_datetime(init_set)
    eval_set = _ensure_datetime(eval_set)

    # Train DeepAR with provided horizon
    models = [DeepAR(h=horizon, max_steps=1000, random_seed=seed)]
    nf = NeuralForecast(models=models, freq=freq)
    nf.fit(df=init_set)

    # Predict to max_h once, align to eval via left merge
    max_h = int(eval_set.groupby("unique_id").size().max())
    # NeuralForecast predict uses each model's h; to get max_h, we can predict multiple blocks.
    # Minimal approach: set horizon=max_h at training time. But that can be heavy.
    # Instead, we assume eval horizon == horizon for prob experiments.
    if max_h != horizon:
        raise ValueError(
            f"[DeepAR] eval max horizon ({max_h}) != DeepAR horizon ({horizon}). "
            "For now, set --horizon to eval max horizon for prob runs."
        )

    pred = nf.predict().reset_index()  # returns horizon rows per series
    pred = _ensure_datetime(pred)

    # NeuralForecast returns columns like "DeepAR" as mean; and quantiles if requested via `quantiles` in model config.
    # To keep consistent, we sample by assuming normal around mean is NOT correct. Prefer native quantile outputs.
    # If your DeepAR config doesn't output quantiles, this baseline isn't valid.
    needed = [f"DeepAR-q{int(q*100)}" for q in quantiles]
    if not all(c in pred.columns for c in needed):
        raise ValueError(
            "[DeepAR] Expected quantile columns not found. "
            "Configure DeepAR to output quantiles (NeuralForecast supports prediction intervals/quantiles depending on version). "
            f"Missing: {[c for c in needed if c not in pred.columns]}"
        )

    dfm = eval_set.merge(pred, on=["unique_id", "ds"], how="left")
    for q in quantiles:
        dfm[f"q_{q}"] = dfm[f"DeepAR-q{int(q*100)}"]

    # prob_zero_predicted not natively provided; approximate as fraction of zeros in train
    p_hat = (init_set["y"] > 0).groupby(init_set["unique_id"]).mean()
    dfm["prob_zero_predicted"] = 1.0 - p_hat.reindex(dfm["unique_id"]).values

    keep = ["unique_id", "ds"] + [f"q_{q}" for q in quantiles] + ["prob_zero_predicted"]
    dfm = dfm[keep]
    dfm["model"] = "DeepAR"
    return dfm


# -----------------------
# Main runners
# -----------------------
def _run_online_retail_prob(args: argparse.Namespace, out_dir: Path) -> None:
    df = preprocess_online_retail(load_online_retail(default_data_file()))
    df = _ensure_datetime(df)

    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)
    init_set = _ensure_datetime(init_set)
    eval_set = _ensure_datetime(eval_set)

    # Select only N series for probabilistic experiments
    init_set_small = _select_n_series(init_set, n_series=args.n_series, seed=args.seed)
    eval_set_small = eval_set[eval_set["unique_id"].isin(init_set_small["unique_id"].unique())].copy()

    # Fit TSB-HB-RA and predict quantiles
    params = fit_tsb_hb(init_set_small, n_regimes=args.n_regimes)
    tsbhb_q = predict_tsb_hb(params, eval_set_small, quantiles=QUANTILES)
    tsbhb_q = _ensure_datetime(tsbhb_q)
    tsbhb_q["model"] = "TSB-HB-RA"

    # Hurdle baseline quantiles
    hurdle_q = fit_predict_hurdle(init_set_small, eval_set_small, quantiles=QUANTILES, n_samples=args.n_samples)
    hurdle_q = _ensure_datetime(hurdle_q)
    hurdle_q["model"] = "Hurdle"

    # StatsForecast probabilistic baselines (interval -> approx quantiles)
    sf_q = _predict_statsforecast_intervals_as_quantiles(
        init_set_small, eval_set_small, freq="D", quantiles=QUANTILES
    )

    # Optional DeepAR (disabled by default for stability)
    parts = [tsbhb_q, hurdle_q, sf_q]
    if args.use_deepar:
        deepar_q = _predict_deepar_quantiles(
            init_set_small,
            eval_set_small,
            freq="D",
            quantiles=QUANTILES,
            horizon=args.horizon,
            seed=args.seed,
        )
        parts.append(deepar_q)

    # Build long table and align to eval index (LEFT merge, never inner)
    all_q = pd.concat(parts, ignore_index=True)

    # Ensure eval index duplicated per model after merge
    eval_index = eval_set_small[["unique_id", "ds", "y"]].copy()
    eval_index["ds"] = pd.to_datetime(eval_index["ds"])

    eval_merged = eval_index.merge(all_q, on=["unique_id", "ds"], how="left")

    # Hard-stop if any model misses any quantile on any eval row
    _assert_prob_coverage(eval_merged, QUANTILES)

    # Save aligned quantiles
    eval_merged.to_csv(out_dir / "prob_quantiles_aligned.csv", index=False)

    # Evaluate
    _evaluate_prob(eval_merged, QUANTILES, out_dir)
    print(f"[Done] Prob metrics saved to {out_dir}")


def _run_m5_prob(args: argparse.Namespace, out_dir: Path) -> None:
    # If you truly want M5 prob on 100 series, this runner mirrors online_retail.
    sales_df, calendar_df = load_m5_long(args.m5_sales, args.m5_calendar)
    df = preprocess_m5(sales_df, calendar_df, sample_size=args.m5_sample_size)
    df = _ensure_datetime(df)

    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=2.0 / 3.0, min_len=1)
    init_set = _ensure_datetime(init_set)
    eval_set = _ensure_datetime(eval_set)

    init_set_small = _select_n_series(init_set, n_series=args.n_series, seed=args.seed)
    eval_set_small = eval_set[eval_set["unique_id"].isin(init_set_small["unique_id"].unique())].copy()

    params = fit_tsb_hb(init_set_small, n_regimes=args.n_regimes)
    tsbhb_q = predict_tsb_hb(params, eval_set_small, quantiles=QUANTILES)
    tsbhb_q = _ensure_datetime(tsbhb_q)
    tsbhb_q["model"] = "TSB-HB-RA"

    hurdle_q = fit_predict_hurdle(init_set_small, eval_set_small, quantiles=QUANTILES, n_samples=args.n_samples)
    hurdle_q = _ensure_datetime(hurdle_q)
    hurdle_q["model"] = "Hurdle"

    sf_q = _predict_statsforecast_intervals_as_quantiles(
        init_set_small, eval_set_small, freq="D", quantiles=QUANTILES
    )

    parts = [tsbhb_q, hurdle_q, sf_q]
    if args.use_deepar:
        deepar_q = _predict_deepar_quantiles(
            init_set_small,
            eval_set_small,
            freq="D",
            quantiles=QUANTILES,
            horizon=args.horizon,
            seed=args.seed,
        )
        parts.append(deepar_q)

    all_q = pd.concat(parts, ignore_index=True)

    eval_index = eval_set_small[["unique_id", "ds", "y"]].copy()
    eval_index["ds"] = pd.to_datetime(eval_index["ds"])

    eval_merged = eval_index.merge(all_q, on=["unique_id", "ds"], how="left")
    _assert_prob_coverage(eval_merged, QUANTILES)

    eval_merged.to_csv(out_dir / "prob_quantiles_aligned_m5.csv", index=False)
    _evaluate_prob(eval_merged, QUANTILES, out_dir)
    print(f"[Done] M5 prob metrics saved to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["online_retail", "m5"], default="online_retail")
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)

    # Prob controls
    ap.add_argument("--n-series", type=int, default=100, help="Number of series to evaluate probabilistically.")
    ap.add_argument("--n-samples", type=int, default=2000, help="MC samples for sampling-based models (TSB-HB/Hurdle).")
    ap.add_argument("--n-regimes", type=int, default=4, help="Number of regimes for TSB-HB GMM clustering.")

    # DeepAR controls (optional)
    ap.add_argument("--use-deepar", action="store_true", help="Enable DeepAR baseline (requires neuralforecast).")
    ap.add_argument("--horizon", type=int, default=30, help="Forecast horizon for DeepAR (must match eval max horizon).")

    # M5 inputs
    ap.add_argument("--m5-sales", type=Path, default=default_m5_sales_file())
    ap.add_argument("--m5-calendar", type=Path, default=default_m5_calendar_file())
    ap.add_argument("--m5-sample-size", type=int, default=None)

    args = ap.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "m5":
        _run_m5_prob(args, args.out)
    else:
        _run_online_retail_prob(args, args.out)


if __name__ == "__main__":
    main()