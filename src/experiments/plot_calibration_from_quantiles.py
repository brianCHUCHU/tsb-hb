"""
Redraw calibration curve from existing prob_quantiles.csv and the same dataset/split.

Uses the same data load and train_eval_split_fixed_origin as run_prob so that
eval_set (unique_id, ds, y) matches the run that produced prob_quantiles.csv.
Only the selected models are plotted (TSB-HB, AutoARIMA, AutoTheta, CP-TSB, CP-IMAPA).

Usage:
  python -m experiments.plot_calibration_from_quantiles --out-dir ../outputs/prob_full_cal_new --dataset online
  python -m experiments.plot_calibration_from_quantiles --out-dir ../outputs/prob_m5_fixed_new --dataset m5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import (
    set_seed,
    default_data_file,
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
from experiments.run_prob import plot_calibration_curve, QUANTILES, _qcols

# Models to show on the calibration curve (rest are omitted)
CALIBRATION_PLOT_MODELS = ["TSB-HB", "AutoARIMA", "AutoTheta", "CP-TSB", "CP-IMAPA"]


def _load_quantiles_filtered(quantiles_path: Path, models: list[str], chunksize: int = 1_000_000):
    """Load prob_quantiles.csv, keeping only rows for the given models (chunked for large files)."""
    qcols = _qcols(QUANTILES)
    usecols = ["model", "unique_id", "ds"] + qcols
    try:
        # Try chunked read for large files
        parts = []
        for chunk in _read_csv_chunks(quantiles_path, usecols=usecols, chunksize=chunksize):
            chunk = chunk[chunk["model"].isin(models)]
            if not chunk.empty:
                parts.append(chunk)
        if not parts:
            return None
        return pd.concat(parts, ignore_index=True)
    except Exception:
        # Fallback: single read (for smaller files)
        dtype = {c: str for c in usecols if c in ("unique_id", "ds")}
        df = pd.read_csv(quantiles_path, usecols=usecols, dtype=dtype, low_memory=False)
        return df[df["model"].isin(models)]


def _read_csv_chunks(path: Path, usecols: list[str], chunksize: int):
    # Keep ds as string to avoid mixed-type warning; caller converts to datetime if needed
    dtype = {c: str for c in usecols if c in ("unique_id", "ds")}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype=dtype, low_memory=False):
        yield chunk


def main() -> None:
    ap = argparse.ArgumentParser(description="Redraw calibration curve from prob_quantiles.csv")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory containing prob_quantiles.csv")
    ap.add_argument("--dataset", choices=["online", "m5"], default="online")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--m5-sales", type=Path, default=None)
    ap.add_argument("--m5-calendar", type=Path, default=None)
    ap.add_argument("--m5-sample-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-ratio", type=float, default=1.0 / 3.0)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--chunksize", type=int, default=1_500_000, help="Chunk size when reading large quantiles CSV")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = args.out_dir
    quantiles_path = out_dir / "prob_quantiles.csv"
    if not quantiles_path.exists():
        raise FileNotFoundError(f"prob_quantiles.csv not found: {quantiles_path}")

    if args.dataset == "online":
        data_path = args.data or default_data_file()
        df_raw = load_online_retail(data_path)
        df = preprocess_online_retail(df_raw)
    else:
        sales_path = args.m5_sales or default_m5_sales_file()
        cal_path = args.m5_calendar or default_m5_calendar_file()
        sales_df, calendar_df = load_m5_long(sales_path, cal_path)
        df = preprocess_m5(sales_df, calendar_df, sample_size=args.m5_sample_size)

    _, eval_set = train_eval_split_fixed_origin(df, init_ratio=args.init_ratio, min_len=args.min_len)
    if eval_set.empty:
        raise ValueError("Evaluation set is empty; check split parameters and data.")

    all_q = _load_quantiles_filtered(quantiles_path, CALIBRATION_PLOT_MODELS, chunksize=args.chunksize)
    if all_q is None or all_q.empty:
        raise ValueError(
            f"No rows found in {quantiles_path} for models {CALIBRATION_PLOT_MODELS}. "
            "Check that prob_quantiles.csv contains these model names."
        )
    # Align ds dtypes for merge (CSV reads ds as object)
    if all_q["ds"].dtype == object or pd.api.types.is_string_dtype(all_q["ds"]):
        all_q["ds"] = pd.to_datetime(all_q["ds"])

    eval_merged = eval_set[["unique_id", "ds", "y"]].merge(all_q, on=["unique_id", "ds"], how="inner")
    if eval_merged.empty:
        raise ValueError("Merge of eval_set and quantiles produced no rows; check that (unique_id, ds) align.")

    plot_calibration_curve(
        eval_merged,
        QUANTILES,
        out_dir,
        models_to_plot=CALIBRATION_PLOT_MODELS,
    )
    print("Calibration curve saved to", out_dir / "calibration_curve.png")


if __name__ == "__main__":
    main()
