from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import set_seed, default_data_file, default_out_dir
from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from models.tsb_hb import fit_tsb_hb, predict_tsb_hb
from models.baselines import fit_predict_baselines

# Lazy import neuralforecast
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import DeepAR
    from neuralforecast.losses.pytorch import DistributionLoss
except ImportError:
    NeuralForecast = None


def _parse_horizons(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x.strip()]


def _rolling_forecast_over_eval_probabilistic(
    nf,
    base_model_col: str,
    quantile_cols: list[str],
    init_hist: pd.DataFrame,
    eval_set: pd.DataFrame,
    h: int,
) -> pd.DataFrame:
    """Forecast the full eval segment by chaining blocks of size h (Probabilistic Version).

    Pred-only chaining: we never refit and we never peek at realized evaluation targets.
    To move beyond the first h steps, we append the model's own point predictions as context.
    """
    if h <= 0:
        raise ValueError("h must be positive.")

    hist = init_hist[["unique_id", "ds", "y"]].copy()

    eval_idx = eval_set[["unique_id", "ds", "y"]].copy()
    eval_idx["k"] = eval_idx.groupby("unique_id").cumcount()
    k_max = int(eval_idx["k"].max()) if not eval_idx.empty else -1

    preds: list[pd.DataFrame] = []
    n_blocks = int(np.ceil((k_max + 1) / h)) if k_max >= 0 else 0
    all_pred_cols = [base_model_col] + quantile_cols

    for _ in range(n_blocks):
        # Predict the next h steps for ALL series
        block_fcst = nf.predict(df=hist).reset_index()
        
        for col in all_pred_cols:
            if col not in block_fcst.columns:
                raise KeyError(f"Forecast output missing expected column '{col}'. Got: {list(block_fcst.columns)}")

        block_out = block_fcst[["unique_id", "ds"] + all_pred_cols].copy()
        preds.append(block_out)

        # Advance the context to support the next block using the point prediction
        advance = block_out.rename(columns={base_model_col: "y"})[["unique_id", "ds", "y"]].copy()
        advance["y"] = advance["y"].fillna(0.0)

        hist = pd.concat([hist, advance], ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    if not preds:
        return pd.DataFrame(columns=["unique_id", "ds"] + all_pred_cols)
    all_preds = pd.concat(preds, ignore_index=True)
    # Keep only the timestamps that exist in the evaluation set
    return eval_set[["unique_id", "ds"]].merge(all_preds, on=["unique_id", "ds"], how="left")


def plot_calibration_curve(eval_merged: pd.DataFrame, quantiles: list[float], out_dir: Path):
    """繪製可靠度圖 (Calibration Curve / Reliability Diagram)"""
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    
    for model, dfm in eval_merged.groupby("model"):
        emp_coverages = []
        for q in quantiles:
            col = f"q_{q}"
            emp_cov = (dfm["y"] <= dfm[col]).mean()
            emp_coverages.append(emp_cov)
            
        plt.plot(quantiles, emp_coverages, marker='o', label=model)
        
    plt.xlabel("Nominal Coverage (Quantile)")
    plt.ylabel("Empirical Coverage")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "calibration_curve.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_pit_histogram(eval_merged: pd.DataFrame, quantiles: list[float], out_dir: Path):
    """繪製 PIT (Probability Integral Transform) 直方圖"""
    models = eval_merged["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1: axes = [axes]
    
    bins = [0.0] + quantiles + [1.0]
    
    for ax, model in zip(axes, models):
        dfm = eval_merged[eval_merged["model"] == model].copy()
        pit_values = []
        
        for _, row in dfm.iterrows():
            y = row["y"]
            if y <= row[f"q_{quantiles[0]}"]:
                pit_values.append(quantiles[0] / 2)
            elif y > row[f"q_{quantiles[-1]}"]:
                pit_values.append((1.0 + quantiles[-1]) / 2)
            else:
                for i in range(len(quantiles) - 1):
                    if row[f"q_{quantiles[i]}"] < y <= row[f"q_{quantiles[i+1]}"]:
                        pit_values.append((quantiles[i] + quantiles[i+1]) / 2)
                        break
                        
        sns.histplot(pit_values, bins=bins, stat="density", ax=ax, color="skyblue")
        ax.axhline(1.0, color='r', linestyle='--', label="Uniform (Ideal)")
        ax.set_title(f"PIT Histogram: {model}")
        ax.set_xlabel("PIT Value")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(out_dir / "pit_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=default_data_file())
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    # 增加與 run_deepar.py 一致的參數
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--init-ratio", type=float, default=1.0 / 3.0)
    ap.add_argument("--horizon", type=int, default=10, help="Block size for DeepAR rolling forecast")
    ap.add_argument("--input-size", type=int, default=14)
    ap.add_argument("--start-padding-enabled", action="store_true", default=True)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_online_retail(args.data)
    df = preprocess_online_retail(df_raw)
    
    # Ensure minimum history
    df["t"] = df.groupby("unique_id").cumcount()
    df["L"] = df.groupby("unique_id")["t"].transform("max") + 1
    df = df[df["L"] >= args.min_len].copy()

    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=args.init_ratio, min_len=args.min_len)

    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

    # ==========================================
    # 1. TSB-HB Probabilistic Forecast
    # ==========================================
    print("Fitting TSB-HB...")
    params = fit_tsb_hb(init_set)
    tsbhb_q = predict_tsb_hb(params, eval_set, quantiles=QUANTILES, n_samples=2000)
    tsbhb_q["model"] = "TSB-HB"

    # ==========================================
    # 2. StatsForecast Baselines (AutoARIMA/AutoTheta)
    # ==========================================
    print("Fitting StatsForecast baselines...")
    eval_h = eval_set["unique_id"].value_counts()
    sf_q = fit_predict_baselines(init_set, eval_h, freq="D", probabilistic=True, levels=[80, 50])
    
    cols_arima = ["AutoARIMA", "AutoARIMA-lo-80", "AutoARIMA-hi-80", "AutoARIMA-lo-50", "AutoARIMA-hi-50"]
    cols_theta = ["AutoTheta", "AutoTheta-lo-80", "AutoTheta-hi-80", "AutoTheta-lo-50", "AutoTheta-hi-50"]
    arima = sf_q[[c for c in cols_arima if c in sf_q.columns] + ["unique_id", "ds"]].copy()
    theta = sf_q[[c for c in cols_theta if c in sf_q.columns] + ["unique_id", "ds"]].copy()
    
    rename_map = {
        "AutoARIMA": "q_0.5", "AutoARIMA-lo-80": "q_0.1", "AutoARIMA-hi-80": "q_0.9",
        "AutoARIMA-lo-50": "q_0.25", "AutoARIMA-hi-50": "q_0.75",
        "AutoTheta": "q_0.5", "AutoTheta-lo-80": "q_0.1", "AutoTheta-hi-80": "q_0.9",
        "AutoTheta-lo-50": "q_0.25", "AutoTheta-hi-50": "q_0.75",
    }
    arima = arima.rename(columns=rename_map)
    theta = theta.rename(columns=rename_map)
    arima["model"] = "AutoARIMA"
    theta["model"] = "AutoTheta"

    # ==========================================
    # 3. NeuralForecast Baseline (DeepAR with Chaining)
    # ==========================================
    deepar_q_df = pd.DataFrame()
    if NeuralForecast is not None:
        print(f"Fitting DeepAR (Chaining with horizon={args.horizon})...")
        # 採用 NegativeBinomial 或 Normal 皆可，此處設定輸出 [50, 80] levels 對應我們的 Quantiles
        loss = DistributionLoss(distribution='NegativeBinomial', level=[50, 80])
        models = [
            DeepAR(
                h=args.horizon, 
                input_size=max(int(args.input_size), 1), 
                loss=loss, 
                scaler_type='robust', 
                start_padding_enabled=args.start_padding_enabled,
                max_steps=500
            )
        ]
        nf = NeuralForecast(models=models, freq='D')
        nf.fit(df=init_set[["unique_id", "ds", "y"]])

        base_col = "DeepAR"
        q_cols = ["DeepAR-lo-80", "DeepAR-hi-80", "DeepAR-lo-50", "DeepAR-hi-50"]
        
        # 執行包含機率欄位的滾動預測！
        nf_preds = _rolling_forecast_over_eval_probabilistic(
            nf=nf,
            base_model_col=base_col,
            quantile_cols=q_cols,
            init_hist=init_set,
            eval_set=eval_set,
            h=args.horizon
        )

        rename_nf = {
            "DeepAR": "q_0.5",
            "DeepAR-lo-80": "q_0.1", "DeepAR-hi-80": "q_0.9",
            "DeepAR-lo-50": "q_0.25", "DeepAR-hi-50": "q_0.75",
        }
        deepar_q_df = nf_preds.rename(columns=rename_nf)[["unique_id", "ds"] + list(rename_nf.values())]
        deepar_q_df["model"] = "DeepAR"

    # ==========================================
    # 4. Merge and Evaluate
    # ==========================================
    qcols = [f"q_{q}" for q in QUANTILES]
    dfs_to_concat = [tsbhb_q, arima, theta]
    if not deepar_q_df.empty:
        dfs_to_concat.append(deepar_q_df)

    for df_ in dfs_to_concat:
        for c in qcols:
            if c not in df_.columns:
                df_[c] = np.nan
        df_.dropna(subset=["unique_id", "ds"], inplace=True)

    all_q = pd.concat([d[["model", "unique_id", "ds"] + qcols] for d in dfs_to_concat], ignore_index=True)
    all_q.to_csv(out_dir / "prob_quantiles.csv", index=False)

    eval_merged = eval_set[["unique_id", "ds", "y"]].copy().merge(all_q, on=["unique_id", "ds"], how="inner")
    
    print("Generating Calibration Curve and PIT Histogram...")
    plot_calibration_curve(eval_merged, QUANTILES, out_dir)
    plot_pit_histogram(eval_merged, QUANTILES, out_dir)

    print("Computing Pinball Loss and CRPS...")
    rows = []
    for model, dfm in eval_merged.groupby("model"):
        model_pinballs = []
        for q in QUANTILES:
            col = f"q_{q}"
            dfx = dfm.dropna(subset=[col]).copy()
            err = dfx["y"] - dfx[col]
            loss = np.maximum(q * err, (q - 1) * err).mean()
            model_pinballs.append(loss)
            rows.append({"model": model, "quantile": q, "pinball": float(loss)})
        

    pinball_df = pd.DataFrame(rows)
    pinball_df.to_csv(out_dir / "prob_pinball.csv", index=False)
    pinball_df.to_csv(out_dir / "probabilistic_forecast_pinball_results.csv", index=False)
    
    print("Evaluation complete. Results and plots saved to:", out_dir)

if __name__ == "__main__":
    main()