"""
搜索最佳 Regime 數量
遍歷不同的 n_regimes（2-8），評估模型性能，找出最佳配置
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loading import (
    load_online_retail,
    preprocess_online_retail,
    train_eval_split_fixed_origin,
)
from metrics import mae, me, rmse, rmsse, wrmsse
from models.tsb_hb import fit_tsb_hb, predict_tsb_hb
from utils import default_data_file, default_out_dir, set_seed


def evaluate_single_regime(
    n_regimes: int, 
    init_set: pd.DataFrame, 
    eval_set: pd.DataFrame,
    verbose: bool = True
) -> dict:
    """評估單個 regime 配置的性能"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing n_regimes = {n_regimes}")
        print(f"{'='*60}")
    
    # 計時
    t_start = time.time()
    
    # 訓練模型
    params = fit_tsb_hb(init_set, n_regimes=n_regimes)
    t_fit = time.time() - t_start
    
    # 點預測
    t_pred_start = time.time()
    tsbhb_point = predict_tsb_hb(params, eval_set, quantiles=None)
    t_pred = time.time() - t_pred_start
    
    # 合併真實值與預測值
    merged = eval_set[["unique_id", "ds", "y"]].merge(
        tsbhb_point, on=["unique_id", "ds"], how="left"
    )
    merged = merged.rename(columns={"yhat": "y_pred"})
    merged = merged.dropna(subset=["y_pred"])
    
    # 計算評估指標
    y_true = merged["y"].values
    y_pred = merged["y_pred"].values
    
    metrics = {
        "n_regimes": n_regimes,
        "ME": me(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "RMSSE": rmsse(init_set, merged),
        "WRMSSE": wrmsse(init_set, merged),
        "fit_time": t_fit,
        "pred_time": t_pred,
        "total_time": t_fit + t_pred,
        "n_items": len(params.p_mean),
    }
    
    if verbose:
        print(f"\nResults for n_regimes = {n_regimes}:")
        print(f"  MAE:    {metrics['MAE']:.4f}")
        print(f"  RMSE:   {metrics['RMSE']:.4f}")
        print(f"  RMSSE:  {metrics['RMSSE']:.4f}")
        print(f"  WRMSSE: {metrics['WRMSSE']:.4f}")
        print(f"  Fit time:  {metrics['fit_time']:.2f}s")
        print(f"  Pred time: {metrics['pred_time']:.2f}s")
    
    return metrics


def plot_regime_comparison(results_df: pd.DataFrame, out_dir: Path):
    """繪製不同 regime 數量的性能對比圖"""
    
    # 設置樣式
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 創建 2x2 子圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. MAE vs n_regimes
    ax = axes[0, 0]
    ax.plot(results_df["n_regimes"], results_df["MAE"], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel("Number of Regimes", fontsize=11)
    ax.set_ylabel("MAE", fontsize=11)
    ax.set_title("Mean Absolute Error", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    best_idx = results_df["MAE"].idxmin()
    ax.axvline(results_df.loc[best_idx, "n_regimes"], color='r', linestyle='--', 
               alpha=0.7, label=f'Best: {results_df.loc[best_idx, "n_regimes"]:.0f}')
    ax.legend()
    
    # 2. RMSSE vs n_regimes
    ax = axes[0, 1]
    ax.plot(results_df["n_regimes"], results_df["RMSSE"], 'o-', linewidth=2, markersize=8, color='#ff7f0e')
    ax.set_xlabel("Number of Regimes", fontsize=11)
    ax.set_ylabel("RMSSE", fontsize=11)
    ax.set_title("Root Mean Squared Scaled Error", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    best_idx = results_df["RMSSE"].idxmin()
    ax.axvline(results_df.loc[best_idx, "n_regimes"], color='r', linestyle='--', 
               alpha=0.7, label=f'Best: {results_df.loc[best_idx, "n_regimes"]:.0f}')
    ax.legend()
    
    # 3. WRMSSE vs n_regimes
    ax = axes[1, 0]
    ax.plot(results_df["n_regimes"], results_df["WRMSSE"], 'o-', linewidth=2, markersize=8, color='#2ca02c')
    ax.set_xlabel("Number of Regimes", fontsize=11)
    ax.set_ylabel("WRMSSE", fontsize=11)
    ax.set_title("Weighted Root Mean Squared Scaled Error", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    best_idx = results_df["WRMSSE"].idxmin()
    ax.axvline(results_df.loc[best_idx, "n_regimes"], color='r', linestyle='--', 
               alpha=0.7, label=f'Best: {results_df.loc[best_idx, "n_regimes"]:.0f}')
    ax.legend()
    
    # 4. Training Time vs n_regimes
    ax = axes[1, 1]
    ax.plot(results_df["n_regimes"], results_df["fit_time"], 'o-', linewidth=2, markersize=8, color='#d62728')
    ax.set_xlabel("Number of Regimes", fontsize=11)
    ax.set_ylabel("Training Time (seconds)", fontsize=11)
    ax.set_title("Computational Cost", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("TSB-HB Regime-Aware: Performance vs Number of Regimes", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_dir / "regime_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Comparison plot saved to {out_dir / 'regime_comparison.png'}")


def plot_metrics_heatmap(results_df: pd.DataFrame, out_dir: Path):
    """繪製標準化指標的熱圖"""
    
    # 選擇要可視化的指標
    metrics_cols = ["MAE", "RMSE", "RMSSE", "WRMSSE"]
    
    # 標準化指標（min-max scaling）到 [0, 1]
    normalized = results_df[metrics_cols].copy()
    for col in metrics_cols:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        if max_val > min_val:
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0.0
    
    normalized.index = results_df["n_regimes"].astype(int)
    
    # 繪製熱圖（藍紫配色）
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        normalized.T, 
        annot=True, 
        fmt='.3f', 
        cmap='BuPu',
        cbar_kws={'label': 'Normalized Error (0=Best, 1=Worst)'},
        linewidths=0.5,
        ax=ax
    )
    ax.set_xlabel("Number of Regimes", fontsize=12)
    ax.set_ylabel("Metric", fontsize=12)
    ax.set_title("Normalized Performance Metrics Across Regime Configurations", 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / "regime_metrics_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Heatmap saved to {out_dir / 'regime_metrics_heatmap.png'}")


def main():
    ap = argparse.ArgumentParser(description="Search for optimal number of regimes in TSB-HB")
    ap.add_argument("--min-regimes", type=int, default=2, help="Minimum number of regimes to test")
    ap.add_argument("--max-regimes", type=int, default=8, help="Maximum number of regimes to test")
    ap.add_argument("--out", type=Path, default=default_out_dir() / "regime_search")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = ap.parse_args()
    
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TSB-HB Regime Search: Finding Optimal Number of Regimes")
    print("="*80)
    print(f"Testing regimes: {args.min_regimes} to {args.max_regimes}")
    print(f"Output directory: {args.out}")
    
    # 載入數據
    print("\nLoading and preprocessing data...")
    df = preprocess_online_retail(load_online_retail(default_data_file()))
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1/3, min_len=30)
    print(f"  Training: {len(init_set)} rows, {init_set['unique_id'].nunique()} items")
    print(f"  Evaluation: {len(eval_set)} rows")
    
    # 遍歷不同的 regime 數量
    results = []
    regime_range = range(args.min_regimes, args.max_regimes + 1)
    
    for n_regimes in regime_range:
        try:
            metrics = evaluate_single_regime(
                n_regimes=n_regimes,
                init_set=init_set,
                eval_set=eval_set,
                verbose=not args.quiet
            )
            results.append(metrics)
        except Exception as e:
            print(f"\n[ERROR] Failed for n_regimes={n_regimes}: {e}")
            continue
    
    # 保存結果
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out / "regime_search_results.csv", index=False)
    print(f"\n{'='*80}")
    print(f"[SUCCESS] Results saved to {args.out / 'regime_search_results.csv'}")
    
    # 顯示摘要
    print(f"\n{'='*80}")
    print("Summary: Best Configurations")
    print(f"{'='*80}")
    
    for metric in ["MAE", "RMSE", "RMSSE", "WRMSSE"]:
        best_idx = results_df[metric].idxmin()
        best_row = results_df.loc[best_idx]
        print(f"\nBest {metric}:")
        print(f"  n_regimes: {best_row['n_regimes']:.0f}")
        print(f"  {metric}: {best_row[metric]:.4f}")
        print(f"  Training time: {best_row['fit_time']:.2f}s")
    
    # 繪製對比圖
    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}")
    plot_regime_comparison(results_df, args.out)
    plot_metrics_heatmap(results_df, args.out)
    
    # 完整結果表格
    print(f"\n{'='*80}")
    print("Complete Results Table:")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("[DONE] Regime search completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
