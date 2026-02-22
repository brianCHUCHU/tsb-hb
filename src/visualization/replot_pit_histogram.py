"""
重新繪製 Randomized PIT Histogram
使用現有的 prob_quantiles.csv 和評估數據，無需重新訓練模型
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin


def plot_pit_histogram_randomized(eval_merged: pd.DataFrame, quantiles: list[float], out_path: Path):
    models = eval_merged["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1: axes = [axes]
    
    bins = np.linspace(0, 1, 11)
    
    for ax, model in zip(axes, models):
        dfm = eval_merged[eval_merged["model"] == model].copy()
        pit_values = []
        
        # 確保有 prob_zero_predicted 這一列，如果沒有則用 q_0.1 近似
        has_p_zero = "prob_zero_predicted" in dfm.columns
        
        for _, row in dfm.iterrows():
            y = row["y"]
            q_values = [row[f"q_{q}"] for q in quantiles]
            
            if y == 0:
                # 關鍵修正 1：如果觀測為 0，PIT 應該在 [0, P(Y<=0)] 之間均勻分布
                # P(Y<=0) 對於 TSB-HB 來說就是 1-p
                upper_bound = row["prob_zero_predicted"] if has_p_zero else quantiles[0]
                lower_bound = 0.0
            else:
                # 關鍵修正 2：使用 side='right' 來處理多個分位數相等的情況
                # 這會找到第一個「嚴格大於」y 的分位數，能更準確地找到 y 的累積機率上界
                idx_right = np.searchsorted(q_values, y, side='right')
                idx_left = np.searchsorted(q_values, y, side='left')
                
                # 下界：第一個小於等於 y 的機率
                if idx_left == 0:
                    lower_bound = row["prob_zero_predicted"] if has_p_zero else 0.0
                else:
                    lower_bound = quantiles[idx_left - 1]
                
                # 上界：第一個大於 y 的機率
                if idx_right >= len(quantiles):
                    upper_bound = 1.0
                else:
                    upper_bound = quantiles[idx_right]
            
            # 隨機化處理
            random_pit = np.random.uniform(lower_bound, max(lower_bound + 1e-5, upper_bound))
            pit_values.append(random_pit)
                        
        sns.histplot(pit_values, bins=bins, stat="density", ax=ax, color="skyblue", edgecolor="black")
        ax.axhline(1.0, color='r', linestyle='--', label="Uniform (Ideal)")
        ax.set_title(f"PIT Histogram: {model}")
        ax.set_ylim(0, 2)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Randomized PIT histogram saved to {out_path}")


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Paths
    data_path = repo_root / "data" / "online_retail.csv"
    prob_quantiles_path = repo_root / "outputs" / "prob_quantiles.csv"
    output_path = repo_root / "outputs" / "pit_randomized_histogram.png"
    
    print("Loading data...")
    # Load and prepare evaluation set to get true values
    df_raw = load_online_retail(data_path)
    df = preprocess_online_retail(df_raw)
    _, eval_set = train_eval_split_fixed_origin(df, init_ratio=1/3, min_len=30)
    
    print(f"Loading probabilistic forecasts from {prob_quantiles_path}...")
    # Load prob_quantiles.csv
    prob_df = pd.read_csv(prob_quantiles_path)
    
    # Parse date column
    prob_df['ds'] = pd.to_datetime(prob_df['ds'])
    eval_set['ds'] = pd.to_datetime(eval_set['ds'])
    
    print("Merging forecasts with true values...")
    # Merge with eval_set to get true values
    eval_merged = prob_df.merge(
        eval_set[['unique_id', 'ds', 'y']],
        on=['unique_id', 'ds'],
        how='inner'
    )
    
    print(f"Merged data shape: {eval_merged.shape}")
    print(f"Models in data: {eval_merged['model'].unique()}")
    
    # Define quantiles (should match the columns in prob_quantiles.csv)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    # Check that all required columns exist
    required_cols = ['model', 'unique_id', 'ds', 'y', 'prob_zero_predicted'] + [f'q_{q}' for q in quantiles]
    missing_cols = [col for col in required_cols if col not in eval_merged.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return
    
    print("\nGenerating Randomized PIT histogram...")
    plot_pit_histogram_randomized(eval_merged, quantiles, output_path)
    
    print("\n[SUCCESS] PIT histogram regenerated successfully!")
    print(f"           Saved to: {output_path}")


if __name__ == "__main__":
    main()
