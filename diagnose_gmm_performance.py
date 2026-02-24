"""診斷 GMM 在大數據集上的性能瓶頸"""
import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=== GMM Performance Diagnosis ===\n")

print("Step 1: Loading data...")
t0 = time.time()
from data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
df_raw = load_online_retail(Path("data/online_retail.csv"))
df = preprocess_online_retail(df_raw)
init_set, _ = train_eval_split_fixed_origin(df, init_ratio=1/3, min_len=30)
print(f"  Loaded in {time.time()-t0:.2f}s: {len(init_set)} rows, {init_set['unique_id'].nunique()} items\n")

print("Step 2: Computing features (ADI, CV2)...")
t1 = time.time()
import numpy as np
import pandas as pd
df_copy = init_set.copy()
df_copy["occ"] = (df_copy["y"] > 0).astype(int)
df_copy["size"] = np.where(df_copy["occ"] == 1, df_copy["y"].astype(float), np.nan)

stats = df_copy.groupby("unique_id").agg(
    total_steps=("ds", "nunique"),
    n_pos=("occ", "sum"),
    mean_size=("size", "mean"),
    std_size=("size", "std")
)
stats["adi"] = stats["total_steps"] / stats["n_pos"].replace(0, 1)
stats["cv2"] = (stats["std_size"] / stats["mean_size"].replace(0, 1))**2
stats = stats.fillna(0)
print(f"  Computed features in {time.time()-t1:.2f}s for {len(stats)} items\n")

print("Step 3: Testing GMM clustering...")
t2 = time.time()
from sklearn.mixture import GaussianMixture
features = np.log(stats[["adi", "cv2"]] + 1e-5)
print(f"  Feature matrix shape: {features.shape}")

for n_regimes in [2, 3, 4]:
    t_gmm = time.time()
    gmm = GaussianMixture(n_components=n_regimes, random_state=42, covariance_type='full')
    gmm.fit(features)
    responsibilities = gmm.predict_proba(features)
    elapsed = time.time() - t_gmm
    print(f"  GMM with {n_regimes} regimes: {elapsed:.2f}s (converged: {gmm.converged_})")

print(f"\nTotal GMM test time: {time.time()-t2:.2f}s\n")

print("Step 4: Testing one regime optimization loop...")
t3 = time.time()
from scipy import optimize as opt
import math

counts_df = pd.DataFrame({"s": stats["n_pos"], "n": stats["total_steps"]})
w_g = responsibilities[:, 0]

def _weighted_beta_binom_log_marginal(counts_df, weights, alpha, beta):
    s = counts_df["s"].values
    n = counts_df["n"].values
    eps = 1e-10
    alpha = max(alpha, eps)
    beta = max(beta, eps)
    def lgamma_vec(x): return np.vectorize(math.lgamma)(x)
    ll = (
        lgamma_vec(n + 1) - lgamma_vec(s + 1) - lgamma_vec(n - s + 1) +
        lgamma_vec(s + alpha) + lgamma_vec(n - s + beta) -
        lgamma_vec(n + alpha + beta) -
        (lgamma_vec(alpha) + lgamma_vec(beta) - lgamma_vec(alpha + beta))
    )
    return -float(np.sum(weights * ll))

res = opt.minimize(
    lambda x: _weighted_beta_binom_log_marginal(counts_df, w_g, x[0], x[1]),
    x0=[1.0, 10.0], bounds=[(1e-6, None), (1e-6, None)], method="L-BFGS-B"
)
print(f"  One regime optimization: {time.time()-t3:.2f}s (success: {res.success})\n")

print(f"=== Total diagnosis time: {time.time()-t0:.2f}s ===")
print("\n[SUCCESS] All diagnostic tests completed!")
