from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.special import gammaln
from sklearn.mixture import GaussianMixture


# =============================================================================
# TSB-HB (Hierarchical Bayesian) with regime mixture
# - Occurrence: Beta-Binomial (per regime hyperprior)
# - Size: LogNormal with Normal-Normal hierarchy (per regime hyperprior on log-mean)
#
# Paper-grade implementation notes:
# 1) Robust handling for s=0 (no occurrences) in ADI feature: ADI -> very large
# 2) Robust handling for n_pos < 2 for var/mean statistics; avoid fillna(0) bias
# 3) Stable optimization for Beta-Binomial hyperparameters via log-parameterization
# 4) Predict:
#    - Point forecast uses mixture posterior means
#    - Prob forecast: coherent sampling (sample regime -> sample p_g -> sample occ -> sample log-size)
# 5) Carefully broadcasts outputs per (uid, ds) without length mismatch
# =============================================================================


# -----------------------------
# Dataclasses
# -----------------------------

@dataclass(frozen=True)
class TSBHBParams:
    # Mixture posterior means per series
    p_mean: pd.Series          # E[p_i | data] (mixture across regimes)
    size_mean: pd.Series       # E[size_i | data] (mixture across regimes), where size is positive magnitude (given occurrence)

    # Global within-series variance of log-size
    sigma_sq: float            # estimated residual variance in log-size

    # Regime membership weights per series (from GMM on features)
    regime_weights: pd.DataFrame  # shape: (n_series, G), rows sum to 1

    # Regime hyperparameters for occurrence (Beta prior)
    alpha: List[float]
    beta: List[float]

    # Regime hyperparameters for size (Normal prior on log-mean)
    mu_g: List[float]
    tau_sq_g: List[float]

    # Per-series per-regime posterior for size log-mean
    posterior_m: pd.DataFrame   # m_{ig}
    posterior_v: pd.DataFrame   # v_{ig}

    # Per-series per-regime posterior mean of occurrence probability
    posterior_p: pd.DataFrame   # p_{ig} = E[p_i | data, regime=g]


# =============================================================================
# Utilities
# =============================================================================

def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, eps))


def _row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, eps)
    return mat / row_sums


# =============================================================================
# Beta–Binomial weighted marginal negative log-likelihood
# =============================================================================

def _beta_binom_logpmf(s: np.ndarray, n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    log p(s | n, alpha, beta) for Beta-Binomial:
    C(n,s) * B(s+alpha, n-s+beta) / B(alpha,beta)
    """
    return (
        gammaln(n + 1)
        - gammaln(s + 1)
        - gammaln(n - s + 1)
        + gammaln(s + alpha)
        + gammaln(n - s + beta)
        - gammaln(n + alpha + beta)
        - (gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))
    )


def _weighted_beta_binom_negloglik_logparams(
    log_ab: np.ndarray,
    s: np.ndarray,
    n: np.ndarray,
    w: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Optimize in unconstrained space: alpha=exp(a), beta=exp(b).
    Returns negative weighted log-likelihood.
    """
    a, b = float(log_ab[0]), float(log_ab[1])
    alpha = float(np.exp(a))
    beta = float(np.exp(b))
    alpha = max(alpha, eps)
    beta = max(beta, eps)

    ll = _beta_binom_logpmf(s, n, alpha, beta)
    # w are soft assignment weights for the regime; allow any nonnegative
    return -float(np.sum(w * ll))


# =============================================================================
# Main fitting
# =============================================================================

def fit_tsb_hb(
    train_df: pd.DataFrame,
    n_regimes: int = 4,
    random_state: int = 0,
    eps: float = 1e-12,
) -> TSBHBParams:
    """
    Fit TSB-HB with regime mixture.

    Required columns in train_df:
      - unique_id: series identifier
      - ds: time index (any hashable)
      - y: nonnegative demand (0 allowed). If negative exists, log-size is invalid.

    Returns:
      TSBHBParams
    """
    required = {"unique_id", "ds", "y"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"train_df is missing required columns: {sorted(missing)}")

    df = train_df.copy()

    # Occurrence and size
    df["occ"] = (df["y"] > 0).astype(int)
    # Only positive y contribute to size distribution
    df["size"] = np.where(df["occ"] == 1, df["y"].astype(float), np.nan)

    # Safety: if any non-positive in occ==1, log is invalid
    bad = df.loc[df["occ"] == 1, "size"] <= 0
    if bool(np.any(bad)):
        raise ValueError("Found non-positive y where occ==1; cannot take log for size model.")

    df["log_size"] = np.log(df["size"])

    # -----------------------------
    # Aggregate per series statistics
    # -----------------------------
    # n: number of unique time points
    # s: number of occurrences
    stats = df.groupby("unique_id").agg(
        n=("ds", "nunique"),
        s=("occ", "sum"),
    )

    # log-size stats among positive observations
    size_stats = df.groupby("unique_id")["log_size"].agg(
        n_pos="count",
        mean_log="mean",
        var_log="var",  # sample variance (ddof=1), NaN if n_pos < 2
    )

    stats = stats.join(size_stats)

    # Keep mean_log as NaN if n_pos==0 (do NOT fill with 0; that biases hyperparameters)
    # Replace var_log NaN (n_pos<2) with 0 for feature computation and sigma pooling logic
    stats["var_log"] = stats["var_log"].fillna(0.0)

    # -----------------------------
    # Regime clustering (ADI, CV^2) with robust handling
    # -----------------------------
    n_arr = stats["n"].astype(float).values
    s_arr = stats["s"].astype(float).values
    npos_arr = stats["n_pos"].fillna(0.0).astype(float).values

    # ADI = n / s. If s==0 => extremely intermittent -> set to large cap.
    # Use cap proportional to series length / eps.
    adi = np.empty_like(n_arr)
    s_pos_mask = s_arr > 0
    adi[s_pos_mask] = n_arr[s_pos_mask] / s_arr[s_pos_mask]
    # large value for no-occurrence series
    adi[~s_pos_mask] = n_arr[~s_pos_mask] / eps

    # CV^2 of size (Syntetos-style) often defined for positive demand sizes.
    # Here we use log-size variability proxy. If mean_log is NaN (n_pos==0),
    # set a neutral small value for feature, and let ADI dominate intermittency.
    mean_log = stats["mean_log"].values  # may contain NaN
    var_log = stats["var_log"].values

    # For feature computation, guard mean_log near 0 without discontinuous replace(0,1).
    denom = np.maximum(np.abs(np.nan_to_num(mean_log, nan=1.0)), 1e-6)

    # sqrt(var_log)/mean_log squared (in log-space proxy). If n_pos<2, var_log==0.
    cv2 = (np.sqrt(np.maximum(var_log, 0.0)) / denom) ** 2

    # Compose features; use log transform for scale stability
    features = np.column_stack([_safe_log(adi + 1e-5), _safe_log(cv2 + 1e-5)])

    gmm = GaussianMixture(n_components=n_regimes, random_state=random_state)
    gmm.fit(features)
    weights = gmm.predict_proba(features)          # shape (N, G)
    weights = _row_normalize(weights, eps=eps)     # ensure sum to 1

    regime_weights = pd.DataFrame(
        weights,
        index=stats.index,
        columns=[f"R{g}" for g in range(n_regimes)],
    )

    # -----------------------------
    # Estimate occurrence hyperparameters (alpha_g, beta_g) by weighted MLE
    # -----------------------------
    alpha_list: List[float] = []
    beta_list: List[float] = []

    s_vals = stats["s"].astype(int).values
    n_vals = stats["n"].astype(int).values

    # Initialize around a prior that suggests sparsity: alpha small-ish, beta larger
    init_alpha, init_beta = 1.0, 10.0
    init_log = np.log([init_alpha, init_beta])

    for g in range(n_regimes):
        w = weights[:, g].astype(float)

        # If regime has almost no mass, fall back to defaults
        if float(np.sum(w)) < 1e-8:
            alpha_list.append(init_alpha)
            beta_list.append(init_beta)
            continue

        res = opt.minimize(
            fun=_weighted_beta_binom_negloglik_logparams,
            x0=init_log,
            args=(s_vals, n_vals, w, eps),
            method="L-BFGS-B",  # unconstrained, stable
        )

        if res.success and np.all(np.isfinite(res.x)):
            a_hat, b_hat = float(res.x[0]), float(res.x[1])
            alpha_hat = float(np.exp(a_hat))
            beta_hat = float(np.exp(b_hat))
            alpha_list.append(max(alpha_hat, 1e-8))
            beta_list.append(max(beta_hat, 1e-8))
        else:
            alpha_list.append(init_alpha)
            beta_list.append(init_beta)

    # -----------------------------
    # Estimate sigma^2 (within-series log-size variance)
    # pooled across series with n_pos >= 2
    # -----------------------------
    valid = (stats["n_pos"].fillna(0) >= 2).values
    if bool(np.any(valid)):
        npos_valid = stats.loc[valid, "n_pos"].astype(float).values
        var_valid = stats.loc[valid, "var_log"].astype(float).values
        num = float(np.sum((npos_valid - 1.0) * var_valid))
        den = float(np.sum(npos_valid - 1.0))
        sigma_sq = num / den if den > 0 else 1e-6
    else:
        sigma_sq = 1e-6

    sigma_sq = float(max(sigma_sq, 1e-8))

    # -----------------------------
    # Estimate regime-specific size hyperparameters mu_g, tau_sq_g
    # Prior: theta_i (log-mean) ~ Normal(mu_g, tau_sq_g)
    # We estimate mu_g, tau_sq_g from observed per-series mean_log,
    # weighting by regime weights and also by reliability ~ n_pos.
    # -----------------------------
    mu_g_list: List[float] = []
    tau_sq_list: List[float] = []

    # Use only series with at least 1 positive size to estimate mu/tau
    has_pos = (stats["n_pos"].fillna(0) >= 1).values
    mean_log_obs = stats["mean_log"].values  # NaN for n_pos==0
    mean_log_obs2 = np.where(has_pos, mean_log_obs, np.nan)

    # Reliability weight: proportional to min(n_pos, cap)
    # (prevents a few long series dominating too much)
    npos_cap = 20.0
    rel = np.minimum(stats["n_pos"].fillna(0).astype(float).values, npos_cap)

    for g in range(n_regimes):
        w = weights[:, g].astype(float)
        # Only keep series with observed mean_log
        mask = np.isfinite(mean_log_obs2)
        w_eff = w[mask] * rel[mask]
        x = mean_log_obs2[mask].astype(float)

        w_sum = float(np.sum(w_eff))
        if w_sum < 1e-8:
            # fallback
            mu_g_list.append(float(np.nanmean(mean_log_obs2)) if np.isfinite(np.nanmean(mean_log_obs2)) else 0.0)
            tau_sq_list.append(1e-6)
            continue

        mu_g = float(np.sum(w_eff * x) / w_sum)
        tau_sq = float(np.sum(w_eff * (x - mu_g) ** 2) / w_sum)
        mu_g_list.append(mu_g)
        tau_sq_list.append(max(tau_sq, 1e-6))

    # -----------------------------
    # Posterior per regime (per series)
    # Occurrence posterior mean: p_{ig} = (alpha_g + s_i) / (alpha_g + beta_g + n_i)
    # Size posterior for theta_i (log-mean):
    #   v_{ig} = (sigma^2 * tau_g^2) / (n_pos * tau_g^2 + sigma^2)
    #   m_{ig} = (n_pos * tau_g^2)/(n_pos*tau_g^2 + sigma^2) * ybar
    #          + sigma^2/(n_pos*tau_g^2 + sigma^2) * mu_g
    # Edge handling:
    #  - if n_pos==0: we set m=mu_g, v=tau_sq (i.e., prior)
    # -----------------------------
    idx = stats.index

    posterior_m = pd.DataFrame(index=idx, columns=[f"R{g}" for g in range(n_regimes)], dtype=float)
    posterior_v = pd.DataFrame(index=idx, columns=[f"R{g}" for g in range(n_regimes)], dtype=float)
    posterior_p = pd.DataFrame(index=idx, columns=[f"R{g}" for g in range(n_regimes)], dtype=float)

    p_mean = pd.Series(0.0, index=idx, dtype=float)
    size_mean = pd.Series(0.0, index=idx, dtype=float)

    n_i = stats["n"].astype(float).values
    s_i = stats["s"].astype(float).values
    n_pos_i = stats["n_pos"].fillna(0.0).astype(float).values
    ybar_i = stats["mean_log"].values  # NaN where n_pos==0

    for g in range(n_regimes):
        alpha_g = float(alpha_list[g])
        beta_g = float(beta_list[g])
        mu_g = float(mu_g_list[g])
        tau_sq = float(tau_sq_list[g])

        # Occurrence posterior mean per series for this regime
        p_g = (alpha_g + s_i) / (alpha_g + beta_g + n_i)
        posterior_p[f"R{g}"] = p_g

        # Size posterior for theta_i (log-mean)
        denom = n_pos_i * tau_sq + sigma_sq

        # For n_pos==0, fallback to prior
        m_g = np.empty_like(n_pos_i, dtype=float)
        v_g = np.empty_like(n_pos_i, dtype=float)

        has = n_pos_i > 0
        # If has positives but ybar might still be NaN (shouldn't), guard
        ybar_safe = np.where(np.isfinite(ybar_i), ybar_i, mu_g)

        v_g[has] = (sigma_sq * tau_sq) / np.maximum(denom[has], eps)
        m_g[has] = (n_pos_i[has] * tau_sq / np.maximum(denom[has], eps)) * ybar_safe[has] + (sigma_sq / np.maximum(denom[has], eps)) * mu_g

        # Prior when no positives
        v_g[~has] = tau_sq
        m_g[~has] = mu_g

        posterior_m[f"R{g}"] = m_g
        posterior_v[f"R{g}"] = v_g

        # Mixture posterior mean aggregation
        w_g = regime_weights[f"R{g}"].values.astype(float)

        # Occurrence mean
        p_mean.values[:] += w_g * p_g

        # Size mean:
        # If log_size ~ Normal(theta_i, sigma_sq) and theta_i posterior ~ Normal(m_g, v_g),
        # then predictive log_size has variance sigma_sq + v_g and mean m_g
        # => E[size] = exp(m_g + 0.5*(sigma_sq + v_g))
        size_mean.values[:] += w_g * np.exp(m_g + 0.5 * (sigma_sq + v_g))

    # Numerical guard
    p_mean = p_mean.clip(lower=0.0, upper=1.0)
    size_mean = size_mean.clip(lower=0.0)

    return TSBHBParams(
        p_mean=p_mean,
        size_mean=size_mean,
        sigma_sq=sigma_sq,
        regime_weights=regime_weights,
        alpha=alpha_list,
        beta=beta_list,
        mu_g=mu_g_list,
        tau_sq_g=tau_sq_list,
        posterior_m=posterior_m,
        posterior_v=posterior_v,
        posterior_p=posterior_p,
    )


# =============================================================================
# Prediction
# =============================================================================

def predict_tsb_hb(
    params: TSBHBParams,
    eval_df: pd.DataFrame,
    quantiles: Optional[List[float]] = None,
    n_samples: int = 2000,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Point or probabilistic prediction.

    Input eval_df must have columns:
      - unique_id
      - ds

    Output:
      - If quantiles is None or empty:
          columns: unique_id, ds, yhat
      - Else:
          columns: unique_id, ds, q_<tau> for tau in quantiles, prob_zero_predicted
          (same quantiles repeated over all ds for each unique_id)
    """
    required = {"unique_id", "ds"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval_df is missing required columns: {sorted(missing)}")

    out_base = eval_df[["unique_id", "ds"]].copy()

    # ------ Point forecast ------
    if quantiles is None or len(quantiles) == 0:
        mean_pred = (params.p_mean * params.size_mean).fillna(0.0)
        out_base["yhat"] = out_base["unique_id"].map(mean_pred).fillna(0.0)
        return out_base

    # ------ Probabilistic forecast (Monte Carlo) ------
    rng = np.random.default_rng(random_state)

    quantiles = list(quantiles)
    n_regimes = len(params.alpha)

    uids = out_base["unique_id"].unique()
    rows = []

    # Pre-fetch per-uid vectors for speed
    # (posterior_m/v/p are DataFrames indexed by uid, columns R0..RG-1)
    for uid in uids:
        if uid not in params.regime_weights.index:
            # unseen series: by default skip (or could output zeros)
            continue

        ds_vals = out_base.loc[out_base["unique_id"] == uid, "ds"].values
        T = len(ds_vals)
        if T == 0:
            continue

        w = params.regime_weights.loc[uid].values.astype(float)  # (G,)
        w = w / max(w.sum(), 1e-12)

        # Sample regimes
        regime_idx = rng.choice(n_regimes, size=n_samples, p=w)

        # Gather per-regime posterior parameters for this uid
        m_g_arr = np.array([float(params.posterior_m.loc[uid, f"R{g}"]) for g in range(n_regimes)], dtype=float)
        v_g_arr = np.array([float(params.posterior_v.loc[uid, f"R{g}"]) for g in range(n_regimes)], dtype=float)
        p_g_arr = np.array([float(params.posterior_p.loc[uid, f"R{g}"]) for g in range(n_regimes)], dtype=float)
        p_g_arr = np.clip(p_g_arr, 0.0, 1.0)

        # Coherent sampling:
        # 1) sample regime
        # 2) sample occurrence using p_{ig} of that sampled regime
        # 3) sample log-size with predictive variance sigma_sq + v_{ig}
        p_draw = p_g_arr[regime_idx]
        occ = rng.binomial(1, p_draw, size=n_samples)

        pred_std = np.sqrt(np.maximum(params.sigma_sq + v_g_arr[regime_idx], 1e-12))
        log_size = rng.normal(loc=m_g_arr[regime_idx], scale=pred_std, size=n_samples)

        samples = occ * np.exp(log_size)

        qvals: Dict[str, float] = {f"q_{q}": float(np.quantile(samples, q)) for q in quantiles}

        # Predicted probability of zero:
        # Under coherent regime-mixture: P(Y=0) = sum_g w_g * (1 - p_{ig})
        prob_zero = float(np.sum(w * (1.0 - p_g_arr)))
        qvals["prob_zero_predicted"] = prob_zero

        # Broadcast scalars to length T properly
        row_dict = {
            "unique_id": np.repeat(uid, T),
            "ds": ds_vals,
        }
        for k, v in qvals.items():
            row_dict[k] = np.repeat(v, T)

        rows.append(pd.DataFrame(row_dict))

    if not rows:
        cols = ["unique_id", "ds"] + [f"q_{q}" for q in quantiles] + ["prob_zero_predicted"]
        return pd.DataFrame(columns=cols)

    return pd.concat(rows, ignore_index=True)