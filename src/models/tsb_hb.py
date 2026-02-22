from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm


@dataclass
class TSBHBParams:
    # Indexed by unique_id
    p_posterior: pd.Series
    shrunk_mean_log: pd.Series
    posterior_var_mu: pd.Series
    sigma_sq_process: float


def _beta_binom_log_marginal(s: int, n: int, alpha: float, beta: float) -> float:
    if alpha <= 0 or beta <= 0 or s < 0 or n < s:
        return -np.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(s + 1)
        - math.lgamma(n - s + 1)
        + math.lgamma(s + alpha)
        + math.lgamma(n - s + beta)
        - math.lgamma(n + alpha + beta)
        - (math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))
    )


def _estimate_beta_hyperparams(counts: pd.DataFrame) -> tuple[float, float]:
    s_arr, n_arr = counts["s"].astype(int).values, counts["n"].astype(int).values
    valid = (n_arr > 0) & (s_arr <= n_arr)
    s_arr, n_arr = s_arr[valid], n_arr[valid]

    def objective(params: np.ndarray) -> float:
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return np.inf
        ll = [_beta_binom_log_marginal(int(si), int(ni), float(alpha), float(beta)) for si, ni in zip(s_arr, n_arr)]
        return -float(np.sum(ll))

    result = opt.minimize(objective, x0=[1.0, 10.0], method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)])
    if not result.success:
        return 1.0, 1.0
    return float(result.x[0]), float(result.x[1])


def fit_tsb_hb(train_df: pd.DataFrame) -> TSBHBParams:
    """Fits TSB-HB model parameters on the initial set.

    Steps mirror the notebook:
    - p: Beta-Binomial HB, posterior mean with shared (alpha, beta)
    - size: LogNormal mean with shrinkage via REML credibility weighting
    Returns series indexed by unique_id and scalar sigma_sq_process.
    """
    init_set = train_df.copy()
    init_set["occ"] = (init_set["y"] > 0).astype(int)
    init_set["size"] = np.where(init_set["occ"] == 1, init_set["y"].astype(float), np.nan)
    init_set["log_size"] = np.log(init_set["size"])  # NaN for zero-demand periods

    g_init = init_set.groupby("unique_id")
    s = g_init["occ"].sum()
    n = g_init["ds"].nunique()

    alpha_hat, beta_hat = _estimate_beta_hyperparams(pd.DataFrame({"s": s, "n": n}))
    p_post_mean = (alpha_hat + s) / (alpha_hat + beta_hat + n)

    item_stats = init_set.groupby("unique_id")["log_size"].agg(n_pos="count", mean_log="mean", var_log="var").reset_index()
    item_stats = item_stats.fillna({"var_log": 0})
    item_stats_filtered = item_stats[item_stats["n_pos"] > 1].copy()

    # Pooled within-item variance (weighted)
    numerator = float(np.sum((item_stats_filtered["n_pos"] - 1) * item_stats_filtered["var_log"]))
    denominator = float(np.sum(item_stats_filtered["n_pos"] - 1))
    sigma_sq = numerator / denominator if denominator > 0 else 1e-6

    y_i = item_stats_filtered["mean_log"].values
    n_i = item_stats_filtered["n_pos"].values

    # Method-of-moments initial tau^2
    observed_var = np.var(y_i, ddof=1) if len(y_i) > 1 else 0.0
    avg_sampling_var = float(np.mean(sigma_sq / n_i)) if len(n_i) > 0 else 0.0
    tau_sq_mom = max(observed_var - avg_sampling_var, 1e-6)

    def reml_neg_log_likelihood(tau_sq: float) -> float:
        if tau_sq <= 0:
            return np.inf
        V_i = tau_sq + sigma_sq / n_i
        weights = 1.0 / V_i
        mu_hat = float(np.sum(weights * y_i) / np.sum(weights))
        return float(np.sum(np.log(V_i)) + np.sum((y_i - mu_hat) ** 2 / V_i) + np.log(np.sum(weights)))

    res = opt.minimize(lambda x: reml_neg_log_likelihood(x[0]), x0=[tau_sq_mom], method="L-BFGS-B", bounds=[(1e-9, None)])
    tau_sq_reml = max(float(res.x[0]) if res.success else tau_sq_mom, 1e-6)

    # GLS estimate of global mean
    V_i_final = tau_sq_reml + sigma_sq / n_i
    weights_final = 1.0 / V_i_final
    global_mean = float(np.sum(weights_final * y_i) / np.sum(weights_final)) if np.sum(weights_final) > 0 else 0.0

    # Credibility weighting
    # credibility = n_pos / (n_pos + k), k = sigma^2 / tau^2
    k_reml = sigma_sq / tau_sq_reml
    item_stats = item_stats.merge(
        pd.DataFrame({
            "unique_id": item_stats_filtered["unique_id"],
            "credibility": n_i / (n_i + (sigma_sq / tau_sq_reml)),
        }),
        on="unique_id",
        how="left",
    ).fillna({"credibility": 0})

    item_stats["shrunk_mean_log"] = (
        item_stats["credibility"] * item_stats["mean_log"].fillna(global_mean)
        + (1 - item_stats["credibility"]) * global_mean
    )
    # Posterior variance for the mean parameter mu_i
    item_stats["posterior_var_mu"] = (sigma_sq * tau_sq_reml) / (item_stats["n_pos"] * tau_sq_reml + sigma_sq)
    item_stats.loc[item_stats["n_pos"] == 0, "posterior_var_mu"] = tau_sq_reml

    params = item_stats.set_index("unique_id")[
        ["shrunk_mean_log", "posterior_var_mu"]
    ]
    params["p_posterior"] = p_post_mean

    return TSBHBParams(
        p_posterior=params["p_posterior"],
        shrunk_mean_log=params["shrunk_mean_log"],
        posterior_var_mu=params["posterior_var_mu"],
        sigma_sq_process=float(sigma_sq),
    )


def predict_tsb_hb(
    params: TSBHBParams,
    eval_df: pd.DataFrame,
    quantiles: Optional[List[float]] = None,
    n_samples: int = 2000,
) -> pd.DataFrame:
    """Predict on the evaluation set.

    - Point forecast: per-series constant mean across horizon, replicated at each ds
    - Probabilistic: Monte Carlo sampling per series to get requested quantiles
    """
    if quantiles is None or len(quantiles) == 0:
        # Point prediction using E[Y] = p * E[size], E[size] under lognormal is exp(mu + sigma^2/2)
        size_mean = np.exp(params.shrunk_mean_log + params.sigma_sq_process / 2.0)
        mean_pred = (params.p_posterior * size_mean).fillna(0.0)
        out = eval_df[["unique_id", "ds"]].copy()
        out["yhat"] = out["unique_id"].map(mean_pred)
        out["yhat"] = out["yhat"].fillna(0.0)
        return out

    # Probabilistic via Monte Carlo following the notebook
    uids = eval_df["unique_id"].unique().tolist()
    qcols = [f"q_{q}" for q in quantiles]
    rows = []
    for uid in uids:
        if uid not in params.p_posterior.index:
            continue
        p = float(params.p_posterior.get(uid, 0.0))
        mu = float(params.shrunk_mean_log.get(uid, 0.0))
        pred_var = float(params.sigma_sq_process + params.posterior_var_mu.get(uid, 0.0))
        pred_std = float(np.sqrt(pred_var))
        # Zero-inflation and lognormal draws
        demand_occurs = np.random.binomial(1, p, n_samples)
        log_samples = norm.rvs(loc=mu, scale=pred_std, size=n_samples)
        size_samples = np.exp(log_samples)
        samples = size_samples * demand_occurs
        qvals = {f"q_{q}": float(np.quantile(samples, q)) for q in quantiles}
        qvals["prob_zero_predicted"] = 1.0 - p

        ds_vals = eval_df.loc[eval_df["unique_id"] == uid, "ds"].values
        tmp = pd.DataFrame({**qvals, "unique_id": uid, "ds": ds_vals})
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["unique_id", "ds"] + qcols)

