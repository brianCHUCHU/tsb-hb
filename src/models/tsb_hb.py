from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm


GLOBAL_GROUP = "__global__"


@dataclass
class TSBHBParams:
    # Posterior terms (indexed by unique_id)
    p_posterior: pd.Series
    shrunk_mean_log: pd.Series
    posterior_var_mu: pd.Series
    sigma_sq_process: pd.Series

    # Grouping/hyperparameters
    group_labels: pd.Series
    alpha_by_group: pd.Series
    beta_by_group: pd.Series
    size_global_mean_by_group: pd.Series
    size_sigma_sq_by_group: pd.Series
    size_tau_sq_by_group: pd.Series

    # Sufficient statistics for online updates / uncertainty propagation
    n_obs: pd.Series
    s_obs: pd.Series
    n_pos: pd.Series
    sum_log: pd.Series
    sum_sq_log: pd.Series

    # Optional bootstrap hyperparameter draws keyed by:
    # alpha, beta, size_mu, size_sigma, size_tau
    # Each value is DataFrame indexed by group, columns draw_*
    bootstrap_group_hypers: Optional[Dict[str, pd.DataFrame]] = None
    # Configuration metadata
    group_shrink_strength: float = 0.0
    dynamic_occurrence: bool = False
    occurrence_discount: float = 1.0
    adaptive_prior_strength: bool = False
    prior_strength_min: float = 1.0
    prior_strength_max: float = 1.0
    prior_strength_power: float = 1.0
    item_variance_mode: str = "group"
    item_variance_shrink_strength: float = 20.0
    item_var_log: Optional[pd.Series] = None


@dataclass
class TSBHBOnlineState:
    n_obs: pd.Series
    s_obs: pd.Series
    n_eff_occ: pd.Series
    s_eff_occ: pd.Series
    n_pos: pd.Series
    sum_log: pd.Series
    sum_sq_log: pd.Series
    group_labels: pd.Series
    alpha_by_group: pd.Series
    beta_by_group: pd.Series
    size_global_mean_by_group: pd.Series
    size_sigma_sq_by_group: pd.Series
    size_tau_sq_by_group: pd.Series
    bootstrap_group_hypers: Optional[Dict[str, pd.DataFrame]] = None
    dynamic_occurrence: bool = False
    occurrence_discount: float = 1.0
    adaptive_prior_strength: bool = False
    prior_strength_min: float = 1.0
    prior_strength_max: float = 1.0
    prior_strength_power: float = 1.0
    item_variance_mode: str = "group"
    item_variance_shrink_strength: float = 20.0


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
    s_arr = counts["s"].astype(int).to_numpy()
    n_arr = counts["n"].astype(int).to_numpy()
    valid = (n_arr > 0) & (s_arr <= n_arr)
    s_arr, n_arr = s_arr[valid], n_arr[valid]
    if len(s_arr) == 0:
        return 1.0, 1.0

    def objective(params: np.ndarray) -> float:
        alpha, beta = float(params[0]), float(params[1])
        if alpha <= 0 or beta <= 0:
            return np.inf
        ll = [_beta_binom_log_marginal(int(si), int(ni), alpha, beta) for si, ni in zip(s_arr, n_arr)]
        return -float(np.sum(ll))

    try:
        result = opt.minimize(
            objective,
            x0=[1.0, 10.0],
            method="L-BFGS-B",
            bounds=[(1e-6, None), (1e-6, None)],
        )
    except Exception:
        return 1.0, 1.0
    if not result.success:
        return 1.0, 1.0
    return float(result.x[0]), float(result.x[1])


def _normalize_group_labels(
    unique_ids: pd.Index,
    group_labels: Optional[pd.Series | Dict[str, str]] = None,
) -> pd.Series:
    if group_labels is None:
        return pd.Series(GLOBAL_GROUP, index=unique_ids, dtype=object)

    if isinstance(group_labels, dict):
        mapped = pd.Series(group_labels, dtype=object)
        out = mapped.reindex(unique_ids)
    elif isinstance(group_labels, pd.Series):
        if group_labels.index.equals(unique_ids):
            out = group_labels.copy()
        else:
            out = group_labels.reindex(unique_ids)
            if out.isna().all() and len(group_labels) == len(unique_ids):
                out = pd.Series(group_labels.to_numpy(), index=unique_ids, dtype=object)
    else:
        raise TypeError("group_labels must be None, dict, or pandas Series.")

    out = out.fillna(GLOBAL_GROUP).astype(str)
    return out


def _clip_occurrence_discount(discount: float) -> float:
    try:
        val = float(discount)
    except Exception:
        val = 1.0
    if not np.isfinite(val):
        val = 1.0
    return float(np.clip(val, 1e-6, 1.0))


def _compute_series_stats(
    train_df: pd.DataFrame,
    group_labels: Optional[pd.Series | Dict[str, str]] = None,
) -> pd.DataFrame:
    data = train_df[["unique_id", "y"]].copy()
    data["occ"] = (data["y"] > 0).astype(int)
    data["log_y"] = np.nan
    pos_mask = data["y"] > 0
    data.loc[pos_mask, "log_y"] = np.log(data.loc[pos_mask, "y"].astype(float))

    g = data.groupby("unique_id", sort=False)
    n_obs = g["y"].size().astype(float)
    s_obs = g["occ"].sum().astype(float)
    n_pos = g["log_y"].count().astype(float)
    sum_log = g["log_y"].sum(min_count=1).fillna(0.0).astype(float)
    sum_sq_log = g["log_y"].apply(lambda x: float(np.square(x.dropna()).sum())).astype(float)

    mean_log = (sum_log / n_pos.replace(0, np.nan)).fillna(np.nan)
    var_num = sum_sq_log - n_pos * (mean_log.fillna(0.0) ** 2)
    var_log = (var_num / (n_pos - 1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    groups = _normalize_group_labels(n_obs.index, group_labels)
    stats = pd.DataFrame(
        {
            "n_obs": n_obs,
            "s_obs": s_obs,
            "n_pos": n_pos,
            "sum_log": sum_log,
            "sum_sq_log": sum_sq_log,
            "mean_log": mean_log,
            "var_log": var_log,
            "group": groups.reindex(n_obs.index).astype(str),
        },
        index=n_obs.index,
    )
    return stats


def _estimate_size_hyper_from_stats(
    item_stats: pd.DataFrame,
    fallback_mu: float = 0.0,
    fallback_sigma: float = 1.0,
    fallback_tau: float = 1.0,
) -> tuple[float, float, float]:
    pos = item_stats[item_stats["n_pos"] > 0].copy()
    if pos.empty:
        return float(fallback_mu), float(max(fallback_sigma, 1e-6)), float(max(fallback_tau, 1e-6))

    with_var = pos[pos["n_pos"] > 1]
    numerator = float(np.sum((with_var["n_pos"] - 1) * with_var["var_log"]))
    denominator = float(np.sum(with_var["n_pos"] - 1))
    if denominator > 0:
        sigma_sq = numerator / denominator
    else:
        sigma_sq = float(pos["var_log"].mean()) if not pos["var_log"].empty else float(fallback_sigma)
    sigma_sq = float(max(sigma_sq, 1e-6))

    y_i = pos["mean_log"].to_numpy(dtype=float)
    n_i = pos["n_pos"].to_numpy(dtype=float)
    if len(y_i) == 0:
        return float(fallback_mu), sigma_sq, float(max(fallback_tau, 1e-6))
    if len(y_i) == 1:
        mu_hat = float(y_i[0]) if np.isfinite(y_i[0]) else float(fallback_mu)
        return mu_hat, sigma_sq, float(max(fallback_tau, 1e-6))

    observed_var = float(np.var(y_i, ddof=1))
    avg_sampling_var = float(np.mean(sigma_sq / np.maximum(n_i, 1.0)))
    tau_sq_mom = max(observed_var - avg_sampling_var, 1e-6)

    def reml_neg_log_likelihood(tau_sq: float) -> float:
        if tau_sq <= 0:
            return np.inf
        v_i = tau_sq + sigma_sq / np.maximum(n_i, 1.0)
        weights = 1.0 / v_i
        mu_hat = float(np.sum(weights * y_i) / np.sum(weights))
        return float(np.sum(np.log(v_i)) + np.sum((y_i - mu_hat) ** 2 / v_i) + np.log(np.sum(weights)))

    try:
        res = opt.minimize(
            lambda x: reml_neg_log_likelihood(float(x[0])),
            x0=[tau_sq_mom],
            method="L-BFGS-B",
            bounds=[(1e-9, None)],
        )
    except Exception:
        res = None

    if res is None or (not res.success):
        tau_sq = tau_sq_mom
    else:
        tau_sq = max(float(res.x[0]), 1e-6)

    v_i = tau_sq + sigma_sq / np.maximum(n_i, 1.0)
    w_i = 1.0 / v_i
    mu_hat = float(np.sum(w_i * y_i) / np.sum(w_i)) if np.sum(w_i) > 0 else float(np.nanmean(y_i))
    if not np.isfinite(mu_hat):
        mu_hat = float(fallback_mu)
    return mu_hat, sigma_sq, tau_sq


def _estimate_group_hypers(
    stats: pd.DataFrame,
    fallback: Optional[tuple[float, float, float, float, float]] = None,
    group_shrink_strength: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    # Fallback tuple: (alpha, beta, size_mu, size_sigma, size_tau)
    if fallback is None:
        alpha_g, beta_g = _estimate_beta_hyperparams(
            pd.DataFrame(
                {
                    "s": stats["s_obs"].astype(int),
                    "n": stats["n_obs"].astype(int),
                }
            )
        )
        size_mu_g, size_sigma_g, size_tau_g = _estimate_size_hyper_from_stats(stats)
    else:
        alpha_g, beta_g, size_mu_g, size_sigma_g, size_tau_g = fallback

    groups = pd.Index(sorted(stats["group"].astype(str).unique()))
    if GLOBAL_GROUP not in groups:
        groups = groups.append(pd.Index([GLOBAL_GROUP]))

    alpha_by_group: dict[str, float] = {}
    beta_by_group: dict[str, float] = {}
    size_mu_by_group: dict[str, float] = {}
    size_sigma_by_group: dict[str, float] = {}
    size_tau_by_group: dict[str, float] = {}

    group_shrink_strength = float(max(group_shrink_strength, 0.0))

    for grp in groups:
        sub = stats[stats["group"] == grp]
        if sub.empty:
            alpha_by_group[grp] = float(alpha_g)
            beta_by_group[grp] = float(beta_g)
            size_mu_by_group[grp] = float(size_mu_g)
            size_sigma_by_group[grp] = float(size_sigma_g)
            size_tau_by_group[grp] = float(size_tau_g)
            continue

        a_hat, b_hat = _estimate_beta_hyperparams(
            pd.DataFrame({"s": sub["s_obs"].astype(int), "n": sub["n_obs"].astype(int)})
        )
        if not np.isfinite(a_hat) or not np.isfinite(b_hat):
            a_hat, b_hat = float(alpha_g), float(beta_g)

        mu_hat, sigma_hat, tau_hat = _estimate_size_hyper_from_stats(
            sub,
            fallback_mu=float(size_mu_g),
            fallback_sigma=float(size_sigma_g),
            fallback_tau=float(size_tau_g),
        )
        # Optional shrink from group-level estimates back to global estimates.
        if grp == GLOBAL_GROUP or group_shrink_strength <= 0.0:
            omega = 1.0
        else:
            n_series = float(len(sub))
            omega = n_series / (n_series + group_shrink_strength)
            omega = float(np.clip(omega, 0.0, 1.0))

        alpha_final = omega * float(max(a_hat, 1e-6)) + (1.0 - omega) * float(max(alpha_g, 1e-6))
        beta_final = omega * float(max(b_hat, 1e-6)) + (1.0 - omega) * float(max(beta_g, 1e-6))
        size_mu_final = omega * float(mu_hat) + (1.0 - omega) * float(size_mu_g)
        size_sigma_final = omega * float(max(sigma_hat, 1e-6)) + (1.0 - omega) * float(max(size_sigma_g, 1e-6))
        size_tau_final = omega * float(max(tau_hat, 1e-6)) + (1.0 - omega) * float(max(size_tau_g, 1e-6))

        alpha_by_group[grp] = float(max(alpha_final, 1e-6))
        beta_by_group[grp] = float(max(beta_final, 1e-6))
        size_mu_by_group[grp] = float(size_mu_final)
        size_sigma_by_group[grp] = float(max(size_sigma_final, 1e-6))
        size_tau_by_group[grp] = float(max(size_tau_final, 1e-6))

    alpha_s = pd.Series(alpha_by_group, dtype=float)
    beta_s = pd.Series(beta_by_group, dtype=float)
    size_mu_s = pd.Series(size_mu_by_group, dtype=float)
    size_sigma_s = pd.Series(size_sigma_by_group, dtype=float)
    size_tau_s = pd.Series(size_tau_by_group, dtype=float)
    return alpha_s, beta_s, size_mu_s, size_sigma_s, size_tau_s


def _compute_posteriors_from_stats(
    stats: pd.DataFrame,
    alpha_by_group: pd.Series,
    beta_by_group: pd.Series,
    size_mu_by_group: pd.Series,
    size_sigma_by_group: pd.Series,
    size_tau_by_group: pd.Series,
    adaptive_prior_strength: bool = False,
    prior_strength_min: float = 1.0,
    prior_strength_max: float = 1.0,
    prior_strength_power: float = 1.0,
    item_variance_mode: str = "group",
    item_variance_shrink_strength: float = 20.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    prior_min = float(max(prior_strength_min, 1e-6))
    prior_max = float(max(prior_strength_max, 1e-6))
    if prior_min > prior_max:
        prior_min, prior_max = prior_max, prior_min
    prior_power = float(max(prior_strength_power, 0.0))
    variance_mode = str(item_variance_mode).lower()
    if variance_mode == "shrink_item":
        # Backward-compatible alias: previous heuristic mode now maps to conjugate variance shrinkage.
        variance_mode = "conjugate"
    if variance_mode not in {"group", "conjugate"}:
        variance_mode = "group"
    variance_prior_df = float(max(item_variance_shrink_strength, 2.1))

    groups = stats["group"].astype(str)

    alpha = groups.map(alpha_by_group).fillna(alpha_by_group.get(GLOBAL_GROUP, alpha_by_group.iloc[0]))
    beta = groups.map(beta_by_group).fillna(beta_by_group.get(GLOBAL_GROUP, beta_by_group.iloc[0]))
    size_mu = groups.map(size_mu_by_group).fillna(size_mu_by_group.get(GLOBAL_GROUP, size_mu_by_group.iloc[0]))
    size_sigma = groups.map(size_sigma_by_group).fillna(size_sigma_by_group.get(GLOBAL_GROUP, size_sigma_by_group.iloc[0]))
    size_tau = groups.map(size_tau_by_group).fillna(size_tau_by_group.get(GLOBAL_GROUP, size_tau_by_group.iloc[0]))

    s_obs = stats["s_obs"].astype(float)
    n_obs = stats["n_obs"].astype(float)
    n_pos = stats["n_pos"].astype(float)
    sum_log = stats["sum_log"].astype(float)
    if "var_log" in stats.columns:
        item_var_log = stats["var_log"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    elif "sum_sq_log" in stats.columns:
        sum_sq_log = stats["sum_sq_log"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean_log = (sum_log / n_pos.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        var_num = sum_sq_log - n_pos * (mean_log.fillna(0.0) ** 2)
        item_var_log = (var_num / (n_pos - 1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    else:
        item_var_log = pd.Series(0.0, index=stats.index, dtype=float)

    n_obs_denom = n_obs.clip(lower=1.0)
    n_pos_denom = n_pos.clip(lower=1.0)
    occ_scale = pd.Series(1.0, index=stats.index, dtype=float)
    size_scale = pd.Series(1.0, index=stats.index, dtype=float)
    if adaptive_prior_strength:
        group_n_obs = stats.groupby("group", sort=False)["n_obs"].median()
        global_ref_n_obs = float(np.nanmedian(n_obs.to_numpy())) if len(n_obs) > 0 else 1.0
        if not np.isfinite(global_ref_n_obs) or global_ref_n_obs <= 0:
            global_ref_n_obs = 1.0
        ref_n_obs = groups.map(group_n_obs).astype(float).replace([np.inf, -np.inf], np.nan).fillna(global_ref_n_obs).clip(lower=1.0)
        occ_scale = ((ref_n_obs / n_obs_denom) ** prior_power).clip(lower=prior_min, upper=prior_max)

        group_n_pos = stats.groupby("group", sort=False)["n_pos"].median()
        positive_n_pos = n_pos[n_pos > 0]
        global_ref_n_pos = float(np.nanmedian(positive_n_pos.to_numpy())) if len(positive_n_pos) > 0 else 1.0
        if not np.isfinite(global_ref_n_pos) or global_ref_n_pos <= 0:
            global_ref_n_pos = 1.0
        ref_n_pos = groups.map(group_n_pos).astype(float).replace([np.inf, -np.inf], np.nan).fillna(global_ref_n_pos)
        ref_n_pos = ref_n_pos.where(ref_n_pos > 0, global_ref_n_pos)
        size_scale = ((ref_n_pos / n_pos_denom) ** prior_power).clip(lower=prior_min, upper=prior_max)

    alpha_eff = (alpha * occ_scale).clip(lower=1e-9)
    beta_eff = (beta * occ_scale).clip(lower=1e-9)

    p_post = (alpha_eff + s_obs) / (alpha_eff + beta_eff + n_obs)
    p_post = p_post.clip(lower=0.0, upper=1.0)

    mean_mle = (sum_log / n_pos.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if variance_mode == "conjugate":
        # Conjugate update for sigma_i^2 under scaled-inv-chi-square prior:
        # E[sigma_i^2 | x] = (nu*s0^2 + (n_i-1)*s_i^2) / (nu + n_i - 3)
        # Treat size_sigma as prior mean and back out s0^2 = size_sigma * (nu-2)/nu.
        nu = variance_prior_df
        prior_numer = (nu - 2.0) * size_sigma
        sse_term = (n_pos - 1.0).clip(lower=0.0) * item_var_log
        denom = nu + n_pos - 3.0
        sigma_post = (prior_numer + sse_term) / denom.replace(0.0, np.nan)
        sigma_sq_process = sigma_post.where((n_pos > 1.0) & (denom > 1e-9), size_sigma)
        sigma_sq_process = sigma_sq_process.fillna(size_sigma).clip(lower=1e-9)
    else:
        sigma_sq_process = size_sigma.clip(lower=1e-9)

    sigma_for_mu = sigma_sq_process
    k_base = (sigma_for_mu / size_tau.clip(lower=1e-9)).clip(lower=1e-9)
    k_eff = (k_base * size_scale).clip(lower=1e-9)

    credibility = n_pos / (n_pos + k_eff)
    credibility = credibility.fillna(0.0).clip(lower=0.0, upper=1.0)

    mean_filled = mean_mle.fillna(size_mu)
    shrunk_mean_log = credibility * mean_filled + (1.0 - credibility) * size_mu

    posterior_var_mu = sigma_for_mu / (n_pos + k_eff)
    posterior_var_mu = posterior_var_mu.where(n_pos > 0, sigma_for_mu / k_eff)
    posterior_var_mu = posterior_var_mu.fillna(sigma_for_mu / k_eff)
    posterior_var_mu = posterior_var_mu.clip(lower=1e-9)

    idx = stats.index
    return (
        p_post.reindex(idx),
        shrunk_mean_log.reindex(idx),
        posterior_var_mu.reindex(idx),
        sigma_sq_process.reindex(idx),
    )


def _fit_bootstrap_group_hypers(
    stats: pd.DataFrame,
    n_draws: int,
    seed: Optional[int],
    base_alpha_by_group: pd.Series,
    base_beta_by_group: pd.Series,
    base_size_mu_by_group: pd.Series,
    base_size_sigma_by_group: pd.Series,
    base_size_tau_by_group: pd.Series,
    group_shrink_strength: float = 0.0,
) -> Optional[Dict[str, pd.DataFrame]]:
    if n_draws <= 0:
        return None

    rng = np.random.default_rng(seed)
    groups = base_alpha_by_group.index
    cols = [f"draw_{i}" for i in range(n_draws)]

    alpha_draws = pd.DataFrame(index=groups, columns=cols, dtype=float)
    beta_draws = pd.DataFrame(index=groups, columns=cols, dtype=float)
    size_mu_draws = pd.DataFrame(index=groups, columns=cols, dtype=float)
    size_sigma_draws = pd.DataFrame(index=groups, columns=cols, dtype=float)
    size_tau_draws = pd.DataFrame(index=groups, columns=cols, dtype=float)

    if stats.empty:
        for col in cols:
            alpha_draws[col] = base_alpha_by_group
            beta_draws[col] = base_beta_by_group
            size_mu_draws[col] = base_size_mu_by_group
            size_sigma_draws[col] = base_size_sigma_by_group
            size_tau_draws[col] = base_size_tau_by_group
        return {
            "alpha": alpha_draws,
            "beta": beta_draws,
            "size_mu": size_mu_draws,
            "size_sigma": size_sigma_draws,
            "size_tau": size_tau_draws,
        }

    for col in cols:
        sample_idx = rng.integers(0, len(stats), size=len(stats))
        sampled = stats.iloc[sample_idx].copy().reset_index(drop=True)
        (
            alpha_b,
            beta_b,
            size_mu_b,
            size_sigma_b,
            size_tau_b,
        ) = _estimate_group_hypers(
            sampled,
            fallback=(
                float(base_alpha_by_group.get(GLOBAL_GROUP, base_alpha_by_group.iloc[0])),
                float(base_beta_by_group.get(GLOBAL_GROUP, base_beta_by_group.iloc[0])),
                float(base_size_mu_by_group.get(GLOBAL_GROUP, base_size_mu_by_group.iloc[0])),
                float(base_size_sigma_by_group.get(GLOBAL_GROUP, base_size_sigma_by_group.iloc[0])),
                float(base_size_tau_by_group.get(GLOBAL_GROUP, base_size_tau_by_group.iloc[0])),
            ),
            group_shrink_strength=group_shrink_strength,
        )
        alpha_draws[col] = alpha_b.reindex(groups).fillna(base_alpha_by_group)
        beta_draws[col] = beta_b.reindex(groups).fillna(base_beta_by_group)
        size_mu_draws[col] = size_mu_b.reindex(groups).fillna(base_size_mu_by_group)
        size_sigma_draws[col] = size_sigma_b.reindex(groups).fillna(base_size_sigma_by_group)
        size_tau_draws[col] = size_tau_b.reindex(groups).fillna(base_size_tau_by_group)

    return {
        "alpha": alpha_draws,
        "beta": beta_draws,
        "size_mu": size_mu_draws,
        "size_sigma": size_sigma_draws,
        "size_tau": size_tau_draws,
    }


def fit_tsb_hb(
    train_df: pd.DataFrame,
    group_labels: Optional[pd.Series | Dict[str, str]] = None,
    bootstrap_draws: int = 0,
    bootstrap_seed: Optional[int] = None,
    group_shrink_strength: float = 0.0,
    dynamic_occurrence: bool = False,
    occurrence_discount: float = 1.0,
    adaptive_prior_strength: bool = False,
    prior_strength_min: float = 1.0,
    prior_strength_max: float = 1.0,
    prior_strength_power: float = 1.0,
    item_variance_mode: str = "group",
    item_variance_shrink_strength: float = 20.0,
) -> TSBHBParams:
    """Fit TSB-HB with optional group-aware priors and hyperparameter bootstrap.

    - `group_labels`: maps unique_id -> group.
      Use this for regime-aware (ADI/CV^2 groups) or hierarchy-aware (M5 group) priors.
    - `bootstrap_draws`: if >0, estimates group-level hyperparameter uncertainty via
      bootstrap resampling across series.
    """
    stats = _compute_series_stats(train_df, group_labels=group_labels)
    (
        alpha_by_group,
        beta_by_group,
        size_mu_by_group,
        size_sigma_by_group,
        size_tau_by_group,
    ) = _estimate_group_hypers(
        stats,
        group_shrink_strength=group_shrink_strength,
    )

    p_post, shrunk_mean_log, posterior_var_mu, sigma_sq_process = _compute_posteriors_from_stats(
        stats=stats,
        alpha_by_group=alpha_by_group,
        beta_by_group=beta_by_group,
        size_mu_by_group=size_mu_by_group,
        size_sigma_by_group=size_sigma_by_group,
        size_tau_by_group=size_tau_by_group,
        adaptive_prior_strength=adaptive_prior_strength,
        prior_strength_min=prior_strength_min,
        prior_strength_max=prior_strength_max,
        prior_strength_power=prior_strength_power,
        item_variance_mode=item_variance_mode,
        item_variance_shrink_strength=item_variance_shrink_strength,
    )

    bootstrap_group_hypers = _fit_bootstrap_group_hypers(
        stats=stats,
        n_draws=int(max(bootstrap_draws, 0)),
        seed=bootstrap_seed,
        base_alpha_by_group=alpha_by_group,
        base_beta_by_group=beta_by_group,
        base_size_mu_by_group=size_mu_by_group,
        base_size_sigma_by_group=size_sigma_by_group,
        base_size_tau_by_group=size_tau_by_group,
        group_shrink_strength=group_shrink_strength,
    )

    return TSBHBParams(
        p_posterior=p_post,
        shrunk_mean_log=shrunk_mean_log,
        posterior_var_mu=posterior_var_mu,
        sigma_sq_process=sigma_sq_process,
        group_labels=stats["group"].astype(str),
        alpha_by_group=alpha_by_group,
        beta_by_group=beta_by_group,
        size_global_mean_by_group=size_mu_by_group,
        size_sigma_sq_by_group=size_sigma_by_group,
        size_tau_sq_by_group=size_tau_by_group,
        n_obs=stats["n_obs"],
        s_obs=stats["s_obs"],
        n_pos=stats["n_pos"],
        sum_log=stats["sum_log"],
        sum_sq_log=stats["sum_sq_log"],
        bootstrap_group_hypers=bootstrap_group_hypers,
        group_shrink_strength=float(max(group_shrink_strength, 0.0)),
        dynamic_occurrence=bool(dynamic_occurrence),
        occurrence_discount=_clip_occurrence_discount(occurrence_discount),
        adaptive_prior_strength=bool(adaptive_prior_strength),
        prior_strength_min=float(prior_strength_min),
        prior_strength_max=float(prior_strength_max),
        prior_strength_power=float(max(prior_strength_power, 0.0)),
        item_variance_mode=str(item_variance_mode).lower(),
        item_variance_shrink_strength=float(max(item_variance_shrink_strength, 1e-6)),
        item_var_log=stats["var_log"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0),
    )


def _resolve_uid_series_value(series: pd.Series, uid: str, default: float) -> float:
    val = series.get(uid, default)
    try:
        out = float(val)
    except Exception:
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    return out


def _resolve_uid_group(params: TSBHBParams, uid: str) -> str:
    grp = params.group_labels.get(uid, GLOBAL_GROUP)
    if pd.isna(grp):
        return GLOBAL_GROUP
    return str(grp)


def predict_tsb_hb(
    params: TSBHBParams,
    eval_df: pd.DataFrame,
    quantiles: Optional[List[float]] = None,
    n_samples: int = 2000,
    include_hyper_uncertainty: bool = True,
) -> pd.DataFrame:
    """Predict on the evaluation set.

    - Point forecast: per-series constant mean across horizon.
    - Probabilistic forecast:
      - default plug-in EB uncertainty (posterior + process variance),
      - optional bootstrap hyperparameter uncertainty if available in params.
    """
    out = eval_df[["unique_id", "ds"]].copy()

    if quantiles is None or len(quantiles) == 0:
        sigma = out["unique_id"].map(params.sigma_sq_process).fillna(params.sigma_sq_process.mean())
        size_mean = np.exp(
            out["unique_id"].map(params.shrunk_mean_log).fillna(params.shrunk_mean_log.mean())
            + sigma / 2.0
        )
        p = out["unique_id"].map(params.p_posterior).fillna(params.p_posterior.mean()).clip(lower=0.0, upper=1.0)
        out["yhat"] = (p * size_mean).fillna(0.0)
        return out

    qcols = [f"q_{q}" for q in quantiles]
    rows: list[pd.DataFrame] = []
    uids = out["unique_id"].dropna().astype(str).unique().tolist()
    rng = np.random.default_rng()
    prior_min = float(max(params.prior_strength_min, 1e-6))
    prior_max = float(max(params.prior_strength_max, 1e-6))
    if prior_min > prior_max:
        prior_min, prior_max = prior_max, prior_min
    prior_power = float(max(params.prior_strength_power, 0.0))
    var_mode = str(params.item_variance_mode).lower()
    if var_mode == "shrink_item":
        var_mode = "conjugate"
    if var_mode not in {"group", "conjugate"}:
        var_mode = "group"
    var_prior_df = float(max(params.item_variance_shrink_strength, 2.1))

    group_ref_n_obs: dict[str, float] = {}
    group_ref_n_pos: dict[str, float] = {}
    global_ref_n_obs = 1.0
    global_ref_n_pos = 1.0
    if params.adaptive_prior_strength:
        ref_df = pd.DataFrame(
            {
                "group": params.group_labels.astype(str),
                "n_obs": params.n_obs.reindex(params.group_labels.index).fillna(0.0),
                "n_pos": params.n_pos.reindex(params.group_labels.index).fillna(0.0),
            }
        )
        if not ref_df.empty:
            group_ref_n_obs = ref_df.groupby("group", sort=False)["n_obs"].median().to_dict()
            group_ref_n_pos = ref_df.groupby("group", sort=False)["n_pos"].median().to_dict()
            global_ref_n_obs = float(np.nanmedian(ref_df["n_obs"].to_numpy()))
            pos_counts = ref_df["n_pos"][ref_df["n_pos"] > 0].to_numpy()
            global_ref_n_pos = float(np.nanmedian(pos_counts)) if len(pos_counts) > 0 else 1.0
        if not np.isfinite(global_ref_n_obs) or global_ref_n_obs <= 0:
            global_ref_n_obs = 1.0
        if not np.isfinite(global_ref_n_pos) or global_ref_n_pos <= 0:
            global_ref_n_pos = 1.0

    use_bootstrap = (
        include_hyper_uncertainty
        and params.bootstrap_group_hypers is not None
        and "alpha" in params.bootstrap_group_hypers
        and not params.bootstrap_group_hypers["alpha"].empty
    )

    for uid in uids:
        ds_vals = out.loc[out["unique_id"].astype(str) == uid, "ds"].to_numpy()
        if len(ds_vals) == 0:
            continue

        grp_for_uid = _resolve_uid_group(params, uid)
        n_obs = _resolve_uid_series_value(params.n_obs, uid, 0.0)
        s_obs = _resolve_uid_series_value(params.s_obs, uid, 0.0)
        n_pos = _resolve_uid_series_value(params.n_pos, uid, 0.0)
        sum_log = _resolve_uid_series_value(params.sum_log, uid, 0.0)
        mean_mle = sum_log / n_pos if n_pos > 0 else np.nan
        item_var = np.nan
        if params.item_var_log is not None:
            item_var = _resolve_uid_series_value(params.item_var_log, uid, np.nan)
        occ_scale = 1.0
        size_scale = 1.0
        if params.adaptive_prior_strength:
            ref_occ = float(group_ref_n_obs.get(grp_for_uid, global_ref_n_obs))
            if not np.isfinite(ref_occ) or ref_occ <= 0:
                ref_occ = global_ref_n_obs
            occ_scale = float(np.clip((ref_occ / max(n_obs, 1.0)) ** prior_power, prior_min, prior_max))

            ref_pos = float(group_ref_n_pos.get(grp_for_uid, global_ref_n_pos))
            if not np.isfinite(ref_pos) or ref_pos <= 0:
                ref_pos = global_ref_n_pos
            size_scale = float(np.clip((ref_pos / max(n_pos, 1.0)) ** prior_power, prior_min, prior_max))

        if use_bootstrap:
            grp = grp_for_uid
            alpha_df = params.bootstrap_group_hypers["alpha"]
            beta_df = params.bootstrap_group_hypers["beta"]
            mu_df = params.bootstrap_group_hypers["size_mu"]
            sigma_df = params.bootstrap_group_hypers["size_sigma"]
            tau_df = params.bootstrap_group_hypers["size_tau"]
            if grp not in alpha_df.index:
                grp = GLOBAL_GROUP if GLOBAL_GROUP in alpha_df.index else alpha_df.index[0]

            alpha_draws = alpha_df.loc[grp].to_numpy(dtype=float)
            beta_draws = beta_df.loc[grp].to_numpy(dtype=float)
            mu_draws = mu_df.loc[grp].to_numpy(dtype=float)
            sigma_draws = np.maximum(sigma_df.loc[grp].to_numpy(dtype=float), 1e-9)
            tau_draws = np.maximum(tau_df.loc[grp].to_numpy(dtype=float), 1e-9)

            if len(alpha_draws) == 0:
                use_bootstrap = False
            else:
                draw_idx = rng.integers(0, len(alpha_draws), size=n_samples)
                alpha_s = alpha_draws[draw_idx]
                beta_s = beta_draws[draw_idx]
                mu_global_s = mu_draws[draw_idx]
                sigma_s = sigma_draws[draw_idx]
                tau_s = tau_draws[draw_idx]
                alpha_s_eff = np.maximum(alpha_s * occ_scale, 1e-9)
                beta_s_eff = np.maximum(beta_s * occ_scale, 1e-9)
                p_s = (alpha_s_eff + s_obs) / (alpha_s_eff + beta_s_eff + n_obs)
                p_s = np.clip(p_s, 0.0, 1.0)

                if var_mode == "conjugate" and np.isfinite(item_var) and n_pos > 1.0:
                    nu = var_prior_df
                    denom = nu + n_pos - 3.0
                    if denom > 1e-9:
                        sigma_s = np.maximum(((nu - 2.0) * sigma_s + (n_pos - 1.0) * item_var) / denom, 1e-9)
                k_s = np.maximum((sigma_s / tau_s) * size_scale, 1e-9)

                if n_pos > 0:
                    credibility = n_pos / (n_pos + k_s)
                    mu_s = credibility * mean_mle + (1.0 - credibility) * mu_global_s
                    var_mu_s = sigma_s / (n_pos + k_s)
                else:
                    mu_s = mu_global_s
                    var_mu_s = sigma_s / k_s
                pred_std_s = np.sqrt(np.maximum(sigma_s + var_mu_s, 1e-9))

                demand_occurs = rng.binomial(1, p_s, n_samples)
                log_samples = rng.normal(loc=mu_s, scale=pred_std_s, size=n_samples)
                samples = np.exp(log_samples) * demand_occurs
                qvals = {f"q_{q}": float(np.quantile(samples, q)) for q in quantiles}
                qvals["prob_zero_predicted"] = float(np.mean(1.0 - p_s))
                tmp = pd.DataFrame({**qvals, "unique_id": uid, "ds": ds_vals})
                rows.append(tmp)
                continue

        p = _resolve_uid_series_value(params.p_posterior, uid, 0.0)
        mu = _resolve_uid_series_value(params.shrunk_mean_log, uid, 0.0)
        sigma = _resolve_uid_series_value(params.sigma_sq_process, uid, 1e-6)
        var_mu = _resolve_uid_series_value(params.posterior_var_mu, uid, 1e-6)
        pred_std = float(np.sqrt(max(sigma + var_mu, 1e-9)))

        demand_occurs = rng.binomial(1, np.clip(p, 0.0, 1.0), n_samples)
        log_samples = rng.normal(loc=mu, scale=pred_std, size=n_samples)
        samples = np.exp(log_samples) * demand_occurs
        qvals = {f"q_{q}": float(np.quantile(samples, q)) for q in quantiles}
        qvals["prob_zero_predicted"] = float(1.0 - np.clip(p, 0.0, 1.0))
        tmp = pd.DataFrame({**qvals, "unique_id": uid, "ds": ds_vals})
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["unique_id", "ds"] + qcols)
    return pd.concat(rows, ignore_index=True)


def _params_from_online_state(state: TSBHBOnlineState) -> TSBHBParams:
    idx = state.n_obs.index
    if state.dynamic_occurrence:
        n_occ = state.n_eff_occ.reindex(idx).fillna(0.0)
        s_occ = state.s_eff_occ.reindex(idx).fillna(0.0)
    else:
        n_occ = state.n_obs.reindex(idx).fillna(0.0)
        s_occ = state.s_obs.reindex(idx).fillna(0.0)

    n_pos = state.n_pos.reindex(idx).fillna(0.0)
    sum_log = state.sum_log.reindex(idx).fillna(0.0)
    sum_sq_log = state.sum_sq_log.reindex(idx).fillna(0.0)
    mean_log = (sum_log / n_pos.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    var_num = sum_sq_log - n_pos * (mean_log.fillna(0.0) ** 2)
    var_log = (var_num / (n_pos - 1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    stats = pd.DataFrame(
        {
            "n_obs": n_occ,
            "s_obs": s_occ,
            "n_pos": n_pos,
            "sum_log": sum_log,
            "sum_sq_log": sum_sq_log,
            "var_log": var_log,
            "group": state.group_labels.reindex(idx).fillna(GLOBAL_GROUP).astype(str),
        },
        index=idx,
    )
    p_post, shrunk_mean_log, posterior_var_mu, sigma_sq_process = _compute_posteriors_from_stats(
        stats=stats,
        alpha_by_group=state.alpha_by_group,
        beta_by_group=state.beta_by_group,
        size_mu_by_group=state.size_global_mean_by_group,
        size_sigma_by_group=state.size_sigma_sq_by_group,
        size_tau_by_group=state.size_tau_sq_by_group,
        adaptive_prior_strength=state.adaptive_prior_strength,
        prior_strength_min=state.prior_strength_min,
        prior_strength_max=state.prior_strength_max,
        prior_strength_power=state.prior_strength_power,
        item_variance_mode=state.item_variance_mode,
        item_variance_shrink_strength=state.item_variance_shrink_strength,
    )

    return TSBHBParams(
        p_posterior=p_post,
        shrunk_mean_log=shrunk_mean_log,
        posterior_var_mu=posterior_var_mu,
        sigma_sq_process=sigma_sq_process,
        group_labels=stats["group"],
        alpha_by_group=state.alpha_by_group,
        beta_by_group=state.beta_by_group,
        size_global_mean_by_group=state.size_global_mean_by_group,
        size_sigma_sq_by_group=state.size_sigma_sq_by_group,
        size_tau_sq_by_group=state.size_tau_sq_by_group,
        n_obs=n_occ,
        s_obs=s_occ,
        n_pos=n_pos,
        sum_log=sum_log,
        sum_sq_log=sum_sq_log,
        bootstrap_group_hypers=state.bootstrap_group_hypers,
        group_shrink_strength=0.0,
        dynamic_occurrence=state.dynamic_occurrence,
        occurrence_discount=_clip_occurrence_discount(state.occurrence_discount),
        adaptive_prior_strength=state.adaptive_prior_strength,
        prior_strength_min=state.prior_strength_min,
        prior_strength_max=state.prior_strength_max,
        prior_strength_power=state.prior_strength_power,
        item_variance_mode=state.item_variance_mode,
        item_variance_shrink_strength=state.item_variance_shrink_strength,
        item_var_log=var_log,
    )


def initialize_online_tsb_hb(
    train_df: pd.DataFrame,
    group_labels: Optional[pd.Series | Dict[str, str]] = None,
    bootstrap_draws: int = 0,
    bootstrap_seed: Optional[int] = None,
    group_shrink_strength: float = 0.0,
    dynamic_occurrence: bool = False,
    occurrence_discount: float = 1.0,
    adaptive_prior_strength: bool = False,
    prior_strength_min: float = 1.0,
    prior_strength_max: float = 1.0,
    prior_strength_power: float = 1.0,
    item_variance_mode: str = "group",
    item_variance_shrink_strength: float = 20.0,
) -> TSBHBOnlineState:
    occ_discount = _clip_occurrence_discount(occurrence_discount)
    params = fit_tsb_hb(
        train_df=train_df,
        group_labels=group_labels,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        group_shrink_strength=group_shrink_strength,
        dynamic_occurrence=dynamic_occurrence,
        occurrence_discount=occ_discount,
        adaptive_prior_strength=adaptive_prior_strength,
        prior_strength_min=prior_strength_min,
        prior_strength_max=prior_strength_max,
        prior_strength_power=prior_strength_power,
        item_variance_mode=item_variance_mode,
        item_variance_shrink_strength=item_variance_shrink_strength,
    )
    return TSBHBOnlineState(
        n_obs=params.n_obs.copy(),
        s_obs=params.s_obs.copy(),
        n_eff_occ=params.n_obs.copy(),
        s_eff_occ=params.s_obs.copy(),
        n_pos=params.n_pos.copy(),
        sum_log=params.sum_log.copy(),
        sum_sq_log=params.sum_sq_log.copy(),
        group_labels=params.group_labels.copy(),
        alpha_by_group=params.alpha_by_group.copy(),
        beta_by_group=params.beta_by_group.copy(),
        size_global_mean_by_group=params.size_global_mean_by_group.copy(),
        size_sigma_sq_by_group=params.size_sigma_sq_by_group.copy(),
        size_tau_sq_by_group=params.size_tau_sq_by_group.copy(),
        bootstrap_group_hypers=params.bootstrap_group_hypers,
        dynamic_occurrence=bool(dynamic_occurrence),
        occurrence_discount=occ_discount,
        adaptive_prior_strength=bool(adaptive_prior_strength),
        prior_strength_min=float(prior_strength_min),
        prior_strength_max=float(prior_strength_max),
        prior_strength_power=float(max(prior_strength_power, 0.0)),
        item_variance_mode=str(item_variance_mode).lower(),
        item_variance_shrink_strength=float(max(item_variance_shrink_strength, 1e-6)),
    )


def predict_online_tsb_hb(
    state: TSBHBOnlineState,
    eval_df: pd.DataFrame,
    quantiles: Optional[List[float]] = None,
    n_samples: int = 2000,
    include_hyper_uncertainty: bool = False,
) -> pd.DataFrame:
    params = _params_from_online_state(state)
    return predict_tsb_hb(
        params=params,
        eval_df=eval_df,
        quantiles=quantiles,
        n_samples=n_samples,
        include_hyper_uncertainty=include_hyper_uncertainty,
    )


def update_online_tsb_hb(state: TSBHBOnlineState, observed_df: pd.DataFrame) -> TSBHBOnlineState:
    if observed_df.empty:
        return state

    obs = observed_df[["unique_id", "y"]].copy()
    obs["occ"] = (obs["y"] > 0).astype(float)
    obs["log_y"] = np.nan
    pos_mask = obs["y"] > 0
    obs.loc[pos_mask, "log_y"] = np.log(obs.loc[pos_mask, "y"].astype(float))

    g = obs.groupby("unique_id", sort=False)
    n_add = g["y"].size().astype(float)
    s_add = g["occ"].sum().astype(float)
    n_pos_add = g["log_y"].count().astype(float)
    sum_log_add = g["log_y"].sum(min_count=1).fillna(0.0).astype(float)
    sum_sq_log_add = g["log_y"].apply(lambda x: float(np.square(x.dropna()).sum())).astype(float)

    state.n_obs = state.n_obs.add(n_add, fill_value=0.0)
    state.s_obs = state.s_obs.add(s_add, fill_value=0.0)
    state.n_pos = state.n_pos.add(n_pos_add, fill_value=0.0)
    state.sum_log = state.sum_log.add(sum_log_add, fill_value=0.0)
    state.sum_sq_log = state.sum_sq_log.add(sum_sq_log_add, fill_value=0.0)

    if state.dynamic_occurrence:
        discount = _clip_occurrence_discount(state.occurrence_discount)
        state.n_eff_occ = state.n_eff_occ.mul(discount, fill_value=0.0).add(n_add, fill_value=0.0)
        state.s_eff_occ = state.s_eff_occ.mul(discount, fill_value=0.0).add(s_add, fill_value=0.0)
    else:
        state.n_eff_occ = state.n_obs.copy()
        state.s_eff_occ = state.s_obs.copy()

    for uid in n_add.index:
        if uid not in state.group_labels.index:
            state.group_labels.loc[uid] = GLOBAL_GROUP

    # Keep deterministic ordering for reproducibility
    state.n_obs = state.n_obs.sort_index()
    state.s_obs = state.s_obs.reindex(state.n_obs.index).fillna(0.0)
    state.n_eff_occ = state.n_eff_occ.reindex(state.n_obs.index).fillna(0.0)
    state.s_eff_occ = state.s_eff_occ.reindex(state.n_obs.index).fillna(0.0)
    state.n_pos = state.n_pos.reindex(state.n_obs.index).fillna(0.0)
    state.sum_log = state.sum_log.reindex(state.n_obs.index).fillna(0.0)
    state.sum_sq_log = state.sum_sq_log.reindex(state.n_obs.index).fillna(0.0)
    state.group_labels = state.group_labels.reindex(state.n_obs.index).fillna(GLOBAL_GROUP).astype(str)
    return state
