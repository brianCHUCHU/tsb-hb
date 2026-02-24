from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class HurdleLocalLogNormalParams:
    p_hat: pd.Series
    mean_log: pd.Series
    var_log: pd.Series


@dataclass
class HurdleGlobalLogNormalParams:
    p_hat: float
    mean_log: float
    var_log: float


def _safe_log_stats(log_values: pd.Series) -> tuple[float, float]:
    if log_values.empty:
        return 0.0, 1.0
    mean_log = float(log_values.mean())
    if len(log_values) <= 1:
        return mean_log, 0.0
    var_log = float(log_values.var(ddof=1))
    if not np.isfinite(var_log) or var_log < 0:
        var_log = 0.0
    return mean_log, var_log


def _mixture_quantile_lognormal(p: float, mean_log: float, var_log: float, q: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    q = float(np.clip(q, 0.0, 1.0))

    if p <= 0:
        return 0.0

    mass_at_zero = 1.0 - p
    if q <= mass_at_zero:
        return 0.0

    q_adj = (q - mass_at_zero) / p
    q_adj = float(np.clip(q_adj, 1e-12, 1 - 1e-12))

    if var_log <= 0:
        return float(np.exp(mean_log))
    return float(np.exp(mean_log + np.sqrt(var_log) * norm.ppf(q_adj)))


def fit_hurdle_local_lognormal(train_df: pd.DataFrame) -> HurdleLocalLogNormalParams:
    data = train_df[["unique_id", "y"]].copy()
    data["occ"] = (data["y"] > 0).astype(int)

    g = data.groupby("unique_id")
    p_hat = (g["occ"].sum() / g["occ"].count()).astype(float)

    size_df = data.loc[data["y"] > 0, ["unique_id", "y"]].copy()
    size_df["log_y"] = np.log(size_df["y"].astype(float))
    size_stats = size_df.groupby("unique_id")["log_y"].agg(n_pos="count", mean_log="mean", var_log="var")

    idx = p_hat.index
    mean_log = size_stats["mean_log"].reindex(idx)
    var_log = size_stats["var_log"].reindex(idx)
    n_pos = size_stats["n_pos"].reindex(idx).fillna(0)

    # Strict local baseline: no global fallback; no positive samples => unused stats (set to 0).
    mean_log = mean_log.fillna(0.0)
    var_log = np.where(n_pos.values == 0, 0.0, var_log.values)
    var_log = np.where(n_pos.values == 1, 0.0, var_log)
    var_log = pd.Series(var_log, index=idx).fillna(0.0).clip(lower=0.0)

    return HurdleLocalLogNormalParams(
        p_hat=p_hat,
        mean_log=mean_log,
        var_log=var_log,
    )


def predict_hurdle_local_lognormal(
    params: HurdleLocalLogNormalParams,
    eval_df: pd.DataFrame,
    quantiles: Iterable[float] | None = None,
) -> pd.DataFrame:
    out = eval_df[["unique_id", "ds"]].copy()

    p = out["unique_id"].map(params.p_hat)
    if p.isna().any():
        missing_ids = out.loc[p.isna(), "unique_id"].drop_duplicates().tolist()
        preview = ", ".join(str(x) for x in missing_ids[:10])
        raise ValueError(
            f"Hurdle local baseline requires per-series training stats; "
            f"found {len(missing_ids)} unseen unique_id(s) in eval: {preview}"
        )
    p = p.clip(0.0, 1.0)
    mean_log = out["unique_id"].map(params.mean_log).fillna(0.0)
    var_log = out["unique_id"].map(params.var_log).fillna(0.0).clip(lower=0.0)

    out["Hurdle-Local-LogNormal"] = p.values * np.exp(mean_log.values + 0.5 * var_log.values)

    qs = list(quantiles) if quantiles is not None else []
    for q in qs:
        out[f"q_{q}"] = [
            _mixture_quantile_lognormal(float(pp), float(mm), float(vv), float(q))
            for pp, mm, vv in zip(p.values, mean_log.values, var_log.values)
        ]

    return out


def fit_hurdle_global_lognormal(train_df: pd.DataFrame) -> HurdleGlobalLogNormalParams:
    data = train_df[["y"]].copy()
    if data.empty:
        return HurdleGlobalLogNormalParams(p_hat=0.0, mean_log=0.0, var_log=1.0)

    occ = (data["y"] > 0).astype(int)
    p_hat = float(occ.mean())

    pos = data.loc[data["y"] > 0, "y"].astype(float)
    mean_log, var_log = _safe_log_stats(np.log(pos) if not pos.empty else pd.Series(dtype=float))

    return HurdleGlobalLogNormalParams(
        p_hat=float(np.clip(p_hat, 0.0, 1.0)),
        mean_log=float(mean_log),
        var_log=float(max(var_log, 0.0)),
    )


def predict_hurdle_global_lognormal(
    params: HurdleGlobalLogNormalParams,
    eval_df: pd.DataFrame,
    quantiles: Iterable[float] | None = None,
) -> pd.DataFrame:
    out = eval_df[["unique_id", "ds"]].copy()

    p = float(np.clip(params.p_hat, 0.0, 1.0))
    mean_log = float(params.mean_log)
    var_log = float(max(params.var_log, 0.0))
    out["Hurdle-Global-LogNormal"] = p * np.exp(mean_log + 0.5 * var_log)

    qs = list(quantiles) if quantiles is not None else []
    for q in qs:
        out[f"q_{q}"] = _mixture_quantile_lognormal(p, mean_log, var_log, float(q))

    return out
