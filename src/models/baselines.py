from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandas as pd


def _import_statsforecast():
    from statsforecast import StatsForecast
    from statsforecast.models import (
        CrostonClassic,
        CrostonSBA,
        TSB,
        ADIDA,
        IMAPA,
        AutoARIMA,
        AutoTheta,
    )

    return StatsForecast, {
        "CrostonClassic": CrostonClassic,
        "CrostonSBA": CrostonSBA,
        "TSB": TSB,
        "ADIDA": ADIDA,
        "IMAPA": IMAPA,
        "AutoARIMA": AutoARIMA,
        "AutoTheta": AutoTheta,
    }


def _prepare_horizons(horizons: pd.Series) -> pd.DataFrame:
    """Normalize horizon series into a two-column frame: [unique_id, h]."""
    if horizons.empty:
        return pd.DataFrame(columns=["unique_id", "h"])

    h = horizons.astype(int)
    h = h[h > 0]
    if h.empty:
        return pd.DataFrame(columns=["unique_id", "h"])

    h_df = h.rename("h").reset_index()
    uid_col = h_df.columns[0]
    if uid_col != "unique_id":
        h_df = h_df.rename(columns={uid_col: "unique_id"})
    return h_df[["unique_id", "h"]]


def _fit_predict_panel(
    train_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    models: list,
    freq: str,
    probabilistic: bool,
    levels: Optional[List[int]],
) -> pd.DataFrame:
    """Fit panel baselines once and trim per-series horizons."""
    if horizon_df.empty:
        return pd.DataFrame()

    StatsForecast, _ = _import_statsforecast()
    valid_uids = set(horizon_df["unique_id"].tolist())
    train_panel = train_df.loc[train_df["unique_id"].isin(valid_uids), ["unique_id", "ds", "y"]]
    if train_panel.empty:
        return pd.DataFrame()

    h_max = int(horizon_df["h"].max())
    sf = StatsForecast(models=models, freq=freq, n_jobs=-1)
    sf.fit(df=train_panel)

    if probabilistic:
        try:
            pred = sf.predict(h=h_max, level=levels or [80, 50])
        except TypeError:
            pred = sf.forecast(df=train_panel, h=h_max, level=levels or [80, 50])
    else:
        pred = sf.predict(h=h_max)

    out = pred.reset_index()
    out["h_step"] = out.groupby("unique_id").cumcount() + 1
    out = out.merge(horizon_df, on="unique_id", how="inner")
    out = out[out["h_step"] <= out["h"]].drop(columns=["h_step", "h"])
    return out.reset_index(drop=True)


def fit_predict_baselines(
    train_df: pd.DataFrame,
    horizons: pd.Series,
    freq: str = "D",
    tsb_grid: Optional[Sequence[Tuple[float, float]]] = None,
    probabilistic: bool = False,
    levels: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Fit StatsForecast baselines on panel data and return predictions."""
    _, M = _import_statsforecast()
    horizon_df = _prepare_horizons(horizons)

    if tsb_grid is not None:
        outputs = []
        for alpha_d, alpha_p in tsb_grid:
            tmp = _fit_predict_panel(
                train_df=train_df,
                horizon_df=horizon_df,
                models=[M["TSB"](alpha_d=alpha_d, alpha_p=alpha_p)],
                freq=freq,
                probabilistic=probabilistic,
                levels=levels,
            )
            if tmp.empty:
                continue
            tmp["alpha_d"] = alpha_d
            tmp["alpha_p"] = alpha_p
            outputs.append(tmp)
        return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()

    if probabilistic:
        models = [
            M["AutoARIMA"](season_length=7),
            M["AutoTheta"](season_length=7),
        ]
    else:
        models = [
            M["CrostonClassic"](),
            M["CrostonSBA"](),
            M["TSB"](alpha_d=0.5, alpha_p=0.45),
            M["ADIDA"](),
            M["IMAPA"](),
            M["AutoTheta"](),
            M["AutoARIMA"](),
        ]

    return _fit_predict_panel(
        train_df=train_df,
        horizon_df=horizon_df,
        models=models,
        freq=freq,
        probabilistic=probabilistic,
        levels=levels,
    )
