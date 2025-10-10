from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
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


def fit_predict_baselines(
    train_df: pd.DataFrame,
    horizons: pd.Series,
    freq: str = "D",
    tsb_grid: Optional[Sequence[Tuple[float, float]]] = None,
    probabilistic: bool = False,
    levels: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Fit StatsForecast baselines per-series and return predictions.

    Parameters
    - train_df: DataFrame with columns [unique_id, ds, y]
    - horizons: pandas Series mapping unique_id -> horizon (int)
    - freq: pandas frequency string
    - tsb_grid: if provided, only fit TSB over the grid of (alpha_d, alpha_p)
    - probabilistic: if True, include prediction intervals via `levels`
    - levels: list of confidence levels, e.g., [80, 50]
    """
    StatsForecast, M = _import_statsforecast()

    uids = train_df["unique_id"].unique().tolist()
    outputs = []

    if tsb_grid is not None:
        # Grid for TSB only
        for alpha_d, alpha_p in tsb_grid:
            model = M["TSB"](alpha_d=alpha_d, alpha_p=alpha_p)
            sf = StatsForecast(models=[model], freq=freq, n_jobs=-1)
            for uid in uids:
                h = int(horizons.get(uid, 0))
                if h <= 0:
                    continue
                df_uid = train_df.loc[train_df["unique_id"] == uid, ["unique_id", "ds", "y"]]
                if df_uid.empty:
                    continue
                sf.fit(df=df_uid)
                if probabilistic:
                    pred = sf.forecast(df=df_uid, h=h, level=levels or [80])
                else:
                    pred = sf.predict(h=h)
                tmp = pred.reset_index()
                tmp["alpha_d"] = alpha_d
                tmp["alpha_p"] = alpha_p
                outputs.append(tmp)
        return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()

    # Default full baseline set
    models = [
        M["CrostonClassic"](),
        M["CrostonSBA"](),
        M["TSB"](alpha_d=0.5, alpha_p=0.45),
        M["ADIDA"](),
        M["IMAPA"](),
        M["AutoTheta"](),
        M["AutoARIMA"](),
    ]
    sf = StatsForecast(models=models, freq=freq, n_jobs=-1)

    for uid in uids:
        h = int(horizons.get(uid, 0))
        if h <= 0:
            continue
        df_uid = train_df.loc[train_df["unique_id"] == uid, ["unique_id", "ds", "y"]]
        if df_uid.empty:
            continue
        sf.fit(df=df_uid)
        if probabilistic:
            pred = sf.forecast(df=df_uid, h=h, level=levels or [80, 50])
        else:
            pred = sf.predict(h=h)
        outputs.append(pred.reset_index())

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()

