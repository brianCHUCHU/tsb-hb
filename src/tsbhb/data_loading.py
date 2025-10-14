from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import random


def load_online_retail(path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(path, encoding="latin1")
    return df_raw


def preprocess_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only required columns and filter
    df = df[["InvoiceDate", "StockCode", "Quantity", "UnitPrice"]].copy()
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

    # Cap extreme quantities at 99.5% quantile
    q_cap = df["Quantity"].quantile(0.995)
    df["Quantity"] = df["Quantity"].clip(upper=q_cap)

    # Date handling and daily aggregation
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["ds"] = df["InvoiceDate"].dt.date
    df = (
        df.groupby(["StockCode", "ds"], as_index=False)["Quantity"].sum()
        .rename(columns={"StockCode": "unique_id", "Quantity": "y"})
    )
    df["ds"] = pd.to_datetime(df["ds"])  # normalize as datetime64[ns]

    # Densify to daily frequency per series
    df = (
        df.set_index("ds").groupby("unique_id")["y"].apply(lambda x: x.asfreq("D", fill_value=0))
        .reset_index()
    )

    # Sort for reproducibility
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df


def train_eval_split_fixed_origin(df: pd.DataFrame, init_ratio: float = 1.0 / 3, min_len: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["t"] = df.groupby("unique_id").cumcount()
    df["L"] = df.groupby("unique_id")["t"].transform("max") + 1
    df = df[df["L"] >= min_len].copy()
    init_mask = df["t"] < np.floor(df["L"] * init_ratio)
    init_set = df[init_mask].copy()
    eval_set = df[~init_mask].copy()
    return init_set, eval_set


def load_m5_long(sales_path: Path, calendar_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sales_df = pd.read_csv(sales_path)
    calendar_df = pd.read_csv(calendar_path)
    return sales_df, calendar_df


def preprocess_m5(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    sample_size: Optional[int] = 5000,
) -> pd.DataFrame:
    df = sales_df.copy()
    cal = calendar_df.copy()

    if "series_id" in df.columns:
        df = df.rename(columns={"series_id": "id"})
    if "sales" in df.columns:
        df = df.rename(columns={"sales": "y"})

    if sample_size is not None and "id" in df.columns:
        unique_ids = df["id"].dropna().unique().tolist()
        if sample_size < len(unique_ids):
            selected = random.sample(unique_ids, sample_size)
            df = df[df["id"].isin(selected)].copy()

    cal = cal[["d", "date"]].copy()
    cal["date"] = pd.to_datetime(cal["date"])

    df = df.merge(cal, on="d", how="left")
    df = df.rename(columns={"id": "unique_id", "date": "ds"})
    df = df.dropna(subset=["ds"]).copy()
    df["ds"] = pd.to_datetime(df["ds"])

    df = df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df

