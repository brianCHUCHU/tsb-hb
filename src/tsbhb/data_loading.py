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


def convert_m5_wide_to_long(sales_wide_path: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
    """Convert M5 wide format (sales_train_evaluation.csv) to long format.
    
    Args:
        sales_wide_path: Path to sales_train_evaluation.csv or sales_train_validation.csv
        output_path: Optional path to save the long format CSV
        
    Returns:
        DataFrame in long format with columns: [series_id, item_id, store_id, d, sales]
    """
    # Read wide format data
    df_wide = pd.read_csv(sales_wide_path)
    
    # Extract metadata columns
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    meta_cols = [c for c in id_cols if c in df_wide.columns]
    
    # Find all day columns (d_1, d_2, ..., d_1941, etc.)
    day_cols = [c for c in df_wide.columns if c.startswith('d_')]
    
    # Melt to long format
    df_long = df_wide.melt(
        id_vars=meta_cols,
        value_vars=day_cols,
        var_name='d',
        value_name='sales'
    )
    
    # Create series_id if not exists (combination of item and store)
    if 'id' in df_long.columns:
        df_long = df_long.rename(columns={'id': 'series_id'})
    elif 'item_id' in df_long.columns and 'store_id' in df_long.columns:
        df_long['series_id'] = df_long['item_id'] + '_' + df_long['store_id']
    
    # Select and order columns
    output_cols = ['series_id', 'item_id', 'store_id', 'd', 'sales']
    output_cols = [c for c in output_cols if c in df_long.columns]
    df_long = df_long[output_cols].copy()
    
    # Sort by series and day
    df_long = df_long.sort_values(['series_id', 'd']).reset_index(drop=True)
    
    # Optionally save to file
    if output_path is not None:
        df_long.to_csv(output_path, index=False)
        print(f"Long format data saved to: {output_path}")
    
    return df_long


def load_m5_long(sales_path: Path, calendar_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load M5 data in long format.
    
    If sales_path is a wide format file (contains d_1, d_2, etc. columns),
    it will be automatically converted to long format.
    """
    calendar_df = pd.read_csv(calendar_path)
    
    # Try to read and detect format
    sales_df = pd.read_csv(sales_path, nrows=0)  # Read only header
    
    # Check if it's wide format (has d_1, d_2, etc. columns)
    day_cols = [c for c in sales_df.columns if c.startswith('d_') and c[2:].isdigit()]
    
    if len(day_cols) > 0:
        # Wide format detected, convert to long
        print(f"Wide format detected. Converting {sales_path} to long format...")
        sales_df = convert_m5_wide_to_long(sales_path)
        print(f"Conversion complete. Shape: {sales_df.shape}")
    else:
        # Already in long format
        sales_df = pd.read_csv(sales_path)
    
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

