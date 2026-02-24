from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from metrics import mae, me, rmse, rmsse, wrmsse


@dataclass
class WalkForwardFrame:
    step: int
    history: pd.DataFrame
    target: pd.DataFrame


def split_train_val_test_fixed_origin(
    df: pd.DataFrame,
    train_ratio: float = 1.0 / 3.0,
    val_ratio: float = 1.0 / 3.0,
    min_len: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each series into fixed-origin train/val/test partitions.

    Ratios are computed on each series length. For each series:
    - train: [0, floor(L * train_ratio))
    - val: [floor(L * train_ratio), floor(L * (train_ratio + val_ratio)))
    - test: remainder
    """
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be in (0, 1).")
    if val_ratio < 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("val_ratio must be >= 0 and train_ratio + val_ratio must be < 1.")

    tmp = df.copy()
    tmp = tmp.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    tmp["t"] = tmp.groupby("unique_id").cumcount()
    tmp["L"] = tmp.groupby("unique_id")["t"].transform("max") + 1
    tmp = tmp[tmp["L"] >= min_len].copy()

    train_cut = np.floor(tmp["L"] * train_ratio)
    val_cut = np.floor(tmp["L"] * (train_ratio + val_ratio))

    train_mask = tmp["t"] < train_cut
    val_mask = (tmp["t"] >= train_cut) & (tmp["t"] < val_cut)
    test_mask = tmp["t"] >= val_cut

    cols = ["unique_id", "ds", "y"]
    train_df = tmp.loc[train_mask, cols].copy()
    val_df = tmp.loc[val_mask, cols].copy()
    test_df = tmp.loc[test_mask, cols].copy()
    return train_df, val_df, test_df


def iter_walk_forward_frames(
    init_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    step_size: int = 1,
) -> Iterator[WalkForwardFrame]:
    """Yield walk-forward frames with synchronized per-series time index.

    At each yielded frame:
    - `history` contains initial data and all realized targets from earlier steps.
    - `target` contains rows to forecast in the current step block.
    """
    if step_size <= 0:
        raise ValueError("step_size must be >= 1.")

    history = init_set[["unique_id", "ds", "y"]].copy()
    history = history.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    eval_sorted = eval_set[["unique_id", "ds", "y"]].copy()
    eval_sorted = eval_sorted.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    eval_sorted["wf_step"] = eval_sorted.groupby("unique_id").cumcount()
    if eval_sorted.empty:
        return

    # Build block ids once to avoid scanning the whole eval frame at every step.
    eval_sorted["wf_block"] = eval_sorted["wf_step"] // step_size
    for _, block in eval_sorted.groupby("wf_block", sort=True):
        target = block[["unique_id", "ds", "y"]].copy()
        if target.empty:
            continue

        # Downstream callers treat `history` as read-only; avoid a full copy each step.
        step = int(block["wf_step"].min())
        yield WalkForwardFrame(step=step, history=history, target=target)

        history = pd.concat(
            [history, target[["unique_id", "ds", "y"]]],
            ignore_index=True,
        ).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def evaluate_point_models(
    init_set: pd.DataFrame,
    merged_eval: pd.DataFrame,
    model_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute point metrics for each model column in merged_eval."""
    if model_cols is None:
        model_cols = [c for c in merged_eval.columns if c not in {"unique_id", "ds", "y"}]

    rows = []
    for model in model_cols:
        tmp = merged_eval[["unique_id", "ds", "y", model]].dropna(subset=[model]).copy()
        if tmp.empty:
            continue
        tmp = tmp.rename(columns={model: "y_pred"})
        rows.append({
            "model": model,
            "ME": me(tmp["y"].to_numpy(), tmp["y_pred"].to_numpy()),
            "MAE": mae(tmp["y"].to_numpy(), tmp["y_pred"].to_numpy()),
            "RMSE": rmse(tmp["y"].to_numpy(), tmp["y_pred"].to_numpy()),
            "RMSSE": rmsse(init_set, tmp),
            "WRMSSE": wrmsse(init_set, tmp),
        })

    return pd.DataFrame(rows)


def evaluate_prob_models(
    merged_eval: pd.DataFrame,
    quantiles: Iterable[float],
) -> pd.DataFrame:
    """Compute pinball loss per model and quantile."""
    rows: list[dict[str, float | str]] = []
    for model, dfm in merged_eval.groupby("model"):
        for q in quantiles:
            col = f"q_{q}"
            if col not in dfm.columns:
                continue
            dfx = dfm.dropna(subset=[col]).copy()
            if dfx.empty:
                continue
            err = dfx["y"] - dfx[col]
            loss = np.maximum(q * err, (q - 1) * err).mean()
            rows.append({"model": model, "quantile": q, "pinball": float(loss)})
    return pd.DataFrame(rows)
