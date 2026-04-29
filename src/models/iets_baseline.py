from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from utils import find_repo_root

IETS_POINT_COL = "iETS"
IETS_PROB_MODEL = "iETS"


def default_iets_r_script() -> Path:
    return find_repo_root() / "scripts" / "iets_panel_forecast.R"


def default_iets_prob_r_script() -> Path:
    return find_repo_root() / "scripts" / "iets_panel_forecast_prob.R"


def _qcols(quantiles: list[float]) -> list[str]:
    return [f"q_{q}" for q in quantiles]


def _enforce_monotonic_quantiles(df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    out = df.copy()
    qcols = _qcols(quantiles)
    for c in qcols:
        if c not in out.columns:
            out[c] = np.nan
    arr = out[qcols].to_numpy(dtype=float)
    arr = np.where(np.isnan(arr), np.nan, np.maximum(arr, 0.0))
    for j in range(1, arr.shape[1]):
        prev = arr[:, j - 1]
        cur = arr[:, j]
        cur = np.where(np.isnan(cur), prev, cur)
        prev_filled = np.where(np.isnan(prev), cur, prev)
        arr[:, j] = np.maximum(cur, prev_filled)
    out[qcols] = arr
    return out


def fit_predict_iets_panel(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    rscript: str = "Rscript",
    script_path: Path | None = None,
    seed: int = 42,
    occurrence: str = "auto",
) -> pd.DataFrame:
    """Run R smooth::adam iETS panel forecasts; returns [unique_id, ds, iETS].

    Requires a working R installation and CRAN packages ``smooth`` and ``greybox``.
    """
    script = script_path or default_iets_r_script()
    if not script.is_file():
        raise FileNotFoundError(f"iETS R script not found: {script}")

    need_train = {"unique_id", "ds", "y"}
    need_eval = {"unique_id", "ds"}
    if not need_train.issubset(train_df.columns):
        raise ValueError(f"train_df must contain columns {need_train}")
    if not need_eval.issubset(eval_df.columns):
        raise ValueError(f"eval_df must contain columns {need_eval}")

    train_csv = eval_csv = out_csv = Path()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_csv = tmp_path / "train.csv"
        eval_csv = tmp_path / "eval.csv"
        out_csv = tmp_path / "iets_out.csv"
        train_df[list(need_train)].to_csv(train_csv, index=False)
        eval_df[["unique_id", "ds"] + (["y"] if "y" in eval_df.columns else [])].to_csv(eval_csv, index=False)

        cmd = [
            rscript,
            str(script),
            str(train_csv),
            str(eval_csv),
            str(out_csv),
            str(int(seed)),
            str(occurrence),
        ]
        env = os.environ.copy()
        r_user_lib = find_repo_root() / ".R" / "library"
        if r_user_lib.is_dir():
            env["R_LIBS_USER"] = str(r_user_lib)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                "iETS (R) failed. Ensure R is on PATH and packages are installed:\n"
                "  install.packages(c('smooth','greybox'))\n"
                f"Command: {' '.join(cmd)}\n{msg}"
            )
        if not out_csv.is_file():
            raise RuntimeError(f"iETS output missing: {out_csv}")

        out = pd.read_csv(out_csv)
        if IETS_POINT_COL not in out.columns:
            raise RuntimeError(f"iETS output missing column '{IETS_POINT_COL}': {list(out.columns)}")
        out = out[["unique_id", "ds", IETS_POINT_COL]].copy()
        out["ds"] = pd.to_datetime(out["ds"])
        return out


def fit_predict_iets_prob_panel(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    quantiles: list[float],
    rscript: str = "Rscript",
    script_path: Path | None = None,
    seed: int = 42,
    occurrence: str = "auto",
    model_name: str = IETS_PROB_MODEL,
) -> pd.DataFrame:
    """Run iETS probabilistic forecasts via R; returns long frame like run_prob baselines.

    Uses ``forecast(..., interval='prediction', level=c(90,80,50))`` mapped to
    quantiles 0.1, 0.25, 0.5, 0.75, 0.9 (same grid as the main probabilistic pipeline).
    """
    script = script_path or default_iets_prob_r_script()
    if not script.is_file():
        raise FileNotFoundError(f"iETS prob R script not found: {script}")

    need_train = {"unique_id", "ds", "y"}
    need_eval = {"unique_id", "ds"}
    if not need_train.issubset(train_df.columns):
        raise ValueError(f"train_df must contain columns {need_train}")
    if not need_eval.issubset(eval_df.columns):
        raise ValueError(f"eval_df must contain columns {need_eval}")

    qcols = _qcols(quantiles)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_csv = tmp_path / "train.csv"
        eval_csv = tmp_path / "eval.csv"
        out_csv = tmp_path / "iets_prob_out.csv"
        train_df[list(need_train)].to_csv(train_csv, index=False)
        eval_df[["unique_id", "ds"] + (["y"] if "y" in eval_df.columns else [])].to_csv(eval_csv, index=False)

        cmd = [
            rscript,
            str(script),
            str(train_csv),
            str(eval_csv),
            str(out_csv),
            str(int(seed)),
            str(occurrence),
        ]
        env = os.environ.copy()
        r_user_lib = find_repo_root() / ".R" / "library"
        if r_user_lib.is_dir():
            env["R_LIBS_USER"] = str(r_user_lib)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                "iETS probabilistic (R) failed. Ensure R and packages smooth/greybox are installed.\n"
                f"Command: {' '.join(cmd)}\n{msg}"
            )
        if not out_csv.is_file():
            raise RuntimeError(f"iETS prob output missing: {out_csv}")

        out = pd.read_csv(out_csv)
        out["ds"] = pd.to_datetime(out["ds"])
        for c in qcols:
            if c not in out.columns:
                out[c] = np.nan
        out = _enforce_monotonic_quantiles(out, quantiles=quantiles)
        out["model"] = model_name
        return out[["model", "unique_id", "ds"] + qcols].copy()
