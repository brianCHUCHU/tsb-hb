from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
    timeout_seconds: int | None = 3600,
    per_series_timeout_seconds: int = 10,
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
            str(int(max(per_series_timeout_seconds, 1))),
        ]
        env = os.environ.copy()
        r_user_lib = find_repo_root() / ".R" / "library"
        if r_user_lib.is_dir():
            env["R_LIBS_USER"] = str(r_user_lib)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"iETS (R) failed because '{rscript}' was not found. Install R and put Rscript on PATH, "
                "or pass --iets-rscript /absolute/path/to/Rscript."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"iETS (R) exceeded timeout_seconds={timeout_seconds}. "
                "Use --max-series for a sampled run or lower --iets-per-series-timeout."
            ) from exc
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
    timeout_seconds: int | None = 3600,
    per_series_timeout_seconds: int = 10,
    n_jobs: int = 1,
    cache_path: Path | None = None,
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
    if cache_path is not None and cache_path.is_file():
        cached = pd.read_csv(cache_path)
        cached["ds"] = pd.to_datetime(cached["ds"])
        for c in qcols:
            if c not in cached.columns:
                cached[c] = np.nan
        cached = _enforce_monotonic_quantiles(cached, quantiles=quantiles)
        cached["model"] = model_name
        return cached[["model", "unique_id", "ds"] + qcols].copy()

    n_jobs = int(max(n_jobs, 1))
    if n_jobs > 1:
        uids = np.asarray(eval_df["unique_id"].astype(str).drop_duplicates().to_list(), dtype=object)
        if len(uids) == 0:
            return pd.DataFrame(columns=["model", "unique_id", "ds"] + qcols)
        chunks = [chunk.tolist() for chunk in np.array_split(uids, min(n_jobs, len(uids))) if len(chunk) > 0]
        frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = {}
            for job_idx, chunk_uids in enumerate(chunks, start=1):
                eval_chunk = eval_df[eval_df["unique_id"].astype(str).isin(chunk_uids)].copy()
                train_chunk = train_df[train_df["unique_id"].astype(str).isin(chunk_uids)].copy()
                futures[
                    executor.submit(
                        _run_iets_prob_r_script,
                        train_chunk,
                        eval_chunk,
                        rscript=rscript,
                        script=script,
                        seed=seed,
                        occurrence=occurrence,
                        timeout_seconds=timeout_seconds,
                        per_series_timeout_seconds=per_series_timeout_seconds,
                        quantiles=quantiles,
                    )
                ] = job_idx
            for future in as_completed(futures):
                job_idx = futures[future]
                try:
                    frames.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"iETS probabilistic worker {job_idx}/{len(chunks)} failed.") from exc
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["unique_id", "ds"] + qcols)
        out["ds"] = pd.to_datetime(out["ds"])
        for c in qcols:
            if c not in out.columns:
                out[c] = np.nan
        out = _enforce_monotonic_quantiles(out, quantiles=quantiles)
        out["model"] = model_name
        final = out[["model", "unique_id", "ds"] + qcols].copy()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            final.to_csv(cache_path, index=False)
        return final

    out = _run_iets_prob_r_script(
        train_df,
        eval_df,
        rscript=rscript,
        script=script,
        seed=seed,
        occurrence=occurrence,
        timeout_seconds=timeout_seconds,
        per_series_timeout_seconds=per_series_timeout_seconds,
        quantiles=quantiles,
    )
    out["ds"] = pd.to_datetime(out["ds"])
    for c in qcols:
        if c not in out.columns:
            out[c] = np.nan
    out = _enforce_monotonic_quantiles(out, quantiles=quantiles)
    out["model"] = model_name
    final = out[["model", "unique_id", "ds"] + qcols].copy()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(cache_path, index=False)
    return final


def _run_iets_prob_r_script(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    rscript: str,
    script: Path,
    seed: int,
    occurrence: str,
    timeout_seconds: int | None,
    per_series_timeout_seconds: int,
    quantiles: list[float],
) -> pd.DataFrame:
    need_train = {"unique_id", "ds", "y"}
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
            str(int(max(per_series_timeout_seconds, 1))),
            ",".join(str(q) for q in quantiles),
        ]
        env = os.environ.copy()
        r_user_lib = find_repo_root() / ".R" / "library"
        if r_user_lib.is_dir():
            env["R_LIBS_USER"] = str(r_user_lib)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"iETS probabilistic (R) failed because '{rscript}' was not found. Install R and put Rscript on PATH, "
                "or pass --iets-rscript /absolute/path/to/Rscript."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"iETS probabilistic (R) exceeded timeout_seconds={timeout_seconds}. "
                "Use --max-series for a sampled run or lower --iets-per-series-timeout."
            ) from exc
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                "iETS probabilistic (R) failed. Ensure R and packages smooth/greybox are installed.\n"
                f"Command: {' '.join(cmd)}\n{msg}"
            )
        if not out_csv.is_file():
            raise RuntimeError(f"iETS prob output missing: {out_csv}")

        out = pd.read_csv(out_csv)
        return out
