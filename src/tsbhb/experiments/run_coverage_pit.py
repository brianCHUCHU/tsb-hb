from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tsbhb.utils import set_seed, default_data_file, default_out_dir
from tsbhb.data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from tsbhb.models.tsb_hb import fit_tsb_hb, predict_tsb_hb
from tsbhb.models.baselines import fit_predict_baselines
from tsbhb.metrics import coverage_rate, pit_values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=default_data_file())
    ap.add_argument("--out", type=Path, default=default_out_dir())
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_online_retail(args.data)
    df = preprocess_online_retail(df_raw)
    init_set, eval_set = train_eval_split_fixed_origin(df, init_ratio=1 / 3, min_len=30)

    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
    params = fit_tsb_hb(init_set)
    tsbhb_q = predict_tsb_hb(params, eval_set, quantiles=QUANTILES, n_samples=2000)
    tsbhb_q["model"] = "TSB-HB"

    eval_h = eval_set["unique_id"].value_counts()
    sf_q = fit_predict_baselines(init_set, eval_h, freq="D", probabilistic=True, levels=[80, 50])
    # Build per-model quantiles
    rename_map = {
        "AutoARIMA": "q_0.5", "AutoARIMA-lo-80": "q_0.1", "AutoARIMA-hi-80": "q_0.9",
        "AutoARIMA-lo-50": "q_0.25", "AutoARIMA-hi-50": "q_0.75",
        "AutoTheta": "q_0.5", "AutoTheta-lo-80": "q_0.1", "AutoTheta-hi-80": "q_0.9",
        "AutoTheta-lo-50": "q_0.25", "AutoTheta-hi-50": "q_0.75",
    }
    def _extract(df: pd.DataFrame, model: str) -> pd.DataFrame:
        cols = [model, f"{model}-lo-80", f"{model}-hi-80", f"{model}-lo-50", f"{model}-hi-50"]
        cols = [c for c in cols if c in df.columns]
        out = df[cols + ["unique_id", "ds"]].copy().rename(columns=rename_map)
        for c in [f"q_{q}" for q in QUANTILES]:
            if c not in out.columns:
                out[c] = np.nan
        out["model"] = model
        return out

    arima = _extract(sf_q, "AutoARIMA")
    theta = _extract(sf_q, "AutoTheta")

    all_q = pd.concat([
        tsbhb_q[["unique_id", "ds", "model"] + [f"q_{q}" for q in QUANTILES]],
        arima,
        theta,
    ], ignore_index=True)

    # Coverage summary
    cov_rows = []
    for model, dfm in all_q.groupby("model"):
        df_eval = eval_set[["unique_id", "ds", "y"]].merge(dfm, on=["unique_id", "ds"], how="inner")
        cov = {"model": model}
        cov.update(coverage_rate(df_eval, 0.1, 0.9, 0.8))
        # 95% available only if q_0.025/q_0.975 exist; we don't compute them here
        cov_rows.append(cov)
    pd.DataFrame(cov_rows).to_csv(out_dir / "coverage_summary.csv", index=False)

    # PIT values for TSB-HB only
    tsbhb_eval = eval_set[["unique_id", "ds", "y"]].merge(tsbhb_q, on=["unique_id", "ds"], how="inner")
    pits = pit_values(tsbhb_eval, quantiles=QUANTILES)
    pd.DataFrame({"pit": pits}).to_csv(out_dir / "pit_values.csv", index=False)


if __name__ == "__main__":
    main()

