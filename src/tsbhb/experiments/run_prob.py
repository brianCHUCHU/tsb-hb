from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tsbhb.utils import set_seed, default_data_file, default_out_dir
from tsbhb.data_loading import load_online_retail, preprocess_online_retail, train_eval_split_fixed_origin
from tsbhb.models.tsb_hb import fit_tsb_hb, predict_tsb_hb
from tsbhb.models.baselines import fit_predict_baselines


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

    # TSB-HB probabilistic quantiles (use Monte Carlo per notebook)
    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
    params = fit_tsb_hb(init_set)
    tsbhb_q = predict_tsb_hb(params, eval_set, quantiles=QUANTILES, n_samples=2000)
    tsbhb_q["model"] = "TSB-HB"

    # AutoARIMA / AutoTheta quantile approximations via StatsForecast intervals
    # Map: level 80 -> q0.1/q0.9, level 50 -> q0.25/q0.75, center -> q0.5
    eval_h = eval_set["unique_id"].value_counts()
    sf_q = fit_predict_baselines(init_set, eval_h, freq="D", probabilistic=True, levels=[80, 50])
    # Split per model
    cols_arima = ["AutoARIMA", "AutoARIMA-lo-80", "AutoARIMA-hi-80", "AutoARIMA-lo-50", "AutoARIMA-hi-50"]
    cols_theta = ["AutoTheta", "AutoTheta-lo-80", "AutoTheta-hi-80", "AutoTheta-lo-50", "AutoTheta-hi-50"]
    arima = sf_q[[c for c in cols_arima if c in sf_q.columns] + ["unique_id", "ds"]].copy()
    theta = sf_q[[c for c in cols_theta if c in sf_q.columns] + ["unique_id", "ds"]].copy()
    rename_map = {
        "AutoARIMA": "q_0.5", "AutoARIMA-lo-80": "q_0.1", "AutoARIMA-hi-80": "q_0.9",
        "AutoARIMA-lo-50": "q_0.25", "AutoARIMA-hi-50": "q_0.75",
        "AutoTheta": "q_0.5", "AutoTheta-lo-80": "q_0.1", "AutoTheta-hi-80": "q_0.9",
        "AutoTheta-lo-50": "q_0.25", "AutoTheta-hi-50": "q_0.75",
    }
    arima = arima.rename(columns=rename_map)
    theta = theta.rename(columns=rename_map)
    arima["model"] = "AutoARIMA"
    theta["model"] = "AutoTheta"

    # Make quantile columns present for all by filling missing
    qcols = [f"q_{q}" for q in QUANTILES]
    for df_ in (tsbhb_q, arima, theta):
        for c in qcols:
            if c not in df_.columns:
                df_[c] = np.nan
        df_.dropna(subset=["unique_id", "ds"], inplace=True)

    all_q = pd.concat([
        tsbhb_q[["model", "unique_id", "ds"] + qcols],
        arima[["model", "unique_id", "ds"] + qcols],
        theta[["model", "unique_id", "ds"] + qcols],
    ], ignore_index=True)
    all_q.to_csv(out_dir / "prob_quantiles.csv", index=False)

    # Also compute pinball summary for convenience
    eval_merged = eval_set[["unique_id", "ds", "y"]].copy().merge(all_q, on=["unique_id", "ds"], how="inner")
    rows = []
    for model, dfm in eval_merged.groupby("model"):
        for q in QUANTILES:
            col = f"q_{q}"
            dfx = dfm.dropna(subset=[col]).copy()
            err = dfx["y"] - dfx[col]
            loss = np.maximum(q * err, (q - 1) * err).mean()
            rows.append({"model": model, "quantile": q, "pinball": float(loss)})
    pd.DataFrame(rows).to_csv(out_dir / "prob_pinball.csv", index=False)


if __name__ == "__main__":
    main()

