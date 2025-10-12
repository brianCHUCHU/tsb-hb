import numpy as np
import pandas as pd
from pathlib import Path
import pytest

EPS = 1e-6


def _read(name):
    root = Path(__file__).resolve().parents[1]
    exp_path = root / "experiments" / "expected" / name
    out_dir = root / "experiments" / "outputs"
    out_path = out_dir / name
    # Allow alternative filename for pinball results
    if name == "probabilistic_forecast_pinball_results.csv" and not out_path.exists():
        alt = out_dir / "prob_pinball.csv"
        if alt.exists():
            out_path = alt
    if not exp_path.exists():
        pytest.skip(f"Expected file not found: {exp_path}")
    if not out_path.exists():
        pytest.skip(f"Output file not found: {out_path}")
    exp = pd.read_csv(exp_path)
    out = pd.read_csv(out_path)
    return exp, out


def _assert_frame_close(exp: pd.DataFrame, out: pd.DataFrame):
    commons = list(set(exp.columns) & set(out.columns))
    exp = exp[commons].copy()
    out = out[commons].copy()
    # If time-indexed, sort by unique keys
    keys = [c for c in ["unique_id", "ds", "model", "alpha", "beta", "quantile"] if c in commons]
    if keys:
        exp = exp.sort_values(keys).reset_index(drop=True)
        out = out.sort_values(keys).reset_index(drop=True)
    # Compare numeric columns with allclose
    for c in commons:
        if np.issubdtype(out[c].dtype, np.number) and np.issubdtype(exp[c].dtype, np.number):
            assert np.allclose(exp[c].to_numpy(), out[c].to_numpy(), rtol=1e-6, atol=1e-8), f"Mismatch in {c}"


def test_point_metrics():
    exp, out = _read("point_metrics.csv")
    _assert_frame_close(exp, out)


def test_ablation_metrics():
    exp, out = _read("ablation_metrics.csv")
    _assert_frame_close(exp, out)


def test_grid_summary():
    exp, out = _read("grid_summary.csv")
    _assert_frame_close(exp, out)


def test_prob_quantiles():
    exp, out = _read("prob_quantiles.csv")
    _assert_frame_close(exp, out)


def test_coverage_summary():
    exp, out = _read("coverage_summary.csv")
    _assert_frame_close(exp, out)


def test_pit_values():
    exp, out = _read("pit_values.csv")
    _assert_frame_close(exp, out)


def test_prob_pinball():
    exp, out = _read("probabilistic_forecast_pinball_results.csv")
    _assert_frame_close(exp, out)


def test_segmentation_rmsse():
    exp, out = _read("segmentation_rmsse.csv")
    _assert_frame_close(exp, out)
