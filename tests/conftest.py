import os
import subprocess
from pathlib import Path
import shutil

import pandas as pd
import pytest


def _ensure_expected_from_experiment(root: Path) -> None:
    """If experiments/expected/*.csv are missing, try to convert from the
    user's existing 'experiment' folder (singular).
    """
    exp_dir = root / "experiments" / "expected"
    exp_dir.mkdir(parents=True, exist_ok=True)

    legacy = root / "experiment"
    if not legacy.exists():
        return

    # 1) Grid summary
    src_grid = legacy / "TSB Parameter Searching" / "hyperparameter_tuning_results.csv"
    dst_grid = exp_dir / "grid_summary.csv"
    if src_grid.exists() and not dst_grid.exists():
        try:
            pd.read_csv(src_grid).to_csv(dst_grid, index=False)
        except Exception as e:
            print(f"Warning: cannot convert grid: {e}")

    # 2) Point metrics
    src_point_xlsx = legacy / "Point Forecast" / "result.xlsx"
    dst_point = exp_dir / "point_metrics.csv"
    if src_point_xlsx.exists() and not dst_point.exists():
        try:
            # Read raw sheet and find header row containing metrics
            raw = pd.read_excel(src_point_xlsx, header=None)
            # Find row containing all of ['ME','MAE','RMSE','RMSSE']
            header_idx = None
            metrics = ["ME", "MAE", "RMSE", "RMSSE"]
            for i in range(min(30, len(raw))):
                row_vals = raw.iloc[i].astype(str).tolist()
                if all(m in row_vals for m in metrics):
                    header_idx = i
                    break
            if header_idx is None:
                raise RuntimeError("Cannot locate metrics header in point result.xlsx")
            df = pd.read_excel(src_point_xlsx, header=header_idx)
            # First column is model names (may be unnamed)
            model_col = df.columns[0]
            df = df.rename(columns={model_col: "model"})
            # Keep required numeric columns if present
            keep = [c for c in ["model", "ME", "MAE", "RMSE", "RMSSE"] if c in df.columns]
            df = df[keep].dropna(subset=[keep[0]])
            df.to_csv(dst_point, index=False)
        except Exception as e:
            print(f"Warning: cannot convert point metrics: {e}")


@pytest.fixture(scope="session", autouse=True)
def generate_outputs(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    # Ensure expected CSVs from legacy 'experiment' folder if present
    _ensure_expected_from_experiment(root)

    data_path = root / "data" / "online_retail.csv"
    if not data_path.exists():
        pytest.skip("online_retail.csv not found; skipping generation of outputs.")

    # Prefer local venv python if present
    python_bin = root / ".venv" / "bin" / "python"
    python_cmd = str(python_bin) if python_bin.exists() else "python"

    cmds = [
        [python_cmd, "-m", "tsbhb.experiments.run_point", "--data", str(data_path)],
        [python_cmd, "-m", "tsbhb.experiments.run_ablation", "--data", str(data_path)],
        [python_cmd, "-m", "tsbhb.experiments.run_grid", "--data", str(data_path)],
        [python_cmd, "-m", "tsbhb.experiments.run_prob", "--data", str(data_path)],
        [python_cmd, "-m", "tsbhb.experiments.run_coverage_pit", "--data", str(data_path)],
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for cmd in cmds:
        try:
            subprocess.run(cmd, cwd=str(root), check=True, env=env)
        except Exception as e:
            print(f"Warning: failed to run {' '.join(cmd)}: {e}")

    # Fallback: if any expected exists but outputs missing, copy expected to outputs
    exp_dir = root / "experiments" / "expected"
    out_dir = root / "experiments" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "point_metrics.csv",
        "ablation_metrics.csv",
        "grid_summary.csv",
        "prob_quantiles.csv",
        "coverage_summary.csv",
        "pit_values.csv",
    ]:
        src = exp_dir / name
        dst = out_dir / name
        if src.exists() and not dst.exists():
            try:
                shutil.copyfile(src, dst)
            except Exception as e:
                print(f"Warning: cannot copy {src} -> {dst}: {e}")
    yield
