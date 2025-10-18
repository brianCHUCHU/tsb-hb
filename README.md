# TSB-HB

Reference implementation for the TSB-HB (Teunter–Syntetos–Babai with hierarchical Bayes shrinkage) demand-forecasting experiments. The code base exposes modular data loading, model components, evaluation utilities, and ready-to-run experiment entry points.

## Quick start

1. **Install uv** (once per machine):

	 ```bash
	 curl -LsSf https://astral.sh/uv/install.sh | sh
	 ```

2. **Create the project environment** (Python 3.10) and install the package plus pinned dependencies. A `.venv/` directory will be created automatically next to this README:

	 ```bash
	 uv sync
	 ```

3. *(Optional)* **Enable the DeepAR extras** (required only for `run_deepar.py`):

	 ```bash
	 uv sync --extra deepar
	 ```

4. **Run any script** directly through uv (no manual activation needed):

	 ```bash
	 uv run python -m tsbhb.experiments.run_point --help
	 ```

To work inside the environment interactively, activate the virtualenv created in `.venv/` (`source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\activate` on Windows).

## Dependencies

Core runtime dependencies are pinned in `pyproject.toml`:

| Package | Version | Purpose |
| --- | --- | --- |
| `numpy` | 1.26.4 | Vectorised math utilities across the project |
| `pandas` | 2.3.1 | Data wrangling for loading, preprocessing and evaluation |
| `scipy` | 1.15.3 | Optimisation routines for the hierarchical Bayes estimator |
| `statsforecast` | 2.0.2 | Baseline intermittent-demand models (Croston, TSB, AutoARIMA, etc.) |
| `matplotlib` | 3.10.3 | Shrinkage and PIT diagnostic plots |

Optional extras:

- `deepar`: installs `neuralforecast==3.1.2` and `torch==2.6.0`, which are only needed when you execute `run_deepar.py`.

All transitive packages are resolved automatically by uv during `uv sync`.

## Data

Place raw datasets under the `data/` directory at the repository root. The loaders expect the following filenames; add the appropriate download links and bibliographic citation where indicated:

| Dataset | Expected path | Source URL | Citation |
| --- | --- | --- | --- |
| Online Retail transactional data | `data/online_retail.csv` | `[TODO: add download URL]` | `[TODO: add citation]` |
| M5 sales (evaluation-long format) | `data/m5_evaluation_long.csv` | `[TODO: add download URL]` | `[TODO: add citation]` |
| M5 calendar metadata | `data/calendar.csv` | `[TODO: add download URL]` | `[TODO: add citation]` |

The preprocessing pipeline densifies the series to daily frequency and drops rows with missing timestamps. Outputs from experiments are written to the top-level `outputs/` directory unless you override `--out`.

## Running experiments

Each experiment script lives under `src/tsbhb/experiments/` and can be invoked with uv. Common flags supported by most scripts:

- `--data`: path to the primary dataset (defaults to `data/online_retail.csv`).
- `--out`: destination directory for metrics/plots (defaults to `outputs/`).
- `--seed`: random seed for reproducibility (default `42`).

Generic invocation pattern:

```bash
uv run python -m tsbhb.experiments.<script_name> [options]
```

### Script catalog

- `run_point.py`
	- **Purpose:** Fits the TSB-HB model and a suite of StatsForecast baselines for point forecasts.
	- **Highlights:** Supports `--dataset online_retail|m5`. Generates `point_metrics.csv`, shrinkage plots (`fig_shrink_p.png`, `fig_shrink_size.png`), and, for M5 runs, `point_metrics_m5.csv`.
- `run_prob.py`
	- **Purpose:** Produces probabilistic forecasts (quantiles) for TSB-HB and AutoARIMA/AutoTheta baselines.
	- **Outputs:** `prob_quantiles.csv`, `prob_pinball.csv`, and `probabilistic_forecast_pinball_results.csv` in the chosen `--out` directory.
- `run_grid.py`
	- **Purpose:** Sweeps across `(alpha_d, alpha_p)` combinations for the TSB baseline to compare against the TSB-HB reference.
	- **Outputs:** `grid_summary.csv` summarising ME/MAE/RMSE/RMSSE for each grid point.
- `run_ablation.py`
	- **Purpose:** Compares different shrinkage/likelihood variants (HB LogNormal, MLE LogNormal, HB Gamma) to quantify each modelling choice.
	- **Outputs:** `ablation_metrics.csv` with ME/MAE/RMSE/RMSSE per variant.
- `run_coverage_pit.py`
	- **Purpose:** Evaluates interval coverage and probability integral transform statistics for probabilistic forecasts.
	- **Outputs:** `coverage_summary.csv` (coverage & interval widths) and `pit_values.csv` for histogram diagnostics.
- `run_deepar.py`
	- **Purpose:** Optional neural benchmark using `neuralforecast`’s AutoDeepAR across configurable horizons.
	- **Requirements:** Install extras via `uv sync --extra deepar`. Produces `multi_horizon_comparison_results.csv` with ME/MAE/RMSE/RMSSE per horizon and model.

### Developing custom benchmarks

- Implement deep-learning baselines in `src/tsbhb/models/dl_benchmark.py` (template provided).
- Extend relevant experiment scripts (e.g., `run_point.py`, `run_prob.py`) with new CLI options or model branches.
- Add additional dependencies to `pyproject.toml` (either core or under a new optional extra) and rerun `uv sync` to lock them in.

## Outputs

By default, each experiment writes results, diagnostics, and plots into `outputs/`. Clean the directory between runs if you need a fresh slate, or override `--out` with a dedicated subdirectory for reproducibility.



