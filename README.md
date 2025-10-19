# TSB-HB

Reference implementation for the TSB-HB (Teunter–Syntetos–Babai with hierarchical Bayes shrinkage) demand-forecasting experiments. The project uses a flattened `src/` layout: helper modules sit at the top level (`data_loading.py`, `metrics.py`, `plotting.py`, `utils.py`) alongside subpackages (`models/`, `experiments/`, `tools/`). Packaging metadata lives under `src/tsbhb.egg-info/` because the distribution name remains `tsbhb`.

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
	 uv run python -m experiments.run_point --help
	 ```

To work inside the environment interactively, activate the virtualenv created in `.venv/` (`source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\activate` on Windows). With the flattened layout, you can import modules directly (e.g. `import data_loading`, `from models.tsb_hb import fit_tsb_hb`).

## Data

### Online Retail Dataset

Download the Online Retail Dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)

- Place the data at `data/online_retail.csv`. The loader also accepts the legacy name `Online_Retail.csv`.

### M5 Dataset

To run M5 experiments, download the M5 dataset from [Kaggle M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy/data):

1. Download `sales_train_evaluation.csv` (or `sales_train_validation.csv`)
2. Download `calendar.csv`
3. Place both files in the `data/` directory

The loader automatically detects the wide format and converts it on the fly—no preprocessing required. Your `data/` directory should look like:

```bash
data/
	├── online_retail.csv    # or Online_Retail.csv
	├── sales_train_evaluation.csv  # M5 wide format (auto-converted)
	└── calendar.csv                # M5 calendar metadata
```

**Optional caching:** If you prefer to materialise the long-format file once (to speed up repeated experiments), use the in-package helper:

```bash
uv run python -m tools.convert_m5_to_long \
		--input data/sales_train_evaluation.csv \
		--output data/m5_evaluation_long.csv
```

You can then pass the cached file explicitly with `--m5-sales data/m5_evaluation_long.csv` when invoking scripts.

## Running experiments

Each experiment script lives under `src/experiments/` and can be invoked with uv. Common flags supported by most scripts:

- `--data`: path to the primary dataset (defaults to `data/online_retail.csv`).
- `--out`: destination directory for metrics/plots (defaults to `outputs/`).
- `--seed`: random seed for reproducibility (default `42`).

Generic invocation pattern:

```bash
uv run python -m experiments.<script_name> [options]
```

### Script catalog

- `run_point.py`
	- **Purpose:** Fits the TSB-HB model and a suite of StatsForecast baselines for point forecasts.
	- **Datasets:** Online Retail (default) and M5 via `--dataset m5`.
	- **Highlights:** Generates `point_metrics.csv`, shrinkage plots (`fig_shrink_p.png`, `fig_shrink_size.png`), and `point_metrics_m5.csv` when running on M5.
- `run_prob.py`
	- **Purpose:** Produces probabilistic forecasts (quantiles) for TSB-HB and AutoARIMA/AutoTheta baselines.
	- **Datasets:** Online Retail only (M5 support not implemented).
	- **Outputs:** `prob_quantiles.csv`, `prob_pinball.csv`, and `probabilistic_forecast_pinball_results.csv` in the chosen `--out` directory.
- `run_grid.py`
	- **Purpose:** Sweeps across `(alpha_d, alpha_p)` combinations for the TSB baseline to compare against the TSB-HB reference.
	- **Datasets:** Online Retail only.
	- **Outputs:** `grid_summary.csv` summarising ME/MAE/RMSE/RMSSE for each grid point.
- `run_ablation.py`
	- **Purpose:** Compares different shrinkage/likelihood variants (HB LogNormal, MLE LogNormal, HB Gamma) to quantify each modelling choice.
	- **Datasets:** Online Retail only.
	- **Outputs:** `ablation_metrics.csv` with ME/MAE/RMSE/RMSSE per variant.
- `run_coverage_pit.py`
	- **Purpose:** Evaluates interval coverage and probability integral transform statistics for probabilistic forecasts.
	- **Datasets:** Online Retail only.
	- **Outputs:** `coverage_summary.csv` (coverage & interval widths) and `pit_values.csv` for histogram diagnostics.
- `run_deepar.py`
	- **Purpose:** Optional neural benchmark using `neuralforecast`’s AutoDeepAR across configurable horizons.
	- **Datasets:** Online Retail only (subset sampling controlled via CLI flags).
	- **Requirements:** Install extras via `uv sync --extra deepar`. Produces `multi_horizon_comparison_results.csv` with ME/MAE/RMSE/RMSSE per horizon and model.

## Outputs

By default, each experiment writes results, diagnostics, and plots into `outputs/`. Clean the directory between runs if you need a fresh slate, or override `--out` with a dedicated subdirectory for reproducibility.

## Reference

1. Azul Garza, Max Mergenthaler Canseco, Cristian Challú, & Kin G. Olivares.  
   **StatsForecast: Lightning fast forecasting with statistical and econometric models.**  
   PyCon Salt Lake City, Utah, US, 2022.  
   [https://github.com/Nixtla/statsforecast](https://github.com/Nixtla/statsforecast)

2. Addison Howard, inversion, Spyros Makridakis, & Vangelis.  
   **M5 Forecasting – Accuracy.** Kaggle, 2020.  
   [https://kaggle.com/competitions/m5-forecasting-accuracy](https://kaggle.com/competitions/m5-forecasting-accuracy)

3. Daqing Chen.  
   **Online Retail.** UCI Machine Learning Repository, 2015.  
   DOI: [10.24432/C5BW33](https://doi.org/10.24432/C5BW33)
