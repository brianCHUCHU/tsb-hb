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
	- **Purpose:** Fits the TSB-HB model, StatsForecast baselines, and hurdle baselines (local/global LogNormal) for point forecasts.
	- **Datasets:** Online Retail (default) and M5 via `--dataset m5`.
	- **Protocol options (Online Retail):** `--protocol fixed` (single fixed-origin fit) or `--protocol walk_forward` (sequential updates), configurable with `--walk-step`.
	- **HB controls:** `--hb-regime-aware/--no-hb-regime-aware`, `--hb-group-shrink-strength`, `--hb-online-update/--no-hb-online-update`, `--hb-dynamic-occurrence/--no-hb-dynamic-occurrence`, `--hb-occ-discount`.
	- **M5 hierarchy ablation:** `--m5-hierarchy-mode {off,on,ablation}` toggles hierarchy-aware pooling; ablation mode writes `m5_hierarchy_ablation.csv` with hierarchy vs non-hierarchy TSB-HB rows.
	- **Highlights:** Generates `point_metrics.csv`, shrinkage plots (`fig_shrink_p.png`, `fig_shrink_size.png`), segmentation metrics (`segmentation_rmsse.csv`), slice diagnostics (`point_slice_metrics.csv`), and `point_metrics_m5.csv` on M5.
- `run_prob.py`
	- **Purpose:** Produces probabilistic forecasts (quantiles) for TSB-HB, AutoARIMA/AutoTheta, and hurdle baselines (local/global LogNormal).
	- **Datasets:** Online Retail only (M5 support not implemented).
	- **Protocol options:** `--protocol fixed` or `--protocol walk_forward` (`--walk-step` controls block size).
	- **HB controls:** `--hb-regime-aware/--no-hb-regime-aware`, `--hb-group-shrink-strength`, `--hb-online-update/--no-hb-online-update`, `--hb-dynamic-occurrence/--no-hb-dynamic-occurrence`, `--hb-occ-discount`, `--hb-bootstrap-draws`, `--hb-disable-hyper-uncertainty`.
	- **Calibration control:** `--hb-calibration-mode {none,location_scale}` with `--hb-calibration-ratio` and `--hb-calibration-samples` for optional post-hoc quantile calibration on TSB-HB.
	- **Optional neural baseline:** DeepAR is available only for fixed protocol via `--with-deepar`.
	- **Outputs:** `prob_quantiles.csv`, `prob_pinball.csv`, `probabilistic_forecast_pinball_results.csv`, `coverage_summary.csv`, `pit_values.csv`, `prob_metrics.csv`, `prob_slice_metrics.csv`, plus calibration/PIT plots.
- `run_grid.py`
	- **Purpose:** Sweeps across `(alpha_d, alpha_p)` combinations for the TSB baseline to compare against the TSB-HB reference.
	- **Datasets:** Online Retail only.
	- **Outputs:** `grid_summary.csv` summarising ME/MAE/RMSE/RMSSE for each grid point.
- `run_ablation.py`
	- **Purpose:** Compares different shrinkage/likelihood variants (HB LogNormal, MLE LogNormal, HB Gamma) to quantify each modelling choice.
	- **Datasets:** Online Retail only.
	- **Outputs:** `ablation_metrics.csv` with ME/MAE/RMSE/RMSSE per variant.
- `run_coverage_pit.py`
	- **Purpose:** Compatibility wrapper for probabilistic diagnostics; delegates to `run_prob.py` so coverage/PIT use the same shared pipeline.
	- **Datasets:** Online Retail only.
	- **Outputs:** Same as `run_prob.py` (including `coverage_summary.csv` and `pit_values.csv`).
- `run_deepar.py`
	- **Purpose:** Optional neural benchmark using `neuralforecast`’s AutoDeepAR across configurable horizons.
	- **Datasets:** Online Retail only (subset sampling controlled via CLI flags).
	- **Requirements:** Install extras via `uv sync --extra deepar`. Produces `multi_horizon_comparison_results.csv` with ME/MAE/RMSE/RMSSE per horizon and model.

## Outputs

By default, each experiment writes results, diagnostics, and plots into `outputs/`. Clean the directory between runs if you need a fresh slate, or override `--out` with a dedicated subdirectory for reproducibility.

### Configurable batch runner

For the current v2 setup (group shrink + dynamic occurrence + optional probabilistic calibration), use:

```bash
bash scripts/experiments/run_v2_suite.sh scripts/experiments/config.v2.env
```

Edit `scripts/experiments/config.v2.env` to toggle protocol/dataset runs and all HB switches from one place.

## Latest benchmark snapshot (2026-02-24, DeepAR excluded)

Source run directory:
- `outputs/nightly_full_live_20260224_022743`

Closest statistical comparator is `Hurdle-Local-LogNormal`; deltas below are `TSB-HB - Hurdle-Local-LogNormal`.

### Online Retail point forecasting

| Protocol | TSB-HB MAE | TSB-HB RMSE | TSB-HB WRMSSE | dMAE | dRMSE | dWRMSSE |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 5.7665 | 17.6934 | 1.1769 | -0.2266 | -0.0643 | -0.0016 |
| walk_forward | 5.5853 | 17.2619 | 1.1504 | -0.1304 | -0.0268 | +0.0006 |

Takeaway:
- Fixed protocol: TSB-HB is the best WRMSSE among non-neural baselines in this run.
- Walk protocol: TSB-HB improves MAE/RMSE vs Hurdle-Local but is nearly tied on WRMSSE.

### Online Retail probabilistic forecasting (`@80`)

| Protocol | dPinball | dCoverage@80 | dAIW@80 | dGoalScore@80 |
|---|---:|---:|---:|---:|
| fixed | +0.0028 | +0.0000 | -0.5087 | -0.5599 |
| walk_forward | +0.0040 | -0.0005 | -0.2901 | -0.3257 |

Positive-only (`y>0`) deltas:
- fixed: `dPinball_pos=+0.0582`, `dCoverage@80_pos=+0.0001`, `dAIW@80_pos=-1.2509`, `dGoalScore@80_pos=-1.4772`
- walk_forward: `dPinball_pos=+0.0406`, `dCoverage@80_pos=-0.0018`, `dAIW@80_pos=-0.7115`, `dGoalScore@80_pos=-0.7858`

Takeaway:
- TSB-HB gives sharper intervals (smaller AIW) at nearly the same coverage.
- Pinball remains slightly worse than Hurdle-Local overall and on positive-only slices.

### M5 point ablation (hierarchy on/off)

| Model | MAE | RMSE | WRMSSE |
|---|---:|---:|---:|
| TSB-HB-NoHierarchy | 1.1771 | 2.8286 | 1.1038 |
| TSB-HB-Hierarchy | 1.1824 | 2.8608 | 1.1119 |
| Hurdle-Local-LogNormal | 1.1913 | 2.8272 | 1.1015 |

Takeaway:
- In this run, M5 hierarchy does not improve TSB-HB point metrics; keep hierarchy as an ablation switch instead of a default claim.

### Calibration ablation (`none` vs `location_scale`, fixed)

For TSB-HB, `location_scale` relative to `none`:
- `pinball_mean: +0.0483`
- `Coverage@80: -0.0844` (closer to nominal 0.8)
- `AIW@80: -3.6779`
- `GoalScore@80: -4.6280`
- positive-only `pinball_mean_pos: +0.3064`, `Coverage@80_pos: -0.1275`, `AIW@80_pos: -7.5019`

Takeaway:
- `location_scale` strongly narrows intervals and moves unconditional coverage toward nominal, but degrades pinball (especially positive-only).
- Default recommendation remains `--hb-calibration-mode none`; use `location_scale` only as a controlled calibration ablation.

### Current claim boundary

Supported by current evidence:
- TSB-HB is a strong point baseline and often slightly better than Hurdle-Local in fixed and sparse/cold-start slices.
- TSB-HB provides sharper probabilistic intervals at matched or near-matched coverage.

Not yet supported as a strong claim:
- TSB-HB universally outperforming Hurdle-Local on probabilistic accuracy (pinball), especially conditional on `y>0`.

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
