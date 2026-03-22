# TSB-HB

Reference implementation for `Taxonomy-Conditioned Hierarchical Bayesian TSB Models for Heterogeneous Intermittent Demand Forecasting`.

- Online Retail uses ADI/CV^2 taxonomy-conditioned pooling computed once from the initialization window.
- M5 uses a single global pool in the main reported run.
- Fixed-origin is the main evaluation protocol; walk-forward is kept as a robustness check.
- Probabilistic forecasting uses the released configuration: conjugate variance with `nu=20`, bootstrap averaging with `B=20`, and location-scale calibration on the initialization window only.
- Generated experiment artifacts live under `outputs/` and are not committed.

## Setup

1. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create the project environment:

```bash
uv sync
```

3. Optional DeepAR extras:

```bash
uv sync --extra deepar
```

4. Inspect CLI help:

```bash
uv run python -m experiments.run_point --help
uv run python -m experiments.run_prob --help
```

## Data

### Online Retail

Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail) and place it at `data/online_retail.csv` or `data/Online_Retail.csv`.

### M5

Download `sales_train_evaluation.csv` (or `sales_train_validation.csv`) and `calendar.csv` from the [Kaggle M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy/data), then place both files in `data/`.

Optional one-time conversion to long format:

```bash
uv run python -m tools.convert_m5_to_long \
  --input data/sales_train_evaluation.csv \
  --output data/m5_evaluation_long.csv
```

## Paper Reproduction

Run the canonical paper suite with:

```bash
bash scripts/experiments/run_paper_suite.sh scripts/experiments/config.paper.env
```

The generated outputs map directly to the paper-facing runs:

- `outputs/paper_reproduction/online_retail/point_fixed/point_metrics.csv`
- `outputs/paper_reproduction/online_retail/point_fixed/segmentation_rmsse.csv`
- `outputs/paper_reproduction/online_retail/prob_fixed/prob_metrics.csv`
- `outputs/paper_reproduction/online_retail/prob_fixed/coverage_summary.csv`
- `outputs/paper_reproduction/online_retail/prob_fixed/prob_pinball.csv`
- `outputs/paper_reproduction/online_retail/point_walk_forward/point_metrics.csv`
- `outputs/paper_reproduction/m5/point_fixed/point_metrics_m5.csv`

The exact commands and parameter choices are documented in [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md).

## Main Scripts

- `experiments.run_point`
  Default `--baseline-mode paper` matches the point-forecast baselines used in the paper. `--m5-hierarchy-mode off` is the paper default for M5.
- `experiments.run_prob`
  Default `--baseline-mode paper` plus default `--hb-calibration-mode location_scale` matches the released probabilistic configuration.
- `experiments.run_coverage_pit`
  Thin wrapper around `run_prob` for the same probabilistic pipeline.
- `experiments.run_grid`, `experiments.run_ablation`, `experiments.run_deepar`
  Supplemental experiments that are not required for the canonical reproduction path.

Supplemental baseline modes are still available when needed:

- `--baseline-mode extended`: paper baselines plus hurdle baselines.
- `--baseline-mode hurdle_only`: TSB-HB plus hurdle baselines only.
- `--baseline-mode hb_only`: TSB-HB only.

## Repository Layout

- `src/models/tsb_hb.py`: hierarchical Bayesian TSB model and forecasting logic.
- `src/experiments/`: point, probabilistic, ablation, and benchmark entry points.
- `src/data_loading.py`, `src/metrics.py`, `src/plotting.py`: shared preprocessing, metrics, and figures.
- `scripts/experiments/run_paper_suite.sh`: canonical batch runner for the paper configuration.

## References

1. Azul Garza, Max Mergenthaler Canseco, Cristian Challu, and Kin G. Olivares.  
   **StatsForecast: Lightning fast forecasting with statistical and econometric models.**  
   [https://github.com/Nixtla/statsforecast](https://github.com/Nixtla/statsforecast)

2. Addison Howard, inversion, Spyros Makridakis, and Vangelis.  
   **M5 Forecasting - Accuracy.**  
   [https://kaggle.com/competitions/m5-forecasting-accuracy](https://kaggle.com/competitions/m5-forecasting-accuracy)

3. Daqing Chen.  
   **Online Retail.** UCI Machine Learning Repository, 2015.  
   [https://doi.org/10.24432/C5BW33](https://doi.org/10.24432/C5BW33)
