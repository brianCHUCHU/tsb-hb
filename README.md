# TSB-HB Supplement (Modularized)

This repository provides a modular, reproducible supplement of the TSB-HB experiments refactored from `TSBHB_v3.ipynb`.

## Setup

**Important**: This code requires Python 3.10.x due to Ray and NeuralForecast dependencies.

- Create and activate a Python 3.10 virtual environment:

```bash
# Using conda (recommended)
conda create -n tsbhb_env python=3.10
conda activate tsbhb_env

# Or using venv
python3.10 -m venv .venv && source .venv/bin/activate  # Linux/Mac
# or
py -3.10 -m venv .venv && .venv\Scripts\activate  # Windows
```

- Install pinned dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

### Online Retail Dataset

- Place the Online Retail dataset at `data/online_retail.csv`.

### M5 Dataset (Optional)

To run M5 experiments, download the M5 dataset from [Kaggle M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy/data):

1. Download `sales_train_evaluation.csv` (or `sales_train_validation.csv`)
2. Download `calendar.csv`
3. Place both files in the `data/` directory

The code will **automatically convert** the wide format to long format when loading. No manual preprocessing needed!

```bash
data/
  ├── Online_Retail.csv
  ├── sales_train_evaluation.csv  # M5 wide format (auto-converted)
  └── calendar.csv                 # M5 calendar
```

Alternatively, if you already have the long format data:
```bash
data/
  ├── m5_evaluation_long.csv       # M5 long format (preferred)
  └── calendar.csv
```

**Optional: Pre-convert to long format**

If you want to pre-convert the data (recommended for faster repeated loading):

```bash
python scripts/convert_m5_to_long.py \
    --input data/sales_train_evaluation.csv \
    --output data/m5_evaluation_long.csv
```

## One‑click Re‑run (examples)

Each script supports `--data`, `--out`, and `--seed` (defaults provided).

### Online Retail Experiments

```bash
python -m tsbhb.experiments.run_point --data data/online_retail.csv
python -m tsbhb.experiments.run_ablation --data data/online_retail.csv
python -m tsbhb.experiments.run_grid --data data/online_retail.csv
python -m tsbhb.experiments.run_prob --data data/online_retail.csv
python -m tsbhb.experiments.run_coverage_pit --data data/online_retail.csv
```

### M5 Experiments

```bash
# Point forecast on M5 (auto-detects wide/long format)
python -m tsbhb.experiments.run_point --dataset m5 \
    --m5-sales data/sales_train_evaluation.csv \
    --m5-calendar data/calendar.csv \
    --m5-sample-size 5000

# Or use defaults if files are in data/ directory
python -m tsbhb.experiments.run_point --dataset m5
```

## Tests

Before running tests, generate expected “truth” outputs once by executing the original `TSBHB_v3.ipynb` and saving the following CSVs into `experiments/expected/`:

- `point_metrics.csv`
- `ablation_metrics.csv`
- `grid_summary.csv`
- `prob_quantiles.csv`
- `coverage_summary.csv`
- `pit_values.csv`

Then run:

```bash
pytest -q
```

Tests will run the refactored entrypoints to produce `experiments/outputs/*.csv` and compare to `experiments/expected/*.csv`.

## Deep Learning Benchmark

Implement `DLBenchmark` in `src/tsbhb/models/dl_benchmark.py` by providing `fit()` and `predict()` (optionally probabilistic via quantiles). Extend `run_point.py` / `run_prob.py` with a `--model dl` branch to enable it. Add any extra packages to `requirements.txt` as needed.

## Outputs

- `run_point.py`: writes `experiments/outputs/point_metrics.csv` and shrinkage plots `fig_shrink_p.png`, `fig_shrink_size.png`.
- `run_ablation.py`: writes `experiments/outputs/ablation_metrics.csv`.
- `run_grid.py`: writes `experiments/outputs/grid_summary.csv`.
- `run_prob.py`: writes `experiments/outputs/prob_quantiles.csv` and `experiments/outputs/prob_pinball.csv`.
- `run_coverage_pit.py`: writes `experiments/outputs/coverage_summary.csv` and `experiments/outputs/pit_values.csv`.

