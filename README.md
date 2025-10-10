# TSB-HB Supplement (Modularized)

This repository provides a modular, reproducible supplement of the TSB-HB experiments refactored from `TSBHB_v3.ipynb`.

## Setup

- Create and activate a virtual environment, then install pinned dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

- Place the Online Retail dataset at `data/online_retail.csv`.

## One‑click Re‑run (examples)

Each script supports `--data`, `--out`, and `--seed` (defaults provided).

```bash
python -m tsbhb.experiments.run_point --data data/online_retail.csv
python -m tsbhb.experiments.run_ablation --data data/online_retail.csv
python -m tsbhb.experiments.run_grid --data data/online_retail.csv
python -m tsbhb.experiments.run_prob --data data/online_retail.csv
python -m tsbhb.experiments.run_coverage_pit --data data/online_retail.csv
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

