# Paper Reproduction

This repository is aligned to `Hummer_noHurdle.tex`.

## Canonical assumptions

- Online Retail uses ADI/CV^2 taxonomy-conditioned pooling computed once from the initialization window.
- M5 uses a single global pool in the main reported run.
- Point forecasting uses the conjugate variance configuration with `nu=20`.
- Probabilistic forecasting uses bootstrap averaging with `B=20` and post-hoc location-scale calibration on the initialization window only.
- Generated outputs are written under `outputs/` and are not committed.

## One-command reproduction

```bash
bash scripts/experiments/run_paper_suite.sh scripts/experiments/config.paper.env
```

This produces:

- `outputs/paper_reproduction/online_retail/point_fixed/point_metrics.csv`
- `outputs/paper_reproduction/online_retail/point_fixed/segmentation_rmsse.csv`
- `outputs/paper_reproduction/online_retail/prob_fixed/prob_metrics.csv`
- `outputs/paper_reproduction/online_retail/prob_fixed/coverage_summary.csv`
- `outputs/paper_reproduction/online_retail/prob_fixed/prob_pinball.csv`
- `outputs/paper_reproduction/online_retail/point_walk_forward/point_metrics.csv`
- `outputs/paper_reproduction/m5/point_fixed/point_metrics_m5.csv`

## Direct commands

Online Retail point main table:

```bash
uv run python -m experiments.run_point \
  --dataset online_retail \
  --protocol fixed \
  --baseline-mode paper \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --out outputs/paper_reproduction/online_retail/point_fixed
```

Online Retail probabilistic table:

```bash
uv run python -m experiments.run_prob \
  --dataset online \
  --protocol fixed \
  --baseline-mode paper \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --hb-bootstrap-draws 20 \
  --hb-calibration-mode location_scale \
  --hb-calibration-ratio 0.20 \
  --hb-calibration-samples 1000 \
  --hb-calibration-lambda-min 0.60 \
  --hb-calibration-lambda-max 1.20 \
  --hb-calibration-lambda-steps 13 \
  --hb-calibration-coverage-weight 0.50 \
  --include-conformal-baselines \
  --out outputs/paper_reproduction/online_retail/prob_fixed
```

Walk-forward robustness:

```bash
uv run python -m experiments.run_point \
  --dataset online_retail \
  --protocol walk_forward \
  --baseline-mode paper \
  --walk-step 7 \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --out outputs/paper_reproduction/online_retail/point_walk_forward
```

M5 fixed-origin point table:

```bash
uv run python -m experiments.run_point \
  --dataset m5 \
  --protocol fixed \
  --baseline-mode paper \
  --m5-hierarchy-mode off \
  --m5-sample-size 5000 \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --out outputs/paper_reproduction/m5/point_fixed
```

## Supplemental modes

- `--baseline-mode extended` adds hurdle baselines on top of the paper baselines.
- `--baseline-mode hurdle_only` keeps only TSB-HB and hurdle baselines.
- `--m5-hierarchy-mode ablation` reports hierarchy vs global-pool sensitivity for M5.
