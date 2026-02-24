# TSB-HB 最新版狀態（2026-02-24）

## 0. 版本鎖定（避免混用舊版）
- Branch: `codex/tsbhb-v2-regime-online-bootstrap-hier`
- 本文件只使用以下最新輸出：
  - `outputs/latest_core_fixed_20260224_014308/point`
  - `outputs/latest_core_fixed_20260224_014308/prob`
- 任何 `outputs/` 下其他較早目錄（例如 `v2_*`, `ablation_*`, `quick_*`）不納入本文件結論。

## 1. 目前 TSB-HB 內容（含新增且有效部分）

### 1.1 模型主體
- 結構：occurrence-size 分解，`y_hat = p_hat * s_hat`。
- Occurrence：Beta-Binomial shrinkage（可做 regime-aware pooling）。
- Size：LogNormal（log-space mean shrinkage）。
- Size variance：`conjugate` 模式（目前主版本，`--hb-item-variance-mode conjugate` + `--hb-variance-prior-df`）。
- Walk-forward：支援 online sufficient-stat updates（可選 dynamic occurrence discount）。

### 1.2 目前保留且可用的 calibration
- `none`
- `location_scale`（post-hoc，偏向 coverage/interval trade-off）

### 1.3 這輪整理後已刪除/停用
- 已移除 `quantile_shift`（先前驗證為無效，shift 常學到 0）。
- 已移除舊相容參數：
  - `--walk-baseline-mode`
  - `--hb-item-variance-shrink-strength`
- `run_point.py`/`run_prob.py` 現在只保留主線參數。

### 1.4 新增有效工具（用於貢獻驗證）
- `point_slice_metrics.csv`：依 `regime / sparse / cold_start` 切片的 point 指標。
- `prob_slice_metrics.csv`：同切片的 probabilistic 指標（含 `Coverage/AIW/pinball` 與 positive-only 欄位）。
- 一鍵腳本：`scripts/experiments/run_slice_fixed_hurdle.sh`。

## 2. 正文實驗規劃（已做/未做）

## 2.1 已做（最新版本）

### A. Online Retail, fixed, hurdle-only（主比較）
- 命令：
  - `uv run python -m experiments.run_point --dataset online_retail --protocol fixed --baseline-mode hurdle_only --hb-item-variance-mode conjugate --hb-variance-prior-df 20 --out outputs/latest_core_fixed_20260224_014308/point`
  - `uv run python -m experiments.run_prob --protocol fixed --baseline-mode hurdle_only --hb-item-variance-mode conjugate --hb-variance-prior-df 20 --hb-bootstrap-draws 0 --hb-calibration-mode none --out outputs/latest_core_fixed_20260224_014308/prob`

#### Point（overall）
| model | MAE | RMSE | WRMSSE |
|---|---:|---:|---:|
| TSB-HB-LogNormal | 5.7665 | 17.6934 | 1.1769 |
| Hurdle-Local-LogNormal | 5.9931 | 17.7577 | 1.1785 |
| Hurdle-Global-LogNormal | 7.0861 | 19.3411 | 1.3625 |

- `TSB-HB - Hurdle-Local` 差值：
  - MAE `-0.2266`
  - RMSE `-0.0643`
  - WRMSSE `-0.00155`

#### Prob（overall）
| model | pinball_mean | Coverage@80 | AIW@80 | GoalScore@80 |
|---|---:|---:|---:|---:|
| TSB-HB | 1.9161 | 0.9009 | 10.5467 | 11.6108 |
| Hurdle-Local-LogNormal | 1.9139 | 0.9013 | 11.0775 | 12.1995 |
| Hurdle-Global-LogNormal | 2.2607 | 0.8838 | 8.7724 | 9.5072 |

- `TSB-HB - Hurdle-Local` 差值：
  - pinball_mean `+0.00224`（略差）
  - Coverage@80 `-0.00040`（幾乎持平）
  - AIW@80 `-0.5308`（更窄）
  - GoalScore@80 `-0.5888`（較佳）

### B. 稀疏 / 冷啟動切片（主張 HB 價值的關鍵）
來源：
- `outputs/latest_core_fixed_20260224_014308/point/point_slice_metrics.csv`
- `outputs/latest_core_fixed_20260224_014308/prob/prob_slice_metrics.csv`

#### Point 切片重點（TSB-HB vs Hurdle-Local）
- `Intermittent`：MAE 較好（`-0.0356`），但 RMSE 微差（`+0.0172`）。
- `Lumpy`：MAE/RMSE/WRMSSE 皆較好（MAE `-0.2848`，RMSE `-0.0663`）。
- `CS_2_3`：明顯較好（MAE `-0.1361`，RMSE `-0.5427`，WRMSSE `-0.0385`）。
- `CS_0_1`：略差（MAE `+0.0189`，RMSE `+0.0063`）。

#### Prob 切片重點（TSB-HB vs Hurdle-Local）
- `Intermittent` / `Lumpy`：pinball 微差（約 `+0.002~0.003`），但 `AIW@80` 顯著更窄（約 `-0.17`、`-0.66`）。
- `CS_2_3`：pinball 微幅更好（`-0.00012`），`AIW@80` 更窄（`-0.0347`）。
- `CS_4_5`：TSB-HB 略退步（pinball `+0.00035`，AIW@80 `+0.0198`）。

## 2.2 還沒做（正文建議補齊）
1. Online Retail `walk_forward`（同一組設定：`hurdle_only + conjugate + calibration=none`），並輸出同樣 slice 表。
2. M5 point 主表（含 hierarchy on/off ablation，固定只用最新程式碼重跑）。
3. calibration 作為附錄 ablation：`none` vs `location_scale`（只用最新程式碼重跑，不混舊結果）。

## 3. 目前可用的論文說法（僅限最新版證據）
- 可以主張：
  - 在可比 baseline（Hurdle-Local）下，TSB-HB point 表現可小幅勝出（fixed）。
  - 在相近 coverage 下，TSB-HB 區間更窄（sharpness 較好），尤其在 sparse slice。
  - 在部分 cold-start slice（特別 `CS_2_3`）有明顯優勢。
- 不建議主張：
  - 全面 probabilistic 優於 Hurdle-Local（因 overall pinball 仍略差）。
  - 全域 calibration 已顯著改善（目前主版本仍偏保守）。
