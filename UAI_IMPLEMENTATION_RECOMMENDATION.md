# UAI 實作建議（綜合取捨版）

## 目標
在不大幅增加工程風險的前提下，提升論文可比性與 reviewer 可防守性，並維持正文可壓縮到 UAI 篇幅。

---

## 1. 先做的取捨（建議採用）

### 1.1 正文/附錄重排
- 正文保留：
  - 模型定義（occurrence × size）
  - 兩個核心 posterior mean closed-form
  - 複雜度量級（訓練/推論 big-O）
- 移附錄：
  - shrinkage 性質（monotonicity/bracketing/bias-variance 分解）
  - identifiability/consistency 細節
  - Gamma variant 詳式與完整演算法步驟

### 1.2 實驗協議採「雙 protocol」
- Protocol A: `fixed-origin multi-horizon`（保留你目前主線）
- Protocol B: `walk-forward`（每步吸收新觀測後再預測）

這是最關鍵的公平性補強：
- A 回答冷啟動/靜態外推
- B 回答實務滾動更新

### 1.3 baseline 先擴到「可落地且對齊」
先做（低風險、可快速完成）：
- 已有 intermittent/statistical：`CrostonClassic`, `CrostonSBA`, `TSB`, `ADIDA`, `IMAPA`, `AutoARIMA`, `AutoTheta`
- 已有 deep：`DeepAR`（保留，明確標註在 A/B 協議下的更新規則）
- 新增兩個對齊 baseline：
  - `Hurdle-Local-LogNormal`（每商品各自估）
  - `Hurdle-Global-LogNormal`（完全 pooled）

先不做（次階段）：
- GLMM/Tweedie（需要新依賴與額外穩定性處理，submission 風險較高）

### 1.4 超參策略（最小公平）
- 統一採 `train/val/test` 時序切分
- 只用 `val` 選超參，`test` 只報一次
- 不做 per-item tuning；改做全域一組（或依 ADI/CV2 分群共用）

---

## 2. 具體工程改動（對應目前程式碼）

### 2.1 新增共用評估協議層
新增檔案：`src/experiments/protocols.py`

建議提供：
- `split_train_val_test_fixed_origin(...)`
- `iter_walk_forward_frames(init_df, eval_df, step=1)`
- `evaluate_point_models(...)`
- `evaluate_prob_models(...)`

目的：避免 `run_point.py`, `run_deepar.py`, `run_prob.py` 各自實作不同版本的切分與評估。

### 2.2 新增 hurdle baselines
新增檔案：`src/models/hurdle_baselines.py`

建議先做 point + quantile（用參數抽樣/近似）：
- `fit_hurdle_local_lognormal(train_df)`
- `predict_hurdle_local_lognormal(params, eval_df, quantiles=None)`
- `fit_hurdle_global_lognormal(train_df)`
- `predict_hurdle_global_lognormal(params, eval_df, quantiles=None)`

### 2.3 調整現有實驗腳本
- `src/experiments/run_point.py`
  - 加入 `--protocol {fixed,walk_forward}`
  - 新增 Local/Global hurdle baseline
- `src/experiments/run_prob.py`
  - 輸出 coverage@50/80（95 視 quantile 是否可得）+ AIW + pinball
  - 文案改為「empirically conservative/well-calibrated」
- `src/experiments/run_deepar.py`
  - 明確對齊 protocol A/B 的資訊集合
  - 若是 walk-forward，不需要每步重訓；只更新 context

### 2.4 輸出檔命名統一（避免混淆）
建議輸出路徑格式：
- `outputs/<dataset>/<protocol>/point_metrics.csv`
- `outputs/<dataset>/<protocol>/prob_metrics.csv`
- `outputs/<dataset>/<protocol>/coverage_summary.csv`

---

## 3. 兩週內可交付版本（務實版）

### Week 1
- 完成 `protocols.py`
- 完成 `Hurdle-Local/Global-LogNormal`
- 讓 `run_point.py` 可切換 fixed/walk-forward

### Week 2
- 接上 `run_prob.py`（coverage + pinball + interval width）
- 將 `run_deepar.py` 對齊同一協議
- 出主表（A/B protocol 各一）與附錄表

---

## 4. 論文文字層面的建議（高回報低成本）

- 把 claim 從「calibrated」收斂為：
  - `empirically well-calibrated` 或 `conservative intervals`
- 把「generalizes exponential smoothing」收斂為：
  - `retains multiplicative decomposition; replaces online EWMA updates with fixed-origin EB shrinkage`
- 在方法段前面就先說清楚：
  - 本文核心場景是 fixed-origin（並於實驗補充 walk-forward）

---

## 5. 最終建議（結論）
最值得先做的是：
1. 補齊雙 protocol（fixed + walk-forward）
2. 新增 Hurdle local/global 兩個對齊 baseline
3. 建立統一且節制的 val-only 調參規則

這三項完成後，能在不引入高風險新依賴的情況下，顯著降低「比較不公平」與「設定偏向特定模型」的 reviewer 攻擊面。
