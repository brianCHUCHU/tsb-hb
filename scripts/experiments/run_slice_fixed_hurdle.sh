#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${1:-outputs/slice_fixed_hurdle_$(date +%Y%m%d_%H%M%S)}"
POINT_OUT="${OUT_ROOT}/point"
PROB_OUT="${OUT_ROOT}/prob"
mkdir -p "${POINT_OUT}" "${PROB_OUT}"

run_cmd() {
  echo ""
  echo ">>> $*"
  "$@"
}

COMMON=(
  --protocol fixed
  --baseline-mode hurdle_only
  --hb-item-variance-mode conjugate
  --hb-variance-prior-df 20
)

run_cmd uv run python -m experiments.run_point \
  --dataset online_retail \
  "${COMMON[@]}" \
  --out "${POINT_OUT}"

run_cmd uv run python -m experiments.run_prob \
  "${COMMON[@]}" \
  --hb-bootstrap-draws 0 \
  --hb-calibration-mode none \
  --out "${PROB_OUT}"

OUT_ROOT_ENV="${OUT_ROOT}" run_cmd uv run python - <<'PY'
import pandas as pd
import os
from pathlib import Path
root = Path(os.environ["OUT_ROOT_ENV"])
pt = pd.read_csv(root / "point" / "point_slice_metrics.csv")
pr = pd.read_csv(root / "prob" / "prob_slice_metrics.csv")

pt = pt[pt["model"].isin(["TSB-HB-LogNormal", "Hurdle-Local-LogNormal"])].copy()
pr = pr[pr["model"].isin(["TSB-HB", "Hurdle-Local-LogNormal"])].copy()

def show_point(slice_type: str, slices: list[str]) -> None:
    d = pt[(pt["slice_type"] == slice_type) & (pt["slice"].isin(slices))].copy()
    if d.empty:
        return
    pv = d.pivot_table(index="slice", columns="model", values=["n_series", "MAE", "RMSE", "WRMSSE"])
    pv.columns = [f"{a}_{b}" for a, b in pv.columns]
    pv = pv.reset_index()
    if "MAE_TSB-HB-LogNormal" in pv.columns:
        pv["dMAE(TSB-HB-HL)"] = pv["MAE_TSB-HB-LogNormal"] - pv["MAE_Hurdle-Local-LogNormal"]
        pv["dRMSE(TSB-HB-HL)"] = pv["RMSE_TSB-HB-LogNormal"] - pv["RMSE_Hurdle-Local-LogNormal"]
        pv["dWRMSSE(TSB-HB-HL)"] = pv["WRMSSE_TSB-HB-LogNormal"] - pv["WRMSSE_Hurdle-Local-LogNormal"]
    print(f"\nPOINT [{slice_type}]")
    print(pv.to_string(index=False))

def show_prob(slice_type: str, slices: list[str]) -> None:
    d = pr[(pr["slice_type"] == slice_type) & (pr["slice"].isin(slices))].copy()
    if d.empty:
        return
    pv = d.pivot_table(index="slice", columns="model", values=["n_series", "pinball_mean", "Coverage@80", "AIW@80", "GoalScore@80"])
    pv.columns = [f"{a}_{b}" for a, b in pv.columns]
    pv = pv.reset_index()
    if "pinball_mean_TSB-HB" in pv.columns:
        pv["dPin(TSB-HB-HL)"] = pv["pinball_mean_TSB-HB"] - pv["pinball_mean_Hurdle-Local-LogNormal"]
        pv["dAIW80(TSB-HB-HL)"] = pv["AIW@80_TSB-HB"] - pv["AIW@80_Hurdle-Local-LogNormal"]
        pv["dCov80(TSB-HB-HL)"] = pv["Coverage@80_TSB-HB"] - pv["Coverage@80_Hurdle-Local-LogNormal"]
        pv["dGoal80(TSB-HB-HL)"] = pv["GoalScore@80_TSB-HB"] - pv["GoalScore@80_Hurdle-Local-LogNormal"]
    print(f"\nPROB [{slice_type}]")
    print(pv.to_string(index=False))

show_point("regime", ["Intermittent", "Lumpy"])
show_point("cold_start", ["CS_0_1", "CS_2_3", "CS_4_5", "CS_6_10", "CS_11_plus"])
show_prob("regime", ["Intermittent", "Lumpy"])
show_prob("cold_start", ["CS_0_1", "CS_2_3", "CS_4_5", "CS_6_10", "CS_11_plus"])

print(f"\nDone. Outputs: {root}")
PY
