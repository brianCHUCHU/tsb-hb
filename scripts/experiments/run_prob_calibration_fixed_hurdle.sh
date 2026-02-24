#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${1:-outputs/prob_calibration_fixed_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

run_cmd() {
  echo ""
  echo ">>> $*"
  "$@"
}

COMMON_ARGS=(
  --protocol fixed
  --baseline-mode hurdle_only
  --hb-item-variance-mode conjugate
  --hb-variance-prior-df 20
  --hb-bootstrap-draws 0
)

NONE_OUT="${OUT_ROOT}/none"
LS_OUT="${OUT_ROOT}/location_scale"
mkdir -p "${NONE_OUT}" "${LS_OUT}"

run_cmd uv run python -m experiments.run_prob \
  "${COMMON_ARGS[@]}" \
  --hb-calibration-mode none \
  --out "${NONE_OUT}"

run_cmd uv run python -m experiments.run_prob \
  "${COMMON_ARGS[@]}" \
  --hb-calibration-mode location_scale \
  --hb-calibration-ratio 0.20 \
  --hb-calibration-samples 1000 \
  --hb-calibration-lambda-min 0.40 \
  --hb-calibration-lambda-max 1.10 \
  --hb-calibration-lambda-steps 15 \
  --hb-calibration-coverage-weight 1.00 \
  --out "${LS_OUT}"

for d in "${NONE_OUT}" "${LS_OUT}"; do
  echo ""
  echo "--- $(basename "${d}")"
  awk -F, 'NR==1 || $2=="TSB-HB" || $2=="Hurdle-Local-LogNormal" || $2=="Hurdle-Global-LogNormal"' "${d}/prob_metrics.csv" | column -s, -t
  if [[ -f "${d}/hb_calibration_params.csv" ]]; then
    echo "[hb_calibration_params.csv]"
    cat "${d}/hb_calibration_params.csv"
  fi
done

echo ""
echo "Done. Outputs: ${OUT_ROOT}"
