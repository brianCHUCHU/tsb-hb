#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${1:-outputs/vsv_conjugate_fixed_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

run_cmd() {
  echo ""
  echo ">>> $*"
  "$@"
}

BASE_POINT_OUT="${OUT_ROOT}/point_base"
CONJ_POINT_OUT="${OUT_ROOT}/point_conjugate"
BASE_PROB_OUT="${OUT_ROOT}/prob_base"
CONJ_PROB_OUT="${OUT_ROOT}/prob_conjugate"
mkdir -p "${BASE_POINT_OUT}" "${CONJ_POINT_OUT}" "${BASE_PROB_OUT}" "${CONJ_PROB_OUT}"

run_cmd uv run python -m experiments.run_point \
  --dataset online_retail \
  --protocol fixed \
  --baseline-mode hurdle_only \
  --hb-item-variance-mode group \
  --out "${BASE_POINT_OUT}"

run_cmd uv run python -m experiments.run_point \
  --dataset online_retail \
  --protocol fixed \
  --baseline-mode hurdle_only \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --out "${CONJ_POINT_OUT}"

run_cmd uv run python -m experiments.run_prob \
  --protocol fixed \
  --baseline-mode hurdle_only \
  --hb-item-variance-mode group \
  --out "${BASE_PROB_OUT}"

run_cmd uv run python -m experiments.run_prob \
  --protocol fixed \
  --baseline-mode hurdle_only \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --out "${CONJ_PROB_OUT}"

echo ""
echo "=== Quick comparison (TSB-HB vs Hurdle-Local) ==="
for d in "${BASE_POINT_OUT}" "${CONJ_POINT_OUT}"; do
  echo "--- $(basename "${d}") point"
  awk -F, 'NR==1 || $2=="TSB-HB-LogNormal" || $2=="Hurdle-Local-LogNormal"' "${d}/point_metrics.csv" | column -s, -t
done

for d in "${BASE_PROB_OUT}" "${CONJ_PROB_OUT}"; do
  echo "--- $(basename "${d}") prob"
  awk -F, 'NR==1 || $2=="TSB-HB" || $2=="Hurdle-Local-LogNormal"' "${d}/prob_metrics.csv" | column -s, -t
done

echo ""
echo "Done. Outputs: ${OUT_ROOT}"
