#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-scripts/experiments/config.paper.env}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

is_true() {
  local v="${1:-0}"
  case "${v}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

run_cmd() {
  echo ""
  echo ">>> $*"
  "$@"
}

mkdir -p "${OUT_ROOT}"

POINT_MODE="${POINT_BASELINE_MODE:-paper}"
PROB_MODE="${PROB_BASELINE_MODE:-paper}"

if is_true "${RUN_ONLINE_POINT_FIXED:-1}"; then
  OUT_DIR="${OUT_ROOT}/online_retail/point_fixed"
  mkdir -p "${OUT_DIR}"
  run_cmd uv run python -m experiments.run_point \
    --dataset online_retail \
    --protocol fixed \
    --baseline-mode "${POINT_MODE}" \
    --hb-item-variance-mode conjugate \
    --hb-variance-prior-df 20 \
    --seed "${SEED}" \
    --out "${OUT_DIR}"
fi

if is_true "${RUN_ONLINE_PROB_FIXED:-1}"; then
  OUT_DIR="${OUT_ROOT}/online_retail/prob_fixed"
  mkdir -p "${OUT_DIR}"
  CMD=(
    uv run python -m experiments.run_prob
    --dataset online
    --protocol fixed
    --baseline-mode "${PROB_MODE}"
    --hb-item-variance-mode conjugate
    --hb-variance-prior-df 20
    --hb-bootstrap-draws 20
    --hb-calibration-mode location_scale
    --hb-calibration-ratio 0.20
    --hb-calibration-samples 1000
    --hb-calibration-lambda-min 0.60
    --hb-calibration-lambda-max 1.20
    --hb-calibration-lambda-steps 13
    --hb-calibration-coverage-weight 0.50
    --include-conformal-baselines
    --seed "${SEED}"
    --out "${OUT_DIR}"
  )
  if is_true "${WITH_DEEPAR:-0}"; then
    CMD+=(--with-deepar)
  fi
  run_cmd "${CMD[@]}"
fi

if is_true "${RUN_ONLINE_POINT_WALK:-1}"; then
  OUT_DIR="${OUT_ROOT}/online_retail/point_walk_forward"
  mkdir -p "${OUT_DIR}"
  run_cmd uv run python -m experiments.run_point \
    --dataset online_retail \
    --protocol walk_forward \
    --baseline-mode "${POINT_MODE}" \
    --walk-step "${WALK_STEP}" \
    --hb-item-variance-mode conjugate \
    --hb-variance-prior-df 20 \
    --seed "${SEED}" \
    --out "${OUT_DIR}"
fi

if is_true "${RUN_ONLINE_PROB_WALK:-0}"; then
  OUT_DIR="${OUT_ROOT}/online_retail/prob_walk_forward"
  mkdir -p "${OUT_DIR}"
  run_cmd uv run python -m experiments.run_prob \
    --dataset online \
    --protocol walk_forward \
    --baseline-mode "${PROB_MODE}" \
    --walk-step "${WALK_STEP}" \
    --hb-item-variance-mode conjugate \
    --hb-variance-prior-df 20 \
    --hb-calibration-mode location_scale \
    --include-conformal-baselines \
    --seed "${SEED}" \
    --out "${OUT_DIR}"
fi

if is_true "${RUN_M5_POINT:-1}"; then
  OUT_DIR="${OUT_ROOT}/m5/point_fixed"
  mkdir -p "${OUT_DIR}"
  run_cmd uv run python -m experiments.run_point \
    --dataset m5 \
    --protocol fixed \
    --baseline-mode "${POINT_MODE}" \
    --m5-hierarchy-mode off \
    --m5-sample-size "${M5_SAMPLE_SIZE}" \
    --hb-item-variance-mode conjugate \
    --hb-variance-prior-df 20 \
    --seed "${SEED}" \
    --out "${OUT_DIR}"
fi

echo ""
echo "Paper-facing experiment suite completed."

