#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-scripts/experiments/config.v2.env}"
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

HB_REGIME_FLAG="--no-hb-regime-aware"
if is_true "${HB_REGIME_AWARE:-1}"; then
  HB_REGIME_FLAG="--hb-regime-aware"
fi

HB_ONLINE_FLAG="--no-hb-online-update"
if is_true "${HB_ONLINE_UPDATE:-1}"; then
  HB_ONLINE_FLAG="--hb-online-update"
fi

HB_DYNAMIC_FLAG="--no-hb-dynamic-occurrence"
if is_true "${HB_DYNAMIC_OCCURRENCE:-0}"; then
  HB_DYNAMIC_FLAG="--hb-dynamic-occurrence"
fi

HB_HYPER_UNCERTAINTY_FLAG="--hb-disable-hyper-uncertainty"
if is_true "${HB_USE_HYPER_UNCERTAINTY:-1}"; then
  HB_HYPER_UNCERTAINTY_FLAG=""
fi

if is_true "${RUN_ONLINE_POINT_FIXED:-1}"; then
  OUT_DIR="${OUT_ROOT}/online_retail/point_fixed"
  mkdir -p "${OUT_DIR}"
  run_cmd uv run python -m experiments.run_point \
    --dataset online_retail \
    --protocol fixed \
    --seed "${SEED}" \
    --walk-step "${WALK_STEP}" \
    "${HB_REGIME_FLAG}" \
    --hb-group-shrink-strength "${HB_GROUP_SHRINK_STRENGTH}" \
    --out "${OUT_DIR}"
fi

if is_true "${RUN_ONLINE_POINT_WALK:-1}"; then
  OUT_DIR="${OUT_ROOT}/online_retail/point_walk_forward"
  mkdir -p "${OUT_DIR}"
  run_cmd uv run python -m experiments.run_point \
    --dataset online_retail \
    --protocol walk_forward \
    --seed "${SEED}" \
    --walk-step "${WALK_STEP}" \
    "${HB_REGIME_FLAG}" \
    "${HB_ONLINE_FLAG}" \
    "${HB_DYNAMIC_FLAG}" \
    --hb-group-shrink-strength "${HB_GROUP_SHRINK_STRENGTH}" \
    --hb-occ-discount "${HB_OCC_DISCOUNT}" \
    --out "${OUT_DIR}"
fi

if is_true "${RUN_ONLINE_PROB_FIXED:-1}"; then
  OUT_DIR="${OUT_ROOT}/online_retail/prob_fixed"
  mkdir -p "${OUT_DIR}"
  CMD=(
    uv run python -m experiments.run_prob
    --protocol fixed
    --seed "${SEED}"
    "${HB_REGIME_FLAG}"
    --hb-group-shrink-strength "${HB_GROUP_SHRINK_STRENGTH}"
    --hb-bootstrap-draws "${HB_BOOTSTRAP_DRAWS}"
    --hb-calibration-mode "${HB_CALIBRATION_MODE}"
    --hb-calibration-ratio "${HB_CALIBRATION_RATIO}"
    --hb-calibration-samples "${HB_CALIBRATION_SAMPLES}"
    --out "${OUT_DIR}"
  )
  if [[ -n "${HB_HYPER_UNCERTAINTY_FLAG}" ]]; then
    CMD+=("${HB_HYPER_UNCERTAINTY_FLAG}")
  fi
  if is_true "${WITH_DEEPAR:-0}"; then
    CMD+=(
      --with-deepar
      --horizon "${DEEPAR_HORIZON}"
      --input-size "${DEEPAR_INPUT_SIZE}"
      --max-steps "${DEEPAR_MAX_STEPS}"
    )
  fi
  run_cmd "${CMD[@]}"
fi

if is_true "${RUN_ONLINE_PROB_WALK:-1}"; then
  OUT_DIR="${OUT_ROOT}/online_retail/prob_walk_forward"
  mkdir -p "${OUT_DIR}"
  CMD=(
    uv run python -m experiments.run_prob
    --protocol walk_forward
    --seed "${SEED}"
    --walk-step "${WALK_STEP}"
    "${HB_REGIME_FLAG}"
    "${HB_ONLINE_FLAG}"
    "${HB_DYNAMIC_FLAG}"
    --hb-group-shrink-strength "${HB_GROUP_SHRINK_STRENGTH}"
    --hb-occ-discount "${HB_OCC_DISCOUNT}"
    --hb-bootstrap-draws "${HB_BOOTSTRAP_DRAWS}"
    --hb-calibration-mode "${HB_CALIBRATION_MODE}"
    --hb-calibration-ratio "${HB_CALIBRATION_RATIO}"
    --hb-calibration-samples "${HB_CALIBRATION_SAMPLES}"
    --out "${OUT_DIR}"
  )
  if [[ -n "${HB_HYPER_UNCERTAINTY_FLAG}" ]]; then
    CMD+=("${HB_HYPER_UNCERTAINTY_FLAG}")
  fi
  run_cmd "${CMD[@]}"
fi

if is_true "${RUN_M5_POINT:-1}"; then
  OUT_DIR="${OUT_ROOT}/m5/point_fixed"
  mkdir -p "${OUT_DIR}"
  run_cmd uv run python -m experiments.run_point \
    --dataset m5 \
    --protocol fixed \
    --seed "${SEED}" \
    --m5-sample-size "${M5_SAMPLE_SIZE}" \
    --hb-group-shrink-strength "${HB_GROUP_SHRINK_STRENGTH}" \
    --m5-hierarchy-mode "${M5_HIERARCHY_MODE}" \
    --out "${OUT_DIR}"
fi

echo ""
echo "All configured runs have been launched/completed."

