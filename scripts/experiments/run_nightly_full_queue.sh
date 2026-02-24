#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${1:-outputs/nightly_full_$(date +%Y%m%d_%H%M%S)}"
WALK_STEP="${WALK_STEP:-7}"
SEED="${SEED:-42}"
M5_SAMPLE_SIZE="${M5_SAMPLE_SIZE:-5000}"
DEEPAR_MAX_STEPS="${DEEPAR_MAX_STEPS:-500}"

LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_ROOT}/online_retail" "${OUT_ROOT}/m5"

MASTER_LOG="${OUT_ROOT}/run.log"
STATUS_TSV="${OUT_ROOT}/status.tsv"
SUMMARY_TXT="${OUT_ROOT}/summary.txt"

cat > "${SUMMARY_TXT}" <<TXT
Nightly full experiment queue
- OUT_ROOT: ${OUT_ROOT}
- WALK_STEP: ${WALK_STEP}
- SEED: ${SEED}
- M5_SAMPLE_SIZE: ${M5_SAMPLE_SIZE}
- DEEPAR_MAX_STEPS: ${DEEPAR_MAX_STEPS}

Batches (2 at a time):
1) point_fixed_full + point_walk_full
2) prob_fixed_full + prob_walk_full
3) prob_fixed_full_deepar + m5_point_fixed_ablation
4) prob_fixed_hurdle_none + prob_fixed_hurdle_location_scale
TXT

echo -e "name\tstatus\tstart\tend\telapsed_sec\tlog" > "${STATUS_TSV}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER_LOG}"
}

start_task() {
  local name="$1"
  shift
  local task_log="${LOG_DIR}/${name}.log"
  (
    local start_ts end_ts elapsed ec status
    start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
    local t0
    t0="$(date +%s)"
    {
      echo "[${start_ts}] START ${name}"
      echo "CMD: $*"
    } > "${task_log}"

    set +e
    "$@" >> "${task_log}" 2>&1
    ec=$?
    set -e

    if [[ "$ec" -eq 0 ]]; then
      status="OK"
    else
      status="FAIL"
    fi
    end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
    elapsed="$(( $(date +%s) - t0 ))"
    echo -e "${name}\t${status}\t${start_ts}\t${end_ts}\t${elapsed}\t${task_log}" >> "${STATUS_TSV}"
    exit "$ec"
  ) &
  log "launched ${name} pid=$!"
}

run_batch() {
  local batch_name="$1"
  log "=== ${batch_name} START ==="
  # Wait for all background tasks in this batch. Do not fail-fast; rely on status.tsv.
  wait || true
  log "=== ${batch_name} END ==="
}

log "=== batch1_point START ==="
start_task point_fixed_full \
  uv run python -m experiments.run_point \
  --dataset online_retail \
  --seed "${SEED}" \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --protocol fixed \
  --baseline-mode full \
  --out "${OUT_ROOT}/online_retail/point_fixed_full"
start_task point_walk_full \
  uv run python -m experiments.run_point \
  --dataset online_retail \
  --seed "${SEED}" \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --protocol walk_forward \
  --walk-step "${WALK_STEP}" \
  --baseline-mode full \
  --out "${OUT_ROOT}/online_retail/point_walk_full"
run_batch batch1_point

log "=== batch2_prob START ==="
start_task prob_fixed_full \
  uv run python -m experiments.run_prob \
  --seed "${SEED}" \
  --protocol fixed \
  --baseline-mode full \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --hb-bootstrap-draws 0 \
  --hb-calibration-mode none \
  --out "${OUT_ROOT}/online_retail/prob_fixed_full"
start_task prob_walk_full \
  uv run python -m experiments.run_prob \
  --seed "${SEED}" \
  --protocol walk_forward \
  --walk-step "${WALK_STEP}" \
  --baseline-mode full \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --hb-bootstrap-draws 0 \
  --hb-calibration-mode none \
  --out "${OUT_ROOT}/online_retail/prob_walk_full"
run_batch batch2_prob

log "=== batch3_deepar_m5 START ==="
start_task prob_fixed_full_deepar \
  uv run python -m experiments.run_prob \
  --seed "${SEED}" \
  --protocol fixed \
  --baseline-mode full \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --hb-bootstrap-draws 0 \
  --hb-calibration-mode none \
  --with-deepar \
  --horizon 10 \
  --input-size 14 \
  --max-steps "${DEEPAR_MAX_STEPS}" \
  --out "${OUT_ROOT}/online_retail/prob_fixed_full_deepar"
start_task m5_point_fixed_ablation \
  uv run python -m experiments.run_point \
  --dataset m5 \
  --protocol fixed \
  --seed "${SEED}" \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --m5-sample-size "${M5_SAMPLE_SIZE}" \
  --m5-hierarchy-mode ablation \
  --out "${OUT_ROOT}/m5/point_fixed_ablation"
run_batch batch3_deepar_m5

log "=== batch4_calibration_ablation START ==="
start_task prob_fixed_hurdle_none \
  uv run python -m experiments.run_prob \
  --seed "${SEED}" \
  --protocol fixed \
  --baseline-mode hurdle_only \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --hb-bootstrap-draws 0 \
  --hb-calibration-mode none \
  --out "${OUT_ROOT}/online_retail/prob_fixed_hurdle_none"
start_task prob_fixed_hurdle_location_scale \
  uv run python -m experiments.run_prob \
  --seed "${SEED}" \
  --protocol fixed \
  --baseline-mode hurdle_only \
  --hb-item-variance-mode conjugate \
  --hb-variance-prior-df 20 \
  --hb-bootstrap-draws 0 \
  --hb-calibration-mode location_scale \
  --hb-calibration-ratio 0.20 \
  --hb-calibration-samples 1000 \
  --hb-calibration-lambda-min 0.40 \
  --hb-calibration-lambda-max 1.10 \
  --hb-calibration-lambda-steps 15 \
  --hb-calibration-coverage-weight 1.00 \
  --out "${OUT_ROOT}/online_retail/prob_fixed_hurdle_location_scale"
run_batch batch4_calibration_ablation

log "All batches completed."
log "Status file: ${STATUS_TSV}"
log "Summary file: ${SUMMARY_TXT}"
