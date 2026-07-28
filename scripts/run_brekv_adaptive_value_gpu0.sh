#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVMergeComm || exit 1

GPU=${GPU:-0}
PYTHON=${PYTHON:-/home/xay/miniconda3/envs/ReKV/bin/python}
ROOT=${ROOT:-snapshots/full_matched_budget_fairness_query_sketch}
OUT_DIR=${OUT_DIR:-snapshots/analysis/brekv_adaptive_value}
N_BOOTSTRAP=${N_BOOTSTRAP:-10000}
SEED=${SEED:-42}
SOLVE_THRESHOLD=${SOLVE_THRESHOLD:-1.0}

[[ -x "${PYTHON}" ]] || {
  echo "Python is not executable: ${PYTHON}" >&2
  exit 2
}
[[ -d "${ROOT}" ]] || {
  echo "Input directory does not exist: ${ROOT}" >&2
  exit 2
}

mkdir -p "${OUT_DIR}/logs"
timestamp=$(date +"%m%d_%H%M")
log_file="${OUT_DIR}/logs/brekv_adaptive_value_gpu${GPU}_${timestamp}.log"

echo "This is CPU-only offline analysis; GPU=${GPU} only preserves the requested GPU=0 entry-point convention."
echo "No model or tokenizer will be loaded."
echo "Log: ${log_file}"

{
  echo "######## B-ReKV adaptive-value analysis START $(date '+%F %T') ########"
  echo "CUDA_VISIBLE_DEVICES=${GPU} (unused; CPU-only)"
  echo "ROOT=${ROOT}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
  echo "SEED=${SEED}"
  echo "SOLVE_THRESHOLD=${SOLVE_THRESHOLD}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/analyze_brekv_adaptive_value.py \
    --root "${ROOT}" \
    --out-dir "${OUT_DIR}" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --seed "${SEED}" \
    --solve-threshold "${SOLVE_THRESHOLD}"

  echo "######## B-ReKV adaptive-value analysis DONE $(date '+%F %T') ########"
} 2>&1 | tee "${log_file}"

echo "Results: ${OUT_DIR}"
