#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVMergeComm || exit 1

GPU=${GPU:-1}
PYTHON=${PYTHON:-/home/xay/miniconda3/envs/ReKV/bin/python}
ROOT=${ROOT:-snapshots/full_matched_budget_fairness_query_sketch}
ORACLE_R07_ROOT=${ORACLE_R07_ROOT:-snapshots/brekv_oracle_r07_v1}
SHUFFLED_ROOT=${SHUFFLED_ROOT:-snapshots/brekv_shuffled_budget_v1}
OUT_DIR=${OUT_DIR:-snapshots/analysis/brekv_heldout_shuffled}
N_BOOTSTRAP=${N_BOOTSTRAP:-10000}
SEED=${SEED:-42}
SOLVE_THRESHOLD=${SOLVE_THRESHOLD:-1.0}
BUDGET_TOLERANCE=${BUDGET_TOLERANCE:-1e-6}

[[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 2; }
for input in "${ROOT}" "${ORACLE_R07_ROOT}" "${SHUFFLED_ROOT}"; do
  [[ -d "${input}" ]] || { echo "Input directory does not exist: ${input}" >&2; exit 2; }
done

mkdir -p "${OUT_DIR}/logs"
tag=$(date +"%m%d_%H%M")
log="${OUT_DIR}/logs/brekv_heldout_shuffled_gpu${GPU}_${tag}.log"

echo "CPU-only offline analysis (CUDA_VISIBLE_DEVICES=${GPU} is unused)."
echo "Log: ${log}"
{
  echo "######## B-ReKV held-out analysis START $(date '+%F %T') ########"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    scripts/analyze_brekv_heldout_and_shuffled.py \
    --root "${ROOT}" \
    --oracle-r07-root "${ORACLE_R07_ROOT}" \
    --shuffled-root "${SHUFFLED_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --seed "${SEED}" \
    --solve-threshold "${SOLVE_THRESHOLD}" \
    --budget-tolerance "${BUDGET_TOLERANCE}"
  echo "######## B-ReKV held-out analysis DONE $(date '+%F %T') ########"
} 2>&1 | tee "${log}"

echo "Results: ${OUT_DIR}"
