#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVMergeComm || exit 1

GPU=${GPU:-0}
PYTHON=${PYTHON:-/home/xay/miniconda3/envs/ReKV/bin/python}
FAIRNESS_ROOT=${FAIRNESS_ROOT:-snapshots/full_matched_budget_fairness_query_sketch}
KV_COST_ROOT=${KV_COST_ROOT:-snapshots/query_sketch_cost_v1}
NLD_COST_ROOT=${NLD_COST_ROOT:-snapshots/nld_receiver_aware_cost_v1}
OUT_DIR=${OUT_DIR:-snapshots/analysis/communication_claims}
N_BOOTSTRAP=${N_BOOTSTRAP:-10000}
SEED=${SEED:-42}
BANDWIDTH_GBPS=${BANDWIDTH_GBPS:-"1 10 25 100"}
RTT_MS=${RTT_MS:-"0 1 10 50"}

[[ -x "${PYTHON}" ]] || {
  echo "Python is not executable: ${PYTHON}" >&2
  exit 2
}
for input_root in "${FAIRNESS_ROOT}" "${KV_COST_ROOT}" "${NLD_COST_ROOT}"; do
  [[ -d "${input_root}" ]] || {
    echo "Input directory does not exist: ${input_root}" >&2
    exit 2
  }
done

mkdir -p "${OUT_DIR}/logs"
timestamp=$(date +"%m%d_%H%M")
log_file="${OUT_DIR}/logs/communication_claims_gpu${GPU}_${timestamp}.log"

echo "F3b/M1/M2 is offline CPU analysis; CUDA_VISIBLE_DEVICES=${GPU} is retained for the requested entry point."
echo "Log: ${log_file}"

read -r -a bandwidth_args <<< "${BANDWIDTH_GBPS}"
read -r -a rtt_args <<< "${RTT_MS}"

{
  echo "######## Communication claims analysis START $(date '+%F %T') ########"
  echo "CUDA_VISIBLE_DEVICES=${GPU}"
  echo "FAIRNESS_ROOT=${FAIRNESS_ROOT}"
  echo "KV_COST_ROOT=${KV_COST_ROOT}"
  echo "NLD_COST_ROOT=${NLD_COST_ROOT}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
  echo "SEED=${SEED}"
  echo "BANDWIDTH_GBPS=${BANDWIDTH_GBPS}"
  echo "RTT_MS=${RTT_MS}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/analyze_communication_claims.py \
    --fairness-root "${FAIRNESS_ROOT}" \
    --kv-cost-root "${KV_COST_ROOT}" \
    --nld-cost-root "${NLD_COST_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --seed "${SEED}" \
    --bandwidth-gbps "${bandwidth_args[@]}" \
    --rtt-ms "${rtt_args[@]}"

  echo "######## Communication claims analysis DONE $(date '+%F %T') ########"
} 2>&1 | tee "${log_file}"

echo "Results: ${OUT_DIR}"
