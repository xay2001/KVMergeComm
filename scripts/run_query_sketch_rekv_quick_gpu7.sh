#!/usr/bin/env bash
set -euo pipefail

# Quick deployable-protocol check:
#   receiver        = B query-only Q sketch -> A-local QK scoring
#   receiver_oracle = B attends over full A KV (upper bound)
#
# Cost-profile mode reports accuracy, B->A sketch bytes, A->B selected-KV
# bytes, total communication bytes, latency, and peak memory.

cd /home/xay/KVComm || exit 1

GPU=${GPU:-7}
MODEL_A=${MODEL_A:-/sharedspace/models/Llama-3.1-8B-Instruct}
MODEL_B=${MODEL_B:-/sharedspace/models/Llama-3.1-8B-Instruct}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
LIMIT=${LIMIT:-50}
WARMUP=${WARMUP:-3}
WINDOW=${WINDOW:-8}
RATIO=${RATIO:-0.3}
ROOT=${ROOT:-snapshots/query_sketch_rekv_quick/pair1_llama31_same}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

run_one() {
  local task=$1
  local score_mode=$2
  local label=$3
  local out="${ROOT}/${task}/${label}"

  echo "==== [GPU${GPU}] ${task} ${label} w${WINDOW} r=${RATIO} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${GPU}" python com.py \
    --test_task "${task}" --do_test \
    --profile_cost --profile_limit "${LIMIT}" --profile_warmup "${WARMUP}" \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    --merge --merge_mode evict \
    --score_mode "${score_mode}" --recv_window "${WINDOW}" \
    --merge_ratio "${RATIO}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" \
    --run_name "${label}_w${WINDOW}_r${RATIO}"
}

mkdir -p "${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')
LOG="${ROOT}/logs/gpu${GPU}_query_sketch_quick_${TAG}.log"

{
  echo "######## Query-Sketch ReKV quick test START $(date '+%F %T') ########"
  echo "TASKS=${TASKS} LIMIT=${LIMIT} WARMUP=${WARMUP} WINDOW=${WINDOW} RATIO=${RATIO}"
  for task in ${TASKS}; do
    run_one "${task}" receiver query_sketch_rekv
    run_one "${task}" receiver_oracle full_kv_oracle_rekv
  done
  echo "######## Query-Sketch ReKV quick test DONE $(date '+%F %T') ########"
} 2>&1 | tee "${LOG}"

python scripts/summarize_query_sketch_quick.py

echo "Log: ${LOG}"
echo "Results: ${ROOT}"
