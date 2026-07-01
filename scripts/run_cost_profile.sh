#!/usr/bin/env bash
set -euo pipefail

# Controlled cost profiling for the paper efficiency table.
#
# Default smoke/first-pass setup:
#   GPU=2, pair #1 Llama-3.1-8B same-model, multifieldqa_en + musique.
#
# Copy/run examples:
#   GPU=2 LIMIT=3 WARMUP=1 TASKS="multifieldqa_en" bash scripts/run_cost_profile.sh
#   GPU=2 LIMIT=50 WARMUP=5 TASKS="multifieldqa_en musique" bash scripts/run_cost_profile.sh
#   GPU=2 LIMIT=0 WARMUP=5 RATIOS="0.3 0.5 0.7" TASKS="..." bash scripts/run_cost_profile.sh

cd /home/xay/KVComm || exit 1

GPU=${GPU:-2}
MODEL_A=${MODEL_A:-/sharedspace/models/Llama-3.1-8B-Instruct}
MODEL_B=${MODEL_B:-/sharedspace/models/Llama-3.1-8B-Instruct}
TASKS=${TASKS:-"multifieldqa_en musique"}
LIMIT=${LIMIT:-50}
WARMUP=${WARMUP:-5}
RATIOS=${RATIOS:-"0.3"}
ROOT=${ROOT:-snapshots/cost_profile/pair1_llama31_same}
MAX_BUDGET=${MAX_BUDGET:-0.7}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

common() {
  local task=$1
  local out=$2
  shift 2
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    --profile_cost --profile_limit "${LIMIT}" --profile_warmup "${WARMUP}" \
    --snapshot_path "${out}" "$@"
}

run_kvcomm() {
  local task=$1
  local top=$2
  common "${task}" "${ROOT}/${task}/kvcomm" \
    --top_layers "${top}" \
    --run_name "kvcomm_top${top}_cost"
}

run_rasc() {
  local task=$1
  local win=$2
  local ratio=$3
  common "${task}" "${ROOT}/${task}/mtc_receiver" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 \
    --run_name "recv_w${win}_r${ratio}_cost"
}

run_coverage() {
  local task=$1
  local win=$2
  local tau=$3
  local scale=$4
  common "${task}" "${ROOT}/${task}/coverage" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage --budget_min 0.05 --budget_max "${MAX_BUDGET}" \
    --coverage_tau "${tau}" --coverage_scale "${scale}" \
    --run_name "cov_t${tau}_s${scale}_w${win}_cost"
}

for task in ${TASKS}; do
  for ratio in ${RATIOS}; do
    echo "==== [cost GPU${GPU}] ${task} KVComm top=${ratio} ===="
    run_kvcomm "${task}" "${ratio}"

    echo "==== [cost GPU${GPU}] ${task} RASC w8 r=${ratio} ===="
    run_rasc "${task}" 8 "${ratio}"

    echo "==== [cost GPU${GPU}] ${task} RASC w16 r=${ratio} ===="
    run_rasc "${task}" 16 "${ratio}"
  done

  echo "==== [cost GPU${GPU}] ${task} Coverage w8 tau=0.95 scale=0.75 ===="
  run_coverage "${task}" 8 0.95 0.75

  echo "==== [cost GPU${GPU}] ${task} Coverage w8 tau=0.95 scale=0.85 ===="
  run_coverage "${task}" 8 0.95 0.85
done

echo "==== cost profile runs done -> ${ROOT} ===="
echo "Analyze:"
echo "python scripts/analyze_cost_profile.py --root ${ROOT} --csv ${ROOT}/cost_table.csv"
