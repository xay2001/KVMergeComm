#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

GPU_PAIR1=${GPU_PAIR1:-0}
GPU_PAIR6=${GPU_PAIR6:-1}
PYTHON=${PYTHON:-/home/xay/.conda/envs/kvcomm/bin/python}
ROOT=${ROOT:-snapshots/query_sketch_representation_ablation}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
WINDOWS=${WINDOWS:-"4 8 16 32"}
MODES=${MODES:-"bf16 int8 token_ids"}
RATIO=${RATIO:-0.30}
PROFILE_LIMIT=${PROFILE_LIMIT:-50}
PROFILE_WARMUP=${PROFILE_WARMUP:-3}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

has_done_run() {
  local out=$1 run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] &&
    compgen -G "${out}/${run_name}_*/cost_summary.json" > /dev/null
}

run_one() {
  local gpu=$1 pair=$2 model_a=$3 model_b=$4 task=$5 mode=$6 window=$7
  local out="${ROOT}/${pair}/${task}/${mode}"
  local run_name="rekv_${mode}_w${window}_r${RATIO}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${pair} ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] ${pair} ${task} ${run_name} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test \
    --profile_cost --profile_limit "${PROFILE_LIMIT}" --profile_warmup "${PROFILE_WARMUP}" \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver \
    --query_sketch_mode "${mode}" --recv_window "${window}" \
    --merge_ratio "${RATIO}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_pair() {
  local gpu=$1 pair=$2 model_a=$3 model_b=$4
  for task in ${TASKS}; do
    for mode in ${MODES}; do
      for window in ${WINDOWS}; do
        run_one "${gpu}" "${pair}" "${model_a}" "${model_b}" "${task}" "${mode}" "${window}"
      done
    done
  done
}

mkdir -p "${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')
LOG0="${ROOT}/logs/gpu${GPU_PAIR1}_pair1_${TAG}.log"
LOG1="${ROOT}/logs/gpu${GPU_PAIR6}_pair6_${TAG}.log"

(
  run_pair "${GPU_PAIR1}" pair1_llama31_same \
    /sharedspace/models/Llama-3.1-8B-Instruct \
    /sharedspace/models/Llama-3.1-8B-Instruct
) > "${LOG0}" 2>&1 &
PID0=$!

(
  run_pair "${GPU_PAIR6}" pair6_llama32_abliterated_deepseek3b \
    /sharedspace/models/Llama-3.2-3B-Instruct-abliterated \
    /sharedspace/models/DeepSeek-R1-Distill-Llama-3B
) > "${LOG1}" 2>&1 &
PID1=$!

echo "GPU${GPU_PAIR1} pid=${PID0}: ${LOG0}"
echo "GPU${GPU_PAIR6} pid=${PID1}: ${LOG1}"
wait "${PID0}" "${PID1}"
