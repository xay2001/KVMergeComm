#!/usr/bin/env bash
set -euo pipefail

# Strict adaptive B-ReKV on Table 1 pair #6.
# Uses deployable Query-Sketch scoring and a per-query strict coverage target:
#   tau(Q) = tau_min + (tau_max - tau_min) * normalized_attention_entropy(Q)
# No coverage scale and no hard budget cap are applied.

cd /home/xay/KVComm || exit 1

GPU6=${GPU6:-6}
GPU7=${GPU7:-7}
MODEL_A=${MODEL_A:-/sharedspace/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B=${MODEL_B:-/sharedspace/models/DeepSeek-R1-Distill-Llama-3B}
ROOT=${ROOT:-snapshots/table1_pair6_strict_adaptive_brekv_query_sketch}
TASKS_GPU6=${TASKS_GPU6:-"countries hotpotqa qasper tmath"}
TASKS_GPU7=${TASKS_GPU7:-"tipsheets musique multifieldqa_en twowikimqa"}
LIMIT=${LIMIT:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

has_done_run() {
  local out=$1
  local run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

run_one() {
  local gpu=$1 task=$2 win=$3 tau_min=$4 tau_max=$5
  local out="${ROOT}/${task}/strict_coverage"
  local run_name="strict_adapt_t${tau_min}-${tau_max}_w${win}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ${run_name} ===="
    return
  fi

  echo "==== [GPU${gpu}] ${task} Strict adaptive B-ReKV w${win} tau=[${tau_min},${tau_max}] $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" python com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    --merge --merge_mode evict --score_mode receiver \
    --recv_window "${win}" --merge_sink 4 --merge_recent 8 \
    --budget_mode strict_coverage \
    --coverage_tau_mode adaptive \
    --coverage_tau_min "${tau_min}" \
    --coverage_tau_max "${tau_max}" \
    --budget_min 0.0 --budget_max 1.0 --budget_floor 0.0 \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_task() {
  local gpu=$1 task=$2
  run_one "${gpu}" "${task}" 8 0.70 0.90
  run_one "${gpu}" "${task}" 8 0.80 0.95
  run_one "${gpu}" "${task}" 16 0.80 0.95
}

run_queue() {
  local gpu=$1
  shift
  for task in "$@"; do
    run_task "${gpu}" "${task}"
  done
}

read -r -a QUEUE6 <<< "${TASKS_GPU6}"
read -r -a QUEUE7 <<< "${TASKS_GPU7}"
mkdir -p "${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')
LOG6="${ROOT}/logs/gpu${GPU6}_strict_adaptive_${TAG}.log"
LOG7="${ROOT}/logs/gpu${GPU7}_strict_adaptive_${TAG}.log"

(
  echo "######## GPU${GPU6} Strict adaptive B-ReKV START $(date '+%F %T') ########"
  run_queue "${GPU6}" "${QUEUE6[@]}"
  echo "######## GPU${GPU6} Strict adaptive B-ReKV DONE $(date '+%F %T') ########"
) > "${LOG6}" 2>&1 &
PID6=$!

(
  echo "######## GPU${GPU7} Strict adaptive B-ReKV START $(date '+%F %T') ########"
  run_queue "${GPU7}" "${QUEUE7[@]}"
  echo "######## GPU${GPU7} Strict adaptive B-ReKV DONE $(date '+%F %T') ########"
) > "${LOG7}" 2>&1 &
PID7=$!

echo "GPU${GPU6} pid=${PID6}: ${TASKS_GPU6}"
echo "GPU${GPU7} pid=${PID7}: ${TASKS_GPU7}"
echo "Logs: ${LOG6} ${LOG7}"
echo "Results: ${ROOT}"
wait "${PID6}" "${PID7}"
