#!/usr/bin/env bash
set -euo pipefail

# Re-run the complete Table 1 pair #6 ReKV/B-ReKV matrix with the deployable
# Query-Sketch protocol:
#   B query-only Q sketch -> A-local token scoring -> selected KV -> B.
#
# Pair:
#   S: uihui-ai/Llama-3.2-3B-Instruct-abliterated
#   R: suayptalha/DeepSeek-R1-Distill-Llama-3B
#
# Matrix per task (9 runs):
#   ReKV w8/w16 x r={0.3,0.5,0.7}
#   B-ReKV t0.95_s0.75_w8, t0.95_s0.85_w8, t0.95_s0.90_w16

cd /home/xay/KVComm || exit 1

GPU6=${GPU6:-6}
GPU7=${GPU7:-7}
MODEL_A=${MODEL_A:-/sharedspace/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B=${MODEL_B:-/sharedspace/models/DeepSeek-R1-Distill-Llama-3B}
ROOT=${ROOT:-snapshots/table1_pair6_query_sketch_llama32_abliterated_deepseek3b}
TASKS_GPU6=${TASKS_GPU6:-"countries hotpotqa qasper tmath"}
TASKS_GPU7=${TASKS_GPU7:-"tipsheets musique multifieldqa_en twowikimqa"}
RATIOS=${RATIOS:-"0.3 0.5 0.7"}
LIMIT=${LIMIT:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

has_done_run() {
  local out=$1
  local run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

common_eval() {
  local gpu=$1 task=$2 out=$3 run_name=$4
  shift 4
  CUDA_VISIBLE_DEVICES="${gpu}" python com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    --snapshot_path "${out}" --run_name "${run_name}" \
    "$@"
}

run_rekv() {
  local gpu=$1 task=$2 win=$3 ratio=$4
  local out="${ROOT}/${task}/mtc_receiver"
  local run_name="recv_w${win}_r${ratio}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] ${task} Query-Sketch ReKV w${win} r=${ratio} $(date '+%F %T') ===="
  common_eval "${gpu}" "${task}" "${out}" "${run_name}" \
    --merge --merge_mode evict --score_mode receiver \
    --recv_window "${win}" --merge_ratio "${ratio}" \
    --merge_sink 4 --merge_recent 8
}

run_brekv() {
  local gpu=$1 task=$2 win=$3 tau=$4 scale=$5
  local out="${ROOT}/${task}/coverage"
  local run_name="cov_t${tau}_s${scale}_w${win}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] ${task} Query-Sketch B-ReKV w${win} tau=${tau} scale=${scale} $(date '+%F %T') ===="
  common_eval "${gpu}" "${task}" "${out}" "${run_name}" \
    --merge --merge_mode evict --score_mode receiver \
    --recv_window "${win}" --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage --budget_min 0.05 --budget_max 0.7 \
    --coverage_tau "${tau}" --coverage_scale "${scale}"
}

run_task() {
  local gpu=$1 task=$2
  for win in 8 16; do
    for ratio in ${RATIOS}; do
      run_rekv "${gpu}" "${task}" "${win}" "${ratio}"
    done
  done
  run_brekv "${gpu}" "${task}" 8 0.95 0.75
  run_brekv "${gpu}" "${task}" 8 0.95 0.85
  run_brekv "${gpu}" "${task}" 16 0.95 0.90
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
LOG6="${ROOT}/logs/gpu${GPU6}_table1_pair6_query_sketch_${TAG}.log"
LOG7="${ROOT}/logs/gpu${GPU7}_table1_pair6_query_sketch_${TAG}.log"

(
  echo "######## GPU${GPU6} Table 1 pair #6 Query-Sketch START $(date '+%F %T') ########"
  run_queue "${GPU6}" "${QUEUE6[@]}"
  echo "######## GPU${GPU6} Table 1 pair #6 Query-Sketch DONE $(date '+%F %T') ########"
) > "${LOG6}" 2>&1 &
PID6=$!

(
  echo "######## GPU${GPU7} Table 1 pair #6 Query-Sketch START $(date '+%F %T') ########"
  run_queue "${GPU7}" "${QUEUE7[@]}"
  echo "######## GPU${GPU7} Table 1 pair #6 Query-Sketch DONE $(date '+%F %T') ########"
) > "${LOG7}" 2>&1 &
PID7=$!

echo "GPU${GPU6} pid=${PID6}: ${TASKS_GPU6}"
echo "GPU${GPU7} pid=${PID7}: ${TASKS_GPU7}"
echo "Logs: ${LOG6} ${LOG7}"
echo "Results: ${ROOT}"
wait "${PID6}" "${PID7}"
