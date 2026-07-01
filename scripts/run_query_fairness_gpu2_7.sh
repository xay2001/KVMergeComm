#!/usr/bin/env bash
set -euo pipefail

# Query-aware setting fairness ablations on GPU 2 and GPU 7.
#
# Runs, per dataset and ratio:
#   - Evict / ValueNorm: token-level, query-agnostic
#   - Random-token: token-level random baseline
#   - ReKV: receiver-aware with recv_window={4,8,16,32,all}
#
# Default tasks:
#   GPU2: hotpotqa
#   GPU7: musique multifieldqa_en
#
# Run:
#   bash scripts/run_query_fairness_gpu2_7.sh
#
# Optional:
#   RATIOS="0.3 0.5" SKIP_EXISTING=1 bash scripts/run_query_fairness_gpu2_7.sh

cd /home/xay/KVComm || exit 1

MODEL_A=${MODEL_A:-/sharedspace/models/Llama-3.1-8B-Instruct}
MODEL_B=${MODEL_B:-/sharedspace/models/Llama-3.1-8B-Instruct}
ROOT=${ROOT:-snapshots/query_fairness/pair1_llama31_same}
RATIOS=${RATIOS:-"0.3"}
WINDOWS=${WINDOWS:-"4 8 16 32 0"}
GPU2=${GPU2:-2}
GPU7=${GPU7:-7}
TASKS_GPU2=${TASKS_GPU2:-"hotpotqa"}
TASKS_GPU7=${TASKS_GPU7:-"musique multifieldqa_en"}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

run_one() {
  local task=$1
  local gpu=$2
  local method=$3
  local ratio=$4
  local win=${5:-0}

  local out run_name extra_args
  extra_args=()
  case "${method}" in
    evict)
      out="${ROOT}/${task}/mtc_evict"
      run_name="evict_r${ratio}"
      extra_args=(--merge --merge_mode evict --score_mode value_norm --recv_window 0)
      ;;
    random)
      out="${ROOT}/${task}/mtc_random"
      run_name="random_r${ratio}"
      extra_args=(--merge --merge_mode evict --score_mode random --recv_window 0)
      ;;
    rekv)
      out="${ROOT}/${task}/mtc_receiver"
      run_name="recv_w${win}_r${ratio}"
      extra_args=(--merge --merge_mode evict --score_mode receiver --recv_window "${win}")
      ;;
    *)
      echo "unknown method=${method}" >&2
      exit 1
      ;;
  esac

  if [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null; then
    echo "==== [skip] ${task} ${method} ${run_name} already exists ===="
    return
  fi

  echo "==== [query fairness GPU${gpu}] task=${task} method=${method} ratio=${ratio} win=${win} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${gpu} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    "${extra_args[@]}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
}

run_task() {
  local task=$1
  local gpu=$2
  for ratio in ${RATIOS}; do
    run_one "${task}" "${gpu}" evict "${ratio}"
    run_one "${task}" "${gpu}" random "${ratio}"
    for win in ${WINDOWS}; do
      run_one "${task}" "${gpu}" rekv "${ratio}" "${win}"
    done
  done
}

run_queue() {
  local gpu=$1
  shift
  local tasks=("$@")
  for task in "${tasks[@]}"; do
    run_task "${task}" "${gpu}"
  done
}

read -r -a TASKS2 <<< "${TASKS_GPU2}"
read -r -a TASKS7 <<< "${TASKS_GPU7}"

mkdir -p "${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')

(
  echo "######## query fairness GPU${GPU2} START $(date '+%F %T') ########"
  run_queue "${GPU2}" "${TASKS2[@]}"
  echo "######## query fairness GPU${GPU2} DONE $(date '+%F %T') ########"
) > "${ROOT}/logs/gpu${GPU2}_query_fairness_${TAG}.log" 2>&1 &
P2=$!

(
  echo "######## query fairness GPU${GPU7} START $(date '+%F %T') ########"
  run_queue "${GPU7}" "${TASKS7[@]}"
  echo "######## query fairness GPU${GPU7} DONE $(date '+%F %T') ########"
) > "${ROOT}/logs/gpu${GPU7}_query_fairness_${TAG}.log" 2>&1 &
P7=$!

echo "GPU${GPU2} query fairness pid=${P2} -> ${ROOT}/logs/gpu${GPU2}_query_fairness_${TAG}.log"
echo "GPU${GPU7} query fairness pid=${P7} -> ${ROOT}/logs/gpu${GPU7}_query_fairness_${TAG}.log"
echo "Results root -> ${ROOT}"
