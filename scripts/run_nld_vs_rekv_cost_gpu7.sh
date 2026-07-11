#!/usr/bin/env bash
set -euo pipefail

# Natural-language passing (NLD) cost profile for comparison with ReKV/B-ReKV.
# ReKV/B-ReKV cost summaries already exist under snapshots/cost_profile/.
# This script adds the missing NLD side with the same score/time/memory style.
#
# Default scope is the paper-friendly trio:
#   hotpotqa, musique, multifieldqa_en
#
# Override examples:
#   GPU=7 PAIRS="1 6 7" TASKS="hotpotqa musique" LIMIT=500 \
#     bash scripts/run_nld_vs_rekv_cost_gpu7.sh

cd /home/xay/KVComm || exit 1

GPU=${GPU:-7}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
PAIRS=${PAIRS:-"1 6 7"}
LIMIT=${LIMIT:-500}
WARMUP=${WARMUP:-5}
PHASE1_TOKENS=${PHASE1_TOKENS:-128}

timestamp=$(date +"%m%d_%H%M")
LOG_DIR="snapshots/nld_cost_profile/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/gpu${GPU}_nld_cost_${timestamp}.log"

pair_paths() {
  case "$1" in
    1)
      echo "pair1_llama31_same /sharedspace/models/Llama-3.1-8B-Instruct /sharedspace/models/Llama-3.1-8B-Instruct"
      ;;
    6)
      echo "pair6_llama32_abliterated_deepseek3b /sharedspace/models/Llama-3.2-3B-Instruct-abliterated /sharedspace/models/DeepSeek-R1-Distill-Llama-3B"
      ;;
    7)
      echo "pair7_qwen25_uncensored_bespoke /sharedspace/models/Qwen2.5-7B-Instruct-Uncensored /sharedspace/models/Bespoke-Stratos-7B"
      ;;
    *)
      echo "Unknown pair id: $1" >&2
      return 1
      ;;
  esac
}

run_nld_cost() {
  local pair_slug=$1
  local model_a=$2
  local model_b=$3
  local task=$4
  local root="snapshots/nld_cost_profile/${pair_slug}/${task}/nld"

  echo "==== [GPU${GPU}] ${pair_slug} ${task} NLD cost limit=${LIMIT} ${timestamp} ===="
  CUDA_VISIBLE_DEVICES="${GPU}" python com.py \
    --test_task "${task}" \
    --do_test_nld \
    --profile_cost \
    --profile_limit "${LIMIT}" \
    --profile_warmup "${WARMUP}" \
    --nld_max_tokens_model_A_and_B_phase1 "${PHASE1_TOKENS}" \
    --model_A "${model_a}" \
    --model_B "${model_b}" \
    --snapshot_path "${root}" \
    --run_name "nld_cost"
}

{
  echo "######## NLD cost profile START $(date '+%F %T') ########"
  echo "GPU=${GPU}"
  echo "PAIRS=${PAIRS}"
  echo "TASKS=${TASKS}"
  echo "LIMIT=${LIMIT}"
  echo "WARMUP=${WARMUP}"
  echo "PHASE1_TOKENS=${PHASE1_TOKENS}"

  for pair in ${PAIRS}; do
    read -r pair_slug model_a model_b <<< "$(pair_paths "${pair}")"
    for task in ${TASKS}; do
      run_nld_cost "${pair_slug}" "${model_a}" "${model_b}" "${task}"
    done
  done

  echo "######## NLD cost profile DONE $(date '+%F %T') ########"
} 2>&1 | tee "${LOG_FILE}"
