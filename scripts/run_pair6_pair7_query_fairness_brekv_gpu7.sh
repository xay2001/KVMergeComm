#!/usr/bin/env bash
set -euo pipefail

# Query fairness extension for Table 1 pair #6/#7 on GPU 7.
#
# Tasks:
#   hotpotqa, musique, multifieldqa_en
#
# Methods:
#   - Evict / ValueNorm, r=0.3
#   - Random-token, r=0.3
#   - ReKV, r=0.3, recv_window={8,16}
#   - B-ReKV canonical points:
#       cov_t0.95_s0.75_w8
#       cov_t0.95_s0.85_w8
#       cov_t0.95_s0.90_w16
#
# Run:
#   bash scripts/run_pair6_pair7_query_fairness_brekv_gpu7.sh

cd /home/xay/KVComm || exit 1

GPU=${GPU:-7}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
RATIO=${RATIO:-0.3}
WINDOWS=${WINDOWS:-"8 16"}
SKIP_EXISTING=${SKIP_EXISTING:-1}

PAIR_IDS=${PAIR_IDS:-"6 7"}

MODEL_A_6=${MODEL_A_6:-/sharedspace/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B_6=${MODEL_B_6:-/sharedspace/models/DeepSeek-R1-Distill-Llama-3B}
ROOT_6=${ROOT_6:-snapshots/query_fairness/table1_pair6_llama32_abliterated_deepseek3b}

MODEL_A_7=${MODEL_A_7:-/sharedspace/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B_7=${MODEL_B_7:-/sharedspace/models/Bespoke-Stratos-7B}
ROOT_7=${ROOT_7:-snapshots/query_fairness/table1_pair7_qwen25_uncensored_bespoke}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

model_a_for_pair() {
  local pair=$1
  case "${pair}" in
    6) echo "${MODEL_A_6}" ;;
    7) echo "${MODEL_A_7}" ;;
    *) echo "unknown pair=${pair}" >&2; exit 1 ;;
  esac
}

model_b_for_pair() {
  local pair=$1
  case "${pair}" in
    6) echo "${MODEL_B_6}" ;;
    7) echo "${MODEL_B_7}" ;;
    *) echo "unknown pair=${pair}" >&2; exit 1 ;;
  esac
}

root_for_pair() {
  local pair=$1
  case "${pair}" in
    6) echo "${ROOT_6}" ;;
    7) echo "${ROOT_7}" ;;
    *) echo "unknown pair=${pair}" >&2; exit 1 ;;
  esac
}

has_done_run() {
  local out=$1
  local run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

check_models() {
  local model_a=$1
  local model_b=$2
  if [[ ! -d "${model_a}" ]]; then
    echo "Sender model path does not exist: ${model_a}" >&2
    exit 1
  fi
  if [[ ! -d "${model_b}" ]]; then
    echo "Receiver model path does not exist: ${model_b}" >&2
    exit 1
  fi
}

run_evict() {
  local root=$1 model_a=$2 model_b=$3 task=$4
  local out="${root}/${task}/mtc_evict"
  local run_name="evict_r${RATIO}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} Evict r=${RATIO} ===="
    return
  fi
  echo "==== [GPU${GPU}] ${task} Evict/ValueNorm r=${RATIO} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode value_norm --recv_window 0 \
    --merge_ratio "${RATIO}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
}

run_random() {
  local root=$1 model_a=$2 model_b=$3 task=$4
  local out="${root}/${task}/mtc_random"
  local run_name="random_r${RATIO}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} Random-token r=${RATIO} ===="
    return
  fi
  echo "==== [GPU${GPU}] ${task} Random-token r=${RATIO} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode random --recv_window 0 \
    --merge_ratio "${RATIO}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
}

run_rekv() {
  local root=$1 model_a=$2 model_b=$3 task=$4 win=$5
  local out="${root}/${task}/mtc_receiver"
  local run_name="recv_w${win}_r${RATIO}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ReKV w${win} r=${RATIO} ===="
    return
  fi
  echo "==== [GPU${GPU}] ${task} ReKV w${win} r=${RATIO} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${RATIO}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
}

run_brekv() {
  local root=$1 model_a=$2 model_b=$3 task=$4 win=$5 tau=$6 scale=$7
  local out="${root}/${task}/coverage"
  local run_name="cov_t${tau}_s${scale}_w${win}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} B-ReKV w${win} tau=${tau} scale=${scale} ===="
    return
  fi
  echo "==== [GPU${GPU}] ${task} B-ReKV w${win} tau=${tau} scale=${scale} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage \
    --budget_min 0.05 --budget_max 0.7 \
    --coverage_tau "${tau}" --coverage_scale "${scale}" \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
}

run_pair() {
  local pair=$1
  local model_a model_b root log_dir
  model_a=$(model_a_for_pair "${pair}")
  model_b=$(model_b_for_pair "${pair}")
  root=$(root_for_pair "${pair}")
  check_models "${model_a}" "${model_b}"
  mkdir -p "${root}/logs"

  echo "######## pair${pair} query fairness + B-ReKV START $(date '+%F %T') ########"
  echo "MODEL_A=${model_a}"
  echo "MODEL_B=${model_b}"
  echo "ROOT=${root}"
  echo "TASKS=${TASKS}"

  for task in ${TASKS}; do
    run_evict "${root}" "${model_a}" "${model_b}" "${task}"
    run_random "${root}" "${model_a}" "${model_b}" "${task}"
    for win in ${WINDOWS}; do
      run_rekv "${root}" "${model_a}" "${model_b}" "${task}" "${win}"
    done
    run_brekv "${root}" "${model_a}" "${model_b}" "${task}" 8 0.95 0.75
    run_brekv "${root}" "${model_a}" "${model_b}" "${task}" 8 0.95 0.85
    run_brekv "${root}" "${model_a}" "${model_b}" "${task}" 16 0.95 0.90
  done

  echo "######## pair${pair} query fairness + B-ReKV DONE $(date '+%F %T') ########"
}

TAG=$(date '+%m%d_%H%M')
LOG_ROOT="snapshots/query_fairness/logs"
mkdir -p "${LOG_ROOT}"
LOG_PATH="${LOG_ROOT}/gpu${GPU}_pair6_pair7_query_fairness_brekv_${TAG}.log"

(
  echo "######## GPU${GPU} pair6/pair7 query fairness + B-ReKV START $(date '+%F %T') ########"
  echo "PAIR_IDS=${PAIR_IDS}"
  for pair in ${PAIR_IDS}; do
    run_pair "${pair}"
  done
  echo "######## GPU${GPU} pair6/pair7 query fairness + B-ReKV DONE $(date '+%F %T') ########"
) > "${LOG_PATH}" 2>&1 &

pid=$!
echo "pair6/pair7 query fairness + B-ReKV GPU${GPU} pid=${pid} -> ${LOG_PATH}"
echo "Results roots -> ${ROOT_6} and ${ROOT_7}"
