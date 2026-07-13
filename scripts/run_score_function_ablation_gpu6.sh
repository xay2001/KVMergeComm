#!/usr/bin/env bash
set -euo pipefail

# Score-function ablation for ReKV token selection.
#
# Compares:
#   - receiver attention (ReKV)
#   - receiver attention x value norm
#   - receiver attention + recency prior
#   - value norm / Evict
#   - random-token
#
# Default is intentionally broad but still focused on the reviewer-relevant tasks.
# Override env vars if needed:
#   GPU=6
#   PAIRS="1 6 7"
#   TASKS="hotpotqa musique multifieldqa_en"
#   RATIOS="0.3 0.5"
#   WINDOWS="8 16"
#   LIMIT=0

cd /home/xay/KVComm || exit 1

GPU=${GPU:-6}
SKIP_EXISTING=${SKIP_EXISTING:-1}
LIMIT=${LIMIT:-0}
PAIRS=${PAIRS:-"1 6 7"}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
RATIOS=${RATIOS:-"0.3 0.5"}
WINDOWS=${WINDOWS:-"8 16"}
FOREGROUND=${FOREGROUND:-0}

ROOT=${ROOT:-snapshots/score_function_ablation}
LOG_ROOT="${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')

MODEL_A_1=${MODEL_A_1:-/sharedspace/models/Llama-3.1-8B-Instruct}
MODEL_B_1=${MODEL_B_1:-/sharedspace/models/Llama-3.1-8B-Instruct}

MODEL_A_6=${MODEL_A_6:-/sharedspace/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B_6=${MODEL_B_6:-/sharedspace/models/DeepSeek-R1-Distill-Llama-3B}

MODEL_A_7=${MODEL_A_7:-/sharedspace/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B_7=${MODEL_B_7:-/sharedspace/models/Bespoke-Stratos-7B}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${LOG_ROOT}"

model_a_for_pair() {
  case "$1" in
    1) echo "${MODEL_A_1}" ;;
    6) echo "${MODEL_A_6}" ;;
    7) echo "${MODEL_A_7}" ;;
    *) echo "unknown pair=$1" >&2; exit 1 ;;
  esac
}

model_b_for_pair() {
  case "$1" in
    1) echo "${MODEL_B_1}" ;;
    6) echo "${MODEL_B_6}" ;;
    7) echo "${MODEL_B_7}" ;;
    *) echo "unknown pair=$1" >&2; exit 1 ;;
  esac
}

pair_name() {
  case "$1" in
    1) echo "pair1_llama31_same" ;;
    6) echo "pair6_llama32_abliterated_deepseek3b" ;;
    7) echo "pair7_qwen25_uncensored_bespoke" ;;
    *) echo "pair$1" ;;
  esac
}

check_model() {
  local path=$1
  if [[ ! -d "${path}" ]]; then
    echo "Model path does not exist: ${path}" >&2
    exit 1
  fi
}

has_done_run() {
  local out=$1
  local run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

common_args() {
  local model_a=$1 model_b=$2 task=$3 out=$4 run_name=$5
  shift 5
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${model_a}" --model_B "${model_b}" \
    --limit "${LIMIT}" \
    --snapshot_path "${out}" \
    --run_name "${run_name}" "$@"
}

run_score_mode() {
  local pair=$1 task=$2 ratio=$3 win=$4 score_mode=$5 method_dir=$6 run_prefix=$7
  local model_a model_b pair_root out run_name
  model_a=$(model_a_for_pair "${pair}")
  model_b=$(model_b_for_pair "${pair}")
  pair_root="${ROOT}/$(pair_name "${pair}")"
  out="${pair_root}/${task}/${method_dir}"
  run_name="${run_prefix}_w${win}_r${ratio}"

  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] pair${pair} ${task} ${run_name} ===="
    return
  fi

  echo "==== [GPU${GPU}] pair${pair} ${task} ${score_mode} w${win} r=${ratio} $(date '+%F %T') ===="
  common_args "${model_a}" "${model_b}" "${task}" "${out}" "${run_name}" \
    --merge --merge_mode evict --score_mode "${score_mode}" \
    --query_sketch_mode bf16 --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8
}

run_query_agnostic() {
  local pair=$1 task=$2 ratio=$3 score_mode=$4 method_dir=$5 run_prefix=$6
  local model_a model_b pair_root out run_name
  model_a=$(model_a_for_pair "${pair}")
  model_b=$(model_b_for_pair "${pair}")
  pair_root="${ROOT}/$(pair_name "${pair}")"
  out="${pair_root}/${task}/${method_dir}"
  run_name="${run_prefix}_r${ratio}"

  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] pair${pair} ${task} ${run_name} ===="
    return
  fi

  echo "==== [GPU${GPU}] pair${pair} ${task} ${score_mode} r=${ratio} $(date '+%F %T') ===="
  common_args "${model_a}" "${model_b}" "${task}" "${out}" "${run_name}" \
    --merge --merge_mode evict --score_mode "${score_mode}" --recv_window 0 \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8
}

run_pair() {
  local pair=$1
  local model_a model_b
  model_a=$(model_a_for_pair "${pair}")
  model_b=$(model_b_for_pair "${pair}")
  check_model "${model_a}"
  check_model "${model_b}"

  for task in ${TASKS}; do
    for ratio in ${RATIOS}; do
      run_query_agnostic "${pair}" "${task}" "${ratio}" value_norm mtc_value_norm value_norm
      run_query_agnostic "${pair}" "${task}" "${ratio}" random mtc_random random
      for win in ${WINDOWS}; do
        run_score_mode "${pair}" "${task}" "${ratio}" "${win}" receiver mtc_receiver receiver
        run_score_mode "${pair}" "${task}" "${ratio}" "${win}" receiver_x_value_norm mtc_receiver_x_value_norm receiver_x_value_norm
        run_score_mode "${pair}" "${task}" "${ratio}" "${win}" receiver_recency mtc_receiver_recency receiver_recency
      done
    done
  done
}

LOG_PATH="${LOG_ROOT}/gpu${GPU}_score_function_ablation_${TAG}.log"

run_all() {
  echo "######## Score function ablation START $(date '+%F %T') ########"
  echo "GPU=${GPU}"
  echo "PAIRS=${PAIRS}"
  echo "TASKS=${TASKS}"
  echo "RATIOS=${RATIOS}"
  echo "WINDOWS=${WINDOWS}"
  echo "LIMIT=${LIMIT}"
  echo "ROOT=${ROOT}"
  for pair in ${PAIRS}; do
    run_pair "${pair}"
  done
  echo "######## Score function ablation DONE $(date '+%F %T') ########"
}

if [[ "${FOREGROUND}" == "1" ]]; then
  run_all > "${LOG_PATH}" 2>&1
  echo "score function ablation GPU${GPU} done -> ${LOG_PATH}"
else
  run_all > "${LOG_PATH}" 2>&1 &
  pid=$!
  echo "score function ablation GPU${GPU} pid=${pid} -> ${LOG_PATH}"
fi
echo "root -> ${ROOT}"
