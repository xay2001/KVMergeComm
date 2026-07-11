#!/usr/bin/env bash
set -euo pipefail

# Layer aggregation ablation for receiver-aware ReKV scoring.
#
# Compares how receiver attention is aggregated across B layers:
#   identity: original ReKV, each A layer uses paired B-layer attention
#   last:     all A layers use final B-layer attention
#   mean:     all A layers use mean attention across B layers
#   top4:     all A layers use mean of 4 most concentrated B layers
#   last4:    all A layers use mean of last 4 B layers
#
# Default is pair #1 over all 8 main tasks for a mechanism appendix.
# Override env vars if needed:
#   GPU=5
#   PAIRS="1 6 7"
#   TASKS="hotpotqa musique multifieldqa_en"
#   RATIOS="0.3"
#   WINDOWS="8 16"
#   AGGS="identity last mean top4 last4"

cd /home/xay/KVComm || exit 1

GPU=${GPU:-5}
SKIP_EXISTING=${SKIP_EXISTING:-1}
LIMIT=${LIMIT:-0}
PAIRS=${PAIRS:-"1"}
TASKS=${TASKS:-"countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"}
RATIOS=${RATIOS:-"0.3"}
WINDOWS=${WINDOWS:-"8 16"}
AGGS=${AGGS:-"identity last mean top4 last4"}

ROOT=${ROOT:-snapshots/layer_aggregation_ablation}
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

run_layer_agg() {
  local pair=$1 task=$2 ratio=$3 win=$4 agg=$5
  local model_a model_b pair_root out run_name
  model_a=$(model_a_for_pair "${pair}")
  model_b=$(model_b_for_pair "${pair}")
  pair_root="${ROOT}/$(pair_name "${pair}")"
  out="${pair_root}/${task}/agg_${agg}"
  run_name="recv_${agg}_w${win}_r${ratio}"

  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] pair${pair} ${task} ${run_name} ===="
    return
  fi

  echo "==== [GPU${GPU}] pair${pair} ${task} layer_agg=${agg} w${win} r=${ratio} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${model_a}" --model_B "${model_b}" \
    --limit "${LIMIT}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --receiver_layer_agg "${agg}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
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
      for win in ${WINDOWS}; do
        for agg in ${AGGS}; do
          run_layer_agg "${pair}" "${task}" "${ratio}" "${win}" "${agg}"
        done
      done
    done
  done
}

LOG_PATH="${LOG_ROOT}/gpu${GPU}_layer_aggregation_ablation_${TAG}.log"

(
  echo "######## Layer aggregation ablation START $(date '+%F %T') ########"
  echo "GPU=${GPU}"
  echo "PAIRS=${PAIRS}"
  echo "TASKS=${TASKS}"
  echo "RATIOS=${RATIOS}"
  echo "WINDOWS=${WINDOWS}"
  echo "AGGS=${AGGS}"
  echo "LIMIT=${LIMIT}"
  echo "ROOT=${ROOT}"
  for pair in ${PAIRS}; do
    run_pair "${pair}"
  done
  echo "######## Layer aggregation ablation DONE $(date '+%F %T') ########"
) > "${LOG_PATH}" 2>&1 &

pid=$!
echo "layer aggregation ablation GPU${GPU} pid=${pid} -> ${LOG_PATH}"
echo "root -> ${ROOT}"
