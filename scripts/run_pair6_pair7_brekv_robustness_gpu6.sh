#!/usr/bin/env bash
set -euo pipefail

# B-ReKV small robustness grid for Table 1 pair #6/#7 on GPU 6.
#
# Goal:
#   Show B-ReKV is robust on heterogeneous model pairs, not a cherry-picked
#   canonical point.
#
# Default grid:
#   pairs: 6, 7
#   tasks: hotpotqa, musique
#   window: 8, 16
#   tau: 0.90, 0.95, 0.98
#   scale: 0.65, 0.75, 0.85
#
# Existing canonical runs are skipped if their per_sample.jsonl already exists.
#
# Run:
#   bash scripts/run_pair6_pair7_brekv_robustness_gpu6.sh

cd /home/xay/KVComm || exit 1

GPU=${GPU:-6}
PAIR_IDS=${PAIR_IDS:-"6 7"}
TASKS=${TASKS:-"hotpotqa musique"}
WINDOWS=${WINDOWS:-"8 16"}
TAUS=${TAUS:-"0.90 0.95 0.98"}
SCALES=${SCALES:-"0.65 0.75 0.85"}
MIN_BUDGET=${MIN_BUDGET:-0.05}
MAX_BUDGET=${MAX_BUDGET:-0.7}
SKIP_EXISTING=${SKIP_EXISTING:-1}

MODEL_A_6=${MODEL_A_6:-/sharedspace/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B_6=${MODEL_B_6:-/sharedspace/models/DeepSeek-R1-Distill-Llama-3B}
ROOT_6=${ROOT_6:-snapshots/table1_pair6_llama32_abliterated_deepseek3b}

MODEL_A_7=${MODEL_A_7:-/sharedspace/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B_7=${MODEL_B_7:-/sharedspace/models/Bespoke-Stratos-7B}
ROOT_7=${ROOT_7:-snapshots/table1_pair7_qwen25_uncensored_bespoke}

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

has_done_run() {
  local out=$1
  local run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

run_brekv() {
  local root=$1 model_a=$2 model_b=$3 task=$4 win=$5 tau=$6 scale=$7
  local out="${root}/${task}/coverage"
  local run_name="cov_t${tau}_s${scale}_w${win}"

  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] pair root=${root} task=${task} B-ReKV w${win} tau=${tau} scale=${scale} ===="
    return
  fi

  echo "==== [GPU${GPU}] ${root}/${task} B-ReKV w${win} tau=${tau} scale=${scale} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage \
    --budget_min "${MIN_BUDGET}" --budget_max "${MAX_BUDGET}" \
    --coverage_tau "${tau}" --coverage_scale "${scale}" \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
}

run_pair() {
  local pair=$1
  local model_a model_b root
  model_a=$(model_a_for_pair "${pair}")
  model_b=$(model_b_for_pair "${pair}")
  root=$(root_for_pair "${pair}")
  check_models "${model_a}" "${model_b}"
  mkdir -p "${root}/logs"

  echo "######## pair${pair} B-ReKV robustness START $(date '+%F %T') ########"
  echo "MODEL_A=${model_a}"
  echo "MODEL_B=${model_b}"
  echo "ROOT=${root}"
  echo "TASKS=${TASKS}"
  echo "WINDOWS=${WINDOWS}"
  echo "TAUS=${TAUS}"
  echo "SCALES=${SCALES}"

  for task in ${TASKS}; do
    for win in ${WINDOWS}; do
      for tau in ${TAUS}; do
        for scale in ${SCALES}; do
          run_brekv "${root}" "${model_a}" "${model_b}" "${task}" "${win}" "${tau}" "${scale}"
        done
      done
    done
  done

  echo "######## pair${pair} B-ReKV robustness DONE $(date '+%F %T') ########"
}

TAG=$(date '+%m%d_%H%M')
LOG_ROOT="snapshots/brekv_robustness/logs"
mkdir -p "${LOG_ROOT}"
LOG_PATH="${LOG_ROOT}/gpu${GPU}_pair6_pair7_brekv_robustness_${TAG}.log"

(
  echo "######## GPU${GPU} pair6/pair7 B-ReKV robustness START $(date '+%F %T') ########"
  echo "PAIR_IDS=${PAIR_IDS}"
  for pair in ${PAIR_IDS}; do
    run_pair "${pair}"
  done
  echo "######## GPU${GPU} pair6/pair7 B-ReKV robustness DONE $(date '+%F %T') ########"
) > "${LOG_PATH}" 2>&1 &

pid=$!
echo "pair6/pair7 B-ReKV robustness GPU${GPU} pid=${pid} -> ${LOG_PATH}"
echo "Results roots -> ${ROOT_6} and ${ROOT_7}"
