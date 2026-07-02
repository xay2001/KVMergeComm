#!/usr/bin/env bash
set -euo pipefail

# Full method-coverage cost profiling for Table 1 pair #6/#7 on GPU 2.
#
# Covers all 8 main tasks and these methods:
#   - KVComm top={0.3,0.5,0.7}
#   - Evict/ValueNorm r=0.3
#   - Random-token r=0.3
#   - ReKV-w8/w16 r=0.3
#   - B-ReKV canonical points:
#       cov_t0.95_s0.75_w8
#       cov_t0.95_s0.85_w8
#       cov_t0.95_s0.90_w16
#
# Default LIMIT=50 keeps this as a controlled cost profile. Use LIMIT=0 for
# full-dataset profiling:
#   LIMIT=0 bash scripts/run_pair6_pair7_full_cost_profile_gpu2.sh

cd /home/xay/KVComm || exit 1

GPU=${GPU:-2}
PAIR_IDS=${PAIR_IDS:-"6 7"}
TASKS=${TASKS:-"countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"}
LIMIT=${LIMIT:-50}
WARMUP=${WARMUP:-5}
KVCOMM_TOPS=${KVCOMM_TOPS:-"0.3 0.5 0.7"}
RATIO=${RATIO:-0.3}
SKIP_EXISTING=${SKIP_EXISTING:-1}

MODEL_A_6=${MODEL_A_6:-/sharedspace/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B_6=${MODEL_B_6:-/sharedspace/models/DeepSeek-R1-Distill-Llama-3B}
ROOT_6=${ROOT_6:-snapshots/cost_profile/table1_pair6_llama32_abliterated_deepseek3b_full}

MODEL_A_7=${MODEL_A_7:-/sharedspace/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B_7=${MODEL_B_7:-/sharedspace/models/Bespoke-Stratos-7B}
ROOT_7=${ROOT_7:-snapshots/cost_profile/table1_pair7_qwen25_uncensored_bespoke_full}

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
  [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${out}/${run_name}_*/cost_summary.json" > /dev/null
}

common() {
  local model_a=$1
  local model_b=$2
  local task=$3
  local out=$4
  shift 4
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${model_a}" --model_B "${model_b}" \
    --profile_cost --profile_limit "${LIMIT}" --profile_warmup "${WARMUP}" \
    --snapshot_path "${out}" "$@"
}

run_kvcomm() {
  local model_a=$1 model_b=$2 root=$3 task=$4 top=$5
  local out="${root}/${task}/kvcomm"
  local run_name="kvcomm_top${top}_cost"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} KVComm top=${top} cost ===="
    return
  fi
  echo "==== [cost GPU${GPU}] ${task} KVComm top=${top} $(date '+%F %T') ===="
  common "${model_a}" "${model_b}" "${task}" "${out}" \
    --top_layers "${top}" \
    --run_name "${run_name}"
}

run_evict() {
  local model_a=$1 model_b=$2 root=$3 task=$4
  local out="${root}/${task}/mtc_evict"
  local run_name="evict_r${RATIO}_cost"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} Evict r=${RATIO} cost ===="
    return
  fi
  echo "==== [cost GPU${GPU}] ${task} Evict/ValueNorm r=${RATIO} $(date '+%F %T') ===="
  common "${model_a}" "${model_b}" "${task}" "${out}" \
    --merge --merge_mode evict --score_mode value_norm --recv_window 0 \
    --merge_ratio "${RATIO}" --merge_sink 4 --merge_recent 8 \
    --run_name "${run_name}"
}

run_random() {
  local model_a=$1 model_b=$2 root=$3 task=$4
  local out="${root}/${task}/mtc_random"
  local run_name="random_r${RATIO}_cost"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} Random-token r=${RATIO} cost ===="
    return
  fi
  echo "==== [cost GPU${GPU}] ${task} Random-token r=${RATIO} $(date '+%F %T') ===="
  common "${model_a}" "${model_b}" "${task}" "${out}" \
    --merge --merge_mode evict --score_mode random --recv_window 0 \
    --merge_ratio "${RATIO}" --merge_sink 4 --merge_recent 8 \
    --run_name "${run_name}"
}

run_rekv() {
  local model_a=$1 model_b=$2 root=$3 task=$4 win=$5
  local out="${root}/${task}/mtc_receiver"
  local run_name="recv_w${win}_r${RATIO}_cost"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ReKV w${win} r=${RATIO} cost ===="
    return
  fi
  echo "==== [cost GPU${GPU}] ${task} ReKV w${win} r=${RATIO} $(date '+%F %T') ===="
  common "${model_a}" "${model_b}" "${task}" "${out}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${RATIO}" --merge_sink 4 --merge_recent 8 \
    --run_name "${run_name}"
}

run_brekv() {
  local model_a=$1 model_b=$2 root=$3 task=$4 win=$5 tau=$6 scale=$7
  local out="${root}/${task}/coverage"
  local run_name="cov_t${tau}_s${scale}_w${win}_cost"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} B-ReKV w${win} tau=${tau} scale=${scale} cost ===="
    return
  fi
  echo "==== [cost GPU${GPU}] ${task} B-ReKV w${win} tau=${tau} scale=${scale} $(date '+%F %T') ===="
  common "${model_a}" "${model_b}" "${task}" "${out}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage --budget_min 0.05 --budget_max 0.7 \
    --coverage_tau "${tau}" --coverage_scale "${scale}" \
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

  echo "######## pair${pair} full cost START $(date '+%F %T') ########"
  echo "MODEL_A=${model_a}"
  echo "MODEL_B=${model_b}"
  echo "ROOT=${root}"
  echo "TASKS=${TASKS}"
  echo "LIMIT=${LIMIT}"
  echo "WARMUP=${WARMUP}"

  for task in ${TASKS}; do
    for top in ${KVCOMM_TOPS}; do
      run_kvcomm "${model_a}" "${model_b}" "${root}" "${task}" "${top}"
    done

    run_evict "${model_a}" "${model_b}" "${root}" "${task}"
    run_random "${model_a}" "${model_b}" "${root}" "${task}"
    run_rekv "${model_a}" "${model_b}" "${root}" "${task}" 8
    run_rekv "${model_a}" "${model_b}" "${root}" "${task}" 16
    run_brekv "${model_a}" "${model_b}" "${root}" "${task}" 8 0.95 0.75
    run_brekv "${model_a}" "${model_b}" "${root}" "${task}" 8 0.95 0.85
    run_brekv "${model_a}" "${model_b}" "${root}" "${task}" 16 0.95 0.90
  done

  python scripts/analyze_cost_profile.py --root "${root}" --csv "${root}/cost_table.csv" > "${root}/cost_table.md"
  echo "######## pair${pair} full cost DONE $(date '+%F %T') ########"
  echo "Cost table -> ${root}/cost_table.csv"
}

TAG=$(date '+%m%d_%H%M')
LOG_ROOT="snapshots/cost_profile/logs"
mkdir -p "${LOG_ROOT}"
LOG_PATH="${LOG_ROOT}/gpu${GPU}_pair6_pair7_full_cost_${TAG}.log"

(
  echo "######## GPU${GPU} pair6/pair7 full cost START $(date '+%F %T') ########"
  echo "PAIR_IDS=${PAIR_IDS}"
  for pair in ${PAIR_IDS}; do
    run_pair "${pair}"
  done
  echo "######## GPU${GPU} pair6/pair7 full cost DONE $(date '+%F %T') ########"
) > "${LOG_PATH}" 2>&1 &

pid=$!
echo "pair6/pair7 full cost GPU${GPU} pid=${pid} -> ${LOG_PATH}"
echo "Results roots -> ${ROOT_6} and ${ROOT_7}"
echo "Analyze after completion:"
echo "  python scripts/analyze_cost_profile.py --root ${ROOT_6} --csv ${ROOT_6}/cost_table.csv"
echo "  python scripts/analyze_cost_profile.py --root ${ROOT_7} --csv ${ROOT_7}/cost_table.csv"
