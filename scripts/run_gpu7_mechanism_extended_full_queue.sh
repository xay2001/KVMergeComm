#!/usr/bin/env bash
set -euo pipefail

# Serial GPU7 queue for fuller appendix/mechanism experiments.
#
# Phases:
#   1) Sink/recent token ablation on pair #1, all 8 main tasks.
#   2) Positional coherence ablation (ReKV-S / B-ReKV-S) on pair #1, all 8 main tasks.
#   3) Table 6 extended-task queue on pair #6/#7 with paper-style ReKV/B-ReKV blocks.
#
# Default is intentionally broad. Override env vars to shrink/expand:
#   GPU=7
#   MAIN_TASKS="countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"
#   EXT_TASKS="hotpotqa_full qasper_full musique_full samsum repobench"
#   RUN_SINK_RECENT=1 RUN_POSITIONAL=1 RUN_TABLE6=1
#   RUN_BREKV_SHIFT=0   # keep off until B-ReKV-S shift_back assertion is fixed
#   TABLE6_PAIR_IDS="6 7"
#
# Run:
#   bash scripts/run_gpu7_mechanism_extended_full_queue.sh

cd /home/xay/KVComm || exit 1

GPU=${GPU:-7}
SKIP_EXISTING=${SKIP_EXISTING:-1}

RUN_SINK_RECENT=${RUN_SINK_RECENT:-1}
RUN_POSITIONAL=${RUN_POSITIONAL:-1}
RUN_TABLE6=${RUN_TABLE6:-1}
RUN_BREKV_SHIFT=${RUN_BREKV_SHIFT:-0}

MAIN_TASKS=${MAIN_TASKS:-"countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"}
POSITIONAL_TASKS=${POSITIONAL_TASKS:-"${MAIN_TASKS}"}
EXT_TASKS=${EXT_TASKS:-"hotpotqa_full qasper_full musique_full samsum repobench"}

SINK_RECENT_PAIRS=${SINK_RECENT_PAIRS:-"0:0 4:0 0:8 4:8"}
RATIOS=${RATIOS:-"0.3 0.5 0.7"}
TABLE6_PAIR_IDS=${TABLE6_PAIR_IDS:-"6 7"}

MODEL_A_1=${MODEL_A_1:-/sharedspace/models/Llama-3.1-8B-Instruct}
MODEL_B_1=${MODEL_B_1:-/sharedspace/models/Llama-3.1-8B-Instruct}
ROOT_MECH_1=${ROOT_MECH_1:-snapshots/mechanism/pair1_llama31_same}

MODEL_A_6=${MODEL_A_6:-/sharedspace/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B_6=${MODEL_B_6:-/sharedspace/models/DeepSeek-R1-Distill-Llama-3B}
ROOT_TABLE6_6=${ROOT_TABLE6_6:-snapshots/table6_pair6_llama32_abliterated_deepseek3b}

MODEL_A_7=${MODEL_A_7:-/sharedspace/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B_7=${MODEL_B_7:-/sharedspace/models/Bespoke-Stratos-7B}
ROOT_TABLE6_7=${ROOT_TABLE6_7:-snapshots/table6_pair7_qwen25_uncensored_bespoke}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

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

common_eval() {
  local model_a=$1 model_b=$2 task=$3 out=$4 run_name=$5
  shift 5
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${model_a}" --model_B "${model_b}" \
    --snapshot_path "${out}" \
    --run_name "${run_name}" "$@"
}

run_rekv() {
  local model_a=$1 model_b=$2 root=$3 task=$4 win=$5 ratio=$6 sink=$7 recent=$8 suffix=$9
  local out="${root}/${task}/mtc_receiver"
  local run_name="recv_w${win}_r${ratio}${suffix}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${root}/${task} ReKV ${run_name} ===="
    return
  fi
  echo "==== [GPU${GPU}] ${root}/${task} ReKV w${win} r=${ratio} sink=${sink} recent=${recent} suffix=${suffix} $(date '+%F %T') ===="
  common_eval "${model_a}" "${model_b}" "${task}" "${out}" "${run_name}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink "${sink}" --merge_recent "${recent}"
}

run_rekv_shift() {
  local model_a=$1 model_b=$2 root=$3 task=$4 win=$5 ratio=$6 suffix=$7
  local out="${root}/${task}/mtc_receiver_shift"
  local run_name="recv_w${win}_r${ratio}${suffix}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${root}/${task} ReKV-S ${run_name} ===="
    return
  fi
  echo "==== [GPU${GPU}] ${root}/${task} ReKV-S w${win} r=${ratio} $(date '+%F %T') ===="
  common_eval "${model_a}" "${model_b}" "${task}" "${out}" "${run_name}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 --shift_back
}

run_brekv() {
  local model_a=$1 model_b=$2 root=$3 task=$4 win=$5 tau=$6 scale=$7 sink=$8 recent=$9 suffix=${10}
  local out="${root}/${task}/coverage"
  local run_name="cov_t${tau}_s${scale}_w${win}${suffix}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${root}/${task} B-ReKV ${run_name} ===="
    return
  fi
  echo "==== [GPU${GPU}] ${root}/${task} B-ReKV w${win} tau=${tau} scale=${scale} sink=${sink} recent=${recent} suffix=${suffix} $(date '+%F %T') ===="
  common_eval "${model_a}" "${model_b}" "${task}" "${out}" "${run_name}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink "${sink}" --merge_recent "${recent}" \
    --budget_mode coverage --budget_min 0.05 --budget_max 0.7 \
    --coverage_tau "${tau}" --coverage_scale "${scale}"
}

run_brekv_shift() {
  local model_a=$1 model_b=$2 root=$3 task=$4 win=$5 tau=$6 scale=$7 suffix=$8
  local out="${root}/${task}/coverage_shift"
  local run_name="cov_t${tau}_s${scale}_w${win}${suffix}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${root}/${task} B-ReKV-S ${run_name} ===="
    return
  fi
  echo "==== [GPU${GPU}] ${root}/${task} B-ReKV-S w${win} tau=${tau} scale=${scale} $(date '+%F %T') ===="
  common_eval "${model_a}" "${model_b}" "${task}" "${out}" "${run_name}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage --budget_min 0.05 --budget_max 0.7 \
    --coverage_tau "${tau}" --coverage_scale "${scale}" --shift_back
}

run_sink_recent_phase() {
  check_model "${MODEL_A_1}"
  check_model "${MODEL_B_1}"
  local root="${ROOT_MECH_1}/sink_recent"

  echo "######## Sink/recent ablation START $(date '+%F %T') ########"
  echo "ROOT=${root}"
  echo "TASKS=${MAIN_TASKS}"
  echo "SINK_RECENT_PAIRS=${SINK_RECENT_PAIRS}"

  for task in ${MAIN_TASKS}; do
    for sr in ${SINK_RECENT_PAIRS}; do
      local sink="${sr%%:*}"
      local recent="${sr##*:}"
      local suffix="_sink${sink}_recent${recent}"
      run_rekv "${MODEL_A_1}" "${MODEL_B_1}" "${root}" "${task}" 8 0.3 "${sink}" "${recent}" "${suffix}"
      run_brekv "${MODEL_A_1}" "${MODEL_B_1}" "${root}" "${task}" 8 0.95 0.75 "${sink}" "${recent}" "${suffix}"
    done
  done

  echo "######## Sink/recent ablation DONE $(date '+%F %T') ########"
}

run_positional_phase() {
  check_model "${MODEL_A_1}"
  check_model "${MODEL_B_1}"
  local root="${ROOT_MECH_1}/positional_coherence"

  echo "######## Positional coherence ablation START $(date '+%F %T') ########"
  echo "ROOT=${root}"
  echo "TASKS=${POSITIONAL_TASKS}"
  echo "RUN_BREKV_SHIFT=${RUN_BREKV_SHIFT}"

  for task in ${POSITIONAL_TASKS}; do
    run_rekv "${MODEL_A_1}" "${MODEL_B_1}" "${root}" "${task}" 8 0.3 4 8 "_normal"
    run_rekv_shift "${MODEL_A_1}" "${MODEL_B_1}" "${root}" "${task}" 8 0.3 "_shiftback"
    run_brekv "${MODEL_A_1}" "${MODEL_B_1}" "${root}" "${task}" 8 0.95 0.75 4 8 "_normal"
    if [[ "${RUN_BREKV_SHIFT}" == "1" ]]; then
      run_brekv_shift "${MODEL_A_1}" "${MODEL_B_1}" "${root}" "${task}" 8 0.95 0.75 "_shiftback"
    else
      echo "==== [skip] ${root}/${task} B-ReKV-S shiftback disabled; set RUN_BREKV_SHIFT=1 to retry ===="
    fi
  done

  echo "######## Positional coherence ablation DONE $(date '+%F %T') ########"
}

model_a_for_table6_pair() {
  local pair=$1
  case "${pair}" in
    6) echo "${MODEL_A_6}" ;;
    7) echo "${MODEL_A_7}" ;;
    *) echo "unknown Table 6 pair=${pair}" >&2; exit 1 ;;
  esac
}

model_b_for_table6_pair() {
  local pair=$1
  case "${pair}" in
    6) echo "${MODEL_B_6}" ;;
    7) echo "${MODEL_B_7}" ;;
    *) echo "unknown Table 6 pair=${pair}" >&2; exit 1 ;;
  esac
}

root_for_table6_pair() {
  local pair=$1
  case "${pair}" in
    6) echo "${ROOT_TABLE6_6}" ;;
    7) echo "${ROOT_TABLE6_7}" ;;
    *) echo "unknown Table 6 pair=${pair}" >&2; exit 1 ;;
  esac
}

run_table6_pair() {
  local pair=$1
  local model_a model_b root
  model_a=$(model_a_for_table6_pair "${pair}")
  model_b=$(model_b_for_table6_pair "${pair}")
  root=$(root_for_table6_pair "${pair}")
  check_model "${model_a}"
  check_model "${model_b}"

  echo "######## Table 6 pair${pair} START $(date '+%F %T') ########"
  echo "MODEL_A=${model_a}"
  echo "MODEL_B=${model_b}"
  echo "ROOT=${root}"
  echo "EXT_TASKS=${EXT_TASKS}"
  echo "RATIOS=${RATIOS}"

  for task in ${EXT_TASKS}; do
    for win in 8 16; do
      for ratio in ${RATIOS}; do
        run_rekv "${model_a}" "${model_b}" "${root}" "${task}" "${win}" "${ratio}" 4 8 ""
      done
    done
    run_brekv "${model_a}" "${model_b}" "${root}" "${task}" 8 0.95 0.75 4 8 ""
    run_brekv "${model_a}" "${model_b}" "${root}" "${task}" 8 0.95 0.85 4 8 ""
    run_brekv "${model_a}" "${model_b}" "${root}" "${task}" 16 0.95 0.90 4 8 ""
  done

  echo "######## Table 6 pair${pair} DONE $(date '+%F %T') ########"
}

run_table6_phase() {
  echo "######## Table 6 extended tasks START $(date '+%F %T') ########"
  for pair in ${TABLE6_PAIR_IDS}; do
    run_table6_pair "${pair}"
  done
  echo "######## Table 6 extended tasks DONE $(date '+%F %T') ########"
}

TAG=$(date '+%m%d_%H%M')
LOG_ROOT="snapshots/mechanism/logs"
mkdir -p "${LOG_ROOT}"
LOG_PATH="${LOG_ROOT}/gpu${GPU}_mechanism_extended_full_${TAG}.log"

(
  echo "######## GPU${GPU} mechanism + extended full queue START $(date '+%F %T') ########"
  echo "RUN_SINK_RECENT=${RUN_SINK_RECENT}"
  echo "RUN_POSITIONAL=${RUN_POSITIONAL}"
  echo "RUN_TABLE6=${RUN_TABLE6}"
  echo "RUN_BREKV_SHIFT=${RUN_BREKV_SHIFT}"
  echo "MAIN_TASKS=${MAIN_TASKS}"
  echo "POSITIONAL_TASKS=${POSITIONAL_TASKS}"
  echo "EXT_TASKS=${EXT_TASKS}"

  if [[ "${RUN_SINK_RECENT}" == "1" ]]; then
    run_sink_recent_phase
  fi
  if [[ "${RUN_POSITIONAL}" == "1" ]]; then
    run_positional_phase
  fi
  if [[ "${RUN_TABLE6}" == "1" ]]; then
    run_table6_phase
  fi

  echo "######## GPU${GPU} mechanism + extended full queue DONE $(date '+%F %T') ########"
) > "${LOG_PATH}" 2>&1 &

pid=$!
echo "mechanism + extended full queue GPU${GPU} pid=${pid} -> ${LOG_PATH}"
echo "Main mechanism root -> ${ROOT_MECH_1}"
echo "Table 6 roots -> ${ROOT_TABLE6_6} and ${ROOT_TABLE6_7}"
