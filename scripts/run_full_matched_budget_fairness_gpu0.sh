#!/usr/bin/env bash
set -euo pipefail

# Full matched-budget fairness matrix under the deployable Query-Sketch protocol.
#
# Goal:
#   7 model pairs x 8 tasks:
#     - main B-ReKV: t0.95-s0.75-w8
#     - matched-budget baselines: ReKV, ValueNorm/Evict, Random
#
# Default workload:
#   7 pairs x 8 tasks x (1 B-ReKV + 3 methods x 9 ratios) = 1568 runs.
# This is intentionally large. By default it is split across GPU0/GPU1 at the
# pair-task block level. Use WORKERS, PAIR_IDS, TASKS, RUN_ITEMS, LIMIT, or
# FIXED_RATIOS to shard/smoke-test. Runs are resumable by per_sample.jsonl.
#
# Examples:
#   bash scripts/run_full_matched_budget_fairness_gpu0.sh
#   LIMIT=100 PAIR_IDS="1 7" TASKS="hotpotqa musique" bash scripts/run_full_matched_budget_fairness_gpu0.sh
#   RUN_ITEMS="1:hotpotqa 7:musique" bash scripts/run_full_matched_budget_fairness_gpu0.sh
#   WORKERS=gpu1 bash scripts/run_full_matched_budget_fairness_gpu0.sh

cd /home/xay/KVMergeComm || exit 1

GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
WORKERS=${WORKERS:-both} # both | gpu0 | gpu1
PYTHON=${PYTHON:-/home/xay/miniconda3/envs/ReKV/bin/python}
ROOT=${ROOT:-snapshots/full_matched_budget_fairness_query_sketch}
LIMIT=${LIMIT:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
RUN_ANALYSIS=${RUN_ANALYSIS:-1}

PAIR_IDS=${PAIR_IDS:-"1 2 3 4 5 6 7"}
TASKS=${TASKS:-"countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"}
RUN_ITEMS=${RUN_ITEMS:-}

# The upper ratios make interpolation valid for high actual B-ReKV budgets
# without endpoint clamping.
FIXED_RATIOS=${FIXED_RATIOS:-"0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60"}
FAIRNESS_WINDOW=${FAIRNESS_WINDOW:-8}

BREKV_TAU=${BREKV_TAU:-0.95}
BREKV_SCALE=${BREKV_SCALE:-0.75}
BREKV_WINDOW=${BREKV_WINDOW:-8}
MIN_BUDGET=${MIN_BUDGET:-0.05}
MAX_BUDGET=${MAX_BUDGET:-0.70}

MODEL_A_1=${MODEL_A_1:-/NAS/models/Llama-3.1-8B-Instruct}
MODEL_B_1=${MODEL_B_1:-${MODEL_A_1}}
MODEL_A_2=${MODEL_A_2:-/NAS/models/Llama-3.2-3B-Instruct}
MODEL_B_2=${MODEL_B_2:-${MODEL_A_2}}
MODEL_A_3=${MODEL_A_3:-/NAS/models/Qwen2.5-7B-Instruct}
MODEL_B_3=${MODEL_B_3:-${MODEL_A_3}}
MODEL_A_4=${MODEL_A_4:-/NAS/models/Falcon3-7B-Instruct}
MODEL_B_4=${MODEL_B_4:-${MODEL_A_4}}
MODEL_A_5=${MODEL_A_5:-/NAS/models/EvolCodeLlama-3.1-8B-Instruct}
MODEL_B_5=${MODEL_B_5:-/NAS/models/ToolACE-2-Llama-3.1-8B}
MODEL_A_6=${MODEL_A_6:-/NAS/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B_6=${MODEL_B_6:-/NAS/models/DeepSeek-R1-Distill-Llama-3B}
MODEL_A_7=${MODEL_A_7:-/NAS/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B_7=${MODEL_B_7:-/NAS/models/Bespoke-Stratos-7B}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

pair_name() {
  case "$1" in
    1) echo "pair1_llama31_same" ;;
    2) echo "pair2_llama32_same" ;;
    3) echo "pair3_qwen25_7b_same" ;;
    4) echo "pair4_falcon3_7b_same" ;;
    5) echo "pair5_evolcodellama_toolace" ;;
    6) echo "pair6_llama32_abliterated_deepseek3b" ;;
    7) echo "pair7_qwen25_uncensored_bespoke" ;;
    *) echo "Unknown pair id: $1" >&2; return 2 ;;
  esac
}

model_a() {
  case "$1" in
    1) echo "${MODEL_A_1}" ;;
    2) echo "${MODEL_A_2}" ;;
    3) echo "${MODEL_A_3}" ;;
    4) echo "${MODEL_A_4}" ;;
    5) pair5_runtime_model ;;
    6) echo "${MODEL_A_6}" ;;
    7) echo "${MODEL_A_7}" ;;
    *) return 2 ;;
  esac
}

model_b() {
  case "$1" in
    1) echo "${MODEL_B_1}" ;;
    2) echo "${MODEL_B_2}" ;;
    3) echo "${MODEL_B_3}" ;;
    4) echo "${MODEL_B_4}" ;;
    5) echo "${MODEL_B_5}" ;;
    6) echo "${MODEL_B_6}" ;;
    7) echo "${MODEL_B_7}" ;;
    *) return 2 ;;
  esac
}

pair5_runtime_model() {
  local view="${ROOT}/model_views/$(basename "${MODEL_A_5}")_no_adapter"
  if [[ -f "${MODEL_A_5}/adapter_config.json" && -f "${MODEL_A_5}/model.safetensors.index.json" ]]; then
    mkdir -p "${view}"
    local file base
    for file in "${MODEL_A_5}"/*; do
      base=$(basename "${file}")
      case "${base}" in
        adapter_config.json|adapter_model.bin|adapter_model.safetensors) continue ;;
      esac
      ln -sfn "${file}" "${view}/${base}"
    done
    echo "${view}"
  else
    echo "${MODEL_A_5}"
  fi
}

has_done_run() {
  local out=$1 run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] &&
    compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

run_com() {
  local gpu=$1 pair=$2 task=$3 out_kind=$4 run_name=$5
  shift 5
  local pa pb out
  pa=$(model_a "${pair}")
  pb=$(model_b "${pair}")
  out="${ROOT}/$(pair_name "${pair}")/${task}/${out_kind}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] pair${pair} ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] pair${pair} ${task} ${run_name} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${pa}" --model_B "${pb}" \
    --merge --merge_mode evict --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" --run_name "${run_name}" "$@"
}

run_fixed_curve() {
  local gpu=$1 pair=$2 task=$3 method=$4 ratio
  for ratio in ${FIXED_RATIOS}; do
    case "${method}" in
      rekv)
        run_com "${gpu}" "${pair}" "${task}" fairness_rekv "rekv_w${FAIRNESS_WINDOW}_r${ratio}" \
          --score_mode receiver --query_sketch_mode bf16 \
          --recv_window "${FAIRNESS_WINDOW}" --merge_ratio "${ratio}"
        ;;
      evict)
        run_com "${gpu}" "${pair}" "${task}" fairness_evict "evict_r${ratio}" \
          --score_mode value_norm --recv_window 0 --merge_ratio "${ratio}"
        ;;
      random)
        run_com "${gpu}" "${pair}" "${task}" fairness_random "random_r${ratio}" \
          --score_mode random --recv_window 0 --merge_ratio "${ratio}"
        ;;
      *) echo "Unknown method: ${method}" >&2; return 2 ;;
    esac
  done
}

run_main_brekv() {
  local gpu=$1 pair=$2 task=$3
  run_com "${gpu}" "${pair}" "${task}" coverage "cov_t${BREKV_TAU}_s${BREKV_SCALE}_w${BREKV_WINDOW}" \
    --score_mode receiver --query_sketch_mode bf16 \
    --recv_window "${BREKV_WINDOW}" \
    --budget_mode coverage --budget_min "${MIN_BUDGET}" --budget_max "${MAX_BUDGET}" \
    --coverage_tau "${BREKV_TAU}" --coverage_scale "${BREKV_SCALE}"
}

run_item() {
  local gpu=$1 pair=$2 task=$3
  run_fixed_curve "${gpu}" "${pair}" "${task}" rekv
  run_fixed_curve "${gpu}" "${pair}" "${task}" evict
  run_fixed_curve "${gpu}" "${pair}" "${task}" random
  run_main_brekv "${gpu}" "${pair}" "${task}"
}

iter_items() {
  local pair task
  if [[ -n "${RUN_ITEMS}" ]]; then
    for item in ${RUN_ITEMS}; do
      echo "${item}"
    done
  else
    for pair in ${PAIR_IDS}; do
      for task in ${TASKS}; do
        echo "${pair}:${task}"
      done
    done
  fi
}

validate_model_paths() {
  local pair path_a path_b
  [[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 2; }
  for pair in ${PAIR_IDS}; do
    path_a=$(model_a "${pair}")
    path_b=$(model_b "${pair}")
    [[ -d "${path_a}" ]] || { echo "Pair ${pair} sender path does not exist: ${path_a}" >&2; exit 2; }
    [[ -d "${path_b}" ]] || { echo "Pair ${pair} receiver path does not exist: ${path_b}" >&2; exit 2; }
  done
}

worker_items() {
  local slot=$1 idx=0 item
  while read -r item; do
    if (( idx % 2 == slot )); then
      echo "${item}"
    fi
    idx=$((idx + 1))
  done < <(iter_items)
}

run_worker() {
  local gpu=$1 slot=$2 pair task
  echo "######## Full matched-budget fairness worker GPU${gpu} slot=${slot} START $(date '+%F %T') ########"
  while IFS=: read -r pair task; do
    [[ -n "${pair}" && -n "${task}" ]] || continue
    run_item "${gpu}" "${pair}" "${task}"
  done < <(worker_items "${slot}")
  echo "######## Full matched-budget fairness worker GPU${gpu} slot=${slot} DONE $(date '+%F %T') ########"
}

run_analysis_once() {
  if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    "${PYTHON}" scripts/analyze_stage3_core_reviewer.py \
      --root "${ROOT}" \
      --canonical-tau "${BREKV_TAU}" \
      --canonical-scale "${BREKV_SCALE}" \
      --canonical-window "${BREKV_WINDOW}"
  fi
}

validate_model_paths
mkdir -p "${ROOT}/logs" "${ROOT}/analysis"
TAG=$(date '+%m%d_%H%M')
LOG0="${ROOT}/logs/gpu${GPU0}_full_matched_budget_fairness_${TAG}.log"
LOG1="${ROOT}/logs/gpu${GPU1}_full_matched_budget_fairness_${TAG}.log"
LOG_MAIN="${ROOT}/logs/full_matched_budget_fairness_supervisor_${TAG}.log"

(
  echo "######## Full matched-budget fairness supervisor START $(date '+%F %T') ########"
  echo "ROOT=${ROOT}"
  echo "GPU0=${GPU0}"
  echo "GPU1=${GPU1}"
  echo "WORKERS=${WORKERS}"
  echo "PAIR_IDS=${PAIR_IDS}"
  echo "TASKS=${TASKS}"
  echo "RUN_ITEMS=${RUN_ITEMS}"
  echo "FIXED_RATIOS=${FIXED_RATIOS}"
  echo "B-ReKV=t${BREKV_TAU},s${BREKV_SCALE},w${BREKV_WINDOW}"
  echo "LIMIT=${LIMIT}"

  status=0
  pid0=
  pid1=
  case "${WORKERS}" in
    both)
      run_worker "${GPU0}" 0 > "${LOG0}" 2>&1 & pid0=$!
      run_worker "${GPU1}" 1 > "${LOG1}" 2>&1 & pid1=$!
      echo "GPU${GPU0} pid=${pid0}: ${LOG0}"
      echo "GPU${GPU1} pid=${pid1}: ${LOG1}"
      wait "${pid0}" || status=$?
      wait "${pid1}" || status=$?
      ;;
    gpu0)
      run_worker "${GPU0}" 0 > "${LOG0}" 2>&1 & pid0=$!
      echo "GPU${GPU0} pid=${pid0}: ${LOG0}"
      wait "${pid0}" || status=$?
      ;;
    gpu1)
      run_worker "${GPU1}" 1 > "${LOG1}" 2>&1 & pid1=$!
      echo "GPU${GPU1} pid=${pid1}: ${LOG1}"
      wait "${pid1}" || status=$?
      ;;
    *)
      echo "Unknown WORKERS=${WORKERS}; use both, gpu0, or gpu1." >&2
      exit 2
      ;;
  esac

  if [[ "${status}" == "0" ]]; then
    run_analysis_once || status=$?
  else
    echo "At least one worker failed; skip analysis until rerun completes." >&2
  fi
  echo "######## Full matched-budget fairness supervisor DONE status=${status} $(date '+%F %T') ########"
  exit "${status}"
) > "${LOG_MAIN}" 2>&1 &

pid=$!
echo "full matched-budget fairness supervisor pid=${pid} -> ${LOG_MAIN}"
echo "GPU${GPU0} worker log -> ${LOG0}"
echo "GPU${GPU1} worker log -> ${LOG1}"
echo "Results root -> ${ROOT}"
