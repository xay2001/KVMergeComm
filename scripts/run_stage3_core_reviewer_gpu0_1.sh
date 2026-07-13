#!/usr/bin/env bash
set -euo pipefail

# Stage 3 reviewer evidence under the deployable Query-Sketch protocol.
# Produces matched-budget fairness, B-ReKV Pareto/budget distributions, and
# calibrated-vs-strict evidence. Runs are resumable through per_sample.jsonl.

cd /home/xay/KVMergeComm || exit 1

GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
PYTHON=${PYTHON:-/home/xay/miniconda3/envs/ReKV/bin/python}
ROOT=${ROOT:-snapshots/stage3_core_reviewer_query_sketch}
LIMIT=${LIMIT:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
FIXED_RATIOS=${FIXED_RATIOS:-"0.15 0.20 0.25 0.30 0.35 0.40 0.50"}
FAIRNESS_WINDOW=${FAIRNESS_WINDOW:-8}
WINDOWS=${WINDOWS:-"8 16"}
TAUS=${TAUS:-"0.90 0.95 0.98"}
SCALES=${SCALES:-"0.65 0.75 0.85"}
MIN_BUDGET=${MIN_BUDGET:-0.05}
MAX_BUDGET=${MAX_BUDGET:-0.70}
CANONICAL_TAU=${CANONICAL_TAU:-0.95}
CANONICAL_SCALE=${CANONICAL_SCALE:-0.75}
CANONICAL_WINDOW=${CANONICAL_WINDOW:-8}
RUN_STRICT_PAIR7=${RUN_STRICT_PAIR7:-0}

MODEL_A_1=${MODEL_A_1:-/NAS/models/Llama-3.1-8B-Instruct}
MODEL_B_1=${MODEL_B_1:-/NAS/models/Llama-3.1-8B-Instruct}
MODEL_A_6=${MODEL_A_6:-/NAS/models/Llama-3.2-3B-Instruct-abliterated}
MODEL_B_6=${MODEL_B_6:-/NAS/models/DeepSeek-R1-Distill-Llama-3B}
MODEL_A_7=${MODEL_A_7:-/NAS/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B_7=${MODEL_B_7:-/NAS/models/Bespoke-Stratos-7B}
QUEUE0=${QUEUE0:-"1:hotpotqa 1:musique 1:multifieldqa_en 7:hotpotqa"}
QUEUE1=${QUEUE1:-"6:hotpotqa 6:musique 6:multifieldqa_en 7:musique 7:multifieldqa_en"}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

pair_name() {
  case "$1" in
    1) echo "pair1_llama31_same" ;;
    6) echo "pair6_llama32_abliterated_deepseek3b" ;;
    7) echo "pair7_qwen25_uncensored_bespoke" ;;
    *) echo "Unknown pair: $1" >&2; return 2 ;;
  esac
}

model_a() {
  case "$1" in
    1) echo "${MODEL_A_1}" ;;
    6) echo "${MODEL_A_6}" ;;
    7) echo "${MODEL_A_7}" ;;
    *) return 2 ;;
  esac
}

model_b() {
  case "$1" in
    1) echo "${MODEL_B_1}" ;;
    6) echo "${MODEL_B_6}" ;;
    7) echo "${MODEL_B_7}" ;;
    *) return 2 ;;
  esac
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
      *) echo "Unknown fixed method: ${method}" >&2; return 2 ;;
    esac
  done
}

run_brekv_grid() {
  local gpu=$1 pair=$2 task=$3 win tau scale
  for win in ${WINDOWS}; do
    for tau in ${TAUS}; do
      for scale in ${SCALES}; do
        run_com "${gpu}" "${pair}" "${task}" coverage "cov_t${tau}_s${scale}_w${win}" \
          --score_mode receiver --query_sketch_mode bf16 --recv_window "${win}" \
          --budget_mode coverage --budget_min "${MIN_BUDGET}" --budget_max "${MAX_BUDGET}" \
          --coverage_tau "${tau}" --coverage_scale "${scale}"
      done
    done
  done
}

run_strict_ablation() {
  local gpu=$1 pair=$2 task=$3 spec win tau_min tau_max
  for spec in "8:0.70:0.90" "8:0.80:0.95" "16:0.80:0.95"; do
    IFS=: read -r win tau_min tau_max <<< "${spec}"
    run_com "${gpu}" "${pair}" "${task}" strict_coverage \
      "strict_adapt_t${tau_min}-${tau_max}_w${win}" \
      --score_mode receiver --query_sketch_mode bf16 --recv_window "${win}" \
      --budget_mode strict_coverage --coverage_tau_mode adaptive \
      --coverage_tau_min "${tau_min}" --coverage_tau_max "${tau_max}" \
      --budget_min 0.0 --budget_max 1.0 --budget_floor 0.0
  done
}

run_item() {
  local gpu=$1 pair=$2 task=$3
  run_fixed_curve "${gpu}" "${pair}" "${task}" evict
  run_fixed_curve "${gpu}" "${pair}" "${task}" random
  run_fixed_curve "${gpu}" "${pair}" "${task}" rekv
  run_brekv_grid "${gpu}" "${pair}" "${task}"
  if [[ "${pair}" == "1" ]] || [[ "${pair}" == "7" && "${RUN_STRICT_PAIR7}" == "1" ]]; then
    run_strict_ablation "${gpu}" "${pair}" "${task}"
  fi
}

run_queue() {
  local gpu=$1 queue=$2 item pair task
  for item in ${queue}; do
    IFS=: read -r pair task <<< "${item}"
    run_item "${gpu}" "${pair}" "${task}"
  done
}

[[ -x "${PYTHON}" ]] || { echo "Python interpreter is not executable: ${PYTHON}" >&2; exit 2; }
for path in "${MODEL_A_1}" "${MODEL_B_1}" "${MODEL_A_6}" "${MODEL_B_6}" "${MODEL_A_7}" "${MODEL_B_7}"; do
  [[ -d "${path}" ]] || { echo "Model path does not exist: ${path}" >&2; exit 2; }
done

mkdir -p "${ROOT}/logs" "${ROOT}/analysis"
TAG=$(date '+%m%d_%H%M')
LOG0="${ROOT}/logs/gpu${GPU0}_stage3_${TAG}.log"
LOG1="${ROOT}/logs/gpu${GPU1}_stage3_${TAG}.log"
(run_queue "${GPU0}" "${QUEUE0}") > "${LOG0}" 2>&1 & PID0=$!
(run_queue "${GPU1}" "${QUEUE1}") > "${LOG1}" 2>&1 & PID1=$!
echo "GPU${GPU0} pid=${PID0}: ${LOG0}"
echo "GPU${GPU1} pid=${PID1}: ${LOG1}"
echo "Results: ${ROOT}"
status=0
wait "${PID0}" || status=$?
wait "${PID1}" || status=$?
"${PYTHON}" scripts/analyze_stage3_core_reviewer.py \
  --root "${ROOT}" --canonical-tau "${CANONICAL_TAU}" \
  --canonical-scale "${CANONICAL_SCALE}" \
  --canonical-window "${CANONICAL_WINDOW}" || status=$?
exit "${status}"
