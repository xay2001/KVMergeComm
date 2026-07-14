#!/usr/bin/env bash
set -euo pipefail

# Finish the fast-node share of the Query-Sketch v1 matrix.
#
# GPU0:
#   Table 10 multi-source; Table 6 pair #7 RepoBench/HotpotQA-full;
#   Table 1 pair #7 remaining ReKV + main B-ReKV; pair #7 cost/oracle.
# GPU1:
#   Table 8 pair #5 first four tasks; Table 6 pair #7 remaining tasks;
#   Table 1 pair #1 main B-ReKV; pair #1 cost/oracle.
#
# All runs are resumable. A result is skipped only when its expected output
# file already exists.

cd /home/xay/KVMergeComm || exit 1

GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
WORKERS=${WORKERS:-both}
PYTHON=${PYTHON:-/home/xay/miniconda3/envs/ReKV/bin/python}
LIMIT=${LIMIT:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
PROFILE_LIMIT=${PROFILE_LIMIT:-50}
PROFILE_WARMUP=${PROFILE_WARMUP:-3}

TAU=${TAU:-0.95}
SCALE=${SCALE:-0.75}
BREKV_WINDOW=${BREKV_WINDOW:-8}
RATIOS=${RATIOS:-"0.3 0.5 0.7"}
WINDOWS=${WINDOWS:-"8 16"}

MODEL_LLAMA31=${MODEL_LLAMA31:-/NAS/models/Llama-3.1-8B-Instruct}
MODEL_P7_A=${MODEL_P7_A:-/NAS/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_P7_B=${MODEL_P7_B:-/NAS/models/Bespoke-Stratos-7B}
MODEL_P5_A=${MODEL_P5_A:-/NAS/models/EvolCodeLlama-3.1-8B-Instruct}
MODEL_P5_B=${MODEL_P5_B:-/NAS/models/ToolACE-2-Llama-3.1-8B}

ROOT_T10=${ROOT_T10:-snapshots/table10_multi_source_query_sketch}
ROOT_T6_P7=${ROOT_T6_P7:-snapshots/table6_pair7_query_sketch_qwen25_uncensored_bespoke}
ROOT_T8_P5=${ROOT_T8_P5:-snapshots/table8_pair5_query_sketch_evolcodellama_toolace}
ROOT_T1_P1=${ROOT_T1_P1:-snapshots/table1_pair1_query_sketch_llama31_same}
ROOT_T1_P7=${ROOT_T1_P7:-snapshots/table1_pair7_query_sketch_qwen25_uncensored_bespoke}
ROOT_COST=${ROOT_COST:-snapshots/query_sketch_cost_v1}
ROOT_ORACLE=${ROOT_ORACLE:-snapshots/query_sketch_oracle_gap}
LOG_ROOT=${LOG_ROOT:-snapshots/remaining_fast_gpu0_1/logs}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${LOG_ROOT}"
case "${WORKERS}" in
  both|gpu0|gpu1) ;;
  *) echo "WORKERS must be one of: both, gpu0, gpu1" >&2; exit 2 ;;
esac
exec 9>"${LOG_ROOT}/run_${WORKERS}.lock"
if ! flock -n 9; then
  echo "Another ${WORKERS} run_remaining_fast_gpu0_1.sh process is already active." >&2
  exit 2
fi

[[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 2; }
for model in "${MODEL_LLAMA31}" "${MODEL_P7_A}" "${MODEL_P7_B}" "${MODEL_P5_A}" "${MODEL_P5_B}"; do
  [[ -d "${model}" ]] || { echo "Model path does not exist: ${model}" >&2; exit 2; }
done

has_per_sample() {
  local out=$1 run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] &&
    compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

has_cost_summary() {
  local out=$1 run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] &&
    compgen -G "${out}/${run_name}_*/cost_summary.json" > /dev/null
}

pair5_runtime_model() {
  local view="${ROOT_T8_P5}/model_views/$(basename "${MODEL_P5_A}")_no_adapter"
  if [[ -f "${MODEL_P5_A}/adapter_config.json" && -f "${MODEL_P5_A}/model.safetensors.index.json" ]]; then
    mkdir -p "${view}"
    local file base
    for file in "${MODEL_P5_A}"/*; do
      base=$(basename "${file}")
      case "${base}" in
        adapter_config.json|adapter_model.bin|adapter_model.safetensors) continue ;;
      esac
      ln -sfn "${file}" "${view}/${base}"
    done
    echo "${view}"
  else
    echo "${MODEL_P5_A}"
  fi
}

run_rekv() {
  local gpu=$1 model_a=$2 model_b=$3 root=$4 task=$5 window=$6 ratio=$7
  local out="${root}/${task}/mtc_receiver"
  local run_name="recv_w${window}_r${ratio}"
  if has_per_sample "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] ${task} Query-Sketch ReKV ${run_name} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16 \
    --recv_window "${window}" --merge_ratio "${ratio}" \
    --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_brekv() {
  local gpu=$1 model_a=$2 model_b=$3 root=$4 task=$5 kind=$6
  local out="${root}/${task}/${kind}"
  local run_name
  if [[ "${kind}" == "coverage_frozen" ]]; then
    run_name="frozen_t${TAU}_s${SCALE}_w${BREKV_WINDOW}"
  else
    run_name="cov_t${TAU}_s${SCALE}_w${BREKV_WINDOW}"
  fi
  if has_per_sample "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] ${task} Query-Sketch B-ReKV ${run_name} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16 \
    --recv_window "${BREKV_WINDOW}" --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage --budget_min 0.05 --budget_max 0.70 \
    --coverage_tau "${TAU}" --coverage_scale "${SCALE}" \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_standard_block() {
  local gpu=$1 model_a=$2 model_b=$3 root=$4 task=$5 brekv_kind=$6
  local window ratio
  for window in ${WINDOWS}; do
    for ratio in ${RATIOS}; do
      run_rekv "${gpu}" "${model_a}" "${model_b}" "${root}" "${task}" "${window}" "${ratio}"
    done
  done
  run_brekv "${gpu}" "${model_a}" "${model_b}" "${root}" "${task}" "${brekv_kind}"
}

run_multi_source() {
  local gpu=$1 task=$2 window=$3 ratio=$4
  local out="${ROOT_T10}/${task}/multi_source_rekv"
  local run_name="ms_qs_bf16_w${window}_r${ratio}"
  if has_per_sample "${out}" "${run_name}"; then
    echo "==== [skip] Table10 ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] Table10 ${task} ${run_name} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com_ms.py \
    --test_task "${task}" --do_test \
    --model_A1 "${MODEL_LLAMA31}" --model_A2 "${MODEL_LLAMA31}" --model_B "${MODEL_LLAMA31}" \
    --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16 \
    --recv_window "${window}" --merge_ratio "${ratio}" \
    --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_cost() {
  local gpu=$1 pair=$2 model_a=$3 model_b=$4 task=$5 method=$6
  local out="${ROOT_COST}/${pair}/${task}/${method}"
  local run_name
  local args=(
    --test_task "${task}" --do_test
    --profile_cost --profile_limit "${PROFILE_LIMIT}" --profile_warmup "${PROFILE_WARMUP}"
    --model_A "${model_a}" --model_B "${model_b}"
    --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16
    --merge_sink 4 --merge_recent 8
    --snapshot_path "${out}"
  )
  if [[ "${method}" == "rekv" ]]; then
    run_name="rekv_bf16_w8_r0.30"
    args+=(--recv_window 8 --merge_ratio 0.30)
  else
    run_name="brekv_bf16_t${TAU}_s${SCALE}_w${BREKV_WINDOW}"
    args+=(
      --recv_window "${BREKV_WINDOW}"
      --budget_mode coverage --budget_min 0.05 --budget_max 0.70
      --coverage_tau "${TAU}" --coverage_scale "${SCALE}"
    )
  fi
  if has_cost_summary "${out}" "${run_name}"; then
    echo "==== [skip] cost ${pair} ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] cost ${pair} ${task} ${run_name} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py "${args[@]}" --run_name "${run_name}"
}

run_oracle_brekv() {
  local gpu=$1 pair=$2 model_a=$3 model_b=$4 task=$5 score_mode=$6
  local out="${ROOT_ORACLE}/${pair}/${task}/brekv"
  local run_name="brekv_${score_mode}_t${TAU}_s${SCALE}_w${BREKV_WINDOW}"
  if has_cost_summary "${out}" "${run_name}"; then
    echo "==== [skip] oracle ${pair} ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] oracle ${pair} ${task} ${run_name} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test \
    --profile_cost --profile_limit "${PROFILE_LIMIT}" --profile_warmup "${PROFILE_WARMUP}" \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode "${score_mode}" --query_sketch_mode bf16 \
    --recv_window "${BREKV_WINDOW}" --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage --budget_min 0.05 --budget_max 0.70 \
    --coverage_tau "${TAU}" --coverage_scale "${SCALE}" \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_cost_and_oracle_pair() {
  local gpu=$1 pair=$2 model_a=$3 model_b=$4
  local task method score_mode
  for task in hotpotqa musique multifieldqa_en; do
    for method in rekv brekv; do
      run_cost "${gpu}" "${pair}" "${model_a}" "${model_b}" "${task}" "${method}"
    done
    for score_mode in receiver receiver_oracle; do
      run_oracle_brekv "${gpu}" "${pair}" "${model_a}" "${model_b}" "${task}" "${score_mode}"
    done
  done
}

gpu0_worker() {
  local task window ratio

  # Highest-memory block first.
  for task in hotpotqa musique twowikimqa; do
    for window in ${WINDOWS}; do
      for ratio in ${RATIOS}; do
        run_multi_source "${GPU0}" "${task}" "${window}" "${ratio}"
      done
    done
  done

  # Long-context pair #7 tasks.
  for task in repobench hotpotqa_full; do
    run_standard_block "${GPU0}" "${MODEL_P7_A}" "${MODEL_P7_B}" \
      "${ROOT_T6_P7}" "${task}" coverage_main
  done

  # Pair #7 only lacks two fixed tmath cells, but looping the block is safely resumable.
  for window in ${WINDOWS}; do
    for ratio in ${RATIOS}; do
      run_rekv "${GPU0}" "${MODEL_P7_A}" "${MODEL_P7_B}" \
        "${ROOT_T1_P7}" tmath "${window}" "${ratio}"
    done
  done
  for task in countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath; do
    run_brekv "${GPU0}" "${MODEL_P7_A}" "${MODEL_P7_B}" \
      "${ROOT_T1_P7}" "${task}" coverage_frozen
  done

  run_cost_and_oracle_pair "${GPU0}" pair7_qwen25_uncensored_bespoke \
    "${MODEL_P7_A}" "${MODEL_P7_B}"
}

gpu1_worker() {
  local task
  local pair5_model
  pair5_model=$(pair5_runtime_model)

  # Split pair #5 by task; the other machine owns the remaining four tasks.
  for task in countries tipsheets hotpotqa qasper; do
    run_standard_block "${GPU1}" "${pair5_model}" "${MODEL_P5_B}" \
      "${ROOT_T8_P5}" "${task}" coverage_frozen
  done

  for task in samsum qasper_full musique_full; do
    run_standard_block "${GPU1}" "${MODEL_P7_A}" "${MODEL_P7_B}" \
      "${ROOT_T6_P7}" "${task}" coverage_main
  done

  for task in countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath; do
    run_brekv "${GPU1}" "${MODEL_LLAMA31}" "${MODEL_LLAMA31}" \
      "${ROOT_T1_P1}" "${task}" coverage_frozen
  done

  run_cost_and_oracle_pair "${GPU1}" pair1_llama31_same \
    "${MODEL_LLAMA31}" "${MODEL_LLAMA31}"
}

TAG=$(date '+%m%d_%H%M')
LOG0="${LOG_ROOT}/gpu${GPU0}_remaining_fast_${TAG}.log"
LOG1="${LOG_ROOT}/gpu${GPU1}_remaining_fast_${TAG}.log"

status0=0
status1=0
if [[ "${WORKERS}" == "both" || "${WORKERS}" == "gpu0" ]]; then
  (gpu0_worker) > "${LOG0}" 2>&1 &
  PID0=$!
  echo "GPU${GPU0} pid=${PID0} -> ${LOG0}"
fi
if [[ "${WORKERS}" == "both" || "${WORKERS}" == "gpu1" ]]; then
  (gpu1_worker) > "${LOG1}" 2>&1 &
  PID1=$!
  echo "GPU${GPU1} pid=${PID1} -> ${LOG1}"
fi

if [[ -n "${PID0:-}" ]]; then
  wait "${PID0}" || status0=$?
fi
if [[ -n "${PID1:-}" ]]; then
  wait "${PID1}" || status1=$?
fi

if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
  echo "Fast-node queue failed: workers=${WORKERS}, GPU${GPU0} status=${status0}, GPU${GPU1} status=${status1}" >&2
  exit 1
fi

echo "Fast-node workers=${WORKERS} queue completed."
