#!/usr/bin/env bash
set -euo pipefail

# Stage 5 appendix expansion under the deployable Query-Sketch protocol.
#
# PHASE=all (default):
#   freeze  -> freeze one global B-ReKV config if selection.json is absent
#   table6  -> pair #6/#7, five extended tasks, paper-style 9-run matrix
#   table8  -> pair #2/#3/#4/#5, six ReKV points + one frozen B-ReKV point
#   table10 -> true multi-source Query-Sketch ReKV
#
# Individual phases are resumable:
#   PHASE=freeze|table6|table8|table10 bash scripts/run_stage5_appendix_query_sketch_gpu2_3.sh

cd /home/xay/KVMergeComm || exit 1

GPU2=${GPU2:-2}
GPU3=${GPU3:-3}
PYTHON=${PYTHON:-/home/xay/miniconda3/envs/ReKV/bin/python}
PHASE=${PHASE:-all}
LIMIT=${LIMIT:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SELECTION=${SELECTION:-snapshots/query_sketch_config_freeze/analysis/main_config.json}
LOG_ROOT=${LOG_ROOT:-snapshots/stage5_appendix_query_sketch/logs}

MAIN_TASKS=${MAIN_TASKS:-"countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"}
EXTENDED_TASKS=${EXTENDED_TASKS:-"samsum qasper_full musique_full repobench hotpotqa_full"}
MULTI_SOURCE_TASKS=${MULTI_SOURCE_TASKS:-"hotpotqa musique twowikimqa"}
RATIOS=${RATIOS:-"0.3 0.5 0.7"}
WINDOWS=${WINDOWS:-"8 16"}

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

ROOT_FREEZE=${ROOT_FREEZE:-snapshots/query_sketch_config_freeze}
ROOT_T6_6=${ROOT_T6_6:-snapshots/table6_pair6_query_sketch_llama32_abliterated_deepseek3b}
ROOT_T6_7=${ROOT_T6_7:-snapshots/table6_pair7_query_sketch_qwen25_uncensored_bespoke}
ROOT_T8_2=${ROOT_T8_2:-snapshots/table8_pair2_query_sketch_llama32_same}
ROOT_T8_3=${ROOT_T8_3:-snapshots/table8_pair3_query_sketch_qwen25_7b_same}
ROOT_T8_4=${ROOT_T8_4:-snapshots/table8_pair4_query_sketch_falcon3_7b_same}
ROOT_T8_5=${ROOT_T8_5:-snapshots/table8_pair5_query_sketch_evolcodellama_toolace}
ROOT_T10=${ROOT_T10:-snapshots/table10_multi_source_query_sketch}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

pair_model_a() { eval "echo \${MODEL_A_$1}"; }
pair_model_b() { eval "echo \${MODEL_B_$1}"; }
pair_root_t6() { eval "echo \${ROOT_T6_$1}"; }
pair_root_t8() { eval "echo \${ROOT_T8_$1}"; }

check_model() {
  [[ -d "$1" ]] || { echo "Model path does not exist: $1" >&2; return 2; }
}

has_done_run() {
  local out=$1 run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] &&
    compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

pair5_runtime_model() {
  local view="${ROOT_T8_5}/model_views/$(basename "${MODEL_A_5}")_no_adapter"
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

run_single_rekv() {
  local gpu=$1 model_a=$2 model_b=$3 root=$4 task=$5 win=$6 ratio=$7
  local out="${root}/${task}/mtc_receiver" run_name="recv_w${win}_r${ratio}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${root} ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] ${root} ${task} ${run_name} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16 \
    --recv_window "${win}" --merge_ratio "${ratio}" \
    --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_single_brekv() {
  local gpu=$1 model_a=$2 model_b=$3 root=$4 task=$5 win=$6 tau=$7 scale=$8 kind=${9:-coverage}
  local out="${root}/${task}/${kind}" run_name
  if [[ "${kind}" == "coverage_frozen" ]]; then
    run_name="frozen_t${tau}_s${scale}_w${win}"
  else
    run_name="cov_t${tau}_s${scale}_w${win}"
  fi
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${root} ${task} ${run_name} ===="
    return
  fi
  echo "==== [GPU${gpu}] ${root} ${task} ${run_name} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16 \
    --recv_window "${win}" --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage --budget_min 0.05 --budget_max 0.70 \
    --coverage_tau "${tau}" --coverage_scale "${scale}" \
    --snapshot_path "${out}" --run_name "${run_name}"
}

selection_is_accepted() {
  [[ -f "${SELECTION}" ]] && "${PYTHON}" - "${SELECTION}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
raise SystemExit(0 if data.get("accepted") and data.get("selection") else 1)
PY
}

load_frozen_config() {
  if ! selection_is_accepted; then
    echo "No accepted frozen config: ${SELECTION}" >&2
    return 2
  fi
  read -r FROZEN_TAU FROZEN_SCALE FROZEN_WINDOW < <(
    "${PYTHON}" - "${SELECTION}" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))["selection"]
print(row["tau"], row["scale"], row["window"])
PY
  )
  export FROZEN_TAU FROZEN_SCALE FROZEN_WINDOW
  echo "Frozen B-ReKV: tau=${FROZEN_TAU} scale=${FROZEN_SCALE} window=${FROZEN_WINDOW}"
}

freeze_worker() {
  local gpu=$1 pair=$2 model_a model_b pair_name task ratio candidate tau scale
  local LIMIT=100
  model_a=$(pair_model_a "${pair}")
  model_b=$(pair_model_b "${pair}")
  pair_name="pair${pair}_$( [[ "${pair}" == "1" ]] && echo llama31_same || echo llama32_abliterated_deepseek3b )"
  for task in hotpotqa musique multifieldqa_en; do
    for ratio in 0.20 0.25 0.30 0.35 0.40; do
      run_single_rekv "${gpu}" "${model_a}" "${model_b}" \
        "${ROOT_FREEZE}/${pair_name}" "${task}" 8 "${ratio}"
    done
    for candidate in 0.90:0.95 0.95:0.85 0.95:0.95 0.95:1.00 0.98:0.95 0.98:1.00; do
      IFS=: read -r tau scale <<< "${candidate}"
      local out="${ROOT_FREEZE}/${pair_name}/${task}/coverage"
      local run_name="brekv_t${tau}_s${scale}_w8"
      if has_done_run "${out}" "${run_name}"; then
        echo "==== [skip] ${pair_name} ${task} ${run_name} ===="
        continue
      fi
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com.py \
        --test_task "${task}" --do_test --limit 100 \
        --model_A "${model_a}" --model_B "${model_b}" \
        --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16 \
        --recv_window 8 --merge_sink 4 --merge_recent 8 \
        --budget_mode coverage --budget_min 0.05 --budget_max 0.70 \
        --coverage_tau "${tau}" --coverage_scale "${scale}" \
        --snapshot_path "${out}" --run_name "${run_name}"
    done
  done
}

run_freeze_phase() {
  if selection_is_accepted; then
    echo "Accepted frozen config already exists: ${SELECTION}"
    return
  fi
  launch_two freeze freeze_worker 1 6
  "${PYTHON}" scripts/analyze_query_sketch_config_freeze.py --root "${ROOT_FREEZE}"
  load_frozen_config
}

table6_worker() {
  local gpu=$1 pair=$2 model_a model_b root task win ratio
  model_a=$(pair_model_a "${pair}"); model_b=$(pair_model_b "${pair}"); root=$(pair_root_t6 "${pair}")
  check_model "${model_a}"; check_model "${model_b}"
  for task in ${EXTENDED_TASKS}; do
    for win in ${WINDOWS}; do
      for ratio in ${RATIOS}; do
        run_single_rekv "${gpu}" "${model_a}" "${model_b}" "${root}" "${task}" "${win}" "${ratio}"
      done
    done
    run_single_brekv "${gpu}" "${model_a}" "${model_b}" "${root}" "${task}" 8 0.95 0.75 coverage_main
  done
}

table8_worker() {
  local gpu=$1 pairs=$2 pair model_a model_b root task win ratio
  load_frozen_config
  for pair in ${pairs}; do
    model_a=$(pair_model_a "${pair}"); model_b=$(pair_model_b "${pair}"); root=$(pair_root_t8 "${pair}")
    [[ "${pair}" == "5" ]] && model_a=$(pair5_runtime_model)
    check_model "${model_a}"; check_model "${model_b}"
    for task in ${MAIN_TASKS}; do
      for win in ${WINDOWS}; do
        for ratio in ${RATIOS}; do
          run_single_rekv "${gpu}" "${model_a}" "${model_b}" "${root}" "${task}" "${win}" "${ratio}"
        done
      done
      run_single_brekv "${gpu}" "${model_a}" "${model_b}" "${root}" "${task}" \
        "${FROZEN_WINDOW}" "${FROZEN_TAU}" "${FROZEN_SCALE}" coverage_frozen
    done
  done
}

table10_worker() {
  local gpu=$1 tasks=$2 task win ratio
  check_model "${MODEL_A_1}"
  for task in ${tasks}; do
    for win in ${WINDOWS}; do
      for ratio in ${RATIOS}; do
        local out="${ROOT_T10}/${task}/multi_source_rekv"
        local run_name="ms_qs_bf16_w${win}_r${ratio}"
        if has_done_run "${out}" "${run_name}"; then
          echo "==== [skip] Table10 ${task} ${run_name} ===="
          continue
        fi
        echo "==== [GPU${gpu}] Table10 ${task} ${run_name} $(date '+%F %T') ===="
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" com_ms.py \
          --test_task "${task}" --do_test --limit "${LIMIT}" \
          --model_A1 "${MODEL_A_1}" --model_A2 "${MODEL_A_1}" --model_B "${MODEL_B_1}" \
          --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16 \
          --recv_window "${win}" --merge_ratio "${ratio}" \
          --merge_sink 4 --merge_recent 8 \
          --snapshot_path "${out}" --run_name "${run_name}"
      done
    done
  done
}

launch_two() {
  local label=$1 worker=$2 queue2=$3 queue3=$4 tag log2 log3 pid2 pid3 status=0
  mkdir -p "${LOG_ROOT}"
  tag=$(date '+%m%d_%H%M')
  log2="${LOG_ROOT}/gpu${GPU2}_${label}_${tag}.log"
  log3="${LOG_ROOT}/gpu${GPU3}_${label}_${tag}.log"
  ("${worker}" "${GPU2}" "${queue2}") > "${log2}" 2>&1 & pid2=$!
  ("${worker}" "${GPU3}" "${queue3}") > "${log3}" 2>&1 & pid3=$!
  echo "${label}: GPU${GPU2} pid=${pid2} -> ${log2}"
  echo "${label}: GPU${GPU3} pid=${pid3} -> ${log3}"
  wait "${pid2}" || status=$?
  wait "${pid3}" || status=$?
  return "${status}"
}

run_table6_phase() { launch_two table6 table6_worker 6 7; }
run_table8_phase() { launch_two table8 table8_worker "2 4" "3 5"; }
run_table10_phase() { launch_two table10 table10_worker "hotpotqa musique" "twowikimqa"; }

case "${PHASE}" in
  freeze) run_freeze_phase ;;
  table6) run_table6_phase ;;
  table8) load_frozen_config; run_table8_phase ;;
  table10) run_table10_phase ;;
  all)
    run_freeze_phase
    run_table6_phase
    run_table8_phase
    run_table10_phase
    ;;
  *)
    echo "Unknown PHASE=${PHASE}; expected all|freeze|table6|table8|table10" >&2
    exit 2
    ;;
esac
