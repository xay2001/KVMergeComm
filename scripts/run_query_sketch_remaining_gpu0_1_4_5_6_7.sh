#!/usr/bin/env bash
set -Eeuo pipefail

# Six-GPU queue for the former R0-R3 workload.
# GPUs: 0, 1, 4, 5, 6, 7
# Mandatory matrix: 298 runs. Optional Pair #6 Strict: 9 runs.
#
# Canonical paper method:
#   Query-Sketch BF16 + calibrated B-ReKV tau=0.95, scale=0.75, window=8.
#
# Usage:
#   bash scripts/run_query_sketch_remaining_gpu0_1_4_5_6_7.sh
#   RUN_STRICT=1 bash scripts/run_query_sketch_remaining_gpu0_1_4_5_6_7.sh
#   DRY_RUN=1 bash scripts/run_query_sketch_remaining_gpu0_1_4_5_6_7.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
cd "${PROJECT_ROOT}"

GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
GPU4=${GPU4:-4}
GPU5=${GPU5:-5}
GPU6=${GPU6:-6}
GPU7=${GPU7:-7}

if [[ -z "${PYTHON:-}" ]]; then
  for candidate in \
    /home/xay/.conda/envs/kvcomm/bin/python \
    /home/xay/miniconda3/envs/ReKV/bin/python; do
    if [[ -x "${candidate}" ]]; then
      PYTHON="${candidate}"
      break
    fi
  done
fi
PYTHON=${PYTHON:-python}

if [[ -z "${MODEL_ROOT:-}" ]]; then
  for candidate in /sharedspace/models /NAS/models; do
    if [[ -d "${candidate}" ]]; then
      MODEL_ROOT="${candidate}"
      break
    fi
  done
fi
MODEL_ROOT=${MODEL_ROOT:-/sharedspace/models}

LIMIT=${LIMIT:-0}
PROFILE_LIMIT=${PROFILE_LIMIT:-50}
PROFILE_WARMUP=${PROFILE_WARMUP:-3}
SKIP_EXISTING=${SKIP_EXISTING:-1}
RUN_STRICT=${RUN_STRICT:-0}
DRY_RUN=${DRY_RUN:-0}
RUN_SUMMARY=${RUN_SUMMARY:-1}

CANONICAL_TAU=${CANONICAL_TAU:-0.95}
CANONICAL_SCALE=${CANONICAL_SCALE:-0.75}
CANONICAL_WINDOW=${CANONICAL_WINDOW:-8}
REKV_WINDOWS=${REKV_WINDOWS:-"8 16"}
REKV_RATIOS=${REKV_RATIOS:-"0.3 0.5 0.7"}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_A_2=${MODEL_A_2:-"${MODEL_ROOT}/Llama-3.2-3B-Instruct"}
MODEL_B_2=${MODEL_B_2:-"${MODEL_A_2}"}
MODEL_A_3=${MODEL_A_3:-"${MODEL_ROOT}/Qwen2.5-7B-Instruct"}
MODEL_B_3=${MODEL_B_3:-"${MODEL_A_3}"}
MODEL_A_4=${MODEL_A_4:-"${MODEL_ROOT}/Falcon3-7B-Instruct"}
MODEL_B_4=${MODEL_B_4:-"${MODEL_A_4}"}
MODEL_A_5=${MODEL_A_5:-"${MODEL_ROOT}/EvolCodeLlama-3.1-8B-Instruct"}
MODEL_B_5=${MODEL_B_5:-"${MODEL_ROOT}/ToolACE-2-Llama-3.1-8B"}
MODEL_A_6=${MODEL_A_6:-"${MODEL_ROOT}/Llama-3.2-3B-Instruct-abliterated"}
MODEL_B_6=${MODEL_B_6:-"${MODEL_ROOT}/DeepSeek-R1-Distill-Llama-3B"}

ROOT_T8_2=${ROOT_T8_2:-snapshots/table8_pair2_query_sketch_llama32_same}
ROOT_T8_3=${ROOT_T8_3:-snapshots/table8_pair3_query_sketch_qwen25_7b_same}
ROOT_T8_4=${ROOT_T8_4:-snapshots/table8_pair4_query_sketch_falcon3_7b_same}
ROOT_T8_5=${ROOT_T8_5:-snapshots/table8_pair5_query_sketch_evolcodellama_toolace}
ROOT_T1_6=${ROOT_T1_6:-snapshots/table1_pair6_query_sketch_llama32_abliterated_deepseek3b}
ROOT_T6_6=${ROOT_T6_6:-snapshots/table6_pair6_query_sketch_llama32_abliterated_deepseek3b}
ROOT_COST=${ROOT_COST:-snapshots/query_sketch_cost_v1}
ROOT_ORACLE=${ROOT_ORACLE:-snapshots/query_sketch_oracle_gap_canonical_t095_s075_w8}
ROOT_STRICT=${ROOT_STRICT:-snapshots/stage3_core_reviewer_query_sketch/pair6_llama32_abliterated_deepseek3b}
LOG_ROOT=${LOG_ROOT:-snapshots/logs/query_sketch_remaining_gpu0_1_4_5_6_7}

PAIR5_RUNTIME_MODEL=""
WORKER_PIDS=()

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

quote_command() {
  printf '%q ' "$@"
  printf '\n'
}

terminate_tree() {
  local pid=$1 child
  [[ -n "${pid}" ]] || return 0
  while read -r child; do
    [[ -n "${child}" ]] && terminate_tree "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill -TERM "${pid}" 2>/dev/null || true
}

cleanup() {
  local status=$? pid
  trap - EXIT INT TERM
  if (( status != 0 )); then
    log "queue interrupted (status=${status}); terminating worker trees"
    for pid in "${WORKER_PIDS[@]:-}"; do
      terminate_tree "${pid}"
    done
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

require_path() {
  local path=$1
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  [[ -e "${path}" ]] || {
    echo "Required path does not exist: ${path}" >&2
    return 2
  }
}

prepare_pair5_model() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    PAIR5_RUNTIME_MODEL="${MODEL_A_5}"
    return
  fi
  if [[ -f "${MODEL_A_5}/adapter_config.json" &&
        -f "${MODEL_A_5}/model.safetensors.index.json" ]]; then
    local view="${ROOT_T8_5}/model_views/$(basename "${MODEL_A_5}")_no_adapter"
    local file base
    mkdir -p "${view}"
    for file in "${MODEL_A_5}"/*; do
      base=$(basename "${file}")
      case "${base}" in
        adapter_config.json|adapter_model.bin|adapter_model.safetensors) continue ;;
      esac
      ln -sfn "${file}" "${view}/${base}"
    done
    PAIR5_RUNTIME_MODEL="${view}"
  else
    PAIR5_RUNTIME_MODEL="${MODEL_A_5}"
  fi
}

has_valid_artifact() {
  local search_root=$1 run_prefix=$2 artifact=$3 expected_protocol=$4
  [[ "${SKIP_EXISTING}" == "1" && "${DRY_RUN}" != "1" ]] || return 1
  "${PYTHON}" - "${search_root}" "${run_prefix}" "${artifact}" "${expected_protocol}" <<'PY'
import glob
import json
import os
import sys

root, prefix, artifact, expected = sys.argv[1:]
patterns = [
    os.path.join(root, "**", f"{prefix}_*", artifact),
    os.path.join(root, f"{prefix}_*", artifact),
]
paths = sorted({path for pattern in patterns for path in glob.glob(pattern, recursive=True)})
for path in paths:
    try:
        if artifact.endswith(".jsonl"):
            meta = {}
            with open(path) as handle:
                for line in handle:
                    row = json.loads(line)
                    if "_meta" in row:
                        meta = row["_meta"]
                        break
        else:
            with open(path) as handle:
                meta = json.load(handle).get("_meta", {})
        if meta.get("protocol_version") == expected:
            raise SystemExit(0)
    except (OSError, ValueError, json.JSONDecodeError):
        continue
raise SystemExit(1)
PY
}

run_python() {
  local gpu=$1
  shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[run][GPU%s] ' "${gpu}"
    quote_command "${PYTHON}" "$@"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "$@"
}

run_rekv() {
  local gpu=$1 model_a=$2 model_b=$3 root=$4 task=$5 window=$6 ratio=$7
  local out="${root}/${task}/mtc_receiver"
  local run_name="recv_w${window}_r${ratio}"
  if has_valid_artifact "${out}" "${run_name}" per_sample.jsonl query_sketch_bf16_v1; then
    log "[skip][GPU${gpu}] ${root} ${task} ${run_name}"
    return
  fi
  log "[run][GPU${gpu}] ${root} ${task} ${run_name}"
  run_python "${gpu}" com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16 \
    --recv_window "${window}" --merge_ratio "${ratio}" \
    --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" --run_name "${run_name}"
}

canonical_brekv_done() {
  local root=$1 task=$2
  local prefix="cov_t${CANONICAL_TAU}_s${CANONICAL_SCALE}_w${CANONICAL_WINDOW}"
  has_valid_artifact "${root}/${task}" "${prefix}" per_sample.jsonl query_sketch_bf16_v1 ||
    has_valid_artifact "${root}/${task}" \
      "frozen_t${CANONICAL_TAU}_s${CANONICAL_SCALE}_w${CANONICAL_WINDOW}" \
      per_sample.jsonl query_sketch_bf16_v1
}

run_brekv() {
  local gpu=$1 model_a=$2 model_b=$3 root=$4 task=$5
  local out="${root}/${task}/coverage_main"
  local run_name="cov_t${CANONICAL_TAU}_s${CANONICAL_SCALE}_w${CANONICAL_WINDOW}"
  if canonical_brekv_done "${root}" "${task}"; then
    log "[skip][GPU${gpu}] ${root} ${task} ${run_name}"
    return
  fi
  log "[run][GPU${gpu}] ${root} ${task} ${run_name}"
  run_python "${gpu}" com.py \
    --test_task "${task}" --do_test --limit "${LIMIT}" \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16 \
    --recv_window "${CANONICAL_WINDOW}" --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage --budget_min 0.05 --budget_max 0.70 \
    --coverage_tau "${CANONICAL_TAU}" --coverage_scale "${CANONICAL_SCALE}" \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_seven_point_task() {
  local gpu=$1 model_a=$2 model_b=$3 root=$4 task=$5 window ratio
  for window in ${REKV_WINDOWS}; do
    for ratio in ${REKV_RATIOS}; do
      run_rekv "${gpu}" "${model_a}" "${model_b}" "${root}" "${task}" "${window}" "${ratio}"
    done
  done
  run_brekv "${gpu}" "${model_a}" "${model_b}" "${root}" "${task}"
}

run_six_fixed_task() {
  local gpu=$1 model_a=$2 model_b=$3 root=$4 task=$5 window ratio
  for window in ${REKV_WINDOWS}; do
    for ratio in ${REKV_RATIOS}; do
      run_rekv "${gpu}" "${model_a}" "${model_b}" "${root}" "${task}" "${window}" "${ratio}"
    done
  done
}

run_cost_method() {
  local gpu=$1 task=$2 method=$3
  local pair=pair6_llama32_abliterated_deepseek3b
  local out="${ROOT_COST}/${pair}/${task}/${method}" run_name
  if [[ "${method}" == "rekv" ]]; then
    run_name="rekv_bf16_w8_r0.30"
  else
    run_name="brekv_bf16_t${CANONICAL_TAU}_s${CANONICAL_SCALE}_w${CANONICAL_WINDOW}"
  fi
  if has_valid_artifact "${out}" "${run_name}" cost_summary.json query_sketch_bf16_v1; then
    log "[skip][GPU${gpu}] cost ${pair} ${task} ${run_name}"
    return
  fi
  local args=(
    com.py --test_task "${task}" --do_test
    --profile_cost --profile_limit "${PROFILE_LIMIT}" --profile_warmup "${PROFILE_WARMUP}"
    --model_A "${MODEL_A_6}" --model_B "${MODEL_B_6}"
    --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16
    --merge_sink 4 --merge_recent 8
    --snapshot_path "${out}" --run_name "${run_name}"
  )
  if [[ "${method}" == "rekv" ]]; then
    args+=(--recv_window 8 --merge_ratio 0.30)
  else
    args+=(
      --recv_window "${CANONICAL_WINDOW}"
      --budget_mode coverage --budget_min 0.05 --budget_max 0.70
      --coverage_tau "${CANONICAL_TAU}" --coverage_scale "${CANONICAL_SCALE}"
    )
  fi
  log "[run][GPU${gpu}] cost ${pair} ${task} ${run_name}"
  run_python "${gpu}" "${args[@]}"
}

run_pair6_cost() {
  local gpu=$1 task method
  for task in hotpotqa musique multifieldqa_en; do
    for method in rekv brekv; do
      run_cost_method "${gpu}" "${task}" "${method}"
    done
  done
}

run_oracle_brekv() {
  local gpu=$1 task=$2 score_mode=$3
  local pair=pair6_llama32_abliterated_deepseek3b
  local out="${ROOT_ORACLE}/${pair}/${task}/brekv"
  local run_name="brekv_${score_mode}_t${CANONICAL_TAU}_s${CANONICAL_SCALE}_w${CANONICAL_WINDOW}"
  local protocol=query_sketch_bf16_v1
  [[ "${score_mode}" == "receiver_oracle" ]] && protocol=full_kv_oracle_v1
  if has_valid_artifact "${out}" "${run_name}" cost_summary.json "${protocol}"; then
    log "[skip][GPU${gpu}] oracle ${pair} ${task} ${run_name}"
    return
  fi
  log "[run][GPU${gpu}] oracle ${pair} ${task} ${run_name}"
  run_python "${gpu}" com.py \
    --test_task "${task}" --do_test \
    --profile_cost --profile_limit "${PROFILE_LIMIT}" --profile_warmup "${PROFILE_WARMUP}" \
    --model_A "${MODEL_A_6}" --model_B "${MODEL_B_6}" \
    --merge --merge_mode evict --score_mode "${score_mode}" --query_sketch_mode bf16 \
    --recv_window "${CANONICAL_WINDOW}" --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage --budget_min 0.05 --budget_max 0.70 \
    --coverage_tau "${CANONICAL_TAU}" --coverage_scale "${CANONICAL_SCALE}" \
    --snapshot_path "${out}" --run_name "${run_name}"
}

run_pair6_oracle_gap() {
  local gpu=$1 task mode
  for task in hotpotqa musique multifieldqa_en; do
    for mode in receiver receiver_oracle; do
      run_oracle_brekv "${gpu}" "${task}" "${mode}"
    done
  done
}

run_strict_pair6() {
  local gpu=$1 task spec window tau_min tau_max out run_name
  [[ "${RUN_STRICT}" == "1" ]] || return 0
  for task in hotpotqa musique multifieldqa_en; do
    for spec in "8:0.70:0.90" "8:0.80:0.95" "16:0.80:0.95"; do
      IFS=: read -r window tau_min tau_max <<< "${spec}"
      out="${ROOT_STRICT}/${task}/strict_coverage"
      run_name="strict_adapt_t${tau_min}-${tau_max}_w${window}"
      if has_valid_artifact "${out}" "${run_name}" per_sample.jsonl query_sketch_bf16_v1; then
        log "[skip][GPU${gpu}] strict pair6 ${task} ${run_name}"
        continue
      fi
      log "[run][GPU${gpu}] strict pair6 ${task} ${run_name}"
      run_python "${gpu}" com.py \
        --test_task "${task}" --do_test --limit "${LIMIT}" \
        --model_A "${MODEL_A_6}" --model_B "${MODEL_B_6}" \
        --merge --merge_mode evict --score_mode receiver --query_sketch_mode bf16 \
        --recv_window "${window}" --merge_sink 4 --merge_recent 8 \
        --budget_mode strict_coverage --coverage_tau_mode adaptive \
        --coverage_tau_min "${tau_min}" --coverage_tau_max "${tau_max}" \
        --budget_min 0.0 --budget_max 1.0 --budget_floor 0.0 \
        --snapshot_path "${out}" --run_name "${run_name}"
    done
  done
}

worker_gpu0() {
  local task
  for task in countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath; do
    run_seven_point_task "${GPU0}" "${MODEL_A_4}" "${MODEL_B_4}" "${ROOT_T8_4}" "${task}"
  done
  run_pair6_cost "${GPU0}"
  run_pair6_oracle_gap "${GPU0}"
}

worker_gpu1() {
  local task
  for task in countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath; do
    run_seven_point_task "${GPU1}" "${MODEL_A_3}" "${MODEL_B_3}" "${ROOT_T8_3}" "${task}"
  done
}

worker_gpu4() {
  local task
  for task in countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath; do
    run_seven_point_task "${GPU4}" "${MODEL_A_2}" "${MODEL_B_2}" "${ROOT_T8_2}" "${task}"
  done
}

worker_gpu5() {
  local task
  for task in musique multifieldqa_en twowikimqa tmath; do
    run_seven_point_task "${GPU5}" "${PAIR5_RUNTIME_MODEL}" "${MODEL_B_5}" "${ROOT_T8_5}" "${task}"
  done
  for task in repobench hotpotqa_full; do
    run_seven_point_task "${GPU5}" "${MODEL_A_6}" "${MODEL_B_6}" "${ROOT_T6_6}" "${task}"
  done
}

worker_gpu6() {
  local task
  for task in countries tipsheets hotpotqa qasper; do
    run_seven_point_task "${GPU6}" "${MODEL_A_6}" "${MODEL_B_6}" "${ROOT_T1_6}" "${task}"
  done
  for task in samsum qasper_full; do
    run_seven_point_task "${GPU6}" "${MODEL_A_6}" "${MODEL_B_6}" "${ROOT_T6_6}" "${task}"
  done
}

worker_gpu7() {
  local task
  for task in musique multifieldqa_en twowikimqa; do
    run_seven_point_task "${GPU7}" "${MODEL_A_6}" "${MODEL_B_6}" "${ROOT_T1_6}" "${task}"
  done
  # Pair #6 tmath already has a canonical v1 B-ReKV result; only refresh fixed ReKV.
  run_six_fixed_task "${GPU7}" "${MODEL_A_6}" "${MODEL_B_6}" "${ROOT_T1_6}" tmath
  run_seven_point_task "${GPU7}" "${MODEL_A_6}" "${MODEL_B_6}" "${ROOT_T6_6}" musique_full
  run_strict_pair6 "${GPU7}"
}

verify_batch() {
  [[ "${DRY_RUN}" != "1" ]] || return
  log "running protocol-aware batch verification"
  "${PYTHON}" - \
    "${ROOT_T8_2}" "${ROOT_T8_3}" "${ROOT_T8_4}" "${ROOT_T8_5}" \
    "${ROOT_T1_6}" "${ROOT_T6_6}" "${ROOT_COST}" "${ROOT_ORACLE}" <<'PY'
import glob
import json
import os
import sys

roots = sys.argv[1:]
counts = {}
for root in roots:
    protocols = {}
    files = glob.glob(os.path.join(root, "**", "per_sample.jsonl"), recursive=True)
    files += glob.glob(os.path.join(root, "**", "cost_summary.json"), recursive=True)
    for path in files:
        try:
            if path.endswith(".jsonl"):
                meta = {}
                with open(path) as handle:
                    for line in handle:
                        row = json.loads(line)
                        if "_meta" in row:
                            meta = row["_meta"]
                            break
            else:
                with open(path) as handle:
                    meta = json.load(handle).get("_meta", {})
            protocol = meta.get("protocol_version", "missing")
            protocols[protocol] = protocols.get(protocol, 0) + 1
        except Exception:
            protocols["unreadable"] = protocols.get("unreadable", 0) + 1
    counts[root] = protocols

print("# Six-GPU Query-Sketch batch protocol counts")
for root, protocols in counts.items():
    print(f"{root}: {protocols}")

allowed = {
    "query_sketch_bf16_v1",
    "full_kv_oracle_v1",
    "query_sketch_bf16_multi_source_v1",
}
bad = []
for root, protocols in counts.items():
    for protocol, count in protocols.items():
        if protocol not in allowed and count:
            bad.append((root, protocol, count))
if bad:
    print("NOTICE: roots also contain pre-existing non-v1 artifacts; summaries must filter by protocol:")
    for item in bad:
        print("  ", item)
PY
}

run_summaries() {
  [[ "${RUN_SUMMARY}" == "1" && "${DRY_RUN}" != "1" ]] || return
  mkdir -p "${ROOT_COST}"
  "${PYTHON}" scripts/analyze_cost_profile.py \
    --root "${ROOT_COST}" --csv "${ROOT_COST}/cost_table.csv" \
    > "${ROOT_COST}/cost_table.md"
  "${PYTHON}" scripts/build_experiment_manifest.py \
    --snapshots snapshots --out-dir snapshots/manifest
  "${PYTHON}" scripts/summarize_query_sketch_rerun.py
}

preflight() {
  require_path "${PYTHON}"
  require_path "${MODEL_A_2}"
  require_path "${MODEL_B_2}"
  require_path "${MODEL_A_3}"
  require_path "${MODEL_B_3}"
  require_path "${MODEL_A_4}"
  require_path "${MODEL_B_4}"
  require_path "${MODEL_A_5}"
  require_path "${MODEL_B_5}"
  require_path "${MODEL_A_6}"
  require_path "${MODEL_B_6}"
  prepare_pair5_model
}

launch_worker() {
  local gpu=$1 name=$2 function_name=$3
  local log_file="${LOG_ROOT}/gpu${gpu}_${name}_${TAG}.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    "${function_name}"
    return
  fi
  ("${function_name}") > "${log_file}" 2>&1 &
  local pid=$!
  WORKER_PIDS+=("${pid}")
  log "GPU${gpu} pid=${pid} -> ${log_file}"
}

main() {
  preflight
  mkdir -p "${LOG_ROOT}"
  TAG=$(date '+%m%d_%H%M')
  export TAG

  log "mandatory plan: GPU0=68, GPU1=56, GPU4=56, GPU5=42, GPU6=42, GPU7=34; total=298"
  [[ "${RUN_STRICT}" == "1" ]] && log "optional Strict pair #6: +9 runs on GPU7"

  if [[ "${DRY_RUN}" == "1" ]]; then
    launch_worker "${GPU0}" pair4_cost_oracle worker_gpu0
    launch_worker "${GPU1}" pair3 worker_gpu1
    launch_worker "${GPU4}" pair2 worker_gpu4
    launch_worker "${GPU5}" pair5_table6_heavy worker_gpu5
    launch_worker "${GPU6}" pair6_short worker_gpu6
    launch_worker "${GPU7}" pair6_long worker_gpu7
    return
  fi

  launch_worker "${GPU0}" pair4_cost_oracle worker_gpu0
  launch_worker "${GPU1}" pair3 worker_gpu1
  launch_worker "${GPU4}" pair2 worker_gpu4
  launch_worker "${GPU5}" pair5_table6_heavy worker_gpu5
  launch_worker "${GPU6}" pair6_short worker_gpu6
  launch_worker "${GPU7}" pair6_long worker_gpu7

  local status=0 pid
  for pid in "${WORKER_PIDS[@]}"; do
    wait "${pid}" || status=$?
  done
  (( status == 0 )) || return "${status}"

  verify_batch
  run_summaries
  log "six-GPU Query-Sketch batch complete"
}

main "$@"
