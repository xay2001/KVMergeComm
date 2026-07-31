#!/usr/bin/env bash
set -euo pipefail

# Canonical, resumable n=8 model-compatibility diagnostic.
#
# Examples:
#   PAIRS="1 2 3 4 5 6 7 9" TASKS="hotpotqa" LIMIT=4 GPU=0 \
#     bash scripts/run_model_compatibility_gpu.sh
#   PAIRS="2 3 4 5 6 9" RUN_ATTENTION=0 LIMIT=20 GPU=0 \
#     bash scripts/run_model_compatibility_gpu.sh
#
# LIMIT controls both Full-KV evaluation samples and attention prompts. Keep it
# small for diagnostics. MAX_LENGTH separately caps attention's quadratic cost.

cd /home/xay/KVComm || exit 1

PYTHON=${PYTHON:-python}
GPU=${GPU:-0}
PAIRS=${PAIRS:-"1 2 3 4 5 6 7 9"}
TASKS=${TASKS:-"hotpotqa"}
LIMIT=${LIMIT:-4}
MAX_LENGTH=${MAX_LENGTH:-512}
TOPK=${TOPK:-32}
ROOT=${ROOT:-snapshots/model_compatibility_diagnostic}
RUN_FULL_KV=${RUN_FULL_KV:-1}
RUN_ATTENTION=${RUN_ATTENTION:-1}
FORCE_FULL_KV=${FORCE_FULL_KV:-0}

MODEL_A_1=${MODEL_A_1:-/sharedspace/models/Llama-3.1-8B-Instruct}
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
MODEL_A_9=${MODEL_A_9:-/NAS/models/Llama-3.1-SuperNova-Lite}
MODEL_B_9=${MODEL_B_9:-/NAS/models/DeepSeek-R1-Distill-Llama-8B}

pair_slug() {
  case "$1" in
    1) echo "pair1_llama31_same" ;;
    2) echo "pair2_llama32_same" ;;
    3) echo "pair3_qwen25_7b_same" ;;
    4) echo "pair4_falcon3_7b_same" ;;
    5) echo "pair5_evolcodellama_toolace" ;;
    6) echo "pair6_llama32_abliterated_deepseek3b" ;;
    7) echo "pair7_qwen25_uncensored_bespoke" ;;
    9) echo "pair9_supernova_deepseek_llama8b" ;;
    *) echo "Unknown pair: $1" >&2; return 2 ;;
  esac
}

model_a() {
  local variable="MODEL_A_$1"
  echo "${!variable}"
}

model_b() {
  local variable="MODEL_B_$1"
  echo "${!variable}"
}

pair5_runtime_model() {
  local model=$1
  local view="${ROOT}/model_views/$(basename "${model}")_no_adapter"
  if [[ -f "${model}/adapter_config.json" && -f "${model}/model.safetensors.index.json" ]]; then
    mkdir -p "${view}"
    local file base
    for file in "${model}"/*; do
      base=$(basename "${file}")
      case "${base}" in
        adapter_config.json|adapter_model.bin|adapter_model.safetensors) continue ;;
      esac
      ln -sfn "${file}" "${view}/${base}"
    done
    echo "${view}"
  else
    echo "${model}"
  fi
}

last_layer() {
  "${PYTHON}" -c \
    'from transformers import AutoConfig; import sys; c=AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True); print(int(c.num_hidden_layers)-1)' \
    "$1"
}

has_full_kv() {
  local parent=$1
  compgen -G "${parent}/full_kv_*/per_sample.jsonl" > /dev/null
}

run_full_kv() {
  local pair=$1 task=$2 sender=$3 receiver=$4 slug=$5
  local parent="${ROOT}/full_kv/${slug}/${task}/full_kv"
  if [[ "${FORCE_FULL_KV}" != "1" ]] && "${PYTHON}" -c \
    'from pathlib import Path; import sys; raise SystemExit(0 if any(Path("snapshots").glob(f"**/{sys.argv[1]}/{sys.argv[2]}/full_kv/*/per_sample.jsonl")) else 1)' \
    "${slug}" "${task}"; then
    echo "[reuse] existing Full-KV ${slug}/${task}"
    return
  fi
  if has_full_kv "${parent}"; then
    echo "[skip] Full-KV ${slug}/${task}"
    return
  fi
  mkdir -p "${parent}"
  local layer_to
  layer_to=$(last_layer "${sender}")
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" com.py \
    --test_task "${task}" --do_test \
    --model_A "${sender}" --model_B "${receiver}" \
    --limit "${LIMIT}" --layer_from 0 --layer_to "${layer_to}" \
    --snapshot_path "${parent}" --run_name "full_kv"
}

run_attention() {
  local pair=$1 task=$2 sender=$3 receiver=$4
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/model_compatibility_diagnostic.py \
    calibrate-attention \
    --pair "${pair}" --task "${task}" \
    --sender "${sender}" --receiver "${receiver}" \
    --device cuda:0 --limit "${LIMIT}" --max-length "${MAX_LENGTH}" --topk "${TOPK}" \
    --out-dir "${ROOT}/calibration"
}

mkdir -p "${ROOT}/logs"
log="${ROOT}/logs/gpu${GPU}_$(date '+%m%d_%H%M').log"

{
  echo "n=8 exploratory diagnostic only; not a general predictor"
  echo "PAIRS=${PAIRS} TASKS=${TASKS} LIMIT=${LIMIT} MAX_LENGTH=${MAX_LENGTH}"
  for pair in ${PAIRS}; do
    slug=$(pair_slug "${pair}")
    sender=$(model_a "${pair}")
    receiver=$(model_b "${pair}")
    if [[ "${pair}" == "5" ]]; then
      sender=$(pair5_runtime_model "${sender}")
    fi
    if [[ ! -f "${sender}/config.json" || ! -f "${receiver}/config.json" ]]; then
      echo "Missing checkpoint config for pair${pair}: ${sender} / ${receiver}" >&2
      exit 1
    fi
    for task in ${TASKS}; do
      echo "==== pair${pair} ${task} $(date '+%F %T') ===="
      if [[ "${RUN_FULL_KV}" == "1" ]]; then
        run_full_kv "${pair}" "${task}" "${sender}" "${receiver}" "${slug}"
      fi
      if [[ "${RUN_ATTENTION}" == "1" ]]; then
        run_attention "${pair}" "${task}" "${sender}" "${receiver}"
      fi
    done
  done
  "${PYTHON}" scripts/model_compatibility_diagnostic.py summarize \
    --snapshot-root snapshots \
    --calibration-root "${ROOT}/calibration" \
    --out-dir "${ROOT}/analysis" \
    --pairs ${PAIRS} --tasks ${TASKS}
} 2>&1 | tee "${log}"

echo "Report: ${ROOT}/analysis/REPORT.md"
