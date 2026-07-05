#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVMergeComm || exit 1

# KVComm Appendix Table 8, model pair #5:
#   M_s = yuvraj17/EvolCodeLlama-3.1-8B-Instruct
#   M_r = Team-ACE/ToolACE-2-Llama-3.1-8B

MODEL_A=${MODEL_A:-/NAS/models/EvolCodeLlama-3.1-8B-Instruct}
MODEL_B=${MODEL_B:-/NAS/models/ToolACE-2-Llama-3.1-8B}
ROOT=${ROOT:-snapshots/table8_pair5_evolcodellama_toolace}
LOGDIR="${ROOT}/logs"
GPU=${GPU:-3}
TAG=$(date '+%m%d_%H%M')

mkdir -p "${LOGDIR}"

if [[ ! -d "${MODEL_A}" ]]; then
  echo "Sender model path does not exist: ${MODEL_A}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_B}" ]]; then
  echo "Receiver model path does not exist: ${MODEL_B}" >&2
  exit 1
fi

# This local checkpoint contains both full weights and PEFT adapter metadata.
# Hide adapter files from AutoModelForCausalLM so it does not try to fetch the
# base model from HuggingFace.
MODEL_A_RUNTIME=${MODEL_A_RUNTIME:-}
if [[ -z "${MODEL_A_RUNTIME}" ]]; then
  if [[ -f "${MODEL_A}/adapter_config.json" && -f "${MODEL_A}/model.safetensors.index.json" ]]; then
    MODEL_A_RUNTIME="${ROOT}/model_views/$(basename "${MODEL_A}")_no_adapter"
    mkdir -p "${MODEL_A_RUNTIME}"
    for f in "${MODEL_A}"/*; do
      base=$(basename "${f}")
      case "${base}" in
        adapter_config.json|adapter_model.bin|adapter_model.safetensors)
          continue
          ;;
      esac
      ln -sfn "${f}" "${MODEL_A_RUNTIME}/${base}"
    done
  else
    MODEL_A_RUNTIME="${MODEL_A}"
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

has_completed_run() {
  local path_glob=$1
  compgen -G "${path_glob}" > /dev/null
}

run_rekv() {
  local task=$1
  local win=$2
  local ratio=$3
  local run_name="recv_w${win}_r${ratio}"
  local snapshot_dir="${ROOT}/${task}/mtc_receiver"

  if has_completed_run "${snapshot_dir}/${run_name}_*/per_sample.jsonl"; then
    echo "==== [pair5 evolcodellama_toolace GPU${GPU}] ${task} ${run_name} SKIP existing per_sample.jsonl ===="
    return
  fi

  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL_A_RUNTIME}" --model_B "${MODEL_B}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${snapshot_dir}" \
    --run_name "${run_name}"
}

run_cov() {
  local task=$1
  local win=$2
  local tau=$3
  local scale=$4
  local run_name="cov_t${tau}_s${scale}_w${win}"
  local snapshot_dir="${ROOT}/${task}/coverage"

  if has_completed_run "${snapshot_dir}/${run_name}_*/per_sample.jsonl"; then
    echo "==== [pair5 evolcodellama_toolace GPU${GPU}] ${task} ${run_name} SKIP existing per_sample.jsonl ===="
    return
  fi

  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL_A_RUNTIME}" --model_B "${MODEL_B}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage \
    --budget_min 0.05 --budget_max 0.7 \
    --coverage_tau "${tau}" --coverage_scale "${scale}" \
    --snapshot_path "${snapshot_dir}" \
    --run_name "${run_name}"
}

run_task() {
  local task=$1

  for win in 8 16; do
    for ratio in 0.3 0.5 0.7; do
      echo "==== [pair5 evolcodellama_toolace GPU${GPU}] ${task} ReKV w${win} r=${ratio} $(date '+%F %T') ===="
      run_rekv "${task}" "${win}" "${ratio}"
    done
  done

  echo "==== [pair5 evolcodellama_toolace GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.75 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.75

  echo "==== [pair5 evolcodellama_toolace GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.85 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.85

  echo "==== [pair5 evolcodellama_toolace GPU${GPU}] ${task} B-ReKV w16 tau=0.95 scale=0.90 $(date '+%F %T') ===="
  run_cov "${task}" 16 0.95 0.90
}

GPU_LOG="${LOGDIR}/gpu${GPU}_pair5_${TAG}.log"

(
  echo "######## [GPU${GPU} pair5 evolcodellama_toolace] START $(date '+%F %T') ########"
  echo "MODEL_A=${MODEL_A}"
  echo "MODEL_A_RUNTIME=${MODEL_A_RUNTIME}"
  echo "MODEL_B=${MODEL_B}"
  echo "ROOT=${ROOT}"
  run_task countries
  run_task tipsheets
  run_task hotpotqa
  run_task musique
  run_task multifieldqa_en
  run_task twowikimqa
  run_task qasper
  run_task tmath
  echo "######## [GPU${GPU} pair5 evolcodellama_toolace] DONE $(date '+%F %T') ########"
) > "${GPU_LOG}" 2>&1 &

P=$!

echo "GPU${GPU} pair5 queue pid=${P} -> ${GPU_LOG}"
echo "Results root -> ${ROOT}"
