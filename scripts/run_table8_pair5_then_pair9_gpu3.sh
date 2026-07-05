#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVMergeComm || exit 1

# Serial GPU3 queue:
#   1. Table 8 pair #5: EvolCodeLlama -> ToolACE
#   2. Table 8 pair #9: SuperNova -> DeepSeek-Llama-8B
#
# Both stages skip runs that already have per_sample.jsonl.

GPU=${GPU:-3}
TAG=$(date '+%m%d_%H%M')

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TASKS=(
  countries
  tipsheets
  hotpotqa
  musique
  multifieldqa_en
  twowikimqa
  qasper
  tmath
)

has_completed_run() {
  local path_glob=$1
  compgen -G "${path_glob}" > /dev/null
}

make_no_adapter_view() {
  local model_dir=$1
  local root=$2

  if [[ -f "${model_dir}/adapter_config.json" && -f "${model_dir}/model.safetensors.index.json" ]]; then
    local view_dir="${root}/model_views/$(basename "${model_dir}")_no_adapter"
    mkdir -p "${view_dir}"
    for f in "${model_dir}"/*; do
      local base
      base=$(basename "${f}")
      case "${base}" in
        adapter_config.json|adapter_model.bin|adapter_model.safetensors)
          continue
          ;;
      esac
      ln -sfn "${f}" "${view_dir}/${base}"
    done
    echo "${view_dir}"
  else
    echo "${model_dir}"
  fi
}

run_rekv() {
  local label=$1
  local root=$2
  local model_a=$3
  local model_b=$4
  local task=$5
  local win=$6
  local ratio=$7
  local run_name="recv_w${win}_r${ratio}"
  local snapshot_dir="${root}/${task}/mtc_receiver"

  if has_completed_run "${snapshot_dir}/${run_name}_*/per_sample.jsonl"; then
    echo "==== [${label} GPU${GPU}] ${task} ${run_name} SKIP existing per_sample.jsonl ===="
    return
  fi

  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${snapshot_dir}" \
    --run_name "${run_name}"
}

run_cov() {
  local label=$1
  local root=$2
  local model_a=$3
  local model_b=$4
  local task=$5
  local win=$6
  local tau=$7
  local scale=$8
  local run_name="cov_t${tau}_s${scale}_w${win}"
  local snapshot_dir="${root}/${task}/coverage"

  if has_completed_run "${snapshot_dir}/${run_name}_*/per_sample.jsonl"; then
    echo "==== [${label} GPU${GPU}] ${task} ${run_name} SKIP existing per_sample.jsonl ===="
    return
  fi

  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${model_a}" --model_B "${model_b}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage \
    --budget_min 0.05 --budget_max 0.7 \
    --coverage_tau "${tau}" --coverage_scale "${scale}" \
    --snapshot_path "${snapshot_dir}" \
    --run_name "${run_name}"
}

run_task() {
  local label=$1
  local root=$2
  local model_a=$3
  local model_b=$4
  local task=$5

  for win in 8 16; do
    for ratio in 0.3 0.5 0.7; do
      echo "==== [${label} GPU${GPU}] ${task} ReKV w${win} r=${ratio} $(date '+%F %T') ===="
      run_rekv "${label}" "${root}" "${model_a}" "${model_b}" "${task}" "${win}" "${ratio}"
    done
  done

  echo "==== [${label} GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.75 $(date '+%F %T') ===="
  run_cov "${label}" "${root}" "${model_a}" "${model_b}" "${task}" 8 0.95 0.75

  echo "==== [${label} GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.85 $(date '+%F %T') ===="
  run_cov "${label}" "${root}" "${model_a}" "${model_b}" "${task}" 8 0.95 0.85

  echo "==== [${label} GPU${GPU}] ${task} B-ReKV w16 tau=0.95 scale=0.90 $(date '+%F %T') ===="
  run_cov "${label}" "${root}" "${model_a}" "${model_b}" "${task}" 16 0.95 0.90
}

run_pair() {
  local label=$1
  local root=$2
  local model_a=$3
  local model_b=$4

  mkdir -p "${root}/logs"

  if [[ ! -d "${model_a}" ]]; then
    echo "Sender model path does not exist: ${model_a}" >&2
    exit 1
  fi
  if [[ ! -d "${model_b}" ]]; then
    echo "Receiver model path does not exist: ${model_b}" >&2
    exit 1
  fi

  local model_a_runtime
  model_a_runtime=$(make_no_adapter_view "${model_a}" "${root}")

  echo "######## [GPU${GPU} ${label}] START $(date '+%F %T') ########"
  echo "MODEL_A=${model_a}"
  echo "MODEL_A_RUNTIME=${model_a_runtime}"
  echo "MODEL_B=${model_b}"
  echo "ROOT=${root}"

  for task in "${TASKS[@]}"; do
    run_task "${label}" "${root}" "${model_a_runtime}" "${model_b}" "${task}"
  done

  echo "######## [GPU${GPU} ${label}] DONE $(date '+%F %T') ########"
}

MASTER_LOG="snapshots/table8_pair5_then_pair9_gpu${GPU}_${TAG}.log"
mkdir -p snapshots

(
  run_pair \
    "pair5 evolcodellama_toolace" \
    "snapshots/table8_pair5_evolcodellama_toolace" \
    "/NAS/models/EvolCodeLlama-3.1-8B-Instruct" \
    "/NAS/models/ToolACE-2-Llama-3.1-8B"

  run_pair \
    "pair9 supernova_deepseek_llama8b" \
    "snapshots/table8_pair9_supernova_deepseek_llama8b" \
    "/NAS/models/Llama-3.1-SuperNova-Lite" \
    "/NAS/models/DeepSeek-R1-Distill-Llama-8B"
) > "${MASTER_LOG}" 2>&1 &

P=$!

echo "GPU${GPU} pair5->pair9 serial queue pid=${P} -> ${MASTER_LOG}"
echo "Pair #5 root -> snapshots/table8_pair5_evolcodellama_toolace"
echo "Pair #9 root -> snapshots/table8_pair9_supernova_deepseek_llama8b"
