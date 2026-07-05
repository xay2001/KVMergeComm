#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVMergeComm || exit 1

# KVComm Appendix Table 8, model pair #9:
#   M_s = arcee-ai/Llama-3.1-SuperNova-Lite
#   M_r = deepseek-ai/DeepSeek-R1-Distill-Llama-8B
#
# Corrected rerun after fixing eval.py to recognize local DeepSeek-R1 paths as
# think models. Results intentionally go to a new root to keep the earlier
# invalid run for audit. After pair #9 completes, this queue runs the HotpotQA
# supporting-facts overlap diagnostic on the same physical GPU.

MODEL_A=${MODEL_A:-/NAS/models/Llama-3.1-SuperNova-Lite}
MODEL_B=${MODEL_B:-/NAS/models/DeepSeek-R1-Distill-Llama-8B}
ROOT=${ROOT:-snapshots/table8_pair9_supernova_deepseek_llama8b_corrected}
LOGDIR="${ROOT}/logs"
GPU=${GPU:-2}
TAG=$(date '+%m%d_%H%M')
SUPPORT_MODEL=${SUPPORT_MODEL:-/NAS/models/Llama-3.1-8B-Instruct}
SUPPORT_LIMIT=${SUPPORT_LIMIT:-200}

mkdir -p "${LOGDIR}"

if [[ ! -d "${MODEL_A}" ]]; then
  echo "Sender model path does not exist: ${MODEL_A}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_B}" ]]; then
  echo "Receiver model path does not exist: ${MODEL_B}" >&2
  exit 1
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
    echo "==== [pair9 corrected GPU${GPU}] ${task} ${run_name} SKIP existing per_sample.jsonl ===="
    return
  fi

  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
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
    echo "==== [pair9 corrected GPU${GPU}] ${task} ${run_name} SKIP existing per_sample.jsonl ===="
    return
  fi

  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
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
      echo "==== [pair9 corrected GPU${GPU}] ${task} ReKV w${win} r=${ratio} $(date '+%F %T') ===="
      run_rekv "${task}" "${win}" "${ratio}"
    done
  done

  echo "==== [pair9 corrected GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.75 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.75

  echo "==== [pair9 corrected GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.85 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.85

  echo "==== [pair9 corrected GPU${GPU}] ${task} B-ReKV w16 tau=0.95 scale=0.90 $(date '+%F %T') ===="
  run_cov "${task}" 16 0.95 0.90
}

GPU_LOG="${LOGDIR}/gpu${GPU}_pair9_corrected_${TAG}.log"

(
  echo "######## [GPU${GPU} pair9 corrected supernova_deepseek_llama8b] START $(date '+%F %T') ########"
  echo "MODEL_A=${MODEL_A}"
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
  echo "######## [GPU${GPU} pair9 corrected supernova_deepseek_llama8b] DONE $(date '+%F %T') ########"
  echo "######## [GPU${GPU} hotpot supporting-overlap] START $(date '+%F %T') ########"
  echo "SUPPORT_MODEL=${SUPPORT_MODEL}"
  echo "SUPPORT_LIMIT=${SUPPORT_LIMIT}"
  CUDA_VISIBLE_DEVICES=${GPU} python scripts/hotpot_supporting_overlap.py \
    --device cuda:0 \
    --model "${SUPPORT_MODEL}" \
    --limit "${SUPPORT_LIMIT}" \
    --out_dir snapshots/supporting_overlap/hotpotqa_pair1_full_context
  echo "######## [GPU${GPU} hotpot supporting-overlap] DONE $(date '+%F %T') ########"
) > "${GPU_LOG}" 2>&1 &

P=$!

echo "GPU${GPU} pair9 corrected queue pid=${P} -> ${GPU_LOG}"
echo "Results root -> ${ROOT}"
echo "Supporting overlap root -> snapshots/supporting_overlap/hotpotqa_pair1_full_context"
