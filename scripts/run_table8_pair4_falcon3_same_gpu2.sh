#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVMergeComm || exit 1

# KVComm Appendix Table 8, model pair #4:
#   M_s = tiiuae/Falcon3-7B-Instruct
#   M_r = tiiuae/Falcon3-7B-Instruct

MODEL=${MODEL:-/NAS/models/Falcon3-7B-Instruct}
ROOT=${ROOT:-snapshots/table8_pair4_falcon3_7b_same}
LOGDIR="${ROOT}/logs"
GPU=${GPU:-2}
TAG=$(date '+%m%d_%H%M')

mkdir -p "${LOGDIR}"

if [[ ! -d "${MODEL}" ]]; then
  echo "Model path does not exist: ${MODEL}" >&2
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
    echo "==== [pair4 falcon3_same GPU${GPU}] ${task} ${run_name} SKIP existing per_sample.jsonl ===="
    return
  fi

  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL}" --model_B "${MODEL}" \
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
    echo "==== [pair4 falcon3_same GPU${GPU}] ${task} ${run_name} SKIP existing per_sample.jsonl ===="
    return
  fi

  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL}" --model_B "${MODEL}" \
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
      echo "==== [pair4 falcon3_same GPU${GPU}] ${task} ReKV w${win} r=${ratio} $(date '+%F %T') ===="
      run_rekv "${task}" "${win}" "${ratio}"
    done
  done

  echo "==== [pair4 falcon3_same GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.75 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.75

  echo "==== [pair4 falcon3_same GPU${GPU}] ${task} B-ReKV w8 tau=0.95 scale=0.85 $(date '+%F %T') ===="
  run_cov "${task}" 8 0.95 0.85

  echo "==== [pair4 falcon3_same GPU${GPU}] ${task} B-ReKV w16 tau=0.95 scale=0.90 $(date '+%F %T') ===="
  run_cov "${task}" 16 0.95 0.90
}

GPU_LOG="${LOGDIR}/gpu${GPU}_pair4_${TAG}.log"

(
  echo "######## [GPU${GPU} pair4 falcon3_same] START $(date '+%F %T') ########"
  echo "MODEL=${MODEL}"
  echo "ROOT=${ROOT}"
  run_task countries
  run_task tipsheets
  run_task hotpotqa
  run_task musique
  run_task multifieldqa_en
  run_task twowikimqa
  run_task qasper
  run_task tmath
  echo "######## [GPU${GPU} pair4 falcon3_same] DONE $(date '+%F %T') ########"
) > "${GPU_LOG}" 2>&1 &

P=$!

echo "GPU${GPU} pair4 queue pid=${P} -> ${GPU_LOG}"
echo "Results root -> ${ROOT}"
