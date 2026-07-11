#!/usr/bin/env bash
set -euo pipefail

# Table 10: Multi-Source ReKV.
#
# Two sender contexts (A1/A2) transmit KV to one receiver (B).  Defaults use the
# same model for all agents; override MODEL_A1/MODEL_A2/MODEL_B for heterogeneous
# multi-source runs.
#
# Example:
#   GPU=2 bash scripts/run_table10_multi_source_rekv_gpu.sh

cd /home/xay/KVMergeComm || exit 1

GPU=${GPU:-0}
MODEL_A1=${MODEL_A1:-/NAS/models/Llama-3.1-8B-Instruct}
MODEL_A2=${MODEL_A2:-${MODEL_A1}}
MODEL_B=${MODEL_B:-${MODEL_A1}}
ROOT=${ROOT:-snapshots/table10_multi_source_rekv}
TASKS=${TASKS:-"hotpotqa musique twowikimqa"}
WINDOWS=${WINDOWS:-"8 16"}
RATIOS=${RATIOS:-"0.3 0.5 0.7"}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOGDIR="${ROOT}/logs"
TAG=$(date '+%m%d_%H%M')
mkdir -p "${LOGDIR}"
LOG_PATH="${LOGDIR}/gpu${GPU}_table10_multi_source_rekv_${TAG}.log"

check_model() {
  local label=$1
  local path=$2
  if [[ ! -d "${path}" ]]; then
    echo "${label} model path does not exist: ${path}" >&2
    exit 1
  fi
}

has_done_run() {
  local out=$1
  local run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

run_rekv() {
  local task=$1
  local win=$2
  local ratio=$3
  local out="${ROOT}/${task}/multi_source_rekv"
  local run_name="ms_recv_w${win}_r${ratio}"

  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} multi-source ReKV w${win} r=${ratio} already has per_sample.jsonl ===="
    return
  fi

  echo "==== [table10 GPU${GPU}] ${task} Multi-Source ReKV w${win} r=${ratio} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${GPU} python com_ms.py \
    --test_task "${task}" --do_test \
    --model_A1 "${MODEL_A1}" --model_A2 "${MODEL_A2}" --model_B "${MODEL_B}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8 \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
}

(
  echo "######## Table10 Multi-Source ReKV GPU${GPU} START $(date '+%F %T') ########"
  echo "MODEL_A1=${MODEL_A1}"
  echo "MODEL_A2=${MODEL_A2}"
  echo "MODEL_B=${MODEL_B}"
  echo "ROOT=${ROOT}"
  echo "TASKS=${TASKS}"
  echo "WINDOWS=${WINDOWS}"
  echo "RATIOS=${RATIOS}"
  echo "SKIP_EXISTING=${SKIP_EXISTING}"

  check_model MODEL_A1 "${MODEL_A1}"
  check_model MODEL_A2 "${MODEL_A2}"
  check_model MODEL_B "${MODEL_B}"

  for task in ${TASKS}; do
    for win in ${WINDOWS}; do
      for ratio in ${RATIOS}; do
        run_rekv "${task}" "${win}" "${ratio}"
      done
    done
  done

  echo "######## Table10 Multi-Source ReKV GPU${GPU} DONE $(date '+%F %T') ########"
) > "${LOG_PATH}" 2>&1 &

pid=$!
echo "table10 multi-source ReKV GPU${GPU} pid=${pid} -> ${LOG_PATH}"
echo "Results root -> ${ROOT}/{hotpotqa,musique,twowikimqa}/multi_source_rekv"
