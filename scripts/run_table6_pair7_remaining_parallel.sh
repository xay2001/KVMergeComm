#!/usr/bin/env bash
set -euo pipefail

# Run remaining Table 6 pair #7 jobs on a selected GPU.
# Each wrapper script sets GPU and RUN_ITEMS.

cd /home/xay/KVMergeComm || exit 1

GPU=${GPU:?GPU must be set}
RUN_ITEMS=${RUN_ITEMS:?RUN_ITEMS must be set}
SKIP_EXISTING=${SKIP_EXISTING:-1}

MODEL_A=${MODEL_A:-/NAS/models/Qwen2.5-7B-Instruct-Uncensored}
MODEL_B=${MODEL_B:-/NAS/models/Bespoke-Stratos-7B}
ROOT=${ROOT:-snapshots/table6_pair7_qwen25_uncensored_bespoke}
LOG_ROOT=${LOG_ROOT:-snapshots/table6_pair7_qwen25_uncensored_bespoke/logs}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${LOG_ROOT}"
TAG=$(date '+%m%d_%H%M')
LOG_PATH="${LOG_ROOT}/gpu${GPU}_table6_pair7_remaining_${TAG}.log"

has_done_run() {
  local out=$1
  local run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] && compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null
}

common_eval() {
  local task=$1 out=$2 run_name=$3
  shift 3
  CUDA_VISIBLE_DEVICES="${GPU}" python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL_A}" --model_B "${MODEL_B}" \
    --snapshot_path "${out}" \
    --run_name "${run_name}" "$@"
}

run_rekv() {
  local task=$1 win=$2 ratio=$3
  local out="${ROOT}/${task}/mtc_receiver"
  local run_name="recv_w${win}_r${ratio}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ${run_name} already complete ===="
    return
  fi
  echo "==== [GPU${GPU}] ${task} ReKV w${win} r=${ratio} $(date '+%F %T') ===="
  common_eval "${task}" "${out}" "${run_name}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_ratio "${ratio}" --merge_sink 4 --merge_recent 8
}

run_brekv() {
  local task=$1 win=$2 tau=$3 scale=$4
  local out="${ROOT}/${task}/coverage"
  local run_name="cov_t${tau}_s${scale}_w${win}"
  if has_done_run "${out}" "${run_name}"; then
    echo "==== [skip] ${task} ${run_name} already complete ===="
    return
  fi
  echo "==== [GPU${GPU}] ${task} B-ReKV w${win} tau=${tau} scale=${scale} $(date '+%F %T') ===="
  common_eval "${task}" "${out}" "${run_name}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage --budget_min 0.05 --budget_max 0.7 \
    --coverage_tau "${tau}" --coverage_scale "${scale}"
}

run_item() {
  local item=$1
  IFS=: read -r task method a b c d <<< "${item}"
  case "${method}" in
    rekv)
      run_rekv "${task}" "${a}" "${b}"
      ;;
    brekv)
      run_brekv "${task}" "${a}" "${b}" "${c}"
      ;;
    *)
      echo "Unknown item method: ${item}" >&2
      exit 1
      ;;
  esac
}

{
  echo "######## Table 6 pair #7 remaining START $(date '+%F %T') ########"
  echo "GPU=${GPU}"
  echo "MODEL_A=${MODEL_A}"
  echo "MODEL_B=${MODEL_B}"
  echo "ROOT=${ROOT}"
  echo "RUN_ITEMS=${RUN_ITEMS}"
  echo "SKIP_EXISTING=${SKIP_EXISTING}"

  for item in ${RUN_ITEMS}; do
    run_item "${item}"
  done

  echo "######## Table 6 pair #7 remaining DONE $(date '+%F %T') ########"
} 2>&1 | tee "${LOG_PATH}"
