#!/usr/bin/env bash
set -uo pipefail

# Layer-vs-Token causal comparison (same implementation, matched fractions).
#
# 2x2 design + controls, all under one CVCommunicator implementation with
# identical byte accounting (per_sample.jsonl carries actual bytes):
#   layer x query-free : kvcomm (calibrated) / random_layer
#   layer x receiver   : recv_layer (NEW: query-sketch layer selection)
#   token x query-free : evict (value_norm)
#   token x receiver   : rekv (correct) / rekv_shuffled (causal control)
#   upper bound        : full_kv + skyline
#
# Usage: GPU=0 TASK=hotpotqa bash scripts/run_layer_vs_token_gpu.sh

cd /home/xay/KVComm || exit 1

GPU=${GPU:?set GPU}
TASK=${TASK:?set TASK}
PYTHON=${PYTHON:-/home/xay/.conda/envs/kvcomm/bin/python}
MODEL_A=${MODEL_A:-/sharedspace/models/Llama-3.1-8B-Instruct}
MODEL_B=${MODEL_B:-/sharedspace/models/Llama-3.1-8B-Instruct}
PAIR=${PAIR:-pair1_llama31_same}
LIMIT=${LIMIT:-200}
FRACTIONS=${FRACTIONS:-"0.1 0.3 0.5"}
WINDOW=${WINDOW:-8}
N_LAYERS_MINUS1=${N_LAYERS_MINUS1:-31}
ROOT=${ROOT:-snapshots/layer_vs_token_v1}

timestamp=$(date +"%m%d_%H%M")
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/gpu${GPU}_${TASK}_${timestamp}.log"

run_com() {
  local method=$1
  local run_name=$2
  shift 2
  local parent="${ROOT}/${PAIR}/${TASK}/${method}"
  if compgen -G "${parent}/${run_name}_*/per_sample.jsonl" > /dev/null; then
    echo "[skip] ${TASK} ${method}/${run_name}"
    return
  fi
  echo "==== [GPU${GPU}] ${TASK} ${method}/${run_name} ===="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" com.py \
    --test_task "${TASK}" \
    --model_A "${MODEL_A}" \
    --model_B "${MODEL_B}" \
    --limit "${LIMIT}" \
    --layer_from 0 --layer_to "${N_LAYERS_MINUS1}" \
    --snapshot_path "${parent}" \
    --run_name "${run_name}" \
    "$@"
  if [ $? -ne 0 ]; then
    echo "[FAIL] ${TASK} ${method}/${run_name}"
  fi
}

{
  echo "######## LAYER-VS-TOKEN START $(date '+%F %T') GPU=${GPU} TASK=${TASK} ########"
  echo "LIMIT=${LIMIT} FRACTIONS=${FRACTIONS} WINDOW=${WINDOW}"

  # 0) Skyline reference (receiver reads raw context; recovered-score denominator)
  run_com skyline "skyline" --do_test_skyline

  # 1) Full KV upper bound (all layers, all tokens)
  run_com full_kv "full_kv" --do_test

  for f in ${FRACTIONS}; do
    # 2) KVComm calibrated layer selection (query-free layer granularity)
    run_com kvcomm "kvcomm_top${f}" --do_test --top_layers "${f}" --calib_size 1

    # 3) Random layer selection
    run_com random_layer "random_layer_top${f}" --do_test --top_layers "${f}" --random_selection

    # 4) Receiver-aware layer selection (receiver-conditioned layer granularity)
    run_com recv_layer "recv_layer_f${f}" --do_test \
      --receiver_layer_fraction "${f}" --score_mode receiver \
      --recv_window "${WINDOW}" --query_sketch_mode bf16

    # 5) Query-free token selection (value-norm evict)
    run_com evict "evict_r${f}" --do_test \
      --merge --merge_mode evict --score_mode value_norm --recv_window 0 \
      --merge_ratio "${f}"

    # 6) ReKV correct receiver query (receiver-conditioned token granularity)
    run_com rekv "rekv_w${WINDOW}_r${f}" --do_test \
      --merge --merge_mode evict --score_mode receiver \
      --recv_window "${WINDOW}" --query_sketch_mode bf16 \
      --merge_ratio "${f}"
  done

  # 7) ReKV shuffled receiver query (causal control, single mid point)
  run_com rekv_shuffled "rekv_shuffled_w${WINDOW}_r0.3" --do_test \
    --merge --merge_mode evict --score_mode receiver \
    --recv_window "${WINDOW}" --query_sketch_mode bf16 \
    --merge_ratio 0.3 --query_condition_mode shuffled

  echo "######## LAYER-VS-TOKEN DONE $(date '+%F %T') GPU=${GPU} TASK=${TASK} ########"
} 2>&1 | tee "${LOG_FILE}"
