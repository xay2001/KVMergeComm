#!/usr/bin/env bash
set -euo pipefail

# Training-free query-aware BM25 text-retrieval baseline.
#
# Protocol:
#   B -> A: raw query text
#   A: BM25 ranks token-window chunks from the original context
#   A -> B: top-k original-text chunks + uint32 chunk indices
#   B: one prefill/generation pass
#
# Defaults produce a paper-ready accuracy/communication Pareto:
#   3 pairs x 3 tasks x top-k {1,2,4,8} = 36 runs.
#
# Fast single-point override:
#   TOPKS="4" bash scripts/run_query_aware_text_retrieval_gpu3.sh

cd /home/xay/KVComm || exit 1

GPU=${GPU:-3}
PYTHON=${PYTHON:-/home/xay/.conda/envs/kvcomm/bin/python}
PAIRS=${PAIRS:-"1 6 7"}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
TOPKS=${TOPKS:-"1 2 4 8"}
LIMIT=${LIMIT:-500}
WARMUP=${WARMUP:-5}
CHUNK_TOKENS=${CHUNK_TOKENS:-128}
CHUNK_STRIDE=${CHUNK_STRIDE:-96}
BM25_K1=${BM25_K1:-1.5}
BM25_B=${BM25_B:-0.75}
ROOT=${ROOT:-snapshots/query_aware_text_retrieval_bm25_v1}

timestamp=$(date +"%m%d_%H%M")
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/gpu${GPU}_text_retrieval_${timestamp}.log"

pair_paths() {
  case "$1" in
    1)
      echo "pair1_llama31_same /sharedspace/models/Llama-3.1-8B-Instruct /sharedspace/models/Llama-3.1-8B-Instruct"
      ;;
    6)
      echo "pair6_llama32_abliterated_deepseek3b /sharedspace/models/Llama-3.2-3B-Instruct-abliterated /sharedspace/models/DeepSeek-R1-Distill-Llama-3B"
      ;;
    7)
      echo "pair7_qwen25_uncensored_bespoke /sharedspace/models/Qwen2.5-7B-Instruct-Uncensored /sharedspace/models/Bespoke-Stratos-7B"
      ;;
    *)
      echo "Unknown pair id: $1" >&2
      return 1
      ;;
  esac
}

has_valid_artifact() {
  local parent=$1
  local expected_k=$2
  "${PYTHON}" - "${parent}" "${expected_k}" "${LIMIT}" <<'PY'
import json
import pathlib
import sys

parent = pathlib.Path(sys.argv[1])
expected_k = int(sys.argv[2])
expected_n = int(sys.argv[3])
for summary_path in parent.glob("*/cost_summary.json"):
    try:
        summary = json.loads(summary_path.read_text())
        meta = summary.get("_meta", {})
        if (
            meta.get("protocol_version") == "query_aware_text_retrieval_bm25_v1"
            and int(meta.get("top_k", -1)) == expected_k
            and (expected_n <= 0 or int(meta.get("n", -1)) == expected_n)
        ):
            raise SystemExit(0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
raise SystemExit(1)
PY
}

run_one() {
  local pair_slug=$1
  local model_a=$2
  local model_b=$3
  local task=$4
  local top_k=$5
  local parent="${ROOT}/${pair_slug}/${task}/bm25_topk${top_k}_c${CHUNK_TOKENS}_s${CHUNK_STRIDE}"

  if has_valid_artifact "${parent}" "${top_k}"; then
    echo "[skip] ${pair_slug} ${task} top-k=${top_k}"
    return
  fi

  echo "==== [GPU${GPU}] ${pair_slug} ${task} BM25 top-k=${top_k} ===="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" com.py \
    --test_task "${task}" \
    --do_test_text_retrieval \
    --profile_cost \
    --profile_limit "${LIMIT}" \
    --profile_warmup "${WARMUP}" \
    --text_retrieval_top_k "${top_k}" \
    --text_retrieval_chunk_tokens "${CHUNK_TOKENS}" \
    --text_retrieval_chunk_stride "${CHUNK_STRIDE}" \
    --text_retrieval_bm25_k1 "${BM25_K1}" \
    --text_retrieval_bm25_b "${BM25_B}" \
    --model_A "${model_a}" \
    --model_B "${model_b}" \
    --snapshot_path "${parent}" \
    --run_name "text_retrieval_bm25_topk${top_k}"
}

{
  echo "######## QUERY-AWARE TEXT RETRIEVAL START $(date '+%F %T') ########"
  echo "GPU=${GPU} PAIRS=${PAIRS} TASKS=${TASKS} TOPKS=${TOPKS}"
  echo "LIMIT=${LIMIT} WARMUP=${WARMUP}"
  echo "CHUNK_TOKENS=${CHUNK_TOKENS} CHUNK_STRIDE=${CHUNK_STRIDE}"
  echo "BM25_K1=${BM25_K1} BM25_B=${BM25_B}"

  for pair in ${PAIRS}; do
    read -r pair_slug model_a model_b <<< "$(pair_paths "${pair}")"
    for task in ${TASKS}; do
      for top_k in ${TOPKS}; do
        run_one "${pair_slug}" "${model_a}" "${model_b}" "${task}" "${top_k}"
      done
    done
  done

  echo "######## QUERY-AWARE TEXT RETRIEVAL DONE $(date '+%F %T') ########"
} 2>&1 | tee "${LOG_FILE}"

