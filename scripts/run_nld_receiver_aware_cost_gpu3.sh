#!/usr/bin/env bash
set -euo pipefail

# Receiver-aware NLD (fair-text baseline) cost/score profile, matched sample-
# for-sample against the existing B-ReKV / ReKV query-sketch cost results in
# snapshots/query_sketch_cost_v1 (PROFILE_LIMIT=50, PROFILE_WARMUP=3, seed=42
# dataloader shuffle -> identical sample ids across methods).
#
# Fixes two review-blocking issues in one run:
#   F1 (protocol asymmetry): sender now sees the receiver's question/query
#     text before generating its 128-token evidence/answer, same information
#     access as B-ReKV's query sketch (--nld_receiver_aware).
#   F3a (unpaired sample counts): uses the SAME limit/warmup/seed as the
#     already-frozen B-ReKV/ReKV query-sketch cost cells, instead of the old
#     nld_cost_profile run (limit=500) which produced unpaired 495/495/145
#     samples vs. the KV side's 50.
#
# Default scope: the paper trio of pairs x tasks (9 cells):
#   pair1_llama31_same, pair6_llama32_abliterated_deepseek3b,
#   pair7_qwen25_uncensored_bespoke
#   x hotpotqa, musique, multifieldqa_en
#
# Usage:
#   bash scripts/run_nld_receiver_aware_cost_gpu3.sh
#
# Overrides:
#   GPU=3 PAIRS="1 6 7" TASKS="hotpotqa musique multifieldqa_en" \
#     PROFILE_LIMIT=50 PROFILE_WARMUP=3 MODE=aware \
#     bash scripts/run_nld_receiver_aware_cost_gpu3.sh
#
#   MODE=aware  -> only receiver-aware NLD (the 9 cells asked for)
#   MODE=blind  -> only query-blind NLD, but re-run at the SAME limit/warmup
#                  as the KV side (old nld_cost_profile used limit=500, so it
#                  is NOT paired with the 50-sample KV cost cells)
#   MODE=both   -> run both variants (18 cells), recommended if you also want
#                  a paired blind-vs-aware-vs-ReKV-vs-B-ReKV comparison

cd /home/xay/KVMergeComm || exit 1

GPU=${GPU:-3}
PYTHON=${PYTHON:-/home/xay/miniconda3/envs/ReKV/bin/python}
PAIRS=${PAIRS:-"1 6 7"}
TASKS=${TASKS:-"hotpotqa musique multifieldqa_en"}
# Matched to snapshots/query_sketch_cost_v1 defaults so sample ids line up.
PROFILE_LIMIT=${PROFILE_LIMIT:-50}
PROFILE_WARMUP=${PROFILE_WARMUP:-3}
PHASE1_TOKENS=${PHASE1_TOKENS:-128}
MODE=${MODE:-aware}
ROOT=${ROOT:-snapshots/nld_receiver_aware_cost_v1}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

timestamp=$(date +"%m%d_%H%M")
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/gpu${GPU}_nld_receiver_aware_cost_${timestamp}.log"

pair_paths() {
  case "$1" in
    1)
      echo "pair1_llama31_same /NAS/models/Llama-3.1-8B-Instruct /NAS/models/Llama-3.1-8B-Instruct"
      ;;
    6)
      echo "pair6_llama32_abliterated_deepseek3b /NAS/models/Llama-3.2-3B-Instruct-abliterated /NAS/models/DeepSeek-R1-Distill-Llama-3B"
      ;;
    7)
      echo "pair7_qwen25_uncensored_bespoke /NAS/models/Qwen2.5-7B-Instruct-Uncensored /NAS/models/Bespoke-Stratos-7B"
      ;;
    *)
      echo "Unknown pair id: $1" >&2
      return 1
      ;;
  esac
}

has_done_run() {
  local out=$1 run_name=$2
  [[ "${SKIP_EXISTING}" == "1" ]] &&
    compgen -G "${out}/${run_name}_*/cost_summary.json" > /dev/null
}

run_nld_cost() {
  local pair_slug=$1 model_a=$2 model_b=$3 task=$4 variant=$5
  local root="${ROOT}/${pair_slug}/${task}/nld_${variant}"
  local run_name="nld_${variant}_cost"

  if has_done_run "${root}" "${run_name}"; then
    echo "==== [skip] ${pair_slug} ${task} ${run_name} ===="
    return
  fi

  local args=(
    --test_task "${task}"
    --do_test_nld
    --profile_cost
    --profile_limit "${PROFILE_LIMIT}"
    --profile_warmup "${PROFILE_WARMUP}"
    --nld_max_tokens_model_A_and_B_phase1 "${PHASE1_TOKENS}"
    --model_A "${model_a}"
    --model_B "${model_b}"
    --snapshot_path "${root}"
    --run_name "${run_name}"
  )
  if [[ "${variant}" == "aware" ]]; then
    args+=(--nld_receiver_aware)
  fi

  echo "==== [GPU${GPU}] ${pair_slug} ${task} ${run_name} limit=${PROFILE_LIMIT} warmup=${PROFILE_WARMUP} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" com.py "${args[@]}"
}

variants_for_mode() {
  case "${MODE}" in
    aware) echo "aware" ;;
    blind) echo "blind" ;;
    both) echo "aware blind" ;;
    *)
      echo "Unknown MODE: ${MODE} (expected aware|blind|both)" >&2
      exit 2
      ;;
  esac
}

{
  echo "######## NLD receiver-aware cost profile START $(date '+%F %T') ########"
  echo "GPU=${GPU}"
  echo "PAIRS=${PAIRS}"
  echo "TASKS=${TASKS}"
  echo "PROFILE_LIMIT=${PROFILE_LIMIT}"
  echo "PROFILE_WARMUP=${PROFILE_WARMUP}"
  echo "PHASE1_TOKENS=${PHASE1_TOKENS}"
  echo "MODE=${MODE}"
  echo "ROOT=${ROOT}"

  variants=$(variants_for_mode)

  for pair in ${PAIRS}; do
    read -r pair_slug model_a model_b <<< "$(pair_paths "${pair}")"
    for task in ${TASKS}; do
      for variant in ${variants}; do
        run_nld_cost "${pair_slug}" "${model_a}" "${model_b}" "${task}" "${variant}"
      done
    done
  done

  echo "######## NLD receiver-aware cost profile DONE $(date '+%F %T') ########"
} 2>&1 | tee "${LOG_FILE}"

echo "Results under: ${ROOT}/<pair>/<task>/nld_aware/nld_aware_cost_*/cost_summary.json"
echo "Compare against: snapshots/query_sketch_cost_v1/<pair>/<task>/{rekv,brekv}/*/cost_summary.json"
