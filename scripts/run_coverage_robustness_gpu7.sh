#!/usr/bin/env bash
set -euo pipefail

# Coverage-BRASC robustness / Pareto sweep on GPU 7.
#
# Goal: show Coverage-BRASC has a stable Pareto region rather than one
# cherry-picked point.
#
# Default first batch:
#   musique:
#     w=8,16; tau=0.90,0.95; scale=0.65,0.75,0.85,0.95
#   hotpotqa:
#     w=8,16; tau=0.95,0.98; scale=0.75,0.85,0.95
#   multifieldqa_en:
#     w=8,16; tau=0.90,0.95; scale=0.65,0.75,0.85
#
# It skips completed runs by default. A run is considered complete if a matching
# per_sample.jsonl exists under snapshots/<task>/coverage/.
#
# Run:
#   bash scripts/run_coverage_robustness_gpu7.sh
#
# Analyze:
#   python scripts/analyze_coverage.py --tasks musique hotpotqa multifieldqa_en --tau 0.5
#   python scripts/plot_coverage_pareto.py --task musique --tau 0.5

cd /home/xay/KVComm || exit 1

GPU=${GPU:-7}
MODEL=${MODEL:-/sharedspace/models/Llama-3.1-8B-Instruct}
MIN_BUDGET=${MIN_BUDGET:-0.05}
MAX_BUDGET=${MAX_BUDGET:-0.7}
SKIP_EXISTING=${SKIP_EXISTING:-1}

run_one() {
  local task=$1
  local win=$2
  local tau=$3
  local scale=$4
  local out="snapshots/${task}/coverage"
  local run_name="cov_t${tau}_s${scale}_w${win}"

  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    if compgen -G "${out}/${run_name}_*/per_sample.jsonl" > /dev/null; then
      echo "==== [skip] ${task} ${run_name} already has per_sample.jsonl ===="
      return
    fi
  fi

  echo "==== [coverage robustness GPU${GPU}] task=${task} win=${win} tau=${tau} scale=${scale} $(date '+%F %T') ===="
  CUDA_VISIBLE_DEVICES=${GPU} python com.py \
    --test_task "${task}" --do_test \
    --model_A "${MODEL}" --model_B "${MODEL}" \
    --merge --merge_mode evict --score_mode receiver --recv_window "${win}" \
    --merge_sink 4 --merge_recent 8 \
    --budget_mode coverage \
    --budget_min "${MIN_BUDGET}" --budget_max "${MAX_BUDGET}" \
    --coverage_tau "${tau}" --coverage_scale "${scale}" \
    --snapshot_path "${out}" \
    --run_name "${run_name}"
}

run_grid() {
  local task=$1
  local wins=$2
  local taus=$3
  local scales=$4

  for win in ${wins}; do
    for tau in ${taus}; do
      for scale in ${scales}; do
        run_one "${task}" "${win}" "${tau}" "${scale}"
      done
    done
  done
}

run_grid musique "8 16" "0.90 0.95" "0.65 0.75 0.85 0.95"
run_grid hotpotqa "8 16" "0.95 0.98" "0.75 0.85 0.95"
run_grid multifieldqa_en "8 16" "0.90 0.95" "0.65 0.75 0.85"

echo "==== coverage robustness sweep done ===="
echo "Analyze:"
echo "python scripts/analyze_coverage.py --tasks musique hotpotqa multifieldqa_en --tau 0.5"
