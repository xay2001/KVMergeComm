#!/usr/bin/env bash
set -euo pipefail

# Full 8-dataset cost profiling on GPU 2.
#
# This runs all samples (LIMIT=0) for:
#   - KVComm top={0.3,0.5,0.7}
#   - RASC w8/w16 r={0.3,0.5,0.7}
#   - Coverage-BRASC t0.95_s0.75_w8 and t0.95_s0.85_w8
#
# Output root:
#   snapshots/cost_profile/pair1_llama31_same_all8_full/
#
# Run:
#   bash scripts/run_cost_profile_all8_gpu2.sh
#
# Analyze:
#   python scripts/analyze_cost_profile.py \
#     --root snapshots/cost_profile/pair1_llama31_same_all8_full \
#     --csv snapshots/cost_profile/pair1_llama31_same_all8_full/cost_table.csv

GPU=${GPU:-2}
LIMIT=${LIMIT:-0}
WARMUP=${WARMUP:-5}
RATIOS=${RATIOS:-"0.3 0.5 0.7"}
TASKS=${TASKS:-"countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath"}
ROOT=${ROOT:-snapshots/cost_profile/pair1_llama31_same_all8_full}
MODEL_A=${MODEL_A:-/sharedspace/models/Llama-3.1-8B-Instruct}
MODEL_B=${MODEL_B:-/sharedspace/models/Llama-3.1-8B-Instruct}

export GPU LIMIT WARMUP RATIOS TASKS ROOT MODEL_A MODEL_B

bash scripts/run_cost_profile.sh
