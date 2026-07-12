#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

# The main accuracy matrix runs first. Cost profiling starts only after all
# three model-pair queues finish, avoiding GPU-memory overlap.
bash scripts/run_query_sketch_table1_main_gpu2_3_7.sh
bash scripts/run_query_sketch_cost_gpu2_3_7.sh
