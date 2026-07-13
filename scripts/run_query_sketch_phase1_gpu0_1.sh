#!/usr/bin/env bash
set -euo pipefail

cd /home/xay/KVComm || exit 1

# Phase 1 is intentionally sequential:
#   1) freeze the global calibrated B-ReKV configuration;
#   2) verify deployable Query-Sketch against the Full-KV Oracle;
#   3) measure sketch window/representation trade-offs.
# The Oracle stage refuses to run if no candidate passes matched-budget checks.

bash scripts/run_query_sketch_config_freeze_gpu0_1.sh
bash scripts/run_query_sketch_representation_ablation_gpu0_1.sh
bash scripts/run_query_sketch_oracle_gap_gpu0_1.sh
