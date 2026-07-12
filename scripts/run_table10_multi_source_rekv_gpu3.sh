#!/usr/bin/env bash
set -euo pipefail

# Table 10: Multi-Source ReKV on GPU 3.
# Override MODEL_A1/MODEL_A2/MODEL_B/ROOT/TASKS/WINDOWS/RATIOS if needed.

cd /home/xay/KVMergeComm || exit 1

GPU=3 bash scripts/run_table10_multi_source_rekv_gpu.sh
