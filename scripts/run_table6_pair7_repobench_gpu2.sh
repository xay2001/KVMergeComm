#!/usr/bin/env bash
set -euo pipefail

# Table 6 pair #7 RepoBench remaining runs on GPU 2.
# Defaults:
#   Sender:   /NAS/models/Qwen2.5-7B-Instruct-Uncensored
#   Receiver: /NAS/models/Bespoke-Stratos-7B
#
# Override RUN_ITEMS to run a smaller low-memory subset if needed.

cd /home/xay/KVMergeComm || exit 1

DEFAULT_RUN_ITEMS="repobench:rekv:8:0.3 repobench:rekv:8:0.5 repobench:rekv:8:0.7 repobench:rekv:16:0.3 repobench:rekv:16:0.5 repobench:rekv:16:0.7 repobench:brekv:8:0.95:0.75 repobench:brekv:8:0.95:0.85 repobench:brekv:16:0.95:0.90"

GPU=2 RUN_ITEMS="${RUN_ITEMS:-${DEFAULT_RUN_ITEMS}}" bash scripts/run_table6_pair7_remaining_parallel.sh
