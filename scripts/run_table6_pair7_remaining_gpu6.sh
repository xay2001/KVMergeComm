#!/usr/bin/env bash
set -euo pipefail

# GPU6: samsum complete paper-style block.

GPU=${GPU:-6}
RUN_ITEMS=${RUN_ITEMS:-"samsum:rekv:8:0.3 samsum:rekv:8:0.5 samsum:rekv:8:0.7 samsum:rekv:16:0.3 samsum:rekv:16:0.5 samsum:rekv:16:0.7 samsum:brekv:8:0.95:0.75 samsum:brekv:8:0.95:0.85 samsum:brekv:16:0.95:0.90"}
export GPU RUN_ITEMS

bash scripts/run_table6_pair7_remaining_parallel.sh
