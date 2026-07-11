#!/usr/bin/env bash
set -euo pipefail

# GPU4: remaining qasper_full jobs. The interrupted w16 r=0.5 run is rerun here.

GPU=${GPU:-4}
RUN_ITEMS=${RUN_ITEMS:-"qasper_full:rekv:16:0.5 qasper_full:rekv:16:0.7 qasper_full:brekv:8:0.95:0.75 qasper_full:brekv:8:0.95:0.85 qasper_full:brekv:16:0.95:0.90"}
export GPU RUN_ITEMS

bash scripts/run_table6_pair7_remaining_parallel.sh
