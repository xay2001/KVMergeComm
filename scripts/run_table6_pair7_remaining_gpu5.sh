#!/usr/bin/env bash
set -euo pipefail

# GPU5: musique_full complete paper-style block.

GPU=${GPU:-5}
RUN_ITEMS=${RUN_ITEMS:-"musique_full:rekv:8:0.3 musique_full:rekv:8:0.5 musique_full:rekv:8:0.7 musique_full:rekv:16:0.3 musique_full:rekv:16:0.5 musique_full:rekv:16:0.7 musique_full:brekv:8:0.95:0.75 musique_full:brekv:8:0.95:0.85 musique_full:brekv:16:0.95:0.90"}
export GPU RUN_ITEMS

bash scripts/run_table6_pair7_remaining_parallel.sh
