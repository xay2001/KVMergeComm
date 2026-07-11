#!/usr/bin/env bash
set -euo pipefail

# GPU7: repobench complete paper-style block.

GPU=${GPU:-7}
RUN_ITEMS=${RUN_ITEMS:-"repobench:rekv:8:0.3 repobench:rekv:8:0.5 repobench:rekv:8:0.7 repobench:rekv:16:0.3 repobench:rekv:16:0.5 repobench:rekv:16:0.7 repobench:brekv:8:0.95:0.75 repobench:brekv:8:0.95:0.85 repobench:brekv:16:0.95:0.90"}
export GPU RUN_ITEMS

bash scripts/run_table6_pair7_remaining_parallel.sh
