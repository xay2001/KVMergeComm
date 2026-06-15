#!/bin/bash
# Run the FULL method suite for ONE dataset, sequentially (one dataset at a time).
# usage:  TASK=2wikimqa GPU=0 bash scripts/run_dataset.sh
set -e
export TASK=${TASK:-hotpotqa}
export GPU=${GPU:-0}
echo "################ dataset = ${TASK} (GPU ${GPU}) ################"
bash scripts/run_baseline.sh
bash scripts/run_merge.sh
bash scripts/run_evict.sh
WIN=8  bash scripts/run_receiver.sh
WIN=16 bash scripts/run_receiver.sh
echo "################ ${TASK} done -> snapshots/${TASK}/ ################"
