#!/bin/bash
# tmath(TMATH 数学推理) GPU1 部分 (从 run_tmath.sh 拆出)
#   GPU 1: baseline(top0.3/0.5/0.7/1.0) + merge(r0.1-0.9) + evict(r0.1-0.9)
# 数据为本地 dataloader/data/TMATH，无需下载
# 用法:  bash scripts/run_tmath_gpu1.sh
#   实时进度:  tail -f snapshots/tmath/run_gpu1.out
set -e
export TASK=tmath
mkdir -p snapshots/tmath

echo "[GPU1] baseline + merge + evict   ->  snapshots/tmath/run_gpu1.out"
( GPU=1 bash scripts/run_baseline.sh
  GPU=1 bash scripts/run_merge.sh
  GPU=1 bash scripts/run_evict.sh ) > snapshots/tmath/run_gpu1.out 2>&1 &
P0=$!

wait ${P0}
echo "==== tmath GPU1 done -> snapshots/tmath/ ===="
