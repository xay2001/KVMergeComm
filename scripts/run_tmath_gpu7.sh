#!/bin/bash
# tmath(TMATH 数学推理) GPU7 部分 (从 run_tmath.sh 拆出)
#   GPU 7: receiver w8(r0.1-0.9) + receiver w16(r0.1-0.9)
# 数据为本地 dataloader/data/TMATH，无需下载
# 用法:  bash scripts/run_tmath_gpu7.sh
#   实时进度:  tail -f snapshots/tmath/run_gpu7.out
set -e
export TASK=tmath
mkdir -p snapshots/tmath

echo "[GPU7] receiver w8 + w16          ->  snapshots/tmath/run_gpu7.out"
( GPU=7 WIN=8  bash scripts/run_receiver.sh
  GPU=7 WIN=16 bash scripts/run_receiver.sh ) > snapshots/tmath/run_gpu7.out 2>&1 &
P0=$!

wait ${P0}
echo "==== tmath GPU7 done -> snapshots/tmath/ ===="
