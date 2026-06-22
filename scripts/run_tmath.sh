#!/bin/bash
# tmath(TMATH 数学推理)全套方法,1/7 两张卡并行  (数据为本地 dataloader/data/TMATH，无需下载)
#   GPU 1: bash scripts/run_tmath_gpu1.sh
#   GPU 7: bash scripts/run_tmath_gpu7.sh
# 用法:  bash scripts/run_tmath.sh          # 两张卡同时跑
#   实时进度:  tail -f snapshots/tmath/run_gpu1.out   (或 run_gpu7.out)
set -e

bash scripts/run_tmath_gpu1.sh &
P0=$!
bash scripts/run_tmath_gpu7.sh &
P1=$!

wait ${P0} ${P1}
echo "==== tmath done -> snapshots/tmath/ ===="
