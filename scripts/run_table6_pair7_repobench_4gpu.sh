#!/usr/bin/env bash
set -euo pipefail

# Run the nine remaining Table 6 pair #7 RepoBench configurations in parallel
# across GPUs 0-3. Completed configurations are skipped by the shared runner.

cd /home/xay/KVMergeComm || exit 1

LOG_ROOT=${LOG_ROOT:-snapshots/table6_pair7_qwen25_uncensored_bespoke/logs}
SKIP_EXISTING=${SKIP_EXISTING:-1}
TAG=$(date '+%m%d_%H%M')

mkdir -p "${LOG_ROOT}"

GPU0_ITEMS="repobench:rekv:8:0.3 repobench:rekv:16:0.5 repobench:brekv:8:0.95:0.75"
GPU1_ITEMS="repobench:rekv:8:0.5 repobench:rekv:16:0.7"
GPU2_ITEMS="repobench:rekv:8:0.7 repobench:brekv:8:0.95:0.85"
GPU3_ITEMS="repobench:rekv:16:0.3 repobench:brekv:16:0.95:0.90"

launch_gpu() {
  local gpu=$1
  local items=$2
  local launcher_log="${LOG_ROOT}/gpu${gpu}_table6_pair7_repobench_launcher_${TAG}.log"

  echo "GPU${gpu}: ${items}"
  GPU="${gpu}" \
  RUN_ITEMS="${items}" \
  SKIP_EXISTING="${SKIP_EXISTING}" \
    bash scripts/run_table6_pair7_remaining_parallel.sh \
    > "${launcher_log}" 2>&1 &

  LAUNCHED_PID=$!
  echo "GPU${gpu} pid=${LAUNCHED_PID} -> ${launcher_log}"
}

declare -a PIDS=()
declare -a GPUS=()

launch_gpu 0 "${GPU0_ITEMS}"
PIDS+=("${LAUNCHED_PID}")
GPUS+=(0)

launch_gpu 1 "${GPU1_ITEMS}"
PIDS+=("${LAUNCHED_PID}")
GPUS+=(1)

launch_gpu 2 "${GPU2_ITEMS}"
PIDS+=("${LAUNCHED_PID}")
GPUS+=(2)

launch_gpu 3 "${GPU3_ITEMS}"
PIDS+=("${LAUNCHED_PID}")
GPUS+=(3)

status=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "GPU${GPUS[$i]} queue completed"
  else
    echo "GPU${GPUS[$i]} queue failed" >&2
    status=1
  fi
done

exit "${status}"
