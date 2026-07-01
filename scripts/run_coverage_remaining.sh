#!/bin/bash
# Run missing B-ReKV experiments for remaining datasets.
#
# Usage:
#   GPU=0 TASK=countries bash scripts/run_coverage_remaining.sh
#   GPU=1 TASK=tipsheets bash scripts/run_coverage_remaining.sh
#   GPU=2 TASK=qasper bash scripts/run_coverage_remaining.sh
#   GPU=5 TASK=tmath bash scripts/run_coverage_remaining.sh
#
# If TASK is unset, runs all remaining tasks sequentially on the same GPU.
set -e

GPU=${GPU:-0}

run_one() {
    local task=$1
    case "${task}" in
        countries|tipsheets)
            # Simple/saturated datasets: sweep both windows lightly.
            TASK=${task} GPU=${GPU} WIN=8  TAUS="0.90 0.95" SCALES="0.6 0.7 0.75 0.8 0.9" bash scripts/run_coverage.sh
            TASK=${task} GPU=${GPU} WIN=16 TAUS="0.90 0.95" SCALES="0.6 0.7 0.75 0.8 0.9" bash scripts/run_coverage.sh
            ;;
        tmath)
            # Math task is less sensitive in prior runs; use a compact conservative sweep.
            TASK=${task} GPU=${GPU} WIN=8  TAUS="0.90 0.95" SCALES="0.7 0.8 0.9 1.0" bash scripts/run_coverage.sh
            TASK=${task} GPU=${GPU} WIN=16 TAUS="0.90 0.95" SCALES="0.7 0.8 0.9 1.0" bash scripts/run_coverage.sh
            ;;
        qasper)
            # No existing ReKV fixed curve in current snapshot; run coverage first.
            # Afterwards run fixed ReKV with scripts/run_receiver.sh if needed.
            TASK=${task} GPU=${GPU} WIN=8  TAUS="0.90 0.95" SCALES="0.6 0.7 0.75 0.8 0.9" bash scripts/run_coverage.sh
            TASK=${task} GPU=${GPU} WIN=16 TAUS="0.90 0.95" SCALES="0.6 0.7 0.75 0.8 0.9" bash scripts/run_coverage.sh
            ;;
        *)
            echo "Unknown TASK=${task}. Use countries | tipsheets | qasper | tmath"
            exit 1
            ;;
    esac
}

if [ -n "${TASK:-}" ]; then
    run_one "${TASK}"
else
    for task in countries tipsheets qasper tmath; do
        run_one "${task}"
    done
fi

echo "==== remaining coverage runs done ===="
echo "Analyze with:"
echo "python scripts/analyze_coverage.py --tasks countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath --tau 0.5"
