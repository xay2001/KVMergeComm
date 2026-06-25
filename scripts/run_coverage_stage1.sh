#!/bin/bash
# Stage-1 Coverage-BRASC queue from the offline pre-check.
#
# Usage:
#   GPU=2 TASK=musique bash scripts/run_coverage_stage1.sh
#   GPU=2 TASK=hotpotqa bash scripts/run_coverage_stage1.sh
#   GPU=2 TASK=twowikimqa bash scripts/run_coverage_stage1.sh
#
# Or run the recommended order:
#   GPU=2 bash scripts/run_coverage_stage1.sh
set -e

GPU=${GPU:-0}
WIN=${WIN:-16}

run_task() {
    local task=$1
    case "${task}" in
        musique)
            # Strongest offline signal: rcap90 x0.7-0.9 and rcap95 x0.8.
            TASK=musique GPU=${GPU} WIN=${WIN} TAUS="0.90" SCALES="0.7 0.75 0.8 0.9" bash scripts/run_coverage.sh
            TASK=musique GPU=${GPU} WIN=${WIN} TAUS="0.95" SCALES="0.8" bash scripts/run_coverage.sh
            ;;
        hotpotqa)
            # Local positive points only; run fewer candidates.
            TASK=hotpotqa GPU=${GPU} WIN=${WIN} TAUS="0.90" SCALES="0.7 0.9" bash scripts/run_coverage.sh
            TASK=hotpotqa GPU=${GPU} WIN=${WIN} TAUS="0.95" SCALES="0.9" bash scripts/run_coverage.sh
            ;;
        twowikimqa)
            # Non-monotonic fixed curve; use as secondary evidence.
            TASK=twowikimqa GPU=${GPU} WIN=${WIN} TAUS="0.90" SCALES="0.4 0.5 0.6 0.7" bash scripts/run_coverage.sh
            ;;
        *)
            echo "Unknown TASK=${task}. Use musique | hotpotqa | twowikimqa"
            exit 1
            ;;
    esac
}

if [ -n "${TASK:-}" ]; then
    run_task "${TASK}"
else
    run_task musique
    run_task hotpotqa
    run_task twowikimqa
fi

echo "==== Stage-1 coverage runs done. Analyze with:"
echo "python scripts/analyze_coverage.py --tasks musique hotpotqa twowikimqa --tau 0.5"
