#!/usr/bin/env bash
set -euo pipefail

# End-to-end TCP replay: one pair x three tasks x 50 samples.
# Usage:
#   bash scripts/run_network_e2e.sh PAIR OUTPUT_DIR \
#     hotpotqa=path/to/per_sample.jsonl \
#     musique=path/to/per_sample.jsonl \
#     multifieldqa_en=path/to/per_sample.jsonl
#
# Optional environment: E2E_HOST, E2E_PORT, LIMIT, CHUNK_BYTES, PAYLOAD_FIELD.
# The 1/10 Gbps and 10/50 ms controls are user-space pacing/sleep emulation,
# not kernel traffic shaping or a substitute for a physical constrained link.

if [[ $# -ne 5 ]]; then
  echo "usage: $0 PAIR OUTPUT_DIR TASK1=JSONL TASK2=JSONL TASK3=JSONL" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PAIR=$1
OUTPUT_DIR=$2
shift 2

E2E_HOST=${E2E_HOST:-127.0.0.1}
E2E_PORT=${E2E_PORT:-29571}
LIMIT=${LIMIT:-50}
CHUNK_BYTES=${CHUNK_BYTES:-65536}
PAYLOAD_FIELD=${PAYLOAD_FIELD:-auto}
mkdir -p "${OUTPUT_DIR}"

SERVER_LOG="${OUTPUT_DIR}/server.log"
python "${ROOT}/scripts/network_e2e.py" server \
  --host "${E2E_HOST}" --port "${E2E_PORT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  python "${ROOT}/scripts/network_e2e.py" shutdown \
    --host "${E2E_HOST}" --port "${E2E_PORT}" >/dev/null 2>&1 || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 100); do
  if [[ -s "${SERVER_LOG}" ]] && rg -q '^READY ' "${SERVER_LOG}"; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "receiver failed to start; see ${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 0.05
done
if ! rg -q '^READY ' "${SERVER_LOG}"; then
  echo "receiver readiness timed out; see ${SERVER_LOG}" >&2
  exit 1
fi

inputs=()
for assignment in "$@"; do
  inputs+=(--input "${assignment}")
done

python "${ROOT}/scripts/network_e2e.py" client \
  --host "${E2E_HOST}" --port "${E2E_PORT}" \
  --pair "${PAIR}" "${inputs[@]}" \
  --limit "${LIMIT}" --chunk-bytes "${CHUNK_BYTES}" \
  --payload-field "${PAYLOAD_FIELD}" \
  --jsonl "${OUTPUT_DIR}/network_e2e.jsonl" \
  --csv "${OUTPUT_DIR}/network_e2e.csv"
