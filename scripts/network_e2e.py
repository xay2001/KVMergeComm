#!/usr/bin/env python3
"""Replay recorded per-sample payload sizes over a real TCP socket.

This is intentionally independent from the model/evaluator code.  Bandwidth
and RTT controls are user-space approximations implemented with chunk pacing
and sleeps; they are not kernel traffic shaping and do not reproduce queueing,
loss, jitter, or TCP behavior of a physical constrained link.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path
from typing import BinaryIO


LENGTH = struct.Struct("!I")
DEFAULT_PROFILES = (
    "localhost:0:0",
    "1gbps-10ms:1:10",
    "1gbps-50ms:1:50",
    "10gbps-10ms:10:10",
    "10gbps-50ms:10:50",
)
PAYLOAD_FIELDS = (
    "total_communication_bytes",
    "nld_text_payload_bytes",
    "a_to_b_communication_bytes",
    "kv_bytes_sent",
)
CSV_FIELDS = (
    "pair",
    "task",
    "sample_idx",
    "sample_id",
    "profile",
    "bandwidth_gbps",
    "rtt_ms",
    "payload_field",
    "payload_bytes",
    "serialization_s",
    "transmission_s",
    "deserialization_s",
    "compute_s",
    "e2e_s",
    "decomposition_s",
    "wire_throughput_mbps",
    "checksum",
    "source",
    "timestamp_utc",
)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(min(remaining, 1024 * 1024))
        if not chunk:
            raise ConnectionError(f"socket closed with {remaining} bytes remaining")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_header(header: dict) -> bytes:
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("protocol header is too large")
    return LENGTH.pack(len(raw)) + raw


def recv_header(sock: socket.socket) -> tuple[dict, float]:
    raw = recv_exact(sock, LENGTH.unpack(recv_exact(sock, LENGTH.size))[0])
    tic = time.perf_counter()
    header = json.loads(raw)
    elapsed = time.perf_counter() - tic
    if not isinstance(header, dict):
        raise ValueError("protocol header must be a JSON object")
    return header, elapsed


def sleep_until(deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def send_payload_paced(
    sock: socket.socket,
    payload: bytes,
    bandwidth_gbps: float,
    chunk_bytes: int,
) -> None:
    if bandwidth_gbps <= 0:
        sock.sendall(payload)
        return
    bytes_per_second = bandwidth_gbps * 1_000_000_000 / 8.0
    started = time.perf_counter()
    sent = 0
    view = memoryview(payload)
    while sent < len(payload):
        end = min(sent + chunk_bytes, len(payload))
        sock.sendall(view[sent:end])
        sent = end
        # Cumulative deadlines avoid accumulating scheduler drift.
        sleep_until(started + sent / bytes_per_second)


def deterministic_payload(size: int, seed: int) -> bytes:
    return bytes((seed & 0xFF,)) * size


def handle_sample(
    conn: socket.socket,
    header: dict,
    header_deserialization_s: float,
    max_payload_bytes: int,
) -> None:
    payload_size = int(header["payload_bytes"])
    bandwidth_gbps = float(header["bandwidth_gbps"])
    rtt_ms = float(header["rtt_ms"])
    if payload_size < 0 or payload_size > max_payload_bytes:
        raise ValueError(
            f"payload_bytes={payload_size} is outside [0, {max_payload_bytes}]"
        )
    if bandwidth_gbps < 0 or rtt_ms < 0:
        raise ValueError("bandwidth_gbps and rtt_ms must be non-negative")

    chunks: list[bytes] = []
    remaining = payload_size
    while remaining:
        chunk = conn.recv(min(remaining, 1024 * 1024))
        if not chunk:
            raise ConnectionError("socket closed while receiving payload")
        chunks.append(chunk)
        remaining -= len(chunk)

    tic = time.perf_counter()
    payload = b"".join(chunks)
    if len(payload) != payload_size:
        raise ValueError("deserialized payload length mismatch")
    deserialization_s = header_deserialization_s + time.perf_counter() - tic

    tic = time.perf_counter()
    checksum = hashlib.sha256(payload).hexdigest()
    compute_s = time.perf_counter() - tic

    # Half the configured RTT is placed on each direction.  The request-side
    # half is applied by the client; this is the response-side half.
    if rtt_ms:
        time.sleep(rtt_ms / 2000.0)
    conn.sendall(
        encode_header(
            {
                "status": "ok",
                "payload_bytes": payload_size,
                "checksum": checksum,
                "deserialization_s": deserialization_s,
                "compute_s": compute_s,
            }
        )
    )


def serve(args: argparse.Namespace) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(args.backlog)
        print(f"READY {listener.getsockname()[0]}:{listener.getsockname()[1]}", flush=True)
        while True:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(args.socket_timeout)
                while True:
                    try:
                        header, header_deserialization_s = recv_header(conn)
                    except ConnectionError:
                        break
                    command = header.get("command")
                    if command == "shutdown":
                        conn.sendall(encode_header({"status": "bye"}))
                        return
                    if command != "sample":
                        raise ValueError(f"unknown command: {command!r}")
                    handle_sample(
                        conn,
                        header,
                        header_deserialization_s,
                        args.max_payload_bytes,
                    )


def parse_assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be TASK=PATH")
    task, raw_path = value.split("=", 1)
    if not task or not raw_path:
        raise argparse.ArgumentTypeError("--input must be TASK=PATH")
    return task, Path(raw_path)


def parse_profile(value: str) -> tuple[str, float, float]:
    try:
        name, bandwidth, rtt = value.rsplit(":", 2)
        bandwidth_gbps = float(bandwidth)
        rtt_ms = float(rtt)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--profile must be NAME:BANDWIDTH_GBPS:RTT_MS"
        ) from exc
    if not name or bandwidth_gbps < 0 or rtt_ms < 0:
        raise argparse.ArgumentTypeError("profile values must be non-negative")
    return name, bandwidth_gbps, rtt_ms


def read_samples(
    task: str,
    path: Path,
    limit: int,
    payload_field: str,
) -> list[dict]:
    rows: list[dict] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "_meta" in row:
                continue
            field = payload_field
            if field == "auto":
                field = next((key for key in PAYLOAD_FIELDS if row.get(key) is not None), "")
            if not field or row.get(field) is None:
                raise ValueError(
                    f"{path}:{line_number}: no usable payload field; tried {PAYLOAD_FIELDS}"
                )
            size = int(row[field])
            if size < 0:
                raise ValueError(f"{path}:{line_number}: negative payload size")
            rows.append(
                {
                    "task": task,
                    "sample_idx": row.get("idx", len(rows)),
                    "sample_id": str(row.get("id", row.get("idx", len(rows)))),
                    "payload_field": field,
                    "payload_bytes": size,
                    "source": str(path),
                }
            )
            if len(rows) >= limit:
                break
    if len(rows) < limit:
        raise ValueError(f"{path}: requested {limit} samples, found {len(rows)}")
    return rows


def result_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row["pair"]),
        str(row["task"]),
        str(row["sample_idx"]),
        str(row["profile"]),
    )


def load_results(path: Path) -> tuple[list[dict], set[tuple[str, str, str, str]]]:
    rows: list[dict] = []
    completed: set[tuple[str, str, str, str]] = set()
    if not path.exists():
        return rows, completed
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed.add(result_key(row))
                rows.append(row)
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"warning: ignoring malformed {path}:{line_number}: {exc}", file=sys.stderr)
    return rows, completed


def write_csv(rows: list[dict], path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def append_jsonl(stream: BinaryIO, row: dict) -> None:
    stream.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
    stream.flush()
    os.fsync(stream.fileno())


def replay_one(
    conn: socket.socket,
    pair: str,
    sample: dict,
    profile: tuple[str, float, float],
    chunk_bytes: int,
) -> dict:
    profile_name, bandwidth_gbps, rtt_ms = profile
    started = time.perf_counter()
    tic = time.perf_counter()
    payload = deterministic_payload(
        sample["payload_bytes"],
        hash((pair, sample["task"], sample["sample_idx"])) & 0xFF,
    )
    request = encode_header(
        {
            "command": "sample",
            "payload_bytes": sample["payload_bytes"],
            "bandwidth_gbps": bandwidth_gbps,
            "rtt_ms": rtt_ms,
        }
    )
    serialization_s = time.perf_counter() - tic

    transmission_started = time.perf_counter()
    if rtt_ms:
        time.sleep(rtt_ms / 2000.0)
    conn.sendall(request)
    send_payload_paced(conn, payload, bandwidth_gbps, chunk_bytes)
    response, _ = recv_header(conn)
    socket_roundtrip_s = time.perf_counter() - transmission_started
    if response.get("status") != "ok":
        raise RuntimeError(f"server error: {response}")

    deserialization_s = float(response["deserialization_s"])
    compute_s = float(response["compute_s"])
    transmission_s = max(0.0, socket_roundtrip_s - deserialization_s - compute_s)
    e2e_s = time.perf_counter() - started
    decomposition_s = serialization_s + transmission_s + deserialization_s + compute_s
    return {
        "pair": pair,
        **sample,
        "profile": profile_name,
        "bandwidth_gbps": bandwidth_gbps,
        "rtt_ms": rtt_ms,
        "serialization_s": serialization_s,
        "transmission_s": transmission_s,
        "deserialization_s": deserialization_s,
        "compute_s": compute_s,
        "e2e_s": e2e_s,
        "decomposition_s": decomposition_s,
        "wire_throughput_mbps": (
            sample["payload_bytes"] * 8.0 / transmission_s / 1_000_000
            if transmission_s > 0
            else None
        ),
        "checksum": response["checksum"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def client(args: argparse.Namespace) -> None:
    if len(args.input) != args.expected_tasks:
        raise ValueError(
            f"expected exactly {args.expected_tasks} --input entries, got {len(args.input)}"
        )
    tasks = [task for task, _ in args.input]
    if len(set(tasks)) != len(tasks):
        raise ValueError("task names in --input must be unique")
    profiles = args.profile or [parse_profile(value) for value in DEFAULT_PROFILES]
    samples = [
        sample
        for task, path in args.input
        for sample in read_samples(task, path, args.limit, args.payload_field)
    ]

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows, completed = load_results(args.jsonl)
    total = len(samples) * len(profiles)
    done = sum(
        (args.pair, sample["task"], str(sample["sample_idx"]), profile[0]) in completed
        for profile in profiles
        for sample in samples
    )
    print(f"resume: {done}/{total} rows already complete", flush=True)

    with socket.create_connection((args.host, args.port), args.connect_timeout) as conn:
        conn.settimeout(args.socket_timeout)
        with args.jsonl.open("ab") as output:
            for profile in profiles:
                for sample in samples:
                    key = (
                        args.pair,
                        sample["task"],
                        str(sample["sample_idx"]),
                        profile[0],
                    )
                    if key in completed:
                        continue
                    row = replay_one(conn, args.pair, sample, profile, args.chunk_bytes)
                    append_jsonl(output, row)
                    rows.append(row)
                    completed.add(key)
                    done += 1
                    print(
                        f"[{done}/{total}] {profile[0]} {sample['task']} "
                        f"idx={sample['sample_idx']} bytes={sample['payload_bytes']} "
                        f"e2e={row['e2e_s']:.6f}s",
                        flush=True,
                    )
    write_csv(rows, args.csv)


def shutdown(args: argparse.Namespace) -> None:
    with socket.create_connection((args.host, args.port), args.connect_timeout) as conn:
        conn.sendall(encode_header({"command": "shutdown"}))
        response, _ = recv_header(conn)
        if response.get("status") != "bye":
            raise RuntimeError(f"unexpected shutdown response: {response}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server = subparsers.add_parser("server", help="run the receiver process")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=29571)
    server.add_argument("--backlog", type=int, default=8)
    server.add_argument("--socket-timeout", type=float, default=300.0)
    server.add_argument("--max-payload-bytes", type=int, default=2 * 1024**3)
    server.set_defaults(func=serve)

    run = subparsers.add_parser("client", help="replay JSONL payloads")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=29571)
    run.add_argument("--pair", required=True)
    run.add_argument("--input", action="append", type=parse_assignment, required=True)
    run.add_argument("--expected-tasks", type=int, default=3)
    run.add_argument("--limit", type=int, default=50)
    run.add_argument("--payload-field", default="auto")
    run.add_argument(
        "--profile",
        action="append",
        type=parse_profile,
        help="NAME:BANDWIDTH_GBPS:RTT_MS; repeatable (default: localhost + 1/10 Gbps x 10/50 ms)",
    )
    run.add_argument("--chunk-bytes", type=int, default=64 * 1024)
    run.add_argument("--connect-timeout", type=float, default=10.0)
    run.add_argument("--socket-timeout", type=float, default=300.0)
    run.add_argument("--jsonl", type=Path, required=True)
    run.add_argument("--csv", type=Path, required=True)
    run.set_defaults(func=client)

    stop = subparsers.add_parser("shutdown", help="stop a receiver cleanly")
    stop.add_argument("--host", default="127.0.0.1")
    stop.add_argument("--port", type=int, default=29571)
    stop.add_argument("--connect-timeout", type=float, default=10.0)
    stop.set_defaults(func=shutdown)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
