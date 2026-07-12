#!/usr/bin/env python3
"""Summarize cost_profile runs into a compact cost table."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


RECV_RE = re.compile(r"recv_w(\d+)_r([0-9.]+)")
COV_RE = re.compile(r"cov_t([0-9.]+)_s([0-9.]+)_w(\d+)")
KVCOMM_RE = re.compile(r"kvcomm_top([0-9.]+)")


def communication_bytes(summary: dict) -> float | None:
    """Prefer complete bidirectional accounting, with legacy fallback."""
    total = summary.get("total_communication_bytes_mean")
    return total if total is not None else summary.get("kv_bytes_sent_mean")


def method_label(run_dir: Path, summary: dict) -> str:
    name = run_dir.name
    parent = run_dir.parent.name
    if m := RECV_RE.search(name):
        return f"ReKV-w{m.group(1)} r={m.group(2)}"
    if m := COV_RE.search(name):
        return f"B-ReKV t={m.group(1)} s={m.group(2)} w{m.group(3)}"
    if m := KVCOMM_RE.search(name):
        return f"KVComm top={m.group(1)}"
    meta = summary.get("_meta", {})
    if parent == "cost_profile":
        return name
    if meta.get("budget_mode") == "coverage":
        return "Coverage"
    if meta.get("score_mode") == "receiver":
        return "ReKV"
    return name


def dataset_from_path(run_dir: Path) -> str:
    for part in run_dir.parts:
        if part in {"countries", "tipsheets", "hotpotqa", "qasper", "musique", "multifieldqa_en", "twowikimqa", "tmath"}:
            return part
    return "unknown"


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("**/cost_summary.json")):
        summary = json.loads(path.read_text())
        run_dir = path.parent
        total_bytes = communication_bytes(summary)
        selected_kv_bytes = summary.get("kv_bytes_sent_mean")
        rows.append({
            "dataset": dataset_from_path(run_dir),
            "method": method_label(run_dir, summary),
            "n": summary.get("_meta", {}).get("n"),
            "score": summary.get("score_mean"),
            "budget": summary.get("budget_mean"),
            "kv_byte_ratio": summary.get("kv_byte_ratio_mean"),
            "total_communication_bytes": total_bytes,
            "total_communication_mb": round(total_bytes / (1024 ** 2), 3) if total_bytes is not None else None,
            "selected_kv_bytes": selected_kv_bytes,
            "selected_kv_mb": round(selected_kv_bytes / (1024 ** 2), 3) if selected_kv_bytes is not None else None,
            "ctx_tokens_A": summary.get("ctx_tokens_A_mean"),
            "query_tokens_B": summary.get("query_tokens_B_mean"),
            "output_tokens": summary.get("output_tokens_mean"),
            "t_a_prefill": summary.get("t_a_prefill_mean"),
            "t_receiver_score": summary.get("t_receiver_score_mean"),
            "t_generate_total": summary.get("t_generate_total_mean"),
            "t_total": summary.get("t_total_mean"),
            "peak_mem_gb": summary.get("peak_mem_gb_mean"),
            "run_dir": str(run_dir),
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows: list[dict]) -> None:
    cols = [
        "dataset",
        "method",
        "n",
        "score",
        "budget",
        "total_communication_mb",
        "selected_kv_mb",
        "output_tokens",
        "t_a_prefill",
        "t_receiver_score",
        "t_generate_total",
        "t_total",
        "peak_mem_gb",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col)
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append("" if val is None else str(val))
        print("| " + " | ".join(vals) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("snapshots/cost_profile"))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.root)
    rows.sort(key=lambda r: (r["dataset"], r["method"]))
    print_markdown(rows)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
