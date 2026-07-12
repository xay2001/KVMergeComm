#!/usr/bin/env python3
"""Summarize the quick Query-Sketch ReKV vs Full-KV Oracle comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("snapshots/query_sketch_rekv_quick/pair1_llama31_same")


def latest_summaries() -> list[dict]:
    latest: dict[tuple[str, str], Path] = {}
    for path in ROOT.glob("*/*/*/cost_summary.json"):
        rel = path.relative_to(ROOT)
        task, method = rel.parts[:2]
        key = (task, method)
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path

    rows = []
    for (task, method), path in sorted(latest.items()):
        data = json.loads(path.read_text())
        rows.append(
            {
                "task": task,
                "method": method,
                "score": data.get("score_mean"),
                "query_sketch_bytes": data.get("query_sketch_bytes_mean"),
                "selected_kv_bytes": data.get("kv_bytes_sent_mean"),
                "oracle_full_kv_bytes": data.get("oracle_full_kv_bytes_mean"),
                "total_communication_bytes": data.get("total_communication_bytes_mean"),
                "scoring_time_s": data.get("t_receiver_score_mean"),
                "total_time_s": data.get("t_total_mean"),
                "peak_mem_gb": data.get("peak_mem_gb_mean"),
                "source": str(path),
            }
        )
    return rows


def main() -> None:
    rows = latest_summaries()
    if not rows:
        raise SystemExit(f"No cost summaries found under {ROOT}")

    out_csv = ROOT / "query_sketch_vs_oracle_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], {})[row["method"]] = row

    report = [
        "# Query-Sketch ReKV vs Full-KV Oracle",
        "",
        "| Task | Query-Sketch | Oracle | Score delta | QS total MB | Oracle full-KV MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, methods in sorted(by_task.items()):
        qs = methods.get("query_sketch_rekv")
        oracle = methods.get("full_kv_oracle_rekv")
        if not qs or not oracle:
            continue
        qs_score = float(qs["score"])
        oracle_score = float(oracle["score"])
        qs_mb = float(qs["total_communication_bytes"]) / 1024**2
        oracle_mb = float(oracle["total_communication_bytes"]) / 1024**2
        report.append(
            f"| {task} | {qs_score:.4f} | {oracle_score:.4f} | "
            f"{qs_score - oracle_score:+.4f} | {qs_mb:.2f} | {oracle_mb:.2f} |"
        )

    out_md = ROOT / "query_sketch_vs_oracle_report.md"
    out_md.write_text("\n".join(report) + "\n")
    print(f"summary: {out_csv}")
    print(f"report: {out_md}")


if __name__ == "__main__":
    main()
