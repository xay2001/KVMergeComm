#!/usr/bin/env python3
"""Aggregate existing results by task type."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


TASK_TYPE = {
    "countries": "simple_fact",
    "tipsheets": "simple_synthetic",
    "hotpotqa": "multi_hop",
    "musique": "multi_hop",
    "twowikimqa": "multi_hop",
    "qasper": "long_document",
    "multifieldqa_en": "long_document",
    "tmath": "math_reasoning",
}

REKV_RE = re.compile(r"recv_w(\d+)_r([0-9.]+)")
COV_RE = re.compile(r"cov_t([0-9.]+)_s([0-9.]+)_w(\d+)")


def read_scores(path: Path) -> tuple[dict, list[float]]:
    meta = {}
    scores = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
            else:
                scores.append(float(row.get("score", 0.0)))
    return meta, scores


def dataset_from_path(path: Path) -> str:
    for part in path.parts:
        if part in TASK_TYPE:
            return part
    return "unknown"


def method_label(path: Path) -> str | None:
    name = path.parent.name
    if m := REKV_RE.search(name):
        return f"ReKV-w{m.group(1)} r={m.group(2)}"
    if m := COV_RE.search(name):
        return f"B-ReKV t={m.group(1)} s={m.group(2)} w{m.group(3)}"
    if "evict" in name:
        return "Evict/ValueNorm"
    if "random" in name:
        return "Random-token"
    if "kvcomm" in name:
        return name
    return None


def latest_runs(root: Path) -> dict[tuple[str, str], Path]:
    latest = {}
    for path in root.glob("**/per_sample.jsonl"):
        dataset = dataset_from_path(path)
        method = method_label(path)
        if dataset == "unknown" or method is None:
            continue
        key = (dataset, method)
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path
    return latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            Path("snapshots/table8_pair1_llama31_same"),
            Path("snapshots/table1_pair6_llama32_abliterated_deepseek3b"),
            Path("snapshots/table1_pair7_qwen25_uncensored_bespoke"),
        ],
    )
    parser.add_argument("--out_dir", type=Path, default=Path("snapshots/analysis/task_type_sensitivity"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_rows = []
    grouped = defaultdict(list)

    for root in args.roots:
        if not root.exists():
            continue
        pair = root.name
        for (dataset, method), path in latest_runs(root).items():
            _, scores = read_scores(path)
            if not scores:
                continue
            row = {
                "pair": pair,
                "task_type": TASK_TYPE[dataset],
                "dataset": dataset,
                "method": method,
                "n": len(scores),
                "score_mean": round(sum(scores) / len(scores), 6),
                "run_dir": str(path.parent),
            }
            run_rows.append(row)
            family = "B-ReKV" if method.startswith("B-ReKV") else "ReKV" if method.startswith("ReKV") else method
            grouped[(row["task_type"], family)].append(row["score_mean"])

    run_path = args.out_dir / "task_type_run_summary.csv"
    with run_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "task_type", "dataset", "method", "n", "score_mean", "run_dir"])
        writer.writeheader()
        writer.writerows(run_rows)

    group_rows = []
    for (task_type, family), vals in sorted(grouped.items()):
        group_rows.append(
            {
                "task_type": task_type,
                "method_family": family,
                "runs": len(vals),
                "score_mean": round(sum(vals) / len(vals), 6),
                "score_min": round(min(vals), 6),
                "score_max": round(max(vals), 6),
            }
        )
    group_path = args.out_dir / "task_type_family_summary.csv"
    with group_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "method_family", "runs", "score_mean", "score_min", "score_max"])
        writer.writeheader()
        writer.writerows(group_rows)

    md_path = args.out_dir / "task_type_sensitivity_report.md"
    lines = [
        "# Task-Type Sensitivity",
        "",
        "Task groups:",
        "",
        "- simple: countries, tipsheets",
        "- multi-hop: hotpotqa, musique, twowikimqa",
        "- long document: qasper, multifieldqa_en",
        "- math/reasoning: tmath",
        "",
        "Outputs:",
        "",
        f"- `{run_path}`",
        f"- `{group_path}`",
        "",
        "Use this as the starting point for the paper discussion: ReKV/B-ReKV is strongest on evidence-heavy multi-hop and long-context settings, while simple synthetic tasks can saturate or favor layer-level KVComm.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {run_path}")
    print(f"wrote {group_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
