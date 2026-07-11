#!/usr/bin/env python3
"""Summarize failure cases from existing ReKV/B-ReKV per-sample outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REKV_RE = re.compile(r"recv_w(\d+)_r([0-9.]+)")
COV_RE = re.compile(r"cov_t([0-9.]+)_s([0-9.]+)_w(\d+)")


def read_rows(path: Path) -> tuple[dict, list[dict]]:
    meta = {}
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
            else:
                rows.append(row)
    return meta, rows


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
    return None


def dataset_from_path(path: Path) -> str:
    known = {
        "countries",
        "tipsheets",
        "hotpotqa",
        "qasper",
        "musique",
        "multifieldqa_en",
        "twowikimqa",
        "tmath",
    }
    for part in path.parts:
        if part in known:
            return part
    return "unknown"


def latest_runs(root: Path) -> list[Path]:
    latest = {}
    for path in root.glob("**/per_sample.jsonl"):
        label = method_label(path)
        if label is None:
            continue
        dataset = dataset_from_path(path)
        key = (dataset, label)
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path
    return list(latest.values())


def failure_bucket(row: dict, method: str) -> str:
    budget = row.get("budget")
    if method.startswith("B-ReKV") and isinstance(budget, (float, int)) and float(budget) < 0.25:
        return "low_dynamic_budget"
    if method.startswith("ReKV") and isinstance(budget, (float, int)) and float(budget) <= 0.33:
        return "fixed_budget_may_be_tight"
    return "receiver_or_evidence_failure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            Path("snapshots/table1_pair6_llama32_abliterated_deepseek3b"),
            Path("snapshots/table1_pair7_qwen25_uncensored_bespoke"),
            Path("snapshots/table8_pair1_llama31_same"),
        ],
    )
    parser.add_argument("--tasks", nargs="+", default=["hotpotqa", "musique", "multifieldqa_en", "qasper", "twowikimqa"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_examples", type=int, default=30)
    parser.add_argument("--out_dir", type=Path, default=Path("snapshots/analysis/failure_cases"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    example_rows = []

    for root in args.roots:
        if not root.exists():
            continue
        pair = root.name
        for path in latest_runs(root):
            dataset = dataset_from_path(path)
            if dataset not in args.tasks:
                continue
            method = method_label(path)
            if method is None or not (method.startswith("ReKV") or method.startswith("B-ReKV")):
                continue
            _, rows = read_rows(path)
            scores = [float(r.get("score", 0.0)) for r in rows]
            failures = [r for r in rows if float(r.get("score", 0.0)) < args.threshold]
            summary_rows.append(
                {
                    "pair": pair,
                    "dataset": dataset,
                    "method": method,
                    "n": len(rows),
                    "score_mean": round(sum(scores) / max(len(scores), 1), 6),
                    "failure_count": len(failures),
                    "failure_rate": round(len(failures) / max(len(rows), 1), 6),
                    "run_dir": str(path.parent),
                }
            )
            for row in failures[: args.max_examples]:
                example_rows.append(
                    {
                        "pair": pair,
                        "dataset": dataset,
                        "method": method,
                        "idx": row.get("idx"),
                        "id": row.get("id", ""),
                        "score": row.get("score"),
                        "budget": row.get("budget", ""),
                        "failure_bucket": failure_bucket(row, method),
                        "run_dir": str(path.parent),
                    }
                )

    summary_path = args.out_dir / "failure_case_summary.csv"
    examples_path = args.out_dir / "failure_case_examples.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "dataset", "method", "n", "score_mean", "failure_count", "failure_rate", "run_dir"])
        writer.writeheader()
        writer.writerows(summary_rows)
    with examples_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "dataset", "method", "idx", "id", "score", "budget", "failure_bucket", "run_dir"])
        writer.writeheader()
        writer.writerows(example_rows)

    md_path = args.out_dir / "failure_case_report.md"
    lines = [
        "# Failure Case Analysis",
        "",
        f"Failure threshold: score < {args.threshold}",
        "",
        "This report is based on existing `per_sample.jsonl` files. It identifies failure-heavy tasks and examples for manual inspection; raw model responses are not stored in these runs.",
        "",
        "## Outputs",
        "",
        f"- `{summary_path}`",
        f"- `{examples_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {summary_path}")
    print(f"wrote {examples_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
