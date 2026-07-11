#!/usr/bin/env python3
"""Plot outputs from the analysis suite."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_supporting_overlap(summary_path: Path, out_dir: Path) -> Path:
    rows = read_csv(summary_path)
    if not rows:
        raise RuntimeError(f"empty summary: {summary_path}")
    row = rows[0]
    labels = ["ReKV", "Evict", "Random"]
    vals = [
        float(row["rekv_support_rate"]),
        float(row["evict_support_rate"]),
        float(row["random_support_rate"]),
    ]
    out = out_dir / "supporting_overlap_bar.png"
    plt.figure(figsize=(5.5, 4.0))
    bars = plt.bar(labels, vals, color=["#4c78a8", "#f58518", "#bab0ac"])
    plt.ylabel("Top-k tokens in supporting facts")
    plt.ylim(0, max(vals) * 1.25)
    plt.title(f"HotpotQA Supporting-Facts Overlap (top-{row['top_k']}, n={row['n']})")
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def plot_task_type(task_summary_path: Path, out_dir: Path) -> Path:
    rows = read_csv(task_summary_path)
    task_types = ["simple_fact", "simple_synthetic", "multi_hop", "long_document", "math_reasoning"]
    families = ["ReKV", "B-ReKV"]
    data = {(r["task_type"], r["method_family"]): float(r["score_mean"]) for r in rows}
    x = list(range(len(task_types)))
    width = 0.36
    out = out_dir / "task_type_sensitivity_bar.png"
    plt.figure(figsize=(8.5, 4.2))
    for i, fam in enumerate(families):
        vals = [data.get((t, fam), 0.0) for t in task_types]
        offs = [v + (i - 0.5) * width for v in x]
        plt.bar(offs, vals, width=width, label=fam)
    plt.xticks(x, ["simple\nfact", "simple\nsynthetic", "multi-hop", "long\ndoc", "math"])
    plt.ylabel("Mean score over runs")
    plt.title("Task-Type Sensitivity")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def family(method: str) -> str:
    if method.startswith("B-ReKV"):
        return "B-ReKV"
    if method.startswith("ReKV"):
        return "ReKV"
    return method


def plot_failure_heatmap(failure_summary_path: Path, out_dir: Path) -> Path:
    rows = read_csv(failure_summary_path)
    acc = defaultdict(list)
    for row in rows:
        fam = family(row["method"])
        if fam not in {"ReKV", "B-ReKV"}:
            continue
        acc[(row["dataset"], fam)].append(float(row["failure_rate"]))
    datasets = sorted({k[0] for k in acc})
    families = ["ReKV", "B-ReKV"]
    matrix = []
    for ds in datasets:
        row = []
        for fam in families:
            vals = acc.get((ds, fam), [])
            row.append(sum(vals) / len(vals) if vals else 0.0)
        matrix.append(row)

    out = out_dir / "failure_rate_heatmap.png"
    plt.figure(figsize=(5.8, max(3.2, 0.42 * len(datasets) + 1.2)))
    im = plt.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="magma_r")
    plt.colorbar(im, label="Failure rate (score < 0.5)")
    plt.xticks(range(len(families)), families)
    plt.yticks(range(len(datasets)), datasets)
    plt.title("Failure Rate by Dataset and Method Family")
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="white" if val > 0.55 else "black")
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supporting_summary", type=Path, default=Path("snapshots/supporting_overlap/hotpotqa_pair1_full_context/supporting_overlap_summary_top20_w8_r0.3.csv"))
    parser.add_argument("--task_summary", type=Path, default=Path("snapshots/analysis/task_type_sensitivity/task_type_family_summary.csv"))
    parser.add_argument("--failure_summary", type=Path, default=Path("snapshots/analysis/failure_cases/failure_case_summary.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("snapshots/analysis/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        plot_supporting_overlap(args.supporting_summary, args.out_dir),
        plot_task_type(args.task_summary, args.out_dir),
        plot_failure_heatmap(args.failure_summary, args.out_dir),
    ]
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
