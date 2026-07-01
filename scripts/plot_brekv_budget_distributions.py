#!/usr/bin/env python3
"""Plot B-ReKV per-query budget distributions for paper figures.

This is a CPU-only analysis script. It reads existing `per_sample.jsonl` files
and shows that B-ReKV uses query-dependent budgets rather than a hidden fixed
retention ratio.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def latest_run(task: str, tau: float, scale: float, window: int, root: Path) -> Path:
    pattern = root / task / "coverage" / f"cov_t{tau}_s{scale}_w{window}_*" / "per_sample.jsonl"
    paths = [Path(p) for p in glob.glob(str(pattern))]
    if not paths:
        raise FileNotFoundError(f"No B-ReKV run found for {task}: {pattern}")
    return max(paths, key=lambda p: p.stat().st_mtime)


def read_budgets(path: Path) -> list[float]:
    budgets = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row or "budget" not in row:
                continue
            budgets.append(float(row["budget"]))
    if not budgets:
        raise ValueError(f"No budget values found in {path}")
    return budgets


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    idx = min(max(int(round((len(xs) - 1) * p)), 0), len(xs) - 1)
    return xs[idx]


def summarize(task: str, path: Path, budgets: list[float]) -> dict:
    mean = sum(budgets) / len(budgets)
    var = sum((x - mean) ** 2 for x in budgets) / max(len(budgets) - 1, 1)
    return {
        "task": task,
        "n": len(budgets),
        "mean": round(mean, 6),
        "std": round(math.sqrt(var), 6),
        "min": round(min(budgets), 6),
        "p25": round(percentile(budgets, 0.25), 6),
        "p50": round(percentile(budgets, 0.50), 6),
        "p75": round(percentile(budgets, 0.75), 6),
        "p90": round(percentile(budgets, 0.90), 6),
        "max": round(max(budgets), 6),
        "frac_below_0.3": round(sum(x < 0.3 for x in budgets) / len(budgets), 6),
        "frac_above_0.5": round(sum(x > 0.5 for x in budgets) / len(budgets), 6),
        "path": str(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["musique", "hotpotqa", "multifieldqa_en"])
    ap.add_argument("--root", type=Path, default=Path("snapshots"))
    ap.add_argument("--tau", type=float, default=0.95)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("snapshots/brekv_budget_distribution.png"))
    ap.add_argument("--csv", type=Path, default=Path("snapshots/brekv_budget_distribution_summary.csv"))
    ap.add_argument("--fixed_refs", nargs="*", type=float, default=[0.3, 0.5])
    args = ap.parse_args()

    data = []
    rows = []
    for task in args.tasks:
        path = latest_run(task, args.tau, args.scale, args.window, args.root)
        budgets = read_budgets(path)
        data.append((task, path, budgets))
        rows.append(summarize(task, path, budgets))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(len(data), 1, figsize=(9.2, 2.7 * len(data)), sharex=True)
    if len(data) == 1:
        axes = [axes]

    for ax, (task, path, budgets), row in zip(axes, data, rows):
        ax.hist(budgets, bins=28, color="#e45756", alpha=0.82, edgecolor="white")
        ax.axvline(row["mean"], color="#222222", lw=2, label=f"B-ReKV mean={row['mean']:.3f}")
        for ref in args.fixed_refs:
            ax.axvline(ref, color="#4c78a8", ls="--", lw=1.4, alpha=0.75, label=f"fixed r={ref:g}")
        ax.set_ylabel(f"{task}\n#samples")
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.98,
            0.92,
            f"N={row['n']}  p25={row['p25']:.3f}  p50={row['p50']:.3f}  p75={row['p75']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"},
        )
        # Keep only one legend to avoid repeated clutter.
        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Actual KV budget per query (kept fraction)")
    fig.suptitle(
        f"B-ReKV per-query budget distribution (tau={args.tau:.2f}, scale={args.scale:g}, w={args.window})\n"
        "Budgets vary per sample under fixed global hyperparameters",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"wrote {args.csv}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
