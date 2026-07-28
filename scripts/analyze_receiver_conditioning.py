#!/usr/bin/env python3
"""Paired analysis for receiver-conditioning causal controls."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = [
    "correct",
    "shuffled",
    "unrelated",
    "sender_text_receiver_encoder",
    "sender_context_q",
    "query_free",
]
BASELINES = VARIANTS[1:]


def read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    rows = []
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            if "_meta" in item:
                meta = item["_meta"]
            else:
                rows.append(item)
    if not rows:
        raise ValueError(f"No samples in {path}")
    return meta, rows


def sample_key(row: dict[str, Any]) -> str:
    index = row.get("idx")
    if index is None:
        raise ValueError(f"Sample is missing idx: {row}")
    return f"{index}::{row.get('id')}"


def discover(root: Path) -> dict[tuple[str, str, str], Path]:
    latest: dict[tuple[str, str, str], Path] = {}
    for path in root.glob("*/*/*/*/per_sample.jsonl"):
        pair, task, variant = path.relative_to(root).parts[:3]
        if variant not in VARIANTS:
            continue
        key = (pair, task, variant)
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path
    cells = {(pair, task) for pair, task, _ in latest}
    for pair, task in cells:
        missing = [
            variant
            for variant in VARIANTS
            if (pair, task, variant) not in latest
        ]
        if missing:
            raise ValueError(f"Incomplete cell {pair}/{task}: missing {missing}")
    if not cells:
        raise ValueError(f"No receiver-conditioning runs found under {root}")
    return latest


def bootstrap_means(
    deltas: np.ndarray,
    count: int,
    rng: np.random.Generator,
    chunk: int = 1000,
) -> np.ndarray:
    output = np.empty(count, dtype=np.float64)
    for start in range(0, count, chunk):
        size = min(chunk, count - start)
        indices = rng.integers(0, deltas.size, size=(size, deltas.size))
        output[start : start + size] = deltas[indices].mean(axis=1)
    return output


def ci(bootstrap: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "p_delta_gt_zero": float(np.mean(bootstrap > 0)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    paths: dict[tuple[str, str, str], Path],
    n_bootstrap: int,
    seed: int,
    budget_tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    cell_rows = []
    boot_by_baseline: dict[str, list[np.ndarray]] = defaultdict(list)
    cells = sorted({(pair, task) for pair, task, _ in paths})
    for pair, task in cells:
        loaded = {}
        meta = {}
        for variant in VARIANTS:
            meta[variant], rows = read_jsonl(paths[(pair, task, variant)])
            if meta[variant].get("query_condition_mode") != variant:
                raise ValueError(
                    f"Variant metadata mismatch in {paths[(pair, task, variant)]}"
                )
            loaded[variant] = {sample_key(row): row for row in rows}
        keys = list(loaded["correct"])
        for variant in BASELINES:
            if set(loaded[variant]) != set(keys):
                raise ValueError(f"Sample mismatch for {pair}/{task}/{variant}")

        correct_scores = np.asarray(
            [float(loaded["correct"][key]["score"]) for key in keys]
        )
        correct_budget = np.asarray(
            [float(loaded["correct"][key]["budget"]) for key in keys]
        )
        for baseline in BASELINES:
            baseline_scores = np.asarray(
                [float(loaded[baseline][key]["score"]) for key in keys]
            )
            baseline_budget = np.asarray(
                [float(loaded[baseline][key]["budget"]) for key in keys]
            )
            replay_target = np.asarray(
                [
                    float(loaded[baseline][key]["replay_target_budget"])
                    for key in keys
                ]
            )
            replay_error = np.abs(baseline_budget - replay_target)
            if float(replay_error.max()) > budget_tolerance + 1e-12:
                raise ValueError(
                    f"Budget replay error for {pair}/{task}/{baseline}: "
                    f"{replay_error.max():.6f} > {budget_tolerance:.6f}"
                )
            deltas = correct_scores - baseline_scores
            bootstrap = bootstrap_means(deltas, n_bootstrap, rng)
            boot_by_baseline[baseline].append(bootstrap)
            sketch_correct = np.asarray(
                [
                    float(loaded["correct"][key].get("query_sketch_bytes", 0))
                    for key in keys
                ]
            )
            sketch_baseline = np.asarray(
                [
                    float(loaded[baseline][key].get("query_sketch_bytes", 0))
                    for key in keys
                ]
            )
            cell_rows.append(
                {
                    "pair": pair,
                    "task": task,
                    "comparison": f"correct - {baseline}",
                    "baseline": baseline,
                    "n": len(keys),
                    "correct_score": float(correct_scores.mean()),
                    "baseline_score": float(baseline_scores.mean()),
                    "mean_delta": float(deltas.mean()),
                    **ci(bootstrap),
                    "correct_budget": float(correct_budget.mean()),
                    "baseline_budget": float(baseline_budget.mean()),
                    "mean_abs_replay_error": float(replay_error.mean()),
                    "max_abs_replay_error": float(replay_error.max()),
                    "sketch_bytes_delta": float(
                        sketch_correct.mean() - sketch_baseline.mean()
                    ),
                }
            )

    macro_rows = []
    for baseline in BASELINES:
        matrix = np.vstack(boot_by_baseline[baseline])
        macro_bootstrap = matrix.mean(axis=0)
        selected = [row for row in cell_rows if row["baseline"] == baseline]
        macro_rows.append(
            {
                "comparison": f"correct - {baseline}",
                "baseline": baseline,
                "n_cells": len(selected),
                "mean_delta": float(
                    np.mean([float(row["mean_delta"]) for row in selected])
                ),
                **ci(macro_bootstrap),
                "mean_budget_delta": float(
                    np.mean(
                        [
                            float(row["correct_budget"])
                            - float(row["baseline_budget"])
                            for row in selected
                        ]
                    )
                ),
                "max_abs_replay_error": max(
                    float(row["max_abs_replay_error"]) for row in selected
                ),
            }
        )
    return cell_rows, macro_rows


def plot_macro(rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["baseline"] for row in rows]
    values = [row["mean_delta"] for row in rows]
    lower = [row["mean_delta"] - row["ci_low"] for row in rows]
    upper = [row["ci_high"] - row["mean_delta"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(range(len(rows)), values, color="#4c78a8")
    ax.errorbar(
        range(len(rows)),
        values,
        yerr=[lower, upper],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(rows)), labels, rotation=20, ha="right")
    ax.set_ylabel("Paired score delta (correct - control)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        path = figure_dir / f"receiver_conditioning_paired_delta.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("snapshots/receiver_conditioning_v1")
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else root / "analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = discover(root)
    cell_rows, macro_rows = analyze(
        paths, args.n_bootstrap, args.seed, args.budget_tolerance
    )
    write_csv(out_dir / "paired_by_cell.csv", cell_rows)
    write_csv(out_dir / "paired_macro.csv", macro_rows)
    figures = plot_macro(macro_rows, out_dir)
    summary = {
        "root": str(root),
        "n_runs": len(paths),
        "n_cells": len({(pair, task) for pair, task, _ in paths}),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "budget_tolerance": args.budget_tolerance,
        "macro": macro_rows,
        "figures": figures,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# Receiver Conditioning 因果对照",
        "",
        f"- Runs: {summary['n_runs']}",
        f"- Cells: {summary['n_cells']}",
        f"- Bootstrap: {args.n_bootstrap}, seed={args.seed}",
        "",
    ]
    for row in macro_rows:
        lines.append(
            f"- correct vs {row['baseline']}: Δ={row['mean_delta']:+.6f}, "
            f"95% CI [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}], "
            f"P(Δ>0)={row['p_delta_gt_zero']:.4f}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"runs={len(paths)} cells={summary['n_cells']} output={out_dir}")


if __name__ == "__main__":
    main()
