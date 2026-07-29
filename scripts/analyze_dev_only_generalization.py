#!/usr/bin/env python
"""Strict dev-only configuration generalization for ReKV / B-ReKV.

Pre-registered protocol (fixed before looking at test cells):
  - Development cells: pair1_llama31_same x {hotpotqa, multifieldqa_en}.
  - Selection rule on dev only:
      dev-selected fixed r* = smallest grid ratio whose mean dev score is
      >= 97% of the best mean dev score (accuracy-efficiency criterion).
      B-ReKV configuration = the single canonical coverage config present in
      the sweep (t=0.95, s=0.75, w=8), frozen as-is.
  - Frozen configs are then evaluated on ALL remaining pair x task cells.
    No test-cell result feeds back into selection.

Comparisons per test cell:
  - global fixed r=0.30
  - dev-selected fixed r*
  - dev-selected B-ReKV (canonical coverage)
  - per-task best fixed oracle (upper bound, cheats by construction)

Data source: snapshots/full_matched_budget_fairness_query_sketch/
"""

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("snapshots/full_matched_budget_fairness_query_sketch")
DEV_PAIR = "pair1_llama31_same"
DEV_TASKS = {"hotpotqa", "multifieldqa_en"}
GLOBAL_FIXED = 0.30
DEV_TOLERANCE = 0.97


def load_run(per_sample_path):
    scores, budgets = [], []
    with open(per_sample_path) as handle:
        for line in handle:
            row = json.loads(line)
            if "_meta" in row:
                continue
            scores.append(float(row["score"]))
            if row.get("budget") is not None:
                budgets.append(float(row["budget"]))
    if not scores:
        return None
    return {
        "score": sum(scores) / len(scores),
        "budget": sum(budgets) / len(budgets) if budgets else None,
        "n": len(scores),
    }


def collect():
    """-> rekv[pair][task][ratio], brekv[pair][task]"""
    rekv = defaultdict(lambda: defaultdict(dict))
    brekv = defaultdict(dict)
    for per_sample in ROOT.rglob("per_sample.jsonl"):
        parts = per_sample.relative_to(ROOT).parts
        if len(parts) < 4:
            continue
        pair, task, method = parts[0], parts[1], parts[2]
        run = per_sample.parent.name
        stats = load_run(per_sample)
        if stats is None:
            continue
        if method == "fairness_rekv":
            m = re.search(r"_r(\d?\.\d+)", run)
            if m:
                rekv[pair][task][float(m.group(1))] = stats
        elif method == "coverage":
            brekv[pair][task] = stats
    return rekv, brekv


def main():
    rekv, brekv = collect()

    # ---- dev selection (dev cells only) ----
    ratios = sorted(
        set.intersection(
            *[set(rekv[DEV_PAIR][t]) for t in DEV_TASKS if t in rekv[DEV_PAIR]]
        )
    )
    dev_mean = {
        r: statistics.mean(rekv[DEV_PAIR][t][r]["score"] for t in DEV_TASKS)
        for r in ratios
    }
    best_dev = max(dev_mean.values())
    dev_selected = min(r for r in ratios if dev_mean[r] >= DEV_TOLERANCE * best_dev)

    print(f"dev cells: {DEV_PAIR} x {sorted(DEV_TASKS)}")
    print(f"grid ratios: {ratios}")
    print("dev mean score by r:", {r: round(v, 4) for r, v in dev_mean.items()})
    print(f"dev-selected fixed r* = {dev_selected} (>= {DEV_TOLERANCE:.0%} of best {best_dev:.4f})")

    # ---- frozen evaluation on test cells ----
    rows = []
    for pair in sorted(rekv):
        for task in sorted(rekv[pair]):
            if pair == DEV_PAIR and task in DEV_TASKS:
                continue
            grid = rekv[pair][task]
            if GLOBAL_FIXED not in grid or dev_selected not in grid:
                continue
            oracle_r = max(grid, key=lambda r: grid[r]["score"])

            # Budget-matched fixed baseline: linearly interpolate the fixed-r
            # score curve at B-ReKV's actual mean budget for this cell.
            fixed_at_brekv_budget = None
            cell_brekv = brekv.get(pair, {}).get(task)
            if cell_brekv and cell_brekv["budget"] is not None:
                target = cell_brekv["budget"]
                pts = sorted(
                    (grid[r]["budget"], grid[r]["score"])
                    for r in grid
                    if grid[r]["budget"] is not None
                )
                if pts:
                    if target <= pts[0][0]:
                        fixed_at_brekv_budget = pts[0][1]
                    elif target >= pts[-1][0]:
                        fixed_at_brekv_budget = pts[-1][1]
                    else:
                        for (b0, s0), (b1, s1) in zip(pts, pts[1:]):
                            if b0 <= target <= b1:
                                w = (target - b0) / (b1 - b0) if b1 > b0 else 0.0
                                fixed_at_brekv_budget = s0 + w * (s1 - s0)
                                break
            cell = {
                "pair": pair,
                "task": task,
                "global_fixed_r0.30_score": round(grid[GLOBAL_FIXED]["score"], 4),
                "dev_selected_score": round(grid[dev_selected]["score"], 4),
                "dev_selected_r": dev_selected,
                "brekv_score": (
                    round(brekv[pair][task]["score"], 4)
                    if task in brekv.get(pair, {})
                    else None
                ),
                "brekv_budget": (
                    round(brekv[pair][task]["budget"], 4)
                    if task in brekv.get(pair, {}) and brekv[pair][task]["budget"]
                    else None
                ),
                "oracle_score": round(grid[oracle_r]["score"], 4),
                "oracle_r": oracle_r,
                "gap_dev_vs_oracle": round(
                    grid[dev_selected]["score"] - grid[oracle_r]["score"], 4
                ),
                "gap_brekv_vs_oracle": (
                    round(brekv[pair][task]["score"] - grid[oracle_r]["score"], 4)
                    if task in brekv.get(pair, {})
                    else None
                ),
                "fixed_at_brekv_budget_score": (
                    round(fixed_at_brekv_budget, 4)
                    if fixed_at_brekv_budget is not None
                    else None
                ),
                "brekv_vs_budget_matched_fixed": (
                    round(brekv[pair][task]["score"] - fixed_at_brekv_budget, 4)
                    if fixed_at_brekv_budget is not None and task in brekv.get(pair, {})
                    else None
                ),
            }
            rows.append(cell)

    out_dir = ROOT / "analysis"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "dev_only_generalization.csv"
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} test cells)")

    # aggregates
    def agg(key):
        values = [row[key] for row in rows if row[key] is not None]
        return round(statistics.mean(values), 4) if values else None

    brekv_budgets = [row["brekv_budget"] for row in rows if row["brekv_budget"]]
    summary = {
        "n_test_cells": len(rows),
        "mean_global_fixed": agg("global_fixed_r0.30_score"),
        "mean_dev_selected": agg("dev_selected_score"),
        "mean_brekv": agg("brekv_score"),
        "mean_oracle": agg("oracle_score"),
        "mean_gap_dev_vs_oracle": agg("gap_dev_vs_oracle"),
        "mean_gap_brekv_vs_oracle": agg("gap_brekv_vs_oracle"),
        "mean_fixed_at_brekv_budget": agg("fixed_at_brekv_budget_score"),
        "mean_brekv_vs_budget_matched_fixed": agg("brekv_vs_budget_matched_fixed"),
        "cells_brekv_wins_budget_matched": sum(
            1 for row in rows if (row["brekv_vs_budget_matched_fixed"] or 0) > 0
        ),
        "cells_brekv_loses_budget_matched": sum(
            1
            for row in rows
            if row["brekv_vs_budget_matched_fixed"] is not None
            and row["brekv_vs_budget_matched_fixed"] < 0
        ),
        "dev_selected_r": dev_selected,
        "brekv_budget_mean": round(statistics.mean(brekv_budgets), 4) if brekv_budgets else None,
        "brekv_budget_min": round(min(brekv_budgets), 4) if brekv_budgets else None,
        "brekv_budget_max": round(max(brekv_budgets), 4) if brekv_budgets else None,
        "brekv_budget_stdev": (
            round(statistics.stdev(brekv_budgets), 4) if len(brekv_budgets) > 1 else None
        ),
    }
    out_json = out_dir / "dev_only_generalization_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
