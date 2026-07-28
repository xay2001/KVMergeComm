#!/usr/bin/env python3
"""Offline held-out, exact-oracle, and shuffled-budget analysis for B-ReKV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PAIRS = [
    "pair1_llama31_same",
    "pair2_llama32_same",
    "pair3_qwen25_7b_same",
    "pair4_falcon3_7b_same",
    "pair5_evolcodellama_toolace",
    "pair6_llama32_abliterated_deepseek3b",
    "pair7_qwen25_uncensored_bespoke",
]
TASKS = [
    "countries", "tipsheets", "hotpotqa", "qasper", "musique",
    "multifieldqa_en", "twowikimqa", "tmath",
]
DEV_CELLS = {(PAIRS[0], "hotpotqa"), (PAIRS[0], "musique")}
HELDOUT_CELLS = [(pair, task) for pair in PAIRS for task in TASKS
                 if (pair, task) not in DEV_CELLS]
POLICY_RATIOS = (0.2, 0.3, 0.4, 0.5, 0.6)
SELECTION_RATIOS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)
ORACLE_RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
FIXED_RE = re.compile(r"rekv_w(?P<window>\d+)_r(?P<ratio>[0-9.]+)(?:_|$)")
BREKV_RE = re.compile(
    r"cov_t(?P<tau>[0-9.]+)_s(?P<scale>[0-9.]+)_w(?P<window>\d+)(?:_|$)"
)
BYTE_FIELDS = (
    "total_communication_bytes",
    "a_to_b_communication_bytes",
    "b_to_a_communication_bytes",
)


@dataclass(frozen=True)
class Run:
    pair: str
    task: str
    method: str
    ratio: float | None
    path: Path


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def sample_key(row: dict[str, Any]) -> str:
    if row.get("idx") is None:
        raise ValueError("Sample row is missing idx")
    return f"{row['idx']}::{row.get('id')}"


def read_samples(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            if "_meta" in item:
                continue
            key = sample_key(item)
            if key in rows:
                raise ValueError(f"Duplicate sample {key} in {path}")
            required = ("score", "budget", *BYTE_FIELDS)
            missing = [field for field in required if item.get(field) is None]
            if missing:
                raise ValueError(f"Missing {missing} at {path}:{line_number}")
            rows[key] = {field: float(item[field]) for field in required}
            if item.get("replay_target_budget") is not None:
                rows[key]["replay_target_budget"] = float(
                    item["replay_target_budget"]
                )
    if not rows:
        raise ValueError(f"No samples in {path}")
    return rows


def _newer(latest: dict[Any, Path], key: Any, path: Path) -> None:
    if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
        latest[key] = path


def discover_main(root: Path) -> list[Run]:
    latest: dict[tuple[Any, ...], Path] = {}
    for path in root.glob("*/*/*/*/per_sample.jsonl"):
        pair, task, _, name = path.relative_to(root).parts[:4]
        fixed = FIXED_RE.match(name)
        adaptive = BREKV_RE.match(name)
        if fixed and int(fixed.group("window")) == 8:
            key = (pair, task, "fixed", float(fixed.group("ratio")))
        elif (adaptive and float(adaptive.group("tau")) == 0.95
              and float(adaptive.group("scale")) == 0.75
              and int(adaptive.group("window")) == 8):
            key = (pair, task, "brekv", None)
        else:
            # Other methods/configurations still count toward the 1568-run validation.
            key = (pair, task, "other", name.split("_0", 1)[0])
        _newer(latest, key, path)
    # Validate the source through its authoritative summary if present, and by
    # expected fixed/B-ReKV coverage below. Other methods need not be loaded.
    summary = root / "analysis" / "final_summary.json"
    if not summary.exists():
        raise ValueError(f"Missing 1568-run completion summary: {summary}")
    completion = json.loads(summary.read_text()).get("completion", {})
    if completion != {"completed": 1568, "expected": 1568}:
        raise ValueError(f"Main sweep is not complete: {completion}")
    runs = [
        Run(pair, task, method, ratio, path)
        for (pair, task, method, ratio), path in latest.items()
        if method in {"fixed", "brekv"}
    ]
    return runs


def discover_r07(root: Path) -> dict[tuple[str, str], Path]:
    latest: dict[tuple[str, str], Path] = {}
    for path in root.glob("*/*/*/*/per_sample.jsonl"):
        pair, task, _, name = path.relative_to(root).parts[:4]
        match = FIXED_RE.match(name)
        if match and int(match.group("window")) == 8 and math.isclose(
            float(match.group("ratio")), 0.7, abs_tol=1e-9
        ):
            _newer(latest, (pair, task), path)
    expected = set(HELDOUT_CELLS)
    if set(latest) != expected:
        raise ValueError(f"r=.7 coverage mismatch; missing={sorted(expected-set(latest))}, "
                         f"extra={sorted(set(latest)-expected)}")
    return latest


def load_cells(runs: list[Run]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"fixed": {}}
    )
    for run in runs:
        cell = indexed[(run.pair, run.task)]
        if run.method == "fixed":
            cell["fixed"][run.ratio] = read_samples(run.path)
        else:
            cell["brekv"] = read_samples(run.path)
    expected = {(pair, task) for pair in PAIRS for task in TASKS}
    if set(indexed) != expected:
        raise ValueError(f"Main cell coverage mismatch: {sorted(expected-set(indexed))}")
    for key, cell in indexed.items():
        if "brekv" not in cell:
            raise ValueError(f"Missing canonical B-ReKV in {key}")
        ratios = set(cell["fixed"])
        needed = set(SELECTION_RATIOS)
        if not needed <= ratios:
            raise ValueError(f"Missing fixed ratios in {key}: {sorted(needed-ratios)}")
        ids = set(cell["brekv"])
        for ratio, rows in cell["fixed"].items():
            if set(rows) != ids:
                raise ValueError(f"Sample mismatch in {key} fixed r={ratio}")
    return dict(indexed)


def stats(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "score": mean(row["score"] for row in rows.values()),
        "actual_budget": mean(row["budget"] for row in rows.values()),
        "total_bytes": mean(row[BYTE_FIELDS[0]] for row in rows.values()),
        "a_to_b_bytes": mean(row[BYTE_FIELDS[1]] for row in rows.values()),
        "b_to_a_bytes": mean(row[BYTE_FIELDS[2]] for row in rows.values()),
    }


def choose_dev_ratio(cells: dict[tuple[str, str], dict[str, Any]]) -> float:
    """Select by development macro score; lower actual budget breaks ties."""
    choices = []
    for ratio in SELECTION_RATIOS:
        summaries = [stats(cells[cell]["fixed"][ratio]) for cell in sorted(DEV_CELLS)]
        choices.append((mean(row["score"] for row in summaries),
                        mean(row["actual_budget"] for row in summaries), ratio))
    return min(choices, key=lambda item: (-item[0], item[1], item[2]))[2]


def per_task_best(cells: dict[tuple[str, str], dict[str, Any]]) -> dict[str, float]:
    selected = {}
    for task in TASKS:
        keys = [cell for cell in HELDOUT_CELLS if cell[1] == task]
        candidates = []
        for ratio in SELECTION_RATIOS:
            rows = [stats(cells[key]["fixed"][ratio]) for key in keys]
            candidates.append((mean(row["score"] for row in rows),
                               mean(row["actual_budget"] for row in rows), ratio))
        selected[task] = min(candidates, key=lambda item: (-item[0], item[1], item[2]))[2]
    return selected


def bootstrap_cell_macro(
    arrays: list[np.ndarray], count: int, rng: np.random.Generator
) -> np.ndarray:
    output = np.zeros(count, dtype=np.float64)
    for values in arrays:
        indices = rng.integers(0, len(values), size=(count, len(values)))
        output += values[indices].mean(axis=1)
    return output / len(arrays)


def ci(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    low, high = np.quantile(values, (0.025, 0.975))
    return {f"{prefix}ci_low": float(low), f"{prefix}ci_high": float(high)}


def build_policies(
    cells: dict[tuple[str, str], dict[str, Any]], dev_ratio: float,
    task_best: dict[str, float], n_bootstrap: int, rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policies: list[tuple[str, Any]] = [
        ("B-ReKV", lambda _: None),
        *[(f"Fixed {ratio:.1f}", lambda _, ratio=ratio: ratio)
          for ratio in POLICY_RATIOS],
        ("Dev-selected Fixed", lambda _: dev_ratio),
        ("Per-task Best Fixed", lambda task: task_best[task]),
    ]
    detail: list[dict[str, Any]] = []
    samples_by_policy: dict[str, list[np.ndarray]] = defaultdict(list)
    task_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for policy, selector in policies:
        for pair, task in HELDOUT_CELLS:
            ratio = selector(task)
            rows = cells[(pair, task)]["brekv"] if ratio is None else cells[(pair, task)]["fixed"][ratio]
            summary = stats(rows)
            best_rows = cells[(pair, task)]["fixed"][task_best[task]]
            best_summary = stats(best_rows)
            relative_regret = (
                (best_summary["score"] - summary["score"]) / abs(best_summary["score"])
                if best_summary["score"] != 0 else math.nan
            )
            detail.append({
                "policy": policy, "pair": pair, "task": task,
                "selected_ratio": ratio, **summary,
                "per_task_best_score": best_summary["score"],
                "relative_per_task_best_regret": relative_regret,
            })
            keys = list(rows)
            samples_by_policy[policy].append(
                np.asarray([rows[key]["score"] for key in keys], dtype=np.float64)
            )
            task_scores[policy][task].append(summary["score"])

    brekv_task = {task: mean(task_scores["B-ReKV"][task]) for task in TASKS}
    summaries = []
    for policy, _ in policies:
        selected = [row for row in detail if row["policy"] == policy]
        score_boot = bootstrap_cell_macro(samples_by_policy[policy], n_bootstrap, rng)
        task_regrets = {
            task: (
                mean(task_scores["Per-task Best Fixed"][task])
                - mean(task_scores[policy][task])
            ) / abs(mean(task_scores["Per-task Best Fixed"][task]))
            if mean(task_scores["Per-task Best Fixed"][task]) != 0 else math.nan
            for task in TASKS
        }
        policy_task = {task: mean(task_scores[policy][task]) for task in TASKS}
        summaries.append({
            "policy": policy,
            "selected_ratio": dev_ratio if policy == "Dev-selected Fixed" else None,
            "n_cells": len(selected),
            "score": mean(row["score"] for row in selected),
            **ci(score_boot, "score_"),
            "actual_budget": mean(row["actual_budget"] for row in selected),
            "total_bytes": mean(row["total_bytes"] for row in selected),
            "a_to_b_bytes": mean(row["a_to_b_bytes"] for row in selected),
            "b_to_a_bytes": mean(row["b_to_a_bytes"] for row in selected),
            "relative_per_task_best_regret": mean(task_regrets.values()),
            "worst_task_regret": max(task_regrets.values()),
            "worst_task": max(task_regrets, key=task_regrets.get),
            "brekv_wins_tasks_vs_policy": sum(
                brekv_task[task] > policy_task[task] + 1e-12 for task in TASKS
            ),
        })

    brekv_arrays = samples_by_policy["B-ReKV"]
    for row in summaries:
        policy = row["policy"]
        deltas = [left - right for left, right in
                  zip(brekv_arrays, samples_by_policy[policy])]
        boot = bootstrap_cell_macro(deltas, n_bootstrap, rng)
        row["brekv_score_delta"] = mean(array.mean() for array in deltas)
        row.update(ci(boot, "brekv_delta_"))
        row["p_brekv_gt_policy"] = float(np.mean(boot > 0))
    return detail, summaries


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks[order] = np.repeat((starts + ends - 1) / 2 + 1, ends - starts)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return math.nan
    x, y = rankdata(x), rankdata(y)
    x, y = x - x.mean(), y - y.mean()
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denom) if denom else math.nan


def build_oracle(
    cells: dict[tuple[str, str], dict[str, Any]], r07: dict[tuple[str, str], Path],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    r07_rows = {cell: read_samples(path) for cell, path in r07.items()}
    samples = []
    for pair, task in HELDOUT_CELLS:
        cell = cells[(pair, task)]
        if set(r07_rows[(pair, task)]) != set(cell["brekv"]):
            raise ValueError(f"r=.7 sample mismatch in {pair}/{task}")
        grid = {ratio: cell["fixed"][ratio] for ratio in ORACLE_RATIOS[:-1]}
        grid[0.7] = r07_rows[(pair, task)]
        for key, brekv in cell["brekv"].items():
            solved = [
                (ratio, rows[key]["budget"]) for ratio, rows in grid.items()
                if rows[key]["score"] >= threshold
            ]
            oracle = min(solved, key=lambda item: (item[1], item[0])) if solved else None
            needed = oracle[1] if oracle else None
            allocation = (
                "unsolved" if needed is None else
                "under" if brekv["budget"] < needed - 1e-12 else
                "over" if brekv["budget"] > needed + 1e-12 else "exact"
            )
            difficulty = (
                "unsolved" if oracle is None else "easy" if oracle[0] <= 0.3
                else "hard" if oracle[0] >= 0.5 else "medium"
            )
            samples.append({
                "pair": pair, "task": task, "sample_key": key,
                "oracle_solved": oracle is not None,
                "oracle_ratio": oracle[0] if oracle else None,
                "oracle_needed_budget": needed,
                "brekv_budget": brekv["budget"], "brekv_score": brekv["score"],
                "allocation": allocation, "difficulty": difficulty,
            })

    groups = [("macro", "all", samples)]
    groups += [("task", task, [row for row in samples if row["task"] == task])
               for task in TASKS]
    groups += [
        ("difficulty", difficulty,
         [row for row in samples if row["difficulty"] == difficulty])
        for difficulty in ("easy", "medium", "hard", "unsolved")
    ]
    groups += [
        ("oracle_ratio", f"{ratio:.1f}",
         [row for row in samples if row["oracle_ratio"] == ratio])
        for ratio in ORACLE_RATIOS
    ]
    summary = []
    for scope, group, rows in groups:
        if not rows:
            continue
        solved = [row for row in rows if row["oracle_solved"]]
        x = np.asarray([row["brekv_budget"] for row in solved])
        y = np.asarray([row["oracle_needed_budget"] for row in solved])
        summary.append({
            "scope": scope, "group": group, "n": len(rows), "n_solved": len(solved),
            "solve_rate": len(solved) / len(rows),
            "spearman": spearman(x, y),
            "oracle_needed_budget": mean(row["oracle_needed_budget"] for row in solved),
            "brekv_budget": mean(row["brekv_budget"] for row in rows),
            "easy_fraction": sum(row["difficulty"] == "easy" for row in rows) / len(rows),
            "medium_fraction": sum(row["difficulty"] == "medium" for row in rows) / len(rows),
            "hard_fraction": sum(row["difficulty"] == "hard" for row in rows) / len(rows),
            "unsolved_fraction": sum(row["difficulty"] == "unsolved" for row in rows) / len(rows),
            "under_fraction_solved": sum(row["allocation"] == "under" for row in solved) / len(solved) if solved else math.nan,
            "over_fraction_solved": sum(row["allocation"] == "over" for row in solved) / len(solved) if solved else math.nan,
        })

    edges = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6])
    solved = [row for row in samples if row["oracle_solved"]]
    bin_rows = []
    for index in range(len(edges) + 1):
        selected = [row for row in solved
                    if int(np.digitize(row["oracle_needed_budget"], edges)) == index]
        if not selected:
            continue
        correct = [int(np.digitize(row["brekv_budget"], edges)) == index
                   for row in selected]
        bin_rows.append({
            "oracle_budget_bin": index, "lower": 0 if index == 0 else edges[index - 1],
            "upper": edges[index] if index < len(edges) else math.inf,
            "n": len(selected), "bin_accuracy": mean(correct),
            "under_fraction": mean(
                int(np.digitize(row["brekv_budget"], edges)) < index for row in selected
            ),
            "over_fraction": mean(
                int(np.digitize(row["brekv_budget"], edges)) > index for row in selected
            ),
        })
    return samples, summary, bin_rows


def discover_shuffled(root: Path) -> dict[tuple[str, str], Path]:
    candidates: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in root.glob("**/per_sample.jsonl"):
        parts = path.relative_to(root).parts
        pair = next((part for part in parts if part in PAIRS), None)
        task = next((part for part in parts if part in TASKS), None)
        if pair and task:
            candidates[(pair, task)].append(path)
    output = {}
    for cell, paths in candidates.items():
        canonical = [path for path in paths if "t0.95_s0.75_w8" in path.parent.name]
        selected = canonical or paths
        output[cell] = max(selected, key=lambda path: path.stat().st_mtime)
    expected = set(HELDOUT_CELLS)
    if not expected <= set(output):
        raise ValueError(f"Shuffled held-out coverage missing: {sorted(expected-set(output))}")
    return {cell: output[cell] for cell in HELDOUT_CELLS}


def build_shuffled(
    cells: dict[tuple[str, str], dict[str, Any]], paths: dict[tuple[str, str], Path],
    count: int, rng: np.random.Generator, tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detail, deltas = [], []
    for pair, task in HELDOUT_CELLS:
        normal, shuffled = cells[(pair, task)]["brekv"], read_samples(paths[(pair, task)])
        if set(normal) != set(shuffled):
            raise ValueError(f"Shuffled sample mismatch in {pair}/{task}")
        keys = list(normal)
        normal_budgets = np.sort([normal[key]["budget"] for key in keys])
        shuffled_targets = np.sort([
            shuffled[key].get("replay_target_budget", shuffled[key]["budget"])
            for key in keys
        ])
        target_error = float(np.max(np.abs(normal_budgets - shuffled_targets)))
        if target_error > tolerance:
            raise ValueError(
                f"Target budget multiset mismatch in {pair}/{task}: {target_error}"
            )
        shuffled_budgets = np.sort([shuffled[key]["budget"] for key in keys])
        realized_error = float(np.max(np.abs(normal_budgets - shuffled_budgets)))
        realized_mean_delta = float(
            np.mean([shuffled[key]["budget"] for key in keys])
            - np.mean([normal[key]["budget"] for key in keys])
        )
        delta = np.asarray([normal[key]["score"] - shuffled[key]["score"] for key in keys])
        deltas.append(delta)
        detail.append({
            "pair": pair, "task": task, "n": len(keys),
            "normal_score": mean(normal[key]["score"] for key in keys),
            "shuffled_score": mean(shuffled[key]["score"] for key in keys),
            "score_delta": float(delta.mean()),
            "target_budget_multiset_max_error": target_error,
            "realized_budget_multiset_max_error": realized_error,
            "realized_mean_budget_delta": realized_mean_delta,
        })
    boot = bootstrap_cell_macro(deltas, count, rng)
    summary = {
        "comparison": "normal B-ReKV - shuffled budgets", "n_cells": len(detail),
        "score_delta": mean(row["score_delta"] for row in detail), **ci(boot),
        "p_delta_gt_zero": float(np.mean(boot > 0)),
        "budget_multisets_identical": True,
        "max_budget_multiset_error": max(
            row["target_budget_multiset_max_error"] for row in detail
        ),
        "max_realized_budget_multiset_error": max(
            row["realized_budget_multiset_max_error"] for row in detail
        ),
        "macro_realized_mean_budget_delta": mean(
            row["realized_mean_budget_delta"] for row in detail
        ),
    }
    return detail, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: "NA" if value is None or (
            isinstance(value, float) and not math.isfinite(value)
        ) else value for key, value in row.items()} for row in rows)


def write_report(path: Path, dev_ratio: float, policies: list[dict[str, Any]],
                 oracle: list[dict[str, Any]], shuffled: dict[str, Any]) -> None:
    macro = next(row for row in oracle if row["scope"] == "macro")
    lines = [
        "# B-ReKV held-out 与 shuffled 分析", "",
        "## 口径", "",
        "- Development cells 固定为 pair1/hotpotqa 与 pair1/musique；其余 54 cells 为 held-out。",
        f"- Development 选择的 fixed ratio：`{dev_ratio:g}`（从原 sweep 的 9 档选择；均分优先，实际预算与 ratio 依次破同分）。",
        "- Canonical B-ReKV：t=.95/s=.75/w8；所有汇总均按 cell 等权。",
        "- Per-task Best Fixed 在 held-out 上按任务选择，是 oracle 上界，不是可部署选择。",
        "- Oracle 使用 exact grid .1/.2/.3/.4/.5/.6/.7，成功定义为 score>=1，取成功点中最小实际预算。",
        "- Easy=oracle ratio<=.3，hard>=.5，.4 为 medium；预算 bins 按实际预算 .2/.3/.4/.5/.6 切分。",
        "- CI 为 cell 内严格配对样本 bootstrap 后的 54-cell 等权 percentile 95% CI。", "",
        "## Held-out policies", "",
    ]
    for row in policies:
        lines.append(
            f"- {row['policy']}: score={row['score']:.6f}, budget={row['actual_budget']:.6f}, "
            f"regret={row['relative_per_task_best_regret']:.6f}, "
            f"B-ReKV wins={row['brekv_wins_tasks_vs_policy']}/8, "
            f"Δ(B-ReKV-policy)={row['brekv_score_delta']:+.6f} "
            f"[{row['brekv_delta_ci_low']:+.6f}, {row['brekv_delta_ci_high']:+.6f}]"
        )
    lines += [
        "", "## Exact oracle", "",
        f"- Solved={macro['n_solved']}/{macro['n']}，Spearman={macro['spearman']:.4f}，"
        f"oracle budget={macro['oracle_needed_budget']:.6f}。",
        f"- Easy/medium/hard/unsolved={macro['easy_fraction']:.4f}/"
        f"{macro['medium_fraction']:.4f}/{macro['hard_fraction']:.4f}/"
        f"{macro['unsolved_fraction']:.4f}。", "",
        "## Shuffled budget", "",
        f"- Δscore={shuffled['score_delta']:+.6f} "
        f"[{shuffled['ci_low']:+.6f}, {shuffled['ci_high']:+.6f}]，"
        f"P(Δ>0)={shuffled['p_delta_gt_zero']:.4f}。",
        f"- 54 cells 预算 multiset 校验通过；最大排序后误差 "
        f"{shuffled['max_budget_multiset_error']:.3g}。",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("snapshots/full_matched_budget_fairness_query_sketch"))
    parser.add_argument("--oracle-r07-root", type=Path,
                        default=Path("snapshots/brekv_oracle_r07_v1"))
    parser.add_argument("--shuffled-root", type=Path,
                        default=Path("snapshots/brekv_shuffled_budget_v1"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("snapshots/analysis/brekv_heldout_shuffled"))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solve-threshold", type=float, default=1.0)
    parser.add_argument("--budget-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    rng = np.random.default_rng(args.seed)
    cells = load_cells(discover_main(args.root.resolve()))
    dev_ratio = choose_dev_ratio(cells)
    best = per_task_best(cells)
    policy_detail, policy_summary = build_policies(
        cells, dev_ratio, best, args.n_bootstrap, rng
    )
    oracle_samples, oracle_summary, bins = build_oracle(
        cells, discover_r07(args.oracle_r07_root.resolve()), args.solve_threshold
    )
    shuffled_detail, shuffled_summary = build_shuffled(
        cells, discover_shuffled(args.shuffled_root.resolve()),
        args.n_bootstrap, rng, args.budget_tolerance,
    )
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "heldout_policy_cells.csv", policy_detail)
    write_csv(out / "heldout_policy_summary.csv", policy_summary)
    write_csv(out / "oracle_samples.csv", oracle_samples)
    write_csv(out / "oracle_summary.csv", oracle_summary)
    write_csv(out / "oracle_budget_bins.csv", bins)
    write_csv(out / "shuffled_cells.csv", shuffled_detail)
    write_csv(out / "shuffled_summary.csv", [shuffled_summary])
    summary = {
        "inputs": {"main_root": str(args.root.resolve()),
                   "oracle_r07_root": str(args.oracle_r07_root.resolve()),
                   "shuffled_root": str(args.shuffled_root.resolve())},
        "development_cells": sorted("/".join(cell) for cell in DEV_CELLS),
        "heldout_cells": len(HELDOUT_CELLS),
        "dev_selected_ratio": dev_ratio,
        "per_task_best_ratios": best,
        "policy_summary": policy_summary,
        "oracle_macro": next(row for row in oracle_summary if row["scope"] == "macro"),
        "shuffled_summary": shuffled_summary,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    write_report(out / "REPORT.md", dev_ratio, policy_summary,
                 oracle_summary, shuffled_summary)
    print(f"Analysis complete: {out}")


if __name__ == "__main__":
    main()
