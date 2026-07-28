#!/usr/bin/env python3
"""Offline analysis of the value of B-ReKV's per-sample adaptive budgets.

The script consumes the complete 1568-run matched-budget sweep, but only loads
sample rows from fixed ReKV and the canonical B-ReKV configuration.  It never
loads a model or tokenizer; context length is therefore reported in
model-independent characters and whitespace-delimited words.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PAIR_ORDER = [
    "pair1_llama31_same",
    "pair2_llama32_same",
    "pair3_qwen25_7b_same",
    "pair4_falcon3_7b_same",
    "pair5_evolcodellama_toolace",
    "pair6_llama32_abliterated_deepseek3b",
    "pair7_qwen25_uncensored_bespoke",
]
TASK_ORDER = [
    "countries",
    "tipsheets",
    "hotpotqa",
    "qasper",
    "musique",
    "multifieldqa_en",
    "twowikimqa",
    "tmath",
]
FOCUS_PAIRS = [
    "pair1_llama31_same",
    "pair6_llama32_abliterated_deepseek3b",
    "pair7_qwen25_uncensored_bespoke",
]
FOCUS_TASKS = ["hotpotqa", "musique", "multifieldqa_en"]
REKV_PATTERN = re.compile(r"rekv_w(?P<window>\d+)_r(?P<ratio>[0-9.]+)_")
BREKV_PATTERN = re.compile(
    r"cov_t(?P<tau>[0-9.]+)_s(?P<scale>[0-9.]+)_w(?P<window>\d+)_"
)
OTHER_PATTERNS = [
    re.compile(r"evict_r(?P<ratio>[0-9.]+)_"),
    re.compile(r"random_r(?P<ratio>[0-9.]+)_"),
]


@dataclass(frozen=True)
class Run:
    pair: str
    task: str
    method: str
    params: tuple[tuple[str, float | int], ...]
    path: Path

    def param(self, name: str) -> float | int | None:
        return dict(self.params).get(name)


@dataclass
class Samples:
    ids: list[str]
    rows: dict[str, dict[str, Any]]


def mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if math.isfinite(float(value))]
    return sum(items) / len(items) if items else math.nan


def sample_key(row: dict[str, Any]) -> str:
    if row.get("idx") is None:
        raise ValueError("Sample row has no idx")
    raw_id = row.get("id")
    return f"{row['idx']}::{raw_id}" if raw_id is not None else str(row["idx"])


def read_samples(path: Path) -> Samples:
    rows: dict[str, dict[str, Any]] = {}
    ids: list[str] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if "_meta" in item:
                continue
            key = sample_key(item)
            if key in rows:
                raise ValueError(f"Duplicate sample key {key!r} in {path}")
            if item.get("score") is None or item.get("budget") is None:
                raise ValueError(f"Missing score/budget for {key!r} in {path}")
            rows[key] = {
                "idx": int(item["idx"]),
                "id": item.get("id"),
                "score": float(item["score"]),
                "budget": float(item["budget"]),
            }
            ids.append(key)
    if not rows:
        raise ValueError(f"No sample rows in {path}")
    return Samples(ids=ids, rows=rows)


def parse_run(name: str) -> tuple[str, tuple[tuple[str, float | int], ...]] | None:
    match = REKV_PATTERN.match(name)
    if match:
        return "ReKV", (
            ("ratio", float(match.group("ratio"))),
            ("window", int(match.group("window"))),
        )
    match = BREKV_PATTERN.match(name)
    if match:
        return "B-ReKV", (
            ("scale", float(match.group("scale"))),
            ("tau", float(match.group("tau"))),
            ("window", int(match.group("window"))),
        )
    for method, pattern in zip(("ValueNorm/Evict", "Random"), OTHER_PATTERNS):
        match = pattern.match(name)
        if match:
            return method, (("ratio", float(match.group("ratio"))),)
    return None


def discover_runs(root: Path) -> list[Run]:
    latest: dict[tuple[Any, ...], Path] = {}
    for path in root.glob("*/*/*/*/per_sample.jsonl"):
        pair, task, _, run_name = path.relative_to(root).parts[:4]
        parsed = parse_run(run_name)
        if parsed is None:
            continue
        method, params = parsed
        key = (pair, task, method, params)
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path
    runs = [
        Run(pair, task, method, params, path)
        for (pair, task, method, params), path in latest.items()
    ]
    runs.sort(key=lambda run: (run.pair, run.task, run.method, run.params))
    if len(runs) != 1568:
        raise ValueError(f"Expected 1568 logical per_sample runs, found {len(runs)}")
    return runs


def canonical_brekv(run: Run, tau: float, scale: float, window: int) -> bool:
    return (
        run.method == "B-ReKV"
        and run.param("tau") == tau
        and run.param("scale") == scale
        and run.param("window") == window
    )


def load_analysis_cells(
    runs: list[Run], tau: float, scale: float, window: int
) -> dict[tuple[str, str], dict[str, Any]]:
    fixed: dict[tuple[str, str], list[Run]] = defaultdict(list)
    brekv: dict[tuple[str, str], Run] = {}
    for run in runs:
        cell = (run.pair, run.task)
        if run.method == "ReKV" and run.param("window") == window:
            fixed[cell].append(run)
        elif canonical_brekv(run, tau, scale, window):
            brekv[cell] = run

    expected = {(pair, task) for pair in PAIR_ORDER for task in TASK_ORDER}
    if set(fixed) != expected or set(brekv) != expected:
        raise ValueError(
            f"Cell coverage mismatch: fixed_missing={sorted(expected - set(fixed))}, "
            f"brekv_missing={sorted(expected - set(brekv))}"
        )

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in sorted(expected):
        fixed_runs = sorted(fixed[cell], key=lambda run: float(run.param("ratio")))
        if len(fixed_runs) != 9:
            raise ValueError(f"Expected 9 fixed ReKV levels for {cell}, got {len(fixed_runs)}")
        fixed_samples = [(run, read_samples(run.path)) for run in fixed_runs]
        adaptive = read_samples(brekv[cell].path)
        expected_ids = set(adaptive.ids)
        for run, samples in fixed_samples:
            if set(samples.ids) != expected_ids:
                raise ValueError(f"Sample mismatch in {cell}: {run.path}")
        cells[cell] = {
            "fixed": fixed_samples,
            "brekv_run": brekv[cell],
            "brekv": adaptive,
        }
    return cells


def run_means(samples: Samples) -> tuple[float, float]:
    return (
        mean(row["score"] for row in samples.rows.values()),
        mean(row["budget"] for row in samples.rows.values()),
    )


def interpolation_bracket(
    points: list[tuple[float, float]], target: float
) -> tuple[int, int, float]:
    """Return indices and interpolation weight on an actual-budget curve."""
    ordered = sorted(enumerate(points), key=lambda item: item[1][1])
    if target < ordered[0][1][1] - 1e-12 or target > ordered[-1][1][1] + 1e-12:
        raise ValueError(
            f"Target budget {target:.6f} outside fixed curve "
            f"[{ordered[0][1][1]:.6f}, {ordered[-1][1][1]:.6f}]"
        )
    for (lower_index, lower), (upper_index, upper) in zip(ordered, ordered[1:]):
        if lower[1] - 1e-12 <= target <= upper[1] + 1e-12:
            span = upper[1] - lower[1]
            weight = 0.0 if span <= 0 else (target - lower[1]) / span
            return lower_index, upper_index, min(max(weight, 0.0), 1.0)
    index = ordered[-1][0]
    return index, index, 0.0


def bootstrap_mean(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator, chunk: int = 500
) -> np.ndarray:
    output = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, chunk):
        count = min(chunk, n_bootstrap - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        output[start : start + count] = values[indices].mean(axis=1)
    return output


def ci_fields(values: np.ndarray, prefix: str = "") -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}ci_low": math.nan,
            f"{prefix}ci_high": math.nan,
            f"{prefix}bootstrap_valid": 0,
        }
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        f"{prefix}ci_low": float(low),
        f"{prefix}ci_high": float(high),
        f"{prefix}bootstrap_valid": int(finite.size),
    }


def cell_delta(
    cell: dict[str, Any], lower_index: int, upper_index: int, weight: float
) -> np.ndarray:
    adaptive: Samples = cell["brekv"]
    lower: Samples = cell["fixed"][lower_index][1]
    upper: Samples = cell["fixed"][upper_index][1]
    return np.asarray(
        [
            adaptive.rows[key]["score"]
            - (
                (1.0 - weight) * lower.rows[key]["score"]
                + weight * upper.rows[key]["score"]
            )
            for key in adaptive.ids
        ],
        dtype=np.float64,
    )


def aggregate_bootstrap(
    deltas: list[np.ndarray], n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    return np.vstack(
        [bootstrap_mean(delta, n_bootstrap, rng) for delta in deltas]
    ).mean(axis=0)


def build_fixed_policies(
    cells: dict[tuple[str, str], dict[str, Any]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for key, cell in cells.items():
        fixed_stats = [run_means(samples) for _, samples in cell["fixed"]]
        brekv_score, brekv_budget = run_means(cell["brekv"])
        cell_stats[key] = {
            "fixed": fixed_stats,
            "brekv_score": brekv_score,
            "brekv_budget": brekv_budget,
        }

    detail: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []

    def add_policy(
        policy: str,
        group: str,
        keys: list[tuple[str, str]],
        target: float,
        points: list[tuple[float, float]],
    ) -> None:
        lower, upper, weight = interpolation_bracket(points, target)
        ratios = [
            float(cells[key]["fixed"][lower][0].param("ratio")) for key in keys
        ]
        upper_ratios = [
            float(cells[key]["fixed"][upper][0].param("ratio")) for key in keys
        ]
        deltas = [cell_delta(cells[key], lower, upper, weight) for key in keys]
        fixed_score = mean(
            (1.0 - weight) * cell_stats[key]["fixed"][lower][0]
            + weight * cell_stats[key]["fixed"][upper][0]
            for key in keys
        )
        fixed_budget = mean(
            (1.0 - weight) * cell_stats[key]["fixed"][lower][1]
            + weight * cell_stats[key]["fixed"][upper][1]
            for key in keys
        )
        row = {
            "policy": policy,
            "group": group,
            "n_cells": len(keys),
            "n_samples": sum(delta.size for delta in deltas),
            "brekv_score": mean(cell_stats[key]["brekv_score"] for key in keys),
            "brekv_budget": mean(cell_stats[key]["brekv_budget"] for key in keys),
            "fixed_score": fixed_score,
            "fixed_budget": fixed_budget,
            "score_delta": mean(delta.mean() for delta in deltas),
            "budget_delta": mean(cell_stats[key]["brekv_budget"] for key in keys)
            - fixed_budget,
            "lower_ratio": mean(ratios),
            "upper_ratio": mean(upper_ratios),
            "interpolation_weight": weight,
        }
        detail.append(row)
        boot = aggregate_bootstrap(deltas, n_bootstrap, rng)
        bootstrap_rows.append(
            {
                "policy": policy,
                "group": group,
                "n_cells": len(keys),
                "score_delta": row["score_delta"],
                **ci_fields(boot),
                "bootstrap_win_probability": float(np.mean(boot > 0.0)),
            }
        )

    all_keys = [(pair, task) for pair in PAIR_ORDER for task in TASK_ORDER]
    global_target = mean(cell_stats[key]["brekv_budget"] for key in all_keys)
    global_points = [
        (
            mean(cell_stats[key]["fixed"][level][0] for key in all_keys),
            mean(cell_stats[key]["fixed"][level][1] for key in all_keys),
        )
        for level in range(9)
    ]
    add_policy("global_budget_matched_fixed", "all", all_keys, global_target, global_points)

    task_policy_deltas: list[np.ndarray] = []
    for task in TASK_ORDER:
        keys = [(pair, task) for pair in PAIR_ORDER]
        target = mean(cell_stats[key]["brekv_budget"] for key in keys)
        points = [
            (
                mean(cell_stats[key]["fixed"][level][0] for key in keys),
                mean(cell_stats[key]["fixed"][level][1] for key in keys),
            )
            for level in range(9)
        ]
        add_policy("per_task_budget_matched_fixed", task, keys, target, points)
        row = detail[-1]
        # Reconstruct the selected per-task policy for an equal-cell macro CI.
        lower, upper, weight = interpolation_bracket(points, target)
        task_policy_deltas.extend(
            cell_delta(cells[key], lower, upper, weight) for key in keys
        )
    task_macro_boot = aggregate_bootstrap(task_policy_deltas, n_bootstrap, rng)
    task_rows = [
        row for row in detail if row["policy"] == "per_task_budget_matched_fixed"
    ]
    bootstrap_rows.append(
        {
            "policy": "per_task_budget_matched_fixed",
            "group": "8-task macro",
            "n_cells": 56,
            "score_delta": mean(row["score_delta"] for row in task_rows),
            **ci_fields(task_macro_boot),
            "bootstrap_win_probability": float(np.mean(task_macro_boot > 0.0)),
        }
    )

    matched_cell_deltas: list[np.ndarray] = []
    best_cell_deltas: list[np.ndarray] = []
    for key in all_keys:
        pair, task = key
        stats = cell_stats[key]
        target = stats["brekv_budget"]
        points = stats["fixed"]
        lower, upper, weight = interpolation_bracket(points, target)
        delta = cell_delta(cells[key], lower, upper, weight)
        matched_cell_deltas.append(delta)
        fixed_score = (1.0 - weight) * points[lower][0] + weight * points[upper][0]
        fixed_budget = (1.0 - weight) * points[lower][1] + weight * points[upper][1]
        detail.append(
            {
                "policy": "per_cell_budget_matched_fixed",
                "group": f"{pair}/{task}",
                "pair": pair,
                "task": task,
                "n_cells": 1,
                "n_samples": delta.size,
                "brekv_score": stats["brekv_score"],
                "brekv_budget": target,
                "fixed_score": fixed_score,
                "fixed_budget": fixed_budget,
                "score_delta": float(delta.mean()),
                "budget_delta": target - fixed_budget,
                "lower_ratio": cells[key]["fixed"][lower][0].param("ratio"),
                "upper_ratio": cells[key]["fixed"][upper][0].param("ratio"),
                "interpolation_weight": weight,
            }
        )
        boot = bootstrap_mean(delta, n_bootstrap, rng)
        bootstrap_rows.append(
            {
                "policy": "per_cell_budget_matched_fixed",
                "group": f"{pair}/{task}",
                "pair": pair,
                "task": task,
                "n_cells": 1,
                "score_delta": float(delta.mean()),
                **ci_fields(boot),
                "bootstrap_win_probability": float(np.mean(boot > 0.0)),
            }
        )

        # Strong, explicitly unmatched oracle baseline: best observed fixed score.
        best = min(
            range(9),
            key=lambda level: (-points[level][0], points[level][1]),
        )
        best_delta = cell_delta(cells[key], best, best, 0.0)
        best_cell_deltas.append(best_delta)
        detail.append(
            {
                "policy": "per_cell_best_fixed_unmatched",
                "group": f"{pair}/{task}",
                "pair": pair,
                "task": task,
                "n_cells": 1,
                "n_samples": best_delta.size,
                "brekv_score": stats["brekv_score"],
                "brekv_budget": target,
                "fixed_score": points[best][0],
                "fixed_budget": points[best][1],
                "score_delta": float(best_delta.mean()),
                "budget_delta": target - points[best][1],
                "lower_ratio": cells[key]["fixed"][best][0].param("ratio"),
                "upper_ratio": cells[key]["fixed"][best][0].param("ratio"),
                "interpolation_weight": 0.0,
            }
        )

    for policy, deltas in (
        ("per_cell_budget_matched_fixed", matched_cell_deltas),
        ("per_cell_best_fixed_unmatched", best_cell_deltas),
    ):
        boot = aggregate_bootstrap(deltas, n_bootstrap, rng)
        relevant = [row for row in detail if row["policy"] == policy]
        bootstrap_rows.append(
            {
                "policy": policy,
                "group": "56-cell macro",
                "n_cells": 56,
                "score_delta": mean(row["score_delta"] for row in relevant),
                **ci_fields(boot),
                "bootstrap_win_probability": float(np.mean(boot > 0.0)),
            }
        )
    return detail, bootstrap_rows


def build_oracle(
    cells: dict[tuple[str, str], dict[str, Any]], solve_threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples_out: list[dict[str, Any]] = []
    for (pair, task), cell in sorted(cells.items()):
        adaptive: Samples = cell["brekv"]
        for key in adaptive.ids:
            levels = []
            for run, samples in cell["fixed"]:
                row = samples.rows[key]
                levels.append(
                    (float(run.param("ratio")), row["budget"], row["score"])
                )
            solved_levels = [level for level in levels if level[2] >= solve_threshold]
            oracle = min(solved_levels, key=lambda level: level[1]) if solved_levels else None
            cap_budget = max(level[1] for level in levels)
            brekv_row = adaptive.rows[key]
            samples_out.append(
                {
                    "pair": pair,
                    "task": task,
                    "sample_key": key,
                    "idx": brekv_row["idx"],
                    "id": brekv_row["id"],
                    "solved": oracle is not None,
                    "oracle_ratio": oracle[0] if oracle else None,
                    "oracle_needed_budget": oracle[1] if oracle else None,
                    "capped_oracle_needed_budget": oracle[1] if oracle else cap_budget,
                    "unsolved_cap_budget": cap_budget,
                    "brekv_budget": brekv_row["budget"],
                    "brekv_score": brekv_row["score"],
                }
            )

    summary: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for pair in PAIR_ORDER:
        for task in TASK_ORDER:
            groups.append(
                (
                    "cell",
                    f"{pair}/{task}",
                    [row for row in samples_out if row["pair"] == pair and row["task"] == task],
                )
            )
    for task in TASK_ORDER:
        groups.append(
            ("task", task, [row for row in samples_out if row["task"] == task])
        )
    groups.append(("macro", "all", samples_out))
    for scope, group, rows in groups:
        solved = [row for row in rows if row["solved"]]
        summary.append(
            {
                "scope": scope,
                "group": group,
                "n": len(rows),
                "n_solved": len(solved),
                "n_unsolved": len(rows) - len(solved),
                "unsolved_rate": (len(rows) - len(solved)) / len(rows),
                "oracle_needed_budget_solved_mean": mean(
                    row["oracle_needed_budget"] for row in solved
                ),
                "oracle_needed_budget_capped_mean": mean(
                    row["capped_oracle_needed_budget"] for row in rows
                ),
                "brekv_budget_mean": mean(row["brekv_budget"] for row in rows),
                "brekv_score_mean": mean(row["brekv_score"] for row in rows),
            }
        )
    return samples_out, summary


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0) + 1]
    ends = np.r_[starts[1:], values.size]
    average_ranks = (starts + ends - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(average_ranks, ends - starts)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return math.nan
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / denominator) if denominator > 0 else math.nan


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not math.isfinite(spearman(x, y)):
        return np.full(n_bootstrap, np.nan, dtype=np.float64)
    output = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        indices = rng.integers(0, x.size, size=x.size)
        output[index] = spearman(x[indices], y[indices])
    return output


def nanmean_columns(matrix: np.ndarray) -> np.ndarray:
    counts = np.sum(np.isfinite(matrix), axis=0)
    totals = np.nansum(matrix, axis=0)
    return np.divide(
        totals,
        counts,
        out=np.full(matrix.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )


def build_correlations(
    oracle_rows: list[dict[str, Any]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in oracle_rows:
        by_cell[(row["pair"], row["task"])].append(row)
    output: list[dict[str, Any]] = []
    cell_boot: dict[tuple[str, str, str], np.ndarray] = {}
    cell_point: dict[tuple[str, str, str], float] = {}
    for sensitivity, field in (
        ("solved_only", "oracle_needed_budget"),
        ("unsolved_capped_at_observed_max", "capped_oracle_needed_budget"),
    ):
        for (pair, task), rows in sorted(by_cell.items()):
            selected = rows if sensitivity != "solved_only" else [
                row for row in rows if row["solved"]
            ]
            x = np.asarray([row["brekv_budget"] for row in selected], dtype=np.float64)
            y = np.asarray([row[field] for row in selected], dtype=np.float64)
            point = spearman(x, y)
            boot = bootstrap_spearman(x, y, n_bootstrap, rng)
            cell_boot[(pair, task, sensitivity)] = boot
            cell_point[(pair, task, sensitivity)] = point
            output.append(
                {
                    "scope": "cell",
                    "group": f"{pair}/{task}",
                    "pair": pair,
                    "task": task,
                    "sensitivity": sensitivity,
                    "n": len(selected),
                    "n_cells": 1,
                    "spearman_rho": point,
                    **ci_fields(boot),
                }
            )
        for task in TASK_ORDER:
            keys = [(pair, task, sensitivity) for pair in PAIR_ORDER]
            matrix = np.vstack([cell_boot[key] for key in keys])
            boot = nanmean_columns(matrix)
            output.append(
                {
                    "scope": "task",
                    "group": task,
                    "task": task,
                    "sensitivity": sensitivity,
                    "n": sum(
                        row["n"]
                        for row in output
                        if row["scope"] == "cell"
                        and row["task"] == task
                        and row["sensitivity"] == sensitivity
                    ),
                    "n_cells": 7,
                    "spearman_rho": mean(cell_point[key] for key in keys),
                    **ci_fields(boot),
                }
            )
        keys = [
            (pair, task, sensitivity)
            for pair in PAIR_ORDER
            for task in TASK_ORDER
        ]
        boot = nanmean_columns(np.vstack([cell_boot[key] for key in keys]))
        output.append(
            {
                "scope": "macro",
                "group": "56-cell equal-weight macro",
                "sensitivity": sensitivity,
                "n": sum(
                    row["n"]
                    for row in output
                    if row["scope"] == "cell" and row["sensitivity"] == sensitivity
                ),
                "n_cells": 56,
                "spearman_rho": mean(cell_point[key] for key in keys),
                **ci_fields(boot),
            }
        )
    return output


def load_task_metadata(task: str) -> tuple[dict[int, dict[str, Any]], str | None]:
    """Best-effort metadata extraction through the repository's dataloaders."""
    try:
        if task == "hotpotqa":
            from dataloader.hotpotqa import HotpotQAEvaluator

            data = HotpotQAEvaluator().data
        elif task == "musique":
            from dataloader.musique import MuSiQueEvaluator

            data = MuSiQueEvaluator().data
        elif task == "multifieldqa_en":
            from dataloader.multifieldqa_en import MultiFieldQAEnEvaluator

            data = MultiFieldQAEnEvaluator().data
        else:
            return {}, f"unsupported metadata task: {task}"
        output: dict[int, dict[str, Any]] = {}
        for index, item in enumerate(data):
            context = str(item.get("prompt_A", item.get("context", "")))
            evidence_count: int | None = None
            hop_count: int | None = None
            if task == "hotpotqa":
                facts = item.get("supporting_facts")
                if isinstance(facts, dict) and isinstance(facts.get("title"), list):
                    evidence_count = len(facts["title"])
            elif task == "musique":
                decomposition = item.get("question_decomposition")
                if isinstance(decomposition, list):
                    hop_count = len(decomposition)
                paragraphs = item.get("paragraphs")
                if isinstance(paragraphs, list):
                    evidence_count = sum(
                        bool(paragraph.get("is_supporting"))
                        for paragraph in paragraphs
                        if isinstance(paragraph, dict)
                    )
            output[index] = {
                "metadata_id": item.get("id", item.get("_id")),
                "context_chars": len(context),
                "context_words": len(context.split()),
                "hop_count": hop_count,
                "evidence_count": evidence_count,
            }
        return output, None
    except Exception as exc:  # Metadata is explicitly best-effort.
        return {}, f"{type(exc).__name__}: {exc}"


def build_focused_groups(
    oracle_rows: list[dict[str, Any]], skip_metadata: bool
) -> tuple[list[dict[str, Any]], dict[str, str | None]]:
    metadata: dict[str, dict[int, dict[str, Any]]] = {}
    metadata_status: dict[str, str | None] = {}
    for task in FOCUS_TASKS:
        if skip_metadata:
            metadata[task], metadata_status[task] = {}, "skipped by --skip-metadata"
        else:
            metadata[task], metadata_status[task] = load_task_metadata(task)

    enriched = []
    for row in oracle_rows:
        if row["pair"] not in FOCUS_PAIRS or row["task"] not in FOCUS_TASKS:
            continue
        item = dict(row)
        item.update(
            metadata[row["task"]].get(
                int(row["idx"]),
                {
                    "metadata_id": None,
                    "context_chars": None,
                    "context_words": None,
                    "hop_count": None,
                    "evidence_count": None,
                },
            )
        )
        if item["context_chars"] is not None:
            metadata_id = item.get("metadata_id")
            if metadata_id is not None and row.get("id") is not None:
                item["metadata_id_match"] = str(metadata_id) == str(row["id"])
            else:
                item["metadata_id_match"] = None
        enriched.append(item)

    cutoffs: dict[str, tuple[float, float]] = {}
    for task in FOCUS_TASKS:
        lengths = sorted(
            [
                int(row["context_chars"])
                for row in enriched
                if row["pair"] == FOCUS_PAIRS[0]
                and row["task"] == task
                and row["context_chars"] is not None
            ]
        )
        if lengths:
            cutoffs[task] = tuple(float(value) for value in np.quantile(lengths, [1 / 3, 2 / 3]))

    for row in enriched:
        length = row["context_chars"]
        if length is None or row["task"] not in cutoffs:
            row["context_length_group"] = "NA"
        else:
            low, high = cutoffs[row["task"]]
            row["context_length_group"] = (
                "short" if length <= low else "medium" if length <= high else "long"
            )

    output: list[dict[str, Any]] = []
    for pair in FOCUS_PAIRS:
        for task in FOCUS_TASKS:
            for group in ("short", "medium", "long", "NA"):
                rows = [
                    row
                    for row in enriched
                    if row["pair"] == pair
                    and row["task"] == task
                    and row["context_length_group"] == group
                ]
                if not rows:
                    continue
                solved = [row for row in rows if row["solved"]]
                x = np.asarray([row["brekv_budget"] for row in solved], dtype=np.float64)
                y = np.asarray(
                    [row["oracle_needed_budget"] for row in solved], dtype=np.float64
                )
                output.append(
                    {
                        "pair": pair,
                        "task": task,
                        "context_length_group": group,
                        "n": len(rows),
                        "context_chars_mean": mean(
                            row["context_chars"] for row in rows
                            if row["context_chars"] is not None
                        ),
                        "context_words_mean": mean(
                            row["context_words"] for row in rows
                            if row["context_words"] is not None
                        ),
                        "hop_count_mean": mean(
                            row["hop_count"] for row in rows
                            if row["hop_count"] is not None
                        ),
                        "hop_count_available_n": sum(
                            row["hop_count"] is not None for row in rows
                        ),
                        "evidence_count_mean": mean(
                            row["evidence_count"] for row in rows
                            if row["evidence_count"] is not None
                        ),
                        "evidence_count_available_n": sum(
                            row["evidence_count"] is not None for row in rows
                        ),
                        "metadata_id_mismatch_n": sum(
                            row.get("metadata_id_match") is False for row in rows
                        ),
                        "unsolved_rate": (len(rows) - len(solved)) / len(rows),
                        "oracle_needed_budget_solved_mean": mean(
                            row["oracle_needed_budget"] for row in solved
                        ),
                        "oracle_needed_budget_capped_mean": mean(
                            row["capped_oracle_needed_budget"] for row in rows
                        ),
                        "brekv_budget_mean": mean(row["brekv_budget"] for row in rows),
                        "brekv_score_mean": mean(row["brekv_score"] for row in rows),
                        "spearman_solved_only": spearman(x, y),
                    }
                )
    return output, metadata_status


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    "NA"
                    if value is None
                    or (isinstance(value, float) and not math.isfinite(value))
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def make_figure(
    out_dir: Path,
    policy_bootstrap: list[dict[str, Any]],
    oracle_summary: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
) -> list[str]:
    mpl_dir = out_dir / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    selected = [
        next(
            row
            for row in policy_bootstrap
            if row["policy"] == policy and row["group"] == group
        )
        for policy, group in (
            ("global_budget_matched_fixed", "all"),
            ("per_task_budget_matched_fixed", "8-task macro"),
            ("per_cell_budget_matched_fixed", "56-cell macro"),
            ("per_cell_best_fixed_unmatched", "56-cell macro"),
        )
    ]
    labels = ["Global\nmatched", "Per-task\nmatched", "Per-cell\nmatched", "Per-cell best\n(unmatched)"]
    values = [row["score_delta"] for row in selected]
    errors = [
        [value - row["ci_low"] for value, row in zip(values, selected)],
        [row["ci_high"] - value for value, row in zip(values, selected)],
    ]
    axes[0].bar(range(4), values, color=["#4c78a8", "#72b7b2", "#54a24b", "#bab0ac"])
    axes[0].errorbar(range(4), values, yerr=errors, fmt="none", color="black", capsize=3)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(range(4), labels)
    axes[0].set_ylabel("B-ReKV score delta")
    axes[0].set_title("Adaptive value vs fixed policies")

    task_oracle = {
        row["group"]: row
        for row in oracle_summary
        if row["scope"] == "task"
    }
    axes[1].bar(
        range(len(TASK_ORDER)),
        [100 * task_oracle[task]["unsolved_rate"] for task in TASK_ORDER],
        color="#e45756",
    )
    axes[1].set_xticks(range(len(TASK_ORDER)), TASK_ORDER, rotation=45, ha="right")
    axes[1].set_ylabel("Unsolved at 9 ReKV levels (%)")
    axes[1].set_title("Exact-score oracle coverage")

    correlation_rows = [
        row for row in correlations if row["scope"] == "task"
    ] + [row for row in correlations if row["scope"] == "macro"]
    correlation_groups = TASK_ORDER + ["macro"]
    for sensitivity, color, marker, offset in (
        ("solved_only", "#f58518", "o", -0.12),
        ("unsolved_capped_at_observed_max", "#4c78a8", "s", 0.12),
    ):
        rows = {
            ("macro" if row["scope"] == "macro" else row["group"]): row
            for row in correlation_rows
            if row["sensitivity"] == sensitivity
        }
        xs = np.arange(len(correlation_groups)) + offset
        ys = [rows[group]["spearman_rho"] for group in correlation_groups]
        low = [value - rows[group]["ci_low"] for value, group in zip(ys, correlation_groups)]
        high = [rows[group]["ci_high"] - value for value, group in zip(ys, correlation_groups)]
        axes[2].errorbar(
            xs,
            ys,
            yerr=[low, high],
            fmt=marker,
            color=color,
            capsize=2,
            label=sensitivity.replace("_", " "),
        )
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks(range(len(correlation_groups)), correlation_groups, rotation=45, ha="right")
    axes[2].set_ylabel("Spearman rho (cell-equal macro)")
    axes[2].set_title("B-ReKV budget vs oracle need")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outputs = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"brekv_adaptive_value.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def write_report(
    path: Path,
    policy_rows: list[dict[str, Any]],
    policy_bootstrap: list[dict[str, Any]],
    oracle_summary: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    metadata_status: dict[str, str | None],
    parameters: dict[str, Any],
    figures: list[str],
) -> None:
    policy_lookup = {
        (row["policy"], row["group"]): row for row in policy_bootstrap
    }
    oracle_macro = next(
        row for row in oracle_summary if row["scope"] == "macro"
    )
    correlation_macro = [
        row for row in correlations if row["scope"] == "macro"
    ]
    lines = [
        "# B-ReKV 自适应预算价值分析",
        "",
        "## 口径",
        "",
        "- 输入：完整 1568-run matched-budget sweep；分析仅加载 canonical B-ReKV 与 9 档 fixed ReKV 的 per-sample 数据。",
        "- fixed budget matching：在实际预算轴上对相邻 fixed ReKV 档位做样本级线性插值。",
        "- `per_cell_best_fixed_unmatched` 是额外的强静态上界：每个 cell 取均分最高的 fixed 档，不保证预算匹配，预算差单独报告。",
        f"- Oracle solved：9 档中存在 `score >= {parameters['solve_threshold']}`；需要预算取所有成功档中的最小实际预算。",
        "- Capped sensitivity：未解样本的未知需求封顶为该样本 9 档中最大的实际预算；这是保守的可观测上限敏感性分析。",
        "- CI：样本严格配对 bootstrap；task/macro 在 cell 内重采样后等权分层聚合，percentile 95% CI。",
        "- Spearman task/macro：先按 cell 计算 rho，再对 cell 等权宏平均；不把 pair/task 的预算尺度差异混入 pooled rho。",
        "- 上下文长度：不加载 tokenizer/model，使用实际 dataloader `prompt_A` 的字符数与空白分词数。",
        "",
        "## Fixed-policy 对比",
        "",
    ]
    for policy, group, label in (
        ("global_budget_matched_fixed", "all", "全局 budget-matched fixed"),
        ("per_task_budget_matched_fixed", "8-task macro", "每任务 budget-matched fixed"),
        ("per_cell_budget_matched_fixed", "56-cell macro", "每 cell budget-matched fixed"),
        ("per_cell_best_fixed_unmatched", "56-cell macro", "每 cell best fixed（非预算匹配）"),
    ):
        row = policy_lookup[(policy, group)]
        lines.append(
            f"- {label}：Δscore={row['score_delta']:+.6f}，"
            f"95% CI [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]，"
            f"P(Δ>0)={row['bootstrap_win_probability']:.4f}"
        )
    unmatched = [
        row for row in policy_rows if row["policy"] == "per_cell_best_fixed_unmatched"
    ]
    lines += [
        "",
        "## Oracle 与相关性",
        "",
        f"- 总样本-cell 观测：{oracle_macro['n']}；未解 {oracle_macro['n_unsolved']} "
        f"({100 * oracle_macro['unsolved_rate']:.2f}%)。",
        f"- Oracle needed budget（仅已解）均值：{oracle_macro['oracle_needed_budget_solved_mean']:.6f}；"
        f"封顶敏感性均值：{oracle_macro['oracle_needed_budget_capped_mean']:.6f}。",
        f"- 每-cell best fixed 的平均预算差（B-ReKV - fixed）："
        f"{mean(row['budget_delta'] for row in unmatched):+.6f}。",
    ]
    for row in correlation_macro:
        lines.append(
            f"- {row['sensitivity']}：rho={row['spearman_rho']:+.4f}，"
            f"95% CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]。"
        )
    lines += ["", "## 聚焦元数据", ""]
    for task in FOCUS_TASKS:
        status = metadata_status[task]
        lines.append(
            f"- {task}：{'已提取' if status is None else 'NA（' + status + '）'}。"
        )
    lines += [
        "- HotpotQA 可提取 supporting-fact/evidence count，但源元数据没有显式 hop count，故 hop 为 NA。",
        "- MuSiQue 从 question decomposition 与 supporting paragraphs 提取 hop/evidence count。",
        "- MultiFieldQA-en 源元数据无 hop/evidence 字段，二者均为 NA。",
        "",
        "## 图",
        "",
    ]
    lines.extend(f"- `{figure}`" for figure in figures)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("snapshots/full_matched_budget_fairness_query_sketch"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("snapshots/analysis/brekv_adaptive_value"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solve-threshold", type=float, default=1.0)
    parser.add_argument("--canonical-tau", type=float, default=0.95)
    parser.add_argument("--canonical-scale", type=float, default=0.75)
    parser.add_argument("--canonical-window", type=int, default=8)
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Emit focused length/hop/evidence fields as NA without loading dataloaders.",
    )
    parser.add_argument(
        "--allow-metadata-download",
        action="store_true",
        help="Allow HuggingFace dataloaders to fetch missing metadata (never models).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    if not math.isfinite(args.solve_threshold):
        raise ValueError("--solve-threshold must be finite")
    if not args.allow_metadata_download:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Discovering and validating the 1568 logical runs...")
    runs = discover_runs(root)
    print("Loading canonical B-ReKV and 9-level fixed ReKV samples...")
    cells = load_analysis_cells(
        runs, args.canonical_tau, args.canonical_scale, args.canonical_window
    )
    print(f"Running paired/stratified policy bootstrap ({args.n_bootstrap} replicates)...")
    policy_rows, policy_bootstrap = build_fixed_policies(
        cells, args.n_bootstrap, rng
    )
    print("Building exact-score minimal-actual-budget oracle...")
    oracle_rows, oracle_summary = build_oracle(cells, args.solve_threshold)
    print(f"Running Spearman bootstrap ({args.n_bootstrap} replicates)...")
    correlations = build_correlations(oracle_rows, args.n_bootstrap, rng)
    print("Extracting focused context/hop/evidence metadata (best effort)...")
    focused_groups, metadata_status = build_focused_groups(
        oracle_rows, args.skip_metadata
    )

    write_csv(out_dir / "fixed_policy_comparisons.csv", policy_rows)
    write_csv(out_dir / "fixed_policy_bootstrap.csv", policy_bootstrap)
    write_csv(out_dir / "oracle_samples.csv", oracle_rows)
    write_csv(out_dir / "oracle_summary.csv", oracle_summary)
    write_csv(out_dir / "budget_oracle_spearman.csv", correlations)
    write_csv(out_dir / "focused_context_groups.csv", focused_groups)
    figures = make_figure(
        out_dir, policy_bootstrap, oracle_summary, correlations
    )

    parameters = {
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "solve_threshold": args.solve_threshold,
        "canonical_brekv": {
            "tau": args.canonical_tau,
            "scale": args.canonical_scale,
            "window": args.canonical_window,
        },
        "metadata_offline": not args.allow_metadata_download,
    }
    summary = {
        "inputs": {"root": str(root), "logical_runs": len(runs)},
        "parameters": parameters,
        "validation": {
            "cells": len(cells),
            "fixed_rekv_levels_per_cell": 9,
            "oracle_sample_cell_rows": len(oracle_rows),
            "focused_cells": len(FOCUS_PAIRS) * len(FOCUS_TASKS),
        },
        "metadata_status": metadata_status,
        "fixed_policy_macro": [
            row
            for row in policy_bootstrap
            if row["group"] in {"all", "8-task macro", "56-cell macro"}
        ],
        "oracle_macro": [
            row for row in oracle_summary if row["scope"] == "macro"
        ],
        "spearman_macro": [
            row for row in correlations if row["scope"] == "macro"
        ],
        "figures": figures,
    }
    write_json(out_dir / "summary.json", summary)
    write_report(
        out_dir / "REPORT.md",
        policy_rows,
        policy_bootstrap,
        oracle_summary,
        correlations,
        metadata_status,
        parameters,
        figures,
    )
    print(f"Analysis complete: {out_dir}")


if __name__ == "__main__":
    main()
