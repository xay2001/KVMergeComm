#!/usr/bin/env python3
"""Offline held-out generalization analysis for canonical B-ReKV.

The fixed policy is selected only on pair1_llama31_same/{hotpotqa,musique}.
All other pair-task cells are held out.  The analysis reads the complete
1568-run sweep for validation, but loads samples only for fixed ReKV (w8) and
canonical B-ReKV (t0.95, s0.75, w8).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    "countries",
    "tipsheets",
    "hotpotqa",
    "qasper",
    "musique",
    "multifieldqa_en",
    "twowikimqa",
    "tmath",
]
DEV_PAIR = "pair1_llama31_same"
DEV_TASKS = {"hotpotqa", "musique"}
DISPLAY_RATIOS = [0.2, 0.3, 0.4, 0.5, 0.6]
FIXED_RE = re.compile(r"rekv_w(?P<window>\d+)_r(?P<ratio>[0-9.]+)_")
BREKV_RE = re.compile(
    r"cov_t(?P<tau>[0-9.]+)_s(?P<scale>[0-9.]+)_w(?P<window>\d+)_"
)
OTHER_RES = [
    re.compile(r"evict_r(?P<ratio>[0-9.]+)_"),
    re.compile(r"random_r(?P<ratio>[0-9.]+)_"),
]
METRICS = ("score", "actual_budget", "transmitted_bytes")


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
    values: np.ndarray  # score, actual budget, total transmitted bytes


def mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else math.nan


def parse_run(name: str) -> tuple[str, tuple[tuple[str, float | int], ...]] | None:
    match = FIXED_RE.match(name)
    if match:
        return "fixed", (
            ("ratio", float(match.group("ratio"))),
            ("window", int(match.group("window"))),
        )
    match = BREKV_RE.match(name)
    if match:
        return "brekv", (
            ("scale", float(match.group("scale"))),
            ("tau", float(match.group("tau"))),
            ("window", int(match.group("window"))),
        )
    for method, pattern in zip(("evict", "random"), OTHER_RES):
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
    if len(runs) != 1568:
        raise ValueError(f"Expected 1568 logical per_sample runs, found {len(runs)}")
    return sorted(runs, key=lambda run: (run.pair, run.task, run.method, run.params))


def sample_key(row: dict[str, Any]) -> str:
    if row.get("idx") is None:
        raise ValueError("Sample row is missing idx")
    return (
        f"{row['idx']}::{row.get('id')}"
        if row.get("id") is not None
        else str(row["idx"])
    )


def read_samples(path: Path) -> Samples:
    ids: list[str] = []
    values: list[list[float]] = []
    seen: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if "_meta" in row:
                continue
            key = sample_key(row)
            if key in seen:
                raise ValueError(f"Duplicate sample {key!r} in {path}")
            required = ("score", "budget", "total_communication_bytes")
            if any(row.get(field) is None for field in required):
                raise ValueError(f"Missing {required} for sample {key!r} in {path}")
            seen.add(key)
            ids.append(key)
            values.append(
                [
                    float(row["score"]),
                    float(row["budget"]),
                    float(row["total_communication_bytes"]),
                ]
            )
    if not values:
        raise ValueError(f"No samples in {path}")
    return Samples(ids, np.asarray(values, dtype=np.float64))


def is_canonical(run: Run, tau: float, scale: float, window: int) -> bool:
    return (
        run.method == "brekv"
        and run.param("tau") == tau
        and run.param("scale") == scale
        and run.param("window") == window
    )


def load_cells(
    runs: list[Run], tau: float, scale: float, window: int
) -> tuple[dict[tuple[str, str], dict[str, Samples]], list[float]]:
    selected: dict[tuple[str, str], dict[str, Run]] = defaultdict(dict)
    ratios: set[float] = set()
    for run in runs:
        cell = (run.pair, run.task)
        if run.method == "fixed" and run.param("window") == window:
            ratio = float(run.param("ratio"))
            ratios.add(ratio)
            selected[cell][f"fixed_{ratio:g}"] = run
        elif is_canonical(run, tau, scale, window):
            selected[cell]["brekv"] = run

    expected_cells = {(pair, task) for pair in PAIRS for task in TASKS}
    if set(selected) != expected_cells:
        raise ValueError(
            f"Cell coverage mismatch; missing={sorted(expected_cells - set(selected))}"
        )
    ratio_list = sorted(ratios)
    if ratio_list != [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
        raise ValueError(f"Unexpected fixed ReKV ratios: {ratio_list}")

    cells: dict[tuple[str, str], dict[str, Samples]] = {}
    expected_methods = {"brekv"} | {f"fixed_{ratio:g}" for ratio in ratio_list}
    for cell in sorted(expected_cells):
        if set(selected[cell]) != expected_methods:
            raise ValueError(f"Method coverage mismatch for {cell}: {sorted(selected[cell])}")
        loaded = {
            method: read_samples(run.path)
            for method, run in sorted(selected[cell].items())
        }
        reference = loaded["brekv"].ids
        if any(samples.ids != reference for samples in loaded.values()):
            raise ValueError(f"Ordered sample alignment mismatch in {cell}")
        cells[cell] = loaded
    return cells, ratio_list


def select_fixed_ratio(
    cells: dict[tuple[str, str], dict[str, Samples]],
    keys: list[tuple[str, str]],
    ratios: list[float],
) -> tuple[float, list[dict[str, float]]]:
    candidates = []
    for ratio in ratios:
        method = f"fixed_{ratio:g}"
        score = mean(cells[key][method].values[:, 0].mean() for key in keys)
        budget = mean(cells[key][method].values[:, 1].mean() for key in keys)
        candidates.append({"ratio": ratio, "score": score, "actual_budget": budget})
    # max score; an exact score tie is broken by lower observed actual budget.
    chosen = min(candidates, key=lambda row: (-row["score"], row["actual_budget"]))
    return chosen["ratio"], candidates


def scopes(heldout: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    return {
        "all_heldout": heldout,
        "unseen_task": [key for key in heldout if key[1] not in DEV_TASKS],
        "unseen_model": [key for key in heldout if key[0] != DEV_PAIR],
        "strict_unseen_task_model": [
            key for key in heldout if key[0] != DEV_PAIR and key[1] not in DEV_TASKS
        ],
    }


def policy_specs(
    dev_ratio: float, task_best: dict[str, float]
) -> list[tuple[str, float | None]]:
    specs = [(f"Fixed {ratio:.1f}", ratio) for ratio in DISPLAY_RATIOS]
    specs += [
        ("Dev-selected Fixed", dev_ratio),
        ("B-ReKV", None),
        ("Per-task Best Fixed", math.nan),
    ]
    return specs


def method_for(policy: str, ratio: float | None, task: str, task_best: dict[str, float]) -> str:
    if policy == "B-ReKV":
        return "brekv"
    selected = task_best[task] if policy == "Per-task Best Fixed" else ratio
    if selected is None or not math.isfinite(selected):
        raise ValueError(f"No fixed ratio for {policy}/{task}")
    return f"fixed_{selected:g}"


def percentile(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def add_cis(row: dict[str, Any], boot: np.ndarray) -> None:
    for index, metric in enumerate(METRICS):
        low, high = percentile(boot[:, index])
        row[f"{metric}_ci_low"] = low
        row[f"{metric}_ci_high"] = high


def bootstrap_cells(
    cells: dict[tuple[str, str], dict[str, Samples]],
    keys: list[tuple[str, str]],
    methods: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
    chunk: int = 250,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    output: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key in keys:
        arrays = np.stack([cells[key][method].values for method in methods], axis=1)
        n = arrays.shape[0]
        boot = np.empty((n_bootstrap, len(methods), len(METRICS)), dtype=np.float64)
        for start in range(0, n_bootstrap, chunk):
            count = min(chunk, n_bootstrap - start)
            indices = rng.integers(0, n, size=(count, n))
            boot[start : start + count] = arrays[indices].mean(axis=1)
        output[key] = {method: boot[:, i, :] for i, method in enumerate(methods)}
    return output


def build_results(
    cells: dict[tuple[str, str], dict[str, Samples]],
    heldout: list[tuple[str, str]],
    scope_keys: dict[str, list[tuple[str, str]]],
    dev_ratio: float,
    task_best: dict[str, float],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    specs = policy_specs(dev_ratio, task_best)
    needed = sorted(
        {
            method_for(policy, ratio, task, task_best)
            for policy, ratio in specs
            for task in TASKS
        }
    )
    cell_boot = bootstrap_cells(cells, heldout, needed, n_bootstrap, rng)
    cell_rows: list[dict[str, Any]] = []
    for scope, keys in scope_keys.items():
        for pair, task in keys:
            best_method = f"fixed_{task_best[task]:g}"
            best_point = cells[(pair, task)][best_method].values.mean(axis=0)
            best_boot = cell_boot[(pair, task)][best_method]
            for policy, ratio in specs:
                method = method_for(policy, ratio, task, task_best)
                point = cells[(pair, task)][method].values.mean(axis=0)
                boot = cell_boot[(pair, task)][method]
                row: dict[str, Any] = {
                    "scope": scope,
                    "pair": pair,
                    "task": task,
                    "policy": policy,
                    "selected_ratio": (
                        task_best[task]
                        if policy == "Per-task Best Fixed"
                        else ratio
                    ),
                    "n_samples": cells[(pair, task)][method].values.shape[0],
                    **{metric: float(point[i]) for i, metric in enumerate(METRICS)},
                    "regret_vs_per_task_best": float(best_point[0] - point[0]),
                }
                add_cis(row, boot)
                regret_boot = best_boot[:, 0] - boot[:, 0]
                row["regret_ci_low"], row["regret_ci_high"] = percentile(regret_boot)
                cell_rows.append(row)

    task_rows: list[dict[str, Any]] = []
    task_boots: dict[tuple[str, str, str], np.ndarray] = {}
    regret_boots: dict[tuple[str, str, str], np.ndarray] = {}
    for scope, keys in scope_keys.items():
        for task in TASKS:
            task_keys = [key for key in keys if key[1] == task]
            if not task_keys:
                continue
            best_method = f"fixed_{task_best[task]:g}"
            best_point = np.mean(
                [cells[key][best_method].values.mean(axis=0) for key in task_keys], axis=0
            )
            best_boot = np.mean(
                [cell_boot[key][best_method] for key in task_keys], axis=0
            )
            for policy, ratio in specs:
                methods = [method_for(policy, ratio, task, task_best)] * len(task_keys)
                point = np.mean(
                    [cells[key][method].values.mean(axis=0) for key, method in zip(task_keys, methods)],
                    axis=0,
                )
                boot = np.mean(
                    [cell_boot[key][method] for key, method in zip(task_keys, methods)],
                    axis=0,
                )
                regret_boot = best_boot[:, 0] - boot[:, 0]
                row = {
                    "scope": scope,
                    "task": task,
                    "policy": policy,
                    "selected_ratio": (
                        task_best[task]
                        if policy == "Per-task Best Fixed"
                        else ratio
                    ),
                    "n_cells": len(task_keys),
                    "n_samples": sum(
                        cells[key][method].values.shape[0]
                        for key, method in zip(task_keys, methods)
                    ),
                    **{metric: float(point[i]) for i, metric in enumerate(METRICS)},
                    "regret_vs_per_task_best": float(best_point[0] - point[0]),
                }
                add_cis(row, boot)
                row["regret_ci_low"], row["regret_ci_high"] = percentile(regret_boot)
                task_rows.append(row)
                task_boots[(scope, task, policy)] = boot
                regret_boots[(scope, task, policy)] = regret_boot

    summary_rows: list[dict[str, Any]] = []
    for scope, keys in scope_keys.items():
        present_tasks = [task for task in TASKS if any(key[1] == task for key in keys)]
        for policy, ratio in specs:
            rows = [
                row
                for row in task_rows
                if row["scope"] == scope and row["policy"] == policy
            ]
            boot = np.mean(
                [task_boots[(scope, task, policy)] for task in present_tasks], axis=0
            )
            regret_boot = np.mean(
                [regret_boots[(scope, task, policy)] for task in present_tasks], axis=0
            )
            worst_boot = np.max(
                [regret_boots[(scope, task, policy)] for task in present_tasks], axis=0
            )
            row = {
                "scope": scope,
                "policy": policy,
                "selected_ratio": (
                    "per-task" if policy == "Per-task Best Fixed" else ratio
                ),
                "n_cells": len(keys),
                "n_tasks": len(present_tasks),
                "n_samples": sum(
                    item["n_samples"]
                    for item in rows
                ),
                **{
                    metric: mean(item[metric] for item in rows)
                    for metric in METRICS
                },
                "regret_vs_per_task_best": mean(
                    item["regret_vs_per_task_best"] for item in rows
                ),
                "worst_task_regret": max(
                    item["regret_vs_per_task_best"] for item in rows
                ),
            }
            add_cis(row, boot)
            row["regret_ci_low"], row["regret_ci_high"] = percentile(regret_boot)
            row["worst_task_regret_ci_low"], row["worst_task_regret_ci_high"] = percentile(
                worst_boot
            )
            summary_rows.append(row)
    return cell_rows, task_rows, summary_rows


def win_counts(
    cells: dict[tuple[str, str], dict[str, Samples]],
    scope_keys: dict[str, list[tuple[str, str]]],
    dev_ratio: float,
) -> list[dict[str, Any]]:
    fixed = f"fixed_{dev_ratio:g}"
    rows = []
    for scope, keys in scope_keys.items():
        deltas = {
            key: float(cells[key]["brekv"].values[:, 0].mean() - cells[key][fixed].values[:, 0].mean())
            for key in keys
        }
        task_delta = {
            task: mean(value for (pair, item_task), value in deltas.items() if item_task == task)
            for task in TASKS
            if any(key[1] == task for key in keys)
        }
        rows.append(
            {
                "scope": scope,
                "n_cells": len(deltas),
                "cell_wins": sum(value > 0 for value in deltas.values()),
                "cell_ties": sum(value == 0 for value in deltas.values()),
                "cell_losses": sum(value < 0 for value in deltas.values()),
                "n_tasks": len(task_delta),
                "task_wins": sum(value > 0 for value in task_delta.values()),
                "task_ties": sum(value == 0 for value in task_delta.values()),
                "task_losses": sum(value < 0 for value in task_delta.values()),
            }
        )
    return rows


def csv_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: csv_value(value) for key, value in row.items()} for row in rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def make_figure(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    wins: list[dict[str, Any]],
) -> list[str]:
    mpl_dir = out_dir / ".matplotlib"
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scope_order = [
        "all_heldout",
        "unseen_task",
        "unseen_model",
        "strict_unseen_task_model",
    ]
    policies = [
        "Dev-selected Fixed",
        "B-ReKV",
        "Per-task Best Fixed",
    ]
    colors = {
        "Dev-selected Fixed": "#4c78a8",
        "B-ReKV": "#f58518",
        "Per-task Best Fixed": "#54a24b",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(scope_order))
    width = 0.24
    lookup = {(row["scope"], row["policy"]): row for row in summary_rows}
    for index, policy in enumerate(policies):
        rows = [lookup[(scope, policy)] for scope in scope_order]
        offset = (index - 1) * width
        axes[0].bar(
            x + offset,
            [row["score"] for row in rows],
            width,
            label=policy,
            color=colors[policy],
        )
        axes[1].bar(
            x + offset,
            [row["regret_vs_per_task_best"] for row in rows],
            width,
            label=policy,
            color=colors[policy],
        )
    axes[0].set_ylabel("Task-macro score")
    axes[0].set_title("Held-out score")
    axes[1].set_ylabel("Score regret")
    axes[1].set_title("Regret vs per-task best fixed")
    win_lookup = {row["scope"]: row for row in wins}
    axes[2].bar(
        x - width / 2,
        [win_lookup[scope]["cell_wins"] / win_lookup[scope]["n_cells"] for scope in scope_order],
        width,
        label="Cell win rate",
        color="#f58518",
    )
    axes[2].bar(
        x + width / 2,
        [win_lookup[scope]["task_wins"] / win_lookup[scope]["n_tasks"] for scope in scope_order],
        width,
        label="Task win rate",
        color="#72b7b2",
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("B-ReKV win fraction")
    axes[2].set_title("B-ReKV vs dev-selected fixed")
    labels = ["All heldout", "Unseen task", "Unseen model", "Strict unseen"]
    for axis in axes:
        axis.set_xticks(x, labels, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    outputs = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"brekv_heldout_generalization.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def write_report(
    path: Path,
    dev_ratio: float,
    dev_candidates: list[dict[str, float]],
    task_best: dict[str, float],
    summary_rows: list[dict[str, Any]],
    wins: list[dict[str, Any]],
    figures: list[str],
    n_bootstrap: int,
) -> None:
    lookup = {(row["scope"], row["policy"]): row for row in summary_rows}
    lines = [
        "# B-ReKV held-out 泛化分析",
        "",
        "## 口径",
        "",
        "- Dev 固定为 `pair1_llama31_same/{hotpotqa,musique}`；其余 54 cells 为 held-out。",
        "- Canonical B-ReKV：`t0.95 s0.75 w8`；fixed ReKV：`w8` 的 9 个观测 ratio。",
        "- Dev-selected Fixed 按两个 dev cells 的等权平均 score 最大选择；精确平分时取平均实际预算更低者。",
        "- Per-task Best Fixed 是事后 oracle：每个测试任务在该任务全部 held-out pairs 上选择均分最高的 fixed ratio，平分仍取实际预算更低者。",
        "- 聚合先在每个 task 内对 cells 等权，再对 tasks 等权。transmitted bytes 为 `total_communication_bytes`。",
        f"- CI：{n_bootstrap} 次 paired/stratified percentile bootstrap；每个 cell 内用同一组样本索引重采样所有方法，再按 cell/task 分层聚合。",
        "- `unseen_task`：任务不在 dev；`unseen_model`：pair 不在 dev；`strict`：两者同时未见。",
        "",
        "## 选择结果",
        "",
        f"- Dev-selected ratio：**{dev_ratio:g}**。",
        "- Dev candidates：" + ", ".join(
            f"r{row['ratio']:g}={row['score']:.6f}" for row in dev_candidates
        ),
        "- Per-task best：" + ", ".join(
            f"{task}=r{task_best[task]:g}" for task in TASKS
        ),
        "",
        "## Held-out 结果",
        "",
    ]
    for scope in (
        "all_heldout",
        "unseen_task",
        "unseen_model",
        "strict_unseen_task_model",
    ):
        lines.append(f"### {scope}")
        lines.append("")
        for policy in ("Dev-selected Fixed", "B-ReKV", "Per-task Best Fixed"):
            row = lookup[(scope, policy)]
            lines.append(
                f"- {policy}: score={row['score']:.6f} "
                f"[{row['score_ci_low']:.6f}, {row['score_ci_high']:.6f}], "
                f"budget={row['actual_budget']:.6f}, bytes={row['transmitted_bytes']:.1f}, "
                f"regret={row['regret_vs_per_task_best']:+.6f}, "
                f"worst-task={row['worst_task_regret']:+.6f}"
            )
        win = next(row for row in wins if row["scope"] == scope)
        lines.append(
            f"- B-ReKV vs Dev-selected Fixed：cell W/T/L="
            f"{win['cell_wins']}/{win['cell_ties']}/{win['cell_losses']}；task W/T/L="
            f"{win['task_wins']}/{win['task_ties']}/{win['task_losses']}。"
        )
        lines.append("")
    lines += ["## 图", ""] + [f"- `{figure}`" for figure in figures]
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
        default=Path("snapshots/analysis/brekv_heldout_generalization"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--canonical-tau", type=float, default=0.95)
    parser.add_argument("--canonical-scale", type=float, default=0.75)
    parser.add_argument("--canonical-window", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Discovering and validating 1568 logical runs...")
    runs = discover_runs(root)
    print("Loading aligned fixed ReKV and canonical B-ReKV samples...")
    cells, ratios = load_cells(
        runs, args.canonical_tau, args.canonical_scale, args.canonical_window
    )
    dev = [(DEV_PAIR, task) for task in TASKS if task in DEV_TASKS]
    heldout = [
        (pair, task)
        for pair in PAIRS
        for task in TASKS
        if (pair, task) not in dev
    ]
    if len(dev) != 2 or len(heldout) != 54:
        raise AssertionError(f"Expected 2 dev and 54 held-out cells, got {len(dev)}/{len(heldout)}")
    dev_ratio, dev_candidates = select_fixed_ratio(cells, dev, ratios)
    task_best = {
        task: select_fixed_ratio(
            cells, [key for key in heldout if key[1] == task], ratios
        )[0]
        for task in TASKS
    }
    scope_keys = scopes(heldout)
    expected_scope_sizes = {
        "all_heldout": 54,
        "unseen_task": 42,
        "unseen_model": 48,
        "strict_unseen_task_model": 36,
    }
    actual_scope_sizes = {scope: len(keys) for scope, keys in scope_keys.items()}
    if actual_scope_sizes != expected_scope_sizes:
        raise AssertionError(f"Unexpected scope sizes: {actual_scope_sizes}")

    print(f"Dev-selected fixed ratio: {dev_ratio:g}")
    print(f"Running {args.n_bootstrap} paired/stratified bootstrap replicates...")
    cell_rows, task_rows, summary_rows = build_results(
        cells,
        heldout,
        scope_keys,
        dev_ratio,
        task_best,
        args.n_bootstrap,
        rng,
    )
    wins = win_counts(cells, scope_keys, dev_ratio)
    write_csv(out_dir / "cell_metrics.csv", cell_rows)
    write_csv(out_dir / "task_metrics.csv", task_rows)
    write_csv(out_dir / "scope_summary.csv", summary_rows)
    write_csv(out_dir / "brekv_vs_dev_fixed_wins.csv", wins)
    figures = make_figure(out_dir, summary_rows, wins)
    summary = {
        "inputs": {"root": str(root), "logical_runs": len(runs)},
        "parameters": {
            "canonical_brekv": {
                "tau": args.canonical_tau,
                "scale": args.canonical_scale,
                "window": args.canonical_window,
            },
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "fixed_ratios_considered_for_selection": ratios,
            "fixed_ratios_displayed": DISPLAY_RATIOS,
        },
        "split": {
            "dev_pair": DEV_PAIR,
            "dev_tasks": sorted(DEV_TASKS),
            "dev_cells": len(dev),
            "heldout_cells": len(heldout),
            "scope_cells": actual_scope_sizes,
        },
        "selection": {
            "dev_selected_ratio": dev_ratio,
            "dev_candidates": dev_candidates,
            "per_task_best_fixed_ratio": task_best,
        },
        "scope_summary": summary_rows,
        "brekv_vs_dev_selected_fixed_wins": wins,
        "outputs": {
            "cell_csv": str(out_dir / "cell_metrics.csv"),
            "task_csv": str(out_dir / "task_metrics.csv"),
            "scope_csv": str(out_dir / "scope_summary.csv"),
            "wins_csv": str(out_dir / "brekv_vs_dev_fixed_wins.csv"),
            "report": str(out_dir / "REPORT.md"),
            "figures": figures,
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_report(
        out_dir / "REPORT.md",
        dev_ratio,
        dev_candidates,
        task_best,
        summary_rows,
        wins,
        figures,
        args.n_bootstrap,
    )
    print(f"Analysis complete: {out_dir}")


if __name__ == "__main__":
    main()
