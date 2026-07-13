#!/usr/bin/env python3
"""Analyze Stage 3 matched-budget, Pareto, and budget-distribution evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


RUN_PATTERNS = {
    "ReKV": re.compile(r"rekv_w(?P<window>\d+)_r(?P<ratio>[0-9.]+)_"),
    "ValueNorm/Evict": re.compile(r"evict_r(?P<ratio>[0-9.]+)_"),
    "Random": re.compile(r"random_r(?P<ratio>[0-9.]+)_"),
    "B-ReKV": re.compile(
        r"cov_t(?P<tau>[0-9.]+)_s(?P<scale>[0-9.]+)_w(?P<window>\d+)_"
    ),
    "Strict Adaptive": re.compile(
        r"strict_adapt_t(?P<tau_min>[0-9.]+)-(?P<tau_max>[0-9.]+)_w(?P<window>\d+)_"
    ),
}


def parse_run_name(name: str) -> tuple[str, dict[str, float | int]] | None:
    for method, pattern in RUN_PATTERNS.items():
        match = pattern.match(name)
        if not match:
            continue
        params: dict[str, float | int] = {}
        for key, value in match.groupdict().items():
            params[key] = int(value) if key == "window" else float(value)
        return method, params
    return None


def logical_key(method: str, params: dict[str, float | int]) -> tuple[Any, ...]:
    return (method, *sorted(params.items()))


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
    return meta, rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(max(round((len(ordered) - 1) * fraction), 0), len(ordered) - 1)
    return ordered[index]


def discover_runs(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], list[float]]]:
    latest: dict[tuple[Any, ...], tuple[Path, str, dict[str, float | int]]] = {}
    for path in root.glob("*/*/*/*/per_sample.jsonl"):
        relative = path.relative_to(root)
        pair, task, kind, run_name = relative.parts[:4]
        parsed = parse_run_name(run_name)
        if parsed is None:
            continue
        method, params = parsed
        key = (pair, task, kind, *logical_key(method, params))
        if key not in latest or path.stat().st_mtime > latest[key][0].stat().st_mtime:
            latest[key] = (path, method, params)

    runs = []
    sample_budgets: dict[tuple[Any, ...], list[float]] = {}
    for path, method, params in latest.values():
        pair, task, kind = path.relative_to(root).parts[:3]
        meta, rows = read_jsonl(path)
        scores = [float(row["score"]) for row in rows if row.get("score") is not None]
        budgets = [float(row["budget"]) for row in rows if row.get("budget") is not None]
        if not scores or not budgets:
            continue
        row: dict[str, Any] = {
            "pair": pair,
            "task": task,
            "kind": kind,
            "method": method,
            **params,
            "n": len(rows),
            "score": mean(scores),
            "actual_budget": mean(budgets),
            "budget_std": math.sqrt(
                sum((value - mean(budgets)) ** 2 for value in budgets)
                / max(len(budgets) - 1, 1)
            ),
            "protocol_version": meta.get("protocol_version"),
            "query_sketch_mode": meta.get("query_sketch_mode"),
            "path": str(path.parent),
        }
        coverage_values = [
            float(item["coverage_achieved"])
            for item in rows
            if item.get("coverage_achieved") is not None
        ]
        satisfied_values = [
            float(item["coverage_satisfied_layer_ratio"])
            for item in rows
            if item.get("coverage_satisfied_layer_ratio") is not None
        ]
        row["coverage_achieved"] = mean(coverage_values)
        row["coverage_satisfied_layer_ratio"] = mean(satisfied_values)
        runs.append(row)
        sample_budgets[(pair, task, *logical_key(method, params))] = budgets
    return runs, sample_budgets


def interpolate_curve(curve: list[dict[str, Any]], target: float) -> tuple[float, str, str]:
    points = sorted(curve, key=lambda row: row["actual_budget"])
    if not points:
        return math.nan, "missing", ""
    if target < points[0]["actual_budget"]:
        return math.nan, "below_range", points[0]["path"]
    if target > points[-1]["actual_budget"]:
        return math.nan, "above_range", points[-1]["path"]
    for lower, upper in zip(points, points[1:]):
        low_budget = lower["actual_budget"]
        high_budget = upper["actual_budget"]
        if low_budget <= target <= high_budget:
            span = high_budget - low_budget
            weight = 0.0 if span <= 0 else (target - low_budget) / span
            score = lower["score"] + weight * (upper["score"] - lower["score"])
            bracket = (
                f"{lower.get('ratio', low_budget):g}.."
                f"{upper.get('ratio', high_budget):g}"
            )
            return score, "matched", bracket
    point = min(points, key=lambda row: abs(row["actual_budget"] - target))
    return point["score"], "matched", str(point.get("ratio", point["actual_budget"]))


def matched_budget_rows(
    runs: list[dict[str, Any]], tau: float, scale: float, window: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    canonical: dict[tuple[str, str], dict[str, Any]] = {}
    for row in runs:
        key = (row["pair"], row["task"])
        if (
            row["method"] == "B-ReKV"
            and row.get("tau") == tau
            and row.get("scale") == scale
            and row.get("window") == window
        ):
            canonical[key] = row
        elif row["method"] in {"ReKV", "ValueNorm/Evict", "Random"}:
            grouped[(*key, row["method"])].append(row)

    output = []
    for (pair, task), brekv in sorted(canonical.items()):
        for method in ("ReKV", "ValueNorm/Evict", "Random"):
            matched_score, status, bracket = interpolate_curve(
                grouped[(pair, task, method)], brekv["actual_budget"]
            )
            output.append(
                {
                    "pair": pair,
                    "task": task,
                    "target_method": "B-ReKV",
                    "target_score": brekv["score"],
                    "target_actual_budget": brekv["actual_budget"],
                    "baseline_method": method,
                    "matched_baseline_score": matched_score,
                    "score_delta": (
                        brekv["score"] - matched_score
                        if not math.isnan(matched_score)
                        else math.nan
                    ),
                    "match_status": status,
                    "fixed_ratio_bracket": bracket,
                }
            )
    return output


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front = []
    best_score = -math.inf
    for row in sorted(rows, key=lambda item: (item["actual_budget"], -item["score"])):
        if row["score"] > best_score + 1e-12:
            front.append(row)
            best_score = row["score"]
    return front


def pareto_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        if row["method"] == "B-ReKV":
            groups[(row["pair"], row["task"])].append(row)
    output = []
    for _, rows in sorted(groups.items()):
        front_paths = {row["path"] for row in pareto_front(rows)}
        for row in sorted(rows, key=lambda item: item["actual_budget"]):
            output.append(
                {
                    "pair": row["pair"],
                    "task": row["task"],
                    "window": row["window"],
                    "tau": row["tau"],
                    "scale": row["scale"],
                    "score": row["score"],
                    "actual_budget": row["actual_budget"],
                    "on_pareto_front": row["path"] in front_paths,
                    "path": row["path"],
                }
            )
    return output


def budget_distribution_rows(
    runs: list[dict[str, Any]],
    budgets_by_run: dict[tuple[Any, ...], list[float]],
    tau: float,
    scale: float,
    window: int,
) -> list[dict[str, Any]]:
    output = []
    for row in runs:
        if not (
            row["method"] == "B-ReKV"
            and row.get("tau") == tau
            and row.get("scale") == scale
            and row.get("window") == window
        ):
            continue
        params = {"tau": tau, "scale": scale, "window": window}
        budgets = budgets_by_run[(row["pair"], row["task"], *logical_key("B-ReKV", params))]
        budget_mean = mean(budgets)
        variance = sum((value - budget_mean) ** 2 for value in budgets) / max(
            len(budgets) - 1, 1
        )
        output.append(
            {
                "pair": row["pair"],
                "task": row["task"],
                "n": len(budgets),
                "mean": budget_mean,
                "std": math.sqrt(variance),
                "min": min(budgets),
                "p10": percentile(budgets, 0.10),
                "p25": percentile(budgets, 0.25),
                "p50": percentile(budgets, 0.50),
                "p75": percentile(budgets, 0.75),
                "p90": percentile(budgets, 0.90),
                "max": max(budgets),
                "unique_budgets": len(set(budgets)),
                "path": row["path"],
            }
        )
    return sorted(output, key=lambda item: (item["pair"], item["task"]))


def strict_rows(
    runs: list[dict[str, Any]], tau: float, scale: float, window: int
) -> list[dict[str, Any]]:
    calibrated = {}
    for row in runs:
        if (
            row["method"] == "B-ReKV"
            and row.get("tau") == tau
            and row.get("scale") == scale
            and row.get("window") == window
        ):
            calibrated[(row["pair"], row["task"])] = row
    output = []
    for row in runs:
        if row["method"] != "Strict Adaptive":
            continue
        base = calibrated.get((row["pair"], row["task"]))
        output.append(
            {
                "pair": row["pair"],
                "task": row["task"],
                "strict_window": row["window"],
                "strict_tau_min": row["tau_min"],
                "strict_tau_max": row["tau_max"],
                "strict_score": row["score"],
                "strict_actual_budget": row["actual_budget"],
                "coverage_achieved": row["coverage_achieved"],
                "coverage_satisfied_layer_ratio": row[
                    "coverage_satisfied_layer_ratio"
                ],
                "calibrated_score": base["score"] if base else math.nan,
                "calibrated_actual_budget": (
                    base["actual_budget"] if base else math.nan
                ),
                "score_delta_strict_minus_calibrated": (
                    row["score"] - base["score"] if base else math.nan
                ),
                "budget_delta_strict_minus_calibrated": (
                    row["actual_budget"] - base["actual_budget"]
                    if base
                    else math.nan
                ),
                "path": row["path"],
            }
        )
    return sorted(
        output,
        key=lambda item: (
            item["pair"],
            item["task"],
            item["strict_window"],
            item["strict_tau_min"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    out_dir: Path,
    runs: list[dict[str, Any]],
    budgets_by_run: dict[tuple[Any, ...], list[float]],
    tau: float,
    scale: float,
    window: int,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is unavailable; CSV/Markdown written, PNG plots skipped")
        return False

    pairs = sorted({row["pair"] for row in runs})
    tasks = sorted({row["task"] for row in runs})
    if pairs and tasks:
        fig, axes = plt.subplots(
            len(pairs), len(tasks), figsize=(4.8 * len(tasks), 3.7 * len(pairs)), squeeze=False
        )
        for pair_index, pair in enumerate(pairs):
            for task_index, task in enumerate(tasks):
                ax = axes[pair_index][task_index]
                points = [
                    row
                    for row in runs
                    if row["pair"] == pair
                    and row["task"] == task
                    and row["method"] == "B-ReKV"
                ]
                for candidate_window, marker in ((8, "o"), (16, "^")):
                    selected = [
                        row for row in points if row["window"] == candidate_window
                    ]
                    ax.scatter(
                        [row["actual_budget"] for row in selected],
                        [row["score"] for row in selected],
                        marker=marker,
                        label=f"w={candidate_window}",
                        alpha=0.8,
                    )
                front = pareto_front(points)
                ax.plot(
                    [row["actual_budget"] for row in front],
                    [row["score"] for row in front],
                    color="black",
                    linewidth=1.2,
                    label="Pareto front",
                )
                ax.set_title(f"{pair} / {task}")
                ax.set_xlabel("Actual KV budget")
                ax.set_ylabel("Task score")
                ax.grid(alpha=0.25)
                if pair_index == 0 and task_index == 0:
                    ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "brekv_accuracy_budget_pareto.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(
            len(pairs), len(tasks), figsize=(4.8 * len(tasks), 3.4 * len(pairs)), squeeze=False
        )
        for pair_index, pair in enumerate(pairs):
            for task_index, task in enumerate(tasks):
                ax = axes[pair_index][task_index]
                params = {"tau": tau, "scale": scale, "window": window}
                budgets = budgets_by_run.get(
                    (pair, task, *logical_key("B-ReKV", params)), []
                )
                if budgets:
                    ax.hist(budgets, bins=25, color="#e45756", alpha=0.82)
                    ax.axvline(mean(budgets), color="black", linewidth=1.5)
                ax.set_title(f"{pair} / {task}")
                ax.set_xlabel("Per-query actual KV budget")
                ax.set_ylabel("Samples")
                ax.grid(axis="y", alpha=0.25)
        fig.suptitle(
            f"B-ReKV budget distribution: tau={tau:g}, scale={scale:g}, w={window}"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "brekv_per_query_budget_distribution.png", dpi=180)
        plt.close(fig)
    return True


def write_report(
    path: Path,
    runs: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    distributions: list[dict[str, Any]],
    strict: list[dict[str, Any]],
    plots_written: bool,
) -> None:
    matched_ok = [row for row in matched if row["match_status"] == "matched"]
    out_of_range = [row for row in matched if row["match_status"] != "matched"]
    protocols = sorted({str(row["protocol_version"]) for row in runs})
    lines = [
        "# Stage 3 核心审稿证据",
        "",
        f"- 完成 runs：{len(runs)}",
        f"- 协议标识：{', '.join(protocols)}",
        f"- matched-budget 比较：{len(matched_ok)}/{len(matched)} 可插值",
        f"- 超出 fixed-r 网格：{len(out_of_range)}",
        f"- B-ReKV Pareto 点：{sum(bool(row['on_pareto_front']) for row in pareto)}",
        f"- 预算分布单元：{len(distributions)}",
        f"- Strict 对比单元：{len(strict)}",
        f"- PNG 图：{'已生成' if plots_written else '未生成（缺少 matplotlib 或无数据）'}",
        "",
        "## Matched-budget fairness",
        "",
    ]
    for row in matched:
        delta = row["score_delta"]
        delta_text = "NA" if math.isnan(delta) else f"{delta:+.4f}"
        lines.append(
            f"- {row['pair']} / {row['task']} / {row['baseline_method']}: "
            f"budget={row['target_actual_budget']:.4f}, delta={delta_text}, "
            f"status={row['match_status']}"
        )
    lines += ["", "## Budget distribution", ""]
    for row in distributions:
        lines.append(
            f"- {row['pair']} / {row['task']}: mean={row['mean']:.4f}, "
            f"std={row['std']:.4f}, p10-p90={row['p10']:.4f}-{row['p90']:.4f}, "
            f"unique={row['unique_budgets']}"
        )
    if out_of_range:
        lines += [
            "",
            "## 警告",
            "",
            "以下 matched-budget 目标超出 fixed-r 网格，需扩展 `FIXED_RATIOS` 后补跑：",
        ]
        for row in out_of_range:
            lines.append(
                f"- {row['pair']} / {row['task']} / {row['baseline_method']}: "
                f"{row['match_status']}"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("snapshots/stage3_core_reviewer_query_sketch")
    )
    parser.add_argument("--canonical-tau", type=float, default=0.95)
    parser.add_argument("--canonical-scale", type=float, default=0.75)
    parser.add_argument("--canonical-window", type=int, default=8)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, budgets_by_run = discover_runs(root)
    matched = matched_budget_rows(
        runs, args.canonical_tau, args.canonical_scale, args.canonical_window
    )
    pareto = pareto_rows(runs)
    distributions = budget_distribution_rows(
        runs,
        budgets_by_run,
        args.canonical_tau,
        args.canonical_scale,
        args.canonical_window,
    )
    strict = strict_rows(
        runs, args.canonical_tau, args.canonical_scale, args.canonical_window
    )

    write_csv(out_dir / "all_runs.csv", runs)
    write_csv(out_dir / "query_fairness_matched_budget.csv", matched)
    write_csv(out_dir / "brekv_accuracy_budget_pareto.csv", pareto)
    write_csv(out_dir / "brekv_budget_distribution.csv", distributions)
    write_csv(out_dir / "calibrated_vs_strict_adaptive.csv", strict)
    plots_written = make_plots(
        out_dir,
        runs,
        budgets_by_run,
        args.canonical_tau,
        args.canonical_scale,
        args.canonical_window,
    )
    write_report(
        out_dir / "REPORT.md",
        runs,
        matched,
        pareto,
        distributions,
        strict,
        plots_written,
    )
    print(f"runs={len(runs)} analysis={out_dir}")


if __name__ == "__main__":
    main()
