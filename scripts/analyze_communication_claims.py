#!/usr/bin/env python3
"""Offline F3b latency, M1 matched-budget, and M2 bootstrap analysis."""

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
COST_PAIRS = [
    "pair1_llama31_same",
    "pair6_llama32_abliterated_deepseek3b",
    "pair7_qwen25_uncensored_bespoke",
]
COST_TASKS = ["hotpotqa", "musique", "multifieldqa_en"]
BASELINE_ORDER = ["ReKV", "ValueNorm/Evict", "Random"]
METHOD_COLORS = {
    "NLD": "#bab0ac",
    "ReKV": "#4c78a8",
    "B-ReKV": "#f58518",
    "ValueNorm/Evict": "#54a24b",
    "Random": "#e45756",
}
RUN_PATTERNS = {
    "ReKV": re.compile(r"rekv_w(?P<window>\d+)_r(?P<ratio>[0-9.]+)_"),
    "ValueNorm/Evict": re.compile(r"evict_r(?P<ratio>[0-9.]+)_"),
    "Random": re.compile(r"random_r(?P<ratio>[0-9.]+)_"),
    "B-ReKV": re.compile(
        r"cov_t(?P<tau>[0-9.]+)_s(?P<scale>[0-9.]+)_w(?P<window>\d+)_"
    ),
}


@dataclass(frozen=True)
class Run:
    pair: str
    task: str
    method: str
    params: tuple[tuple[str, float | int], ...]
    path: Path
    n: int
    score: float
    actual_budget: float

    def param(self, name: str) -> float | int | None:
        return dict(self.params).get(name)


def read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if "_meta" in item:
                meta = item["_meta"]
            else:
                rows.append(item)
    if not rows:
        raise ValueError(f"No sample rows in {path}")
    return meta, rows


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else math.nan


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def parse_run_name(name: str) -> tuple[str, tuple[tuple[str, float | int], ...]] | None:
    for method, pattern in RUN_PATTERNS.items():
        match = pattern.match(name)
        if match is None:
            continue
        params: dict[str, float | int] = {}
        for key, value in match.groupdict().items():
            params[key] = int(value) if key == "window" else float(value)
        return method, tuple(sorted(params.items()))
    return None


def discover_fairness_runs(root: Path) -> list[Run]:
    latest: dict[tuple[Any, ...], Path] = {}
    for path in root.glob("*/*/*/*/per_sample.jsonl"):
        pair, task, _, run_name = path.relative_to(root).parts[:4]
        parsed = parse_run_name(run_name)
        if parsed is None:
            continue
        method, params = parsed
        key = (pair, task, method, params)
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path

    runs: list[Run] = []
    for (pair, task, method, params), path in sorted(latest.items(), key=str):
        _, rows = read_jsonl(path)
        scores = [float(row["score"]) for row in rows if row.get("score") is not None]
        budgets = [float(row["budget"]) for row in rows if row.get("budget") is not None]
        if len(scores) != len(rows) or len(budgets) != len(rows):
            raise ValueError(f"Missing score/budget fields in {path}")
        runs.append(
            Run(
                pair=pair,
                task=task,
                method=method,
                params=params,
                path=path,
                n=len(rows),
                score=mean(scores),
                actual_budget=mean(budgets),
            )
        )
    return runs


def canonical_brekv(run: Run, tau: float, scale: float, window: int) -> bool:
    return (
        run.method == "B-ReKV"
        and run.param("tau") == tau
        and run.param("scale") == scale
        and run.param("window") == window
    )


def interpolation_bracket(curve: list[Run], target: float) -> tuple[Run, Run, float]:
    points = sorted(curve, key=lambda run: run.actual_budget)
    if len(points) < 2:
        raise ValueError("Matched-budget interpolation requires at least two points")
    if target < points[0].actual_budget or target > points[-1].actual_budget:
        raise ValueError(
            f"Target budget {target:.6f} outside "
            f"[{points[0].actual_budget:.6f}, {points[-1].actual_budget:.6f}]"
        )
    for lower, upper in zip(points, points[1:]):
        if lower.actual_budget <= target <= upper.actual_budget:
            span = upper.actual_budget - lower.actual_budget
            weight = 0.0 if span <= 0 else (target - lower.actual_budget) / span
            return lower, upper, weight
    raise AssertionError("Interpolation bracket was not found")


def build_m1(
    runs: list[Run], tau: float, scale: float, window: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str], tuple[Run, Run, float]]]:
    by_cell_method: dict[tuple[str, str, str], list[Run]] = defaultdict(list)
    brekv_by_cell: dict[tuple[str, str], Run] = {}
    for run in runs:
        if canonical_brekv(run, tau, scale, window):
            brekv_by_cell[(run.pair, run.task)] = run
        elif run.method in BASELINE_ORDER:
            by_cell_method[(run.pair, run.task, run.method)].append(run)

    expected_cells = {(pair, task) for pair in PAIR_ORDER for task in TASK_ORDER}
    if set(brekv_by_cell) != expected_cells:
        missing = sorted(expected_cells - set(brekv_by_cell))
        extra = sorted(set(brekv_by_cell) - expected_cells)
        raise ValueError(f"B-ReKV cells mismatch; missing={missing}, extra={extra}")

    matched_rows: list[dict[str, Any]] = []
    brackets: dict[tuple[str, str, str], tuple[Run, Run, float]] = {}
    for pair, task in sorted(expected_cells):
        brekv = brekv_by_cell[(pair, task)]
        for baseline in BASELINE_ORDER:
            curve = by_cell_method[(pair, task, baseline)]
            if len(curve) != 9:
                raise ValueError(f"Expected 9 {baseline} points for {pair}/{task}, got {len(curve)}")
            lower, upper, weight = interpolation_bracket(curve, brekv.actual_budget)
            matched_score = lower.score + weight * (upper.score - lower.score)
            brackets[(pair, task, baseline)] = (lower, upper, weight)
            matched_rows.append(
                {
                    "pair": pair,
                    "task": task,
                    "baseline_method": baseline,
                    "n": brekv.n,
                    "brekv_score": brekv.score,
                    "brekv_actual_budget": brekv.actual_budget,
                    "matched_baseline_score": matched_score,
                    "score_delta": brekv.score - matched_score,
                    "lower_ratio": lower.param("ratio"),
                    "upper_ratio": upper.param("ratio"),
                    "interpolation_weight": weight,
                    "lower_actual_budget": lower.actual_budget,
                    "upper_actual_budget": upper.actual_budget,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    group_specs = [("global", "all", matched_rows)]
    for pair in PAIR_ORDER:
        group_specs.append(("pair", pair, [row for row in matched_rows if row["pair"] == pair]))
    for task in TASK_ORDER:
        group_specs.append(("task", task, [row for row in matched_rows if row["task"] == task]))
    for scope, group, items in group_specs:
        for baseline in BASELINE_ORDER:
            selected = [row for row in items if row["baseline_method"] == baseline]
            deltas = [float(row["score_delta"]) for row in selected]
            summary_rows.append(
                {
                    "scope": scope,
                    "group": group,
                    "baseline_method": baseline,
                    "n_cells": len(selected),
                    "mean_delta": mean(deltas),
                    "median_delta": float(np.median(deltas)),
                    "wins": sum(value > 0 for value in deltas),
                    "losses": sum(value < 0 for value in deltas),
                    "ties": sum(value == 0 for value in deltas),
                    "mean_brekv_budget": mean(
                        float(row["brekv_actual_budget"]) for row in selected
                    ),
                }
            )
    return matched_rows, summary_rows, brackets


def samples_by_id(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    meta, rows = read_jsonl(path)
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_id = row.get("id")
        raw_idx = row.get("idx")
        if raw_idx is None:
            raise ValueError(f"Missing idx in {path}")
        sample_id = f"{raw_idx}::{raw_id}" if raw_id is not None else str(raw_idx)
        if sample_id in mapped:
            raise ValueError(f"Duplicate sample id {sample_id!r} in {path}")
        mapped[sample_id] = row
    return meta, mapped


def aligned_scores(
    brekv_path: Path, lower_path: Path, upper_path: Path, weight: float
) -> tuple[np.ndarray, list[str]]:
    _, brekv = samples_by_id(brekv_path)
    _, lower = samples_by_id(lower_path)
    _, upper = samples_by_id(upper_path)
    ids = list(brekv)
    if set(ids) != set(lower) or set(ids) != set(upper):
        raise ValueError(
            f"Sample ID mismatch: {brekv_path}, {lower_path}, {upper_path}"
        )
    brekv_scores = np.asarray([float(brekv[key]["score"]) for key in ids])
    lower_scores = np.asarray([float(lower[key]["score"]) for key in ids])
    upper_scores = np.asarray([float(upper[key]["score"]) for key in ids])
    return brekv_scores - ((1.0 - weight) * lower_scores + weight * upper_scores), ids


def bootstrap_means(
    deltas: np.ndarray, n_bootstrap: int, rng: np.random.Generator, chunk: int = 1000
) -> np.ndarray:
    if deltas.ndim != 1 or deltas.size == 0:
        raise ValueError("Bootstrap requires a non-empty one-dimensional array")
    output = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, chunk):
        count = min(chunk, n_bootstrap - start)
        indices = rng.integers(0, deltas.size, size=(count, deltas.size))
        output[start : start + count] = deltas[indices].mean(axis=1)
    return output


def ci_fields(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_win_probability": float(np.mean(values > 0.0)),
    }


def build_fairness_bootstrap(
    runs: list[Run],
    brackets: dict[tuple[str, str, str], tuple[Run, Run, float]],
    tau: float,
    scale: float,
    window: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    brekv_paths = {
        (run.pair, run.task): run.path
        for run in runs
        if canonical_brekv(run, tau, scale, window)
    }
    cell_rows: list[dict[str, Any]] = []
    boot_by_baseline: dict[str, list[np.ndarray]] = defaultdict(list)
    for pair in PAIR_ORDER:
        for task in TASK_ORDER:
            for baseline in BASELINE_ORDER:
                lower, upper, weight = brackets[(pair, task, baseline)]
                deltas, _ = aligned_scores(
                    brekv_paths[(pair, task)], lower.path, upper.path, weight
                )
                boot = bootstrap_means(deltas, n_bootstrap, rng)
                boot_by_baseline[baseline].append(boot)
                cell_rows.append(
                    {
                        "comparison": f"B-ReKV - {baseline}",
                        "pair": pair,
                        "task": task,
                        "n_pairs": int(deltas.size),
                        "mean_delta": float(deltas.mean()),
                        **ci_fields(boot),
                        "lower_ratio": lower.param("ratio"),
                        "upper_ratio": upper.param("ratio"),
                        "interpolation_weight": weight,
                    }
                )

    macro_rows = []
    for baseline in BASELINE_ORDER:
        matrix = np.vstack(boot_by_baseline[baseline])
        macro_boot = matrix.mean(axis=0)
        point = mean(
            float(row["mean_delta"])
            for row in cell_rows
            if row["comparison"] == f"B-ReKV - {baseline}"
        )
        macro_rows.append(
            {
                "comparison": f"B-ReKV - {baseline}",
                "scope": "56-cell stratified macro",
                "n_cells": matrix.shape[0],
                "mean_delta": point,
                **ci_fields(macro_boot),
            }
        )
    return cell_rows, macro_rows


def infer_cost_method(path: Path, root: Path) -> tuple[str, str, str]:
    pair, task, kind = path.relative_to(root).parts[:3]
    method = {"rekv": "ReKV", "brekv": "B-ReKV", "nld_aware": "NLD"}.get(kind)
    if method is None:
        raise ValueError(f"Unknown cost method directory {kind!r}: {path}")
    return pair, task, method


def discover_cost_profiles(kv_root: Path, nld_root: Path) -> dict[tuple[str, str, str], Path]:
    latest: dict[tuple[str, str, str], Path] = {}
    for root in (kv_root, nld_root):
        for path in root.glob("*/*/*/*/cost_profile.jsonl"):
            pair, task, method = infer_cost_method(path, root)
            if pair not in COST_PAIRS or task not in COST_TASKS:
                continue
            key = (pair, task, method)
            if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
                latest[key] = path
    expected = {
        (pair, task, method)
        for pair in COST_PAIRS
        for task in COST_TASKS
        for method in ("NLD", "ReKV", "B-ReKV")
    }
    if set(latest) != expected:
        raise ValueError(
            f"Cost profiles mismatch; missing={sorted(expected - set(latest))}, "
            f"extra={sorted(set(latest) - expected)}"
        )
    return latest


def build_f3b(
    profiles: dict[tuple[str, str, str], Path],
    bandwidths: list[float],
    rtts: list[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (pair, task, method), path in sorted(profiles.items()):
        meta, rows = read_jsonl(path)
        if method != "NLD" and meta.get("protocol_version") != "query_sketch_bf16_v1":
            raise ValueError(f"Unexpected KV protocol in {path}: {meta.get('protocol_version')}")
        compute = np.asarray([float(row["t_total"]) for row in rows])
        byte_field = "nld_text_payload_bytes" if method == "NLD" else "total_communication_bytes"
        if any(row.get(byte_field) is None for row in rows):
            raise ValueError(f"Missing {byte_field} in {path}")
        payload = np.asarray([float(row[byte_field]) for row in rows])
        for bandwidth in bandwidths:
            transfer = payload * 8.0 / (bandwidth * 1e9)
            for rtt in rtts:
                network = transfer + rtt / 1000.0
                e2e = compute + network
                output.append(
                    {
                        "pair": pair,
                        "task": task,
                        "method": method,
                        "n": len(rows),
                        "bandwidth_gbps": bandwidth,
                        "rtt_ms": rtt,
                        "rtt_rounds": 1,
                        "payload_bytes_mean": float(payload.mean()),
                        "compute_s_mean": float(compute.mean()),
                        "transfer_s_mean": float(transfer.mean()),
                        "network_s_mean": float(network.mean()),
                        "e2e_s_mean": float(e2e.mean()),
                        "e2e_s_median": float(np.median(e2e)),
                        "e2e_s_p95": float(np.quantile(e2e, 0.95)),
                        "source": str(path),
                    }
                )
    return output


def build_cost_bootstrap(
    profiles: dict[tuple[str, str, str], Path],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_rows: list[dict[str, Any]] = []
    boot_by_comparison: dict[str, list[np.ndarray]] = defaultdict(list)
    for pair in COST_PAIRS:
        for task in COST_TASKS:
            samples = {}
            for method in ("NLD", "ReKV", "B-ReKV"):
                _, samples[method] = samples_by_id(profiles[(pair, task, method)])
            ids = list(samples["NLD"])
            for method in ("ReKV", "B-ReKV"):
                if set(ids) != set(samples[method]):
                    raise ValueError(f"Cost sample ID mismatch for {pair}/{task}/{method}")
            nld_scores = np.asarray([float(samples["NLD"][key]["score"]) for key in ids])
            for baseline in ("ReKV", "B-ReKV"):
                baseline_scores = np.asarray(
                    [float(samples[baseline][key]["score"]) for key in ids]
                )
                deltas = nld_scores - baseline_scores
                boot = bootstrap_means(deltas, n_bootstrap, rng)
                comparison = f"NLD - {baseline}"
                boot_by_comparison[comparison].append(boot)
                cell_rows.append(
                    {
                        "comparison": comparison,
                        "pair": pair,
                        "task": task,
                        "n_pairs": len(ids),
                        "mean_delta": float(deltas.mean()),
                        **ci_fields(boot),
                    }
                )

    macro_rows = []
    for comparison, boots in sorted(boot_by_comparison.items()):
        matrix = np.vstack(boots)
        macro_boot = matrix.mean(axis=0)
        macro_rows.append(
            {
                "comparison": comparison,
                "scope": "9-cell stratified macro",
                "n_cells": matrix.shape[0],
                "mean_delta": mean(
                    float(row["mean_delta"])
                    for row in cell_rows
                    if row["comparison"] == comparison
                ),
                **ci_fields(macro_boot),
            }
        )
    return cell_rows, macro_rows


def fixed_rekv_curve(runs: list[Run]) -> list[dict[str, Any]]:
    by_ratio: dict[float, list[Run]] = defaultdict(list)
    for run in runs:
        if run.method == "ReKV" and run.param("window") == 8:
            by_ratio[float(run.param("ratio"))].append(run)
    output = []
    for ratio, points in sorted(by_ratio.items()):
        if len(points) != 56:
            raise ValueError(f"Expected 56 ReKV points at r={ratio}, got {len(points)}")
        output.append(
            {
                "ratio": ratio,
                "n_cells": len(points),
                "actual_budget": mean(point.actual_budget for point in points),
                "score": mean(point.score for point in points),
            }
        )
    if len(output) != 9:
        raise ValueError(f"Expected 9 fixed-r macro points, got {len(output)}")
    return output


def make_plots(
    out_dir: Path,
    f3b_rows: list[dict[str, Any]],
    runs: list[Run],
    curve_rows: list[dict[str, Any]],
    tau: float,
    scale: float,
    window: int,
) -> list[str]:
    matplotlib_config = out_dir / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    fig, axes = plt.subplots(1, len(rtts := sorted({row["rtt_ms"] for row in f3b_rows})), figsize=(5 * len(rtts), 4), squeeze=False)
    macro: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    for row in f3b_rows:
        macro[(row["method"], row["bandwidth_gbps"], row["rtt_ms"])].append(
            float(row["e2e_s_mean"])
        )
    for index, rtt in enumerate(rtts):
        ax = axes[0][index]
        for method in ("NLD", "ReKV", "B-ReKV"):
            bandwidths = sorted({key[1] for key in macro if key[0] == method})
            values = [mean(macro[(method, bandwidth, rtt)]) for bandwidth in bandwidths]
            ax.plot(
                bandwidths,
                values,
                marker="o",
                label=method,
                color=METHOD_COLORS[method],
            )
        ax.set_xscale("log")
        ax.set_xlabel("Bandwidth (Gbps)")
        ax.set_ylabel("End-to-end latency (s)")
        ax.set_title(f"RTT={rtt:g} ms")
        ax.grid(alpha=0.25)
        if index == 0:
            ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figure_dir / f"f3b_latency_curves.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)

    brekv = [
        run
        for run in runs
        if canonical_brekv(run, tau, scale, window)
    ]
    brekv_budget = mean(run.actual_budget for run in brekv)
    brekv_score = mean(run.score for run in brekv)
    matched_score = math.nan
    for lower, upper in zip(curve_rows, curve_rows[1:]):
        if lower["actual_budget"] <= brekv_budget <= upper["actual_budget"]:
            span = upper["actual_budget"] - lower["actual_budget"]
            weight = (
                0.0
                if span <= 0
                else (brekv_budget - lower["actual_budget"]) / span
            )
            matched_score = lower["score"] + weight * (
                upper["score"] - lower["score"]
            )
            break
    if math.isnan(matched_score):
        raise ValueError("Macro B-ReKV budget is outside the fixed ReKV curve")

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.plot(
        [row["actual_budget"] for row in curve_rows],
        [row["score"] for row in curve_rows],
        marker="o",
        label="Fixed ReKV (macro)",
        color=METHOD_COLORS["ReKV"],
    )
    ax.scatter(
        [brekv_budget],
        [brekv_score],
        marker="*",
        s=170,
        label="B-ReKV",
        color=METHOD_COLORS["B-ReKV"],
        zorder=3,
    )
    ax.scatter(
        [brekv_budget],
        [matched_score],
        marker="x",
        s=90,
        label="Matched fixed ReKV",
        color="black",
        zorder=3,
    )
    ax.set_xlabel("Actual KV budget")
    ax.set_ylabel("Task score (56-cell macro)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = figure_dir / f"m1_rekv_pareto_macro.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def write_report(
    path: Path,
    f3b_rows: list[dict[str, Any]],
    m1_summary: list[dict[str, Any]],
    bootstrap_macro: list[dict[str, Any]],
    figures: list[str],
    n_runs: int,
) -> None:
    global_m1 = [row for row in m1_summary if row["scope"] == "global"]
    lines = [
        "# F3b / M1 / M2 离线分析",
        "",
        "## 完整性",
        "",
        f"- Matched-budget runs：{n_runs}/1568",
        f"- F3b 聚合行：{len(f3b_rows)}",
        "- RTT 假设：每个方法均按一次请求-响应 RTT；序列化时间使用现有 wire payload。",
        "- Bootstrap：paired percentile 95% CI；分层宏平均保持每个 pair-task 等权。",
        "",
        "## M1 matched-budget 点估计",
        "",
    ]
    for row in global_m1:
        lines.append(
            f"- B-ReKV vs {row['baseline_method']}: "
            f"Δ={row['mean_delta']:+.6f}, "
            f"W/L/T={row['wins']}/{row['losses']}/{row['ties']}"
        )
    lines += ["", "## M2 paired bootstrap", ""]
    for row in bootstrap_macro:
        lines.append(
            f"- {row['comparison']} ({row['scope']}): "
            f"Δ={row['mean_delta']:+.6f}, "
            f"95% CI [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}], "
            f"P(Δ>0)={row['bootstrap_win_probability']:.4f}"
        )
    lines += ["", "## 图", ""]
    lines.extend(f"- `{figure}`" for figure in figures)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fairness-root",
        type=Path,
        default=Path("snapshots/full_matched_budget_fairness_query_sketch"),
    )
    parser.add_argument(
        "--kv-cost-root", type=Path, default=Path("snapshots/query_sketch_cost_v1")
    )
    parser.add_argument(
        "--nld-cost-root",
        type=Path,
        default=Path("snapshots/nld_receiver_aware_cost_v1"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("snapshots/analysis/communication_claims")
    )
    parser.add_argument("--bandwidth-gbps", nargs="+", type=float, default=[1, 10, 25, 100])
    parser.add_argument("--rtt-ms", nargs="+", type=float, default=[0, 1, 10, 50])
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
    if any(value <= 0 for value in args.bandwidth_gbps):
        raise ValueError("Bandwidth values must be positive")
    if any(value < 0 for value in args.rtt_ms):
        raise ValueError("RTT values cannot be negative")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading 1568-run matched-budget sweep...")
    runs = discover_fairness_runs(args.fairness_root.resolve())
    if len(runs) != 1568:
        raise ValueError(f"Expected 1568 logical runs, found {len(runs)}")
    m1_rows, m1_summary, brackets = build_m1(
        runs, args.canonical_tau, args.canonical_scale, args.canonical_window
    )
    if len(m1_rows) != 168:
        raise ValueError(f"Expected 168 matched rows, found {len(m1_rows)}")
    curve_rows = fixed_rekv_curve(runs)

    print(f"Running fairness paired bootstrap ({args.n_bootstrap} replicates)...")
    fairness_cells, fairness_macro = build_fairness_bootstrap(
        runs,
        brackets,
        args.canonical_tau,
        args.canonical_scale,
        args.canonical_window,
        args.n_bootstrap,
        rng,
    )

    print("Loading paired cost profiles...")
    profiles = discover_cost_profiles(
        args.kv_cost_root.resolve(), args.nld_cost_root.resolve()
    )
    f3b_rows = build_f3b(profiles, args.bandwidth_gbps, args.rtt_ms)
    print(f"Running cost paired bootstrap ({args.n_bootstrap} replicates)...")
    cost_cells, cost_macro = build_cost_bootstrap(profiles, args.n_bootstrap, rng)

    write_csv(out_dir / "f3b_latency.csv", f3b_rows)
    write_csv(out_dir / "m1_matched_budget.csv", m1_rows)
    write_csv(out_dir / "m1_summary.csv", m1_summary)
    write_csv(out_dir / "m1_fixed_rekv_curve.csv", curve_rows)
    m2_cells = fairness_cells + cost_cells
    m2_macro = fairness_macro + cost_macro
    write_csv(out_dir / "m2_paired_bootstrap.csv", m2_cells)
    write_csv(out_dir / "m2_paired_bootstrap_macro.csv", m2_macro)

    figures = make_plots(
        out_dir,
        f3b_rows,
        runs,
        curve_rows,
        args.canonical_tau,
        args.canonical_scale,
        args.canonical_window,
    )
    summary = {
        "inputs": {
            "fairness_root": str(args.fairness_root.resolve()),
            "kv_cost_root": str(args.kv_cost_root.resolve()),
            "nld_cost_root": str(args.nld_cost_root.resolve()),
        },
        "parameters": {
            "bandwidth_gbps": args.bandwidth_gbps,
            "rtt_ms": args.rtt_ms,
            "rtt_rounds": 1,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "canonical_brekv": {
                "tau": args.canonical_tau,
                "scale": args.canonical_scale,
                "window": args.canonical_window,
            },
        },
        "validation": {
            "logical_runs": len(runs),
            "matched_rows": len(m1_rows),
            "fairness_cells": len(fairness_cells),
            "cost_profiles": len(profiles),
            "cost_bootstrap_cells": len(cost_cells),
            "f3b_rows": len(f3b_rows),
        },
        "m1_global": [row for row in m1_summary if row["scope"] == "global"],
        "m2_macro": m2_macro,
        "figures": figures,
    }
    write_json(out_dir / "summary.json", summary)
    write_report(
        out_dir / "REPORT.md",
        f3b_rows,
        m1_summary,
        m2_macro,
        figures,
        len(runs),
    )
    print(f"Analysis complete: {out_dir}")


if __name__ == "__main__":
    main()
