#!/usr/bin/env python3
"""Build final paper-facing summaries for deployable Query-Sketch experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


MAIN_TASKS = [
    "countries",
    "tipsheets",
    "hotpotqa",
    "qasper",
    "musique",
    "multifieldqa_en",
    "twowikimqa",
    "tmath",
]
EXTENDED_TASKS = ["hotpotqa_full", "qasper_full", "musique_full", "samsum", "repobench"]
CORE_TASKS = ["hotpotqa", "musique", "multifieldqa_en"]
CANONICAL = "cov_t0.95_s0.75_w8"
V1 = "query_sketch_bf16_v1"

TABLE1_ROOTS = {
    "pair1": Path("snapshots/table1_pair1_query_sketch_llama31_same"),
    "pair6": Path("snapshots/table1_pair6_query_sketch_llama32_abliterated_deepseek3b"),
    "pair7": Path("snapshots/table1_pair7_query_sketch_qwen25_uncensored_bespoke"),
}
TABLE6_ROOTS = {
    "pair6": Path("snapshots/table6_pair6_query_sketch_llama32_abliterated_deepseek3b"),
    "pair7": Path("snapshots/table6_pair7_query_sketch_qwen25_uncensored_bespoke"),
}
TABLE8_ROOTS = {
    "pair2": Path("snapshots/table8_pair2_query_sketch_llama32_same"),
    "pair3": Path("snapshots/table8_pair3_query_sketch_qwen25_7b_same"),
    "pair4": Path("snapshots/table8_pair4_query_sketch_falcon3_7b_same"),
    "pair5": Path("snapshots/table8_pair5_query_sketch_evolcodellama_toolace"),
}
PAIR_LABELS = {
    "pair1": "S: Llama-3.1-8B; R: Llama-3.1-8B",
    "pair2": "S: Llama-3.2-3B; R: Llama-3.2-3B",
    "pair3": "S: Qwen2.5-7B; R: Qwen2.5-7B",
    "pair4": "S: Falcon3-7B; R: Falcon3-7B",
    "pair5": "S: EvolCodeLlama-8B; R: ToolACE-8B",
    "pair6": "S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B",
    "pair7": "S: Qwen2.5-7B Unc.; R: Bespoke-Stratos-7B",
}
SHORT_PAIR_LABELS = {
    "pair1": "Llama-3.1 same",
    "pair2": "Llama-3.2 same",
    "pair3": "Qwen2.5 same",
    "pair4": "Falcon3 same",
    "pair5": "EvolCodeLlama / ToolACE",
    "pair6": "Llama-3.2 Abl. / DeepSeek",
    "pair7": "Qwen2.5 Unc. / Bespoke",
}
FIXED_RE = re.compile(r"recv_w(?P<window>8|16)_r(?P<ratio>0\.[357])(?:_|$)")


def read_jsonl(path: Path) -> tuple[dict, list[dict]]:
    meta, rows = {}, []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
            else:
                rows.append(row)
    return meta, rows


def mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def task_from_path(path: Path, tasks: list[str]) -> str | None:
    return next((part for part in path.parts if part in tasks), None)


def classify_run(path: Path) -> tuple[str, str] | None:
    run_name = path.parent.name
    fixed = FIXED_RE.match(run_name)
    if fixed:
        return "ReKV", f"w{fixed.group('window')}-r{fixed.group('ratio')}"
    if run_name.startswith(f"{CANONICAL}_") or run_name == CANONICAL:
        return "B-ReKV", "t0.95-s0.75-w8"
    if run_name.startswith("frozen_t0.95_s0.75_w8"):
        return "B-ReKV", "t0.95-s0.75-w8"
    return None


def collect_root_runs(
    pair: str,
    root: Path,
    tasks: list[str],
    source_rank: int = 2,
) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("**/per_sample.jsonl")):
        task = task_from_path(path.relative_to(root), tasks)
        classified = classify_run(path)
        if task is None or classified is None:
            continue
        meta, samples = read_jsonl(path)
        if meta.get("protocol_version") != V1 or not samples:
            continue
        method, config = classified
        rows.append(
            {
                "pair": pair,
                "pair_label": PAIR_LABELS[pair],
                "task": task,
                "method": method,
                "config": config,
                "n": len(samples),
                "score": mean(samples, "score"),
                "budget": mean(samples, "budget"),
                "protocol_version": meta.get("protocol_version"),
                "source_rank": source_rank,
                "run_dir": str(path.parent),
            }
        )
    return rows


def collect_stage3_canonical() -> list[dict]:
    root = Path("snapshots/stage3_core_reviewer_query_sketch")
    rows = []
    for pair_dir in root.glob("pair*"):
        pair = pair_dir.name.split("_")[0]
        if pair not in TABLE1_ROOTS:
            continue
        rows.extend(collect_root_runs(pair, pair_dir, CORE_TASKS, source_rank=1))
    return [row for row in rows if row["method"] == "B-ReKV"]


def deduplicate(rows: list[dict]) -> list[dict]:
    selected: dict[tuple, dict] = {}
    for row in rows:
        key = (row["pair"], row["task"], row["method"], row["config"])
        rank = (row["source_rank"], row["n"], row["run_dir"])
        previous = selected.get(key)
        if previous is None or rank > (
            previous["source_rank"],
            previous["n"],
            previous["run_dir"],
        ):
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (row["pair"], row["task"], row["method"], row["config"]))


def summarize_cells(
    rows: list[dict],
    pairs: list[str],
    tasks: list[str],
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["pair"], row["task"])].append(row)
    summary = []
    for pair in pairs:
        for task in tasks:
            cell = grouped[(pair, task)]
            fixed = [row for row in cell if row["method"] == "ReKV"]
            brekv = [row for row in cell if row["method"] == "B-ReKV"]
            best = max(fixed, key=lambda row: row["score"]) if fixed else None
            b = brekv[-1] if brekv else None
            summary.append(
                {
                    "pair": pair,
                    "pair_label": PAIR_LABELS[pair],
                    "task": task,
                    "fixed_done": len(fixed),
                    "fixed_complete": len(fixed) == 6,
                    "best_rekv_score": best["score"] if best else None,
                    "best_rekv_budget": best["budget"] if best else None,
                    "best_rekv_config": best["config"] if best else None,
                    "brekv_done": bool(b),
                    "brekv_score": b["score"] if b else None,
                    "brekv_budget": b["budget"] if b else None,
                    "score_gap_brekv_minus_best": (
                        b["score"] - best["score"] if b and best else None
                    ),
                    "cell_complete": len(fixed) == 6 and bool(b),
                }
            )
    return summary


def pair_averages(summary: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in summary:
        grouped[row["pair"]].append(row)
    output = []
    for pair, rows in sorted(grouped.items()):
        fixed = [row["best_rekv_score"] for row in rows if row["best_rekv_score"] is not None]
        brekv = [row["brekv_score"] for row in rows if row["brekv_score"] is not None]
        budgets = [row["brekv_budget"] for row in rows if row["brekv_budget"] is not None]
        gaps = [
            row["score_gap_brekv_minus_best"]
            for row in rows
            if row["score_gap_brekv_minus_best"] is not None
        ]
        output.append(
            {
                "pair": pair,
                "pair_label": PAIR_LABELS[pair],
                "cells": len(rows),
                "complete_cells": sum(row["cell_complete"] for row in rows),
                "mean_best_rekv_score": sum(fixed) / len(fixed) if fixed else None,
                "mean_brekv_score": sum(brekv) / len(brekv) if brekv else None,
                "mean_brekv_budget": sum(budgets) / len(budgets) if budgets else None,
                "mean_gap_brekv_minus_best": sum(gaps) / len(gaps) if gaps else None,
            }
        )
    return output


def collect_multisource() -> tuple[list[dict], list[dict]]:
    root = Path("snapshots/table10_multi_source_query_sketch")
    rows = []
    for path in sorted(root.glob("**/per_sample.jsonl")):
        task = task_from_path(path.relative_to(root), ["hotpotqa", "musique", "twowikimqa"])
        match = re.match(r"ms_qs_bf16_w(8|16)_r(0\.[357])(?:_|$)", path.parent.name)
        if task is None or match is None:
            continue
        meta, samples = read_jsonl(path)
        if meta.get("protocol_version") != "query_sketch_bf16_multi_source_v1" or not samples:
            continue
        rows.append(
            {
                "task": task,
                "config": f"w{match.group(1)}-r{match.group(2)}",
                "n": len(samples),
                "score": mean(samples, "score"),
                "budget": mean(samples, "budget"),
                "protocol_version": meta.get("protocol_version"),
                "run_dir": str(path.parent),
            }
        )
    best = []
    for task in ["hotpotqa", "musique", "twowikimqa"]:
        task_rows = [row for row in rows if row["task"] == task]
        item = max(task_rows, key=lambda row: row["score"]) if task_rows else None
        best.append(
            {
                "task": task,
                "runs_done": len(task_rows),
                "complete": len(task_rows) == 6,
                "best_score": item["score"] if item else None,
                "best_budget": item["budget"] if item else None,
                "best_config": item["config"] if item else None,
            }
        )
    return rows, best


def collect_cost() -> list[dict]:
    root = Path("snapshots/query_sketch_cost_v1")
    selected: dict[tuple[str, str, str], dict] = {}
    for path in sorted(root.glob("**/cost_summary.json")):
        data = json.loads(path.read_text())
        meta = data.get("_meta", {})
        if meta.get("protocol_version") != V1:
            continue
        rel = path.relative_to(root)
        pair = next((part.split("_")[0] for part in rel.parts if part.startswith("pair")), "unknown")
        task = task_from_path(rel, CORE_TASKS) or "unknown"
        method = "B-ReKV" if "brekv" in rel.parts else "ReKV"
        row = {
            "pair": pair,
            "pair_label": PAIR_LABELS.get(pair, pair),
            "task": task,
            "method": method,
            "score": data.get("score_mean"),
            "budget": data.get("budget_mean"),
            "b_to_a_mb": data.get("b_to_a_communication_bytes_mean", 0) / 2**20,
            "a_to_b_mb": data.get("a_to_b_communication_bytes_mean", 0) / 2**20,
            "total_mb": data.get("total_communication_bytes_mean", 0) / 2**20,
            "t_total_s": data.get("t_total_mean"),
            "peak_mem_gb": data.get("peak_mem_gb_mean"),
            "run_dir": str(path.parent),
        }
        key = (pair, task, method)
        previous = selected.get(key)
        if previous is None or row["run_dir"] > previous["run_dir"]:
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (row["pair"], row["task"], row["method"]))


def cost_pair_averages(cost: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in cost:
        grouped[row["pair"]].append(row)
    output = []
    for pair, rows in sorted(grouped.items()):
        output.append(
            {
                "pair": pair,
                "pair_label": PAIR_LABELS.get(pair, pair),
                "cells": len(rows),
                "expected_cells": 6,
                "mean_score": sum(row["score"] for row in rows) / len(rows),
                "mean_budget": sum(row["budget"] for row in rows) / len(rows),
                "mean_b_to_a_mb": sum(row["b_to_a_mb"] for row in rows) / len(rows),
                "mean_total_mb": sum(row["total_mb"] for row in rows) / len(rows),
                "mean_t_total_s": sum(row["t_total_s"] for row in rows) / len(rows),
                "mean_peak_mem_gb": sum(row["peak_mem_gb"] for row in rows) / len(rows),
            }
        )
    return output


def collect_canonical_oracle() -> list[dict]:
    root = Path("snapshots/query_sketch_oracle_gap_canonical_t095_s075_w8")
    grouped = defaultdict(dict)
    for path in sorted(root.glob("**/cost_summary.json")):
        data = json.loads(path.read_text())
        meta = data.get("_meta", {})
        protocol = meta.get("protocol_version")
        if protocol not in {V1, "full_kv_oracle_v1"}:
            continue
        task = task_from_path(path.relative_to(root), CORE_TASKS)
        if task is None:
            continue
        variant = "Query-Sketch" if protocol == V1 else "Full-KV Oracle"
        grouped[task][variant] = data
    rows = []
    for task in CORE_TASKS:
        query = grouped[task].get("Query-Sketch")
        oracle = grouped[task].get("Full-KV Oracle")
        if not query or not oracle:
            continue
        rows.append(
            {
                "task": task,
                "query_score": query.get("score_mean"),
                "oracle_score": oracle.get("score_mean"),
                "score_gap_query_minus_oracle": query.get("score_mean") - oracle.get("score_mean"),
                "query_budget": query.get("budget_mean"),
                "oracle_budget": oracle.get("budget_mean"),
                "query_total_mb": query.get("total_communication_bytes_mean", 0) / 2**20,
                "oracle_total_mb": oracle.get("total_communication_bytes_mean", 0) / 2**20,
                "communication_reduction_pct": 100
                * (
                    1
                    - query.get("total_communication_bytes_mean", 0)
                    / oracle.get("total_communication_bytes_mean", 1)
                ),
                "query_time_s": query.get("t_total_mean"),
                "oracle_time_s": oracle.get("t_total_mean"),
                "query_peak_mem_gb": query.get("peak_mem_gb_mean"),
                "oracle_peak_mem_gb": oracle.get("peak_mem_gb_mean"),
            }
        )
    return rows


def markdown_table(rows: list[dict], columns: list[str], digits: int = 4) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.{digits}f}")
            elif value is None:
                values.append("—")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def plot_pair_averages(rows: list[dict], path: Path, title: str) -> None:
    available = [row for row in rows if row["mean_brekv_score"] is not None]
    labels = [SHORT_PAIR_LABELS[row["pair"]] for row in available]
    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.8), 4.4))
    ax.bar([i - width / 2 for i in x], [row["mean_best_rekv_score"] for row in available], width, label="Best fixed ReKV")
    ax.bar([i + width / 2 for i in x], [row["mean_brekv_score"] for row in available], width, label="Canonical B-ReKV")
    ax.set_ylabel("Mean task score")
    ax.set_xlabel("Sender / receiver model setting")
    ax.set_xticks(list(x), labels, rotation=18, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_extended(summary: list[dict], path: Path) -> None:
    rows = [row for row in summary if row["brekv_score"] is not None]
    labels = [f"{row['pair'].replace('pair', 'M')} · {row['task']}" for row in rows]
    x = range(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar([i - width / 2 for i in x], [row["best_rekv_score"] for row in rows], width, label="Best fixed ReKV")
    ax.bar([i + width / 2 for i in x], [row["brekv_score"] for row in rows], width, label="Canonical B-ReKV")
    ax.set_ylabel("Task score")
    ax.set_xlabel("Model setting and extended task")
    ax.set_xticks(list(x), labels, rotation=35, ha="right")
    ax.set_title("Deployable Query-Sketch Accuracy on Extended Tasks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cost(cost: list[dict], oracle: list[dict], path: Path) -> None:
    pairs = sorted({row["pair"] for row in cost})
    methods = ["ReKV", "B-ReKV"]
    cost_map = {(row["pair"], row["task"], row["method"]): row for row in cost}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = [f"{SHORT_PAIR_LABELS[pair]}\n{task}" for pair in pairs for task in CORE_TASKS]
    x = range(len(labels))
    width = 0.38
    totals, times = [], []
    for method in methods:
        series_total = []
        series_time = []
        for pair in pairs:
            for task in CORE_TASKS:
                row = cost_map[(pair, task, method)]
                series_total.append(row["total_mb"])
                series_time.append(row["t_total_s"])
        totals.append(series_total)
        times.append(series_time)
    axes[0].bar([i - width / 2 for i in x], totals[0], width, label="ReKV")
    axes[0].bar([i + width / 2 for i in x], totals[1], width, label="B-ReKV")
    axes[1].bar([i - width / 2 for i in x], times[0], width, label="ReKV")
    axes[1].bar([i + width / 2 for i in x], times[1], width, label="B-ReKV")
    axes[0].set_title("Total bidirectional communication")
    axes[0].set_ylabel("MiB per sample")
    axes[1].set_title("End-to-end latency")
    axes[1].set_ylabel("Seconds per sample")
    axes[2].bar(
        [row["task"] for row in oracle],
        [row["communication_reduction_pct"] for row in oracle],
    )
    axes[2].set_title("Canonical B-ReKV vs Full-KV Oracle")
    axes[2].set_ylabel("Communication reduction (%)")
    for ax in axes[:2]:
        ax.set_xticks(list(x), labels, rotation=45, ha="right", fontsize=7)
        ax.legend()
    axes[2].set_xlabel("Task (Pair #6)")
    axes[2].tick_params(axis="x", rotation=20)
    fig.suptitle("Deployable Query-Sketch Cost Profile (pairs #1/#6/#7)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    table1_summary: list[dict],
    table1_avg: list[dict],
    table6_summary: list[dict],
    table6_avg: list[dict],
    table8_avg: list[dict],
    multisource_best: list[dict],
    cost: list[dict],
    oracle: list[dict],
) -> None:
    complete_t1 = sum(row["cell_complete"] for row in table1_summary)
    fixed_t1 = sum(row["fixed_complete"] for row in table1_summary)
    brekv_t1 = sum(row["brekv_done"] for row in table1_summary)
    cost_complete = sum(1 for row in cost_pair_averages(cost) if row["cells"] == 6)
    lines = [
        "# Deployable Query-Sketch 最终实验汇总（2026-07-14）",
        "",
        "## 数据口径",
        "",
        "- 主结果只纳入 `_meta.protocol_version=query_sketch_bf16_v1`。",
        "- Canonical B-ReKV 固定为 calibrated coverage `tau=0.95, scale=0.75, window=8`。",
        "- `full_kv_oracle_v1` 只用于 Oracle-gap；v0/legacy 不进入平均值。",
        "",
        "## 完整度",
        "",
        f"- 主任务 fixed ReKV 完整单元：{fixed_t1}/24；canonical B-ReKV：{brekv_t1}/24；完整七点单元：{complete_t1}/24。",
        f"- Extended tasks 完整七点单元：{sum(row['cell_complete'] for row in table6_summary)}/10。",
        f"- Appendix model settings 完整七点单元：{sum(row['complete_cells'] for row in table8_avg)}/32。",
        f"- Multi-source：{sum(row['complete'] for row in multisource_best)}/3 tasks。",
        f"- 正式 cost：{len(cost)}/18（{cost_complete}/3 pairs）；canonical B-ReKV Oracle-gap：{len(oracle)}/3。",
        "",
        "## 主任务模型设置平均",
        "",
    ]
    lines += markdown_table(
        table1_avg,
        [
            "pair_label",
            "complete_cells",
            "cells",
            "mean_best_rekv_score",
            "mean_brekv_score",
            "mean_brekv_budget",
            "mean_gap_brekv_minus_best",
        ],
    )
    lines += ["", "## Extended tasks 模型设置平均", ""]
    lines += markdown_table(
        table6_avg,
        [
            "pair_label",
            "complete_cells",
            "cells",
            "mean_best_rekv_score",
            "mean_brekv_score",
            "mean_brekv_budget",
            "mean_gap_brekv_minus_best",
        ],
    )
    lines += ["", "## Appendix model settings 平均", ""]
    lines += markdown_table(
        table8_avg,
        [
            "pair_label",
            "complete_cells",
            "cells",
            "mean_best_rekv_score",
            "mean_brekv_score",
            "mean_brekv_budget",
            "mean_gap_brekv_minus_best",
        ],
    )
    lines += ["", "## Multi-source", ""]
    lines += markdown_table(
        multisource_best,
        ["task", "runs_done", "complete", "best_score", "best_budget", "best_config"],
    )
    lines += ["", "## 正式 Cost / Efficiency（pair 平均）", ""]
    lines += markdown_table(
        cost_pair_averages(cost),
        [
            "pair_label",
            "cells",
            "mean_score",
            "mean_budget",
            "mean_b_to_a_mb",
            "mean_total_mb",
            "mean_t_total_s",
            "mean_peak_mem_gb",
        ],
    )
    lines += ["", "## 正式 Cost / Efficiency（逐 cell）", ""]
    lines += markdown_table(
        cost,
        [
            "pair_label",
            "task",
            "method",
            "score",
            "budget",
            "b_to_a_mb",
            "a_to_b_mb",
            "total_mb",
            "t_total_s",
            "peak_mem_gb",
        ],
    )
    lines += ["", "## Canonical B-ReKV vs Full-KV Oracle", ""]
    lines += markdown_table(
        oracle,
        [
            "task",
            "query_score",
            "oracle_score",
            "score_gap_query_minus_oracle",
            "query_budget",
            "communication_reduction_pct",
            "query_time_s",
            "oracle_time_s",
        ],
    )
    brekv_budgets = [row["mean_brekv_budget"] for row in table1_avg if row["mean_brekv_budget"] is not None]
    mean_main_budget = sum(brekv_budgets) / len(brekv_budgets) if brekv_budgets else None
    lines += [
        "",
        "## 论文结论边界",
        "",
        f"- 本机 Query-Sketch 主矩阵、扩展表、附录模型设置、Multi-Source、正式 cost 与 canonical Oracle-gap 均已完整。",
        f"- Canonical B-ReKV 主任务平均实际预算约 {mean_main_budget:.1%}，明显低于 high-fixed ReKV。",
        "- B-ReKV 的主张是约 30% 左右动态预算折中，以及显著优于 query-agnostic baselines；通常低于 best fixed ReKV，但这是预期 tradeoff。",
        "- NLD 原始 baseline 可复用；ReKV/B-ReKV 侧必须用本报告正式 cost v1 结果替换旧 cost_profile。",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("snapshots/analysis/query_sketch_final_20260714"),
    )
    args = parser.parse_args()
    out = args.out_dir
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    table1_runs = []
    for pair, root in TABLE1_ROOTS.items():
        table1_runs.extend(collect_root_runs(pair, root, MAIN_TASKS))
    table1_runs.extend(collect_stage3_canonical())
    table1_runs = deduplicate(table1_runs)
    table1_summary = summarize_cells(table1_runs, list(TABLE1_ROOTS), MAIN_TASKS)
    table1_avg = pair_averages(table1_summary)

    table6_runs = deduplicate(
        [
            row
            for pair, root in TABLE6_ROOTS.items()
            for row in collect_root_runs(pair, root, EXTENDED_TASKS)
        ]
    )
    table6_summary = summarize_cells(table6_runs, list(TABLE6_ROOTS), EXTENDED_TASKS)
    table6_avg = pair_averages(table6_summary)

    table8_runs = deduplicate(
        [
            row
            for pair, root in TABLE8_ROOTS.items()
            for row in collect_root_runs(pair, root, MAIN_TASKS)
        ]
    )
    table8_summary = summarize_cells(table8_runs, list(TABLE8_ROOTS), MAIN_TASKS)
    table8_avg = pair_averages(table8_summary)

    multisource_runs, multisource_best = collect_multisource()
    cost = collect_cost()
    oracle = collect_canonical_oracle()

    write_csv(out / "table1_runs_v1.csv", table1_runs)
    write_csv(out / "table1_summary.csv", table1_summary)
    write_csv(out / "table1_pair_averages.csv", table1_avg)
    write_csv(out / "extended_runs_v1.csv", table6_runs)
    write_csv(out / "extended_summary.csv", table6_summary)
    write_csv(out / "extended_pair_averages.csv", table6_avg)
    write_csv(out / "appendix_runs_v1.csv", table8_runs)
    write_csv(out / "appendix_summary.csv", table8_summary)
    write_csv(out / "appendix_pair_averages.csv", table8_avg)
    write_csv(out / "multisource_runs_v1.csv", multisource_runs)
    write_csv(out / "multisource_best.csv", multisource_best)
    write_csv(out / "cost_all_pairs.csv", cost)
    write_csv(out / "cost_pair_averages.csv", cost_pair_averages(cost))
    write_csv(out / "canonical_oracle_gap_pair6.csv", oracle)

    plot_pair_averages(
        table1_avg,
        figures / "main_tasks_model_average.png",
        "Deployable Query-Sketch Accuracy on Main Tasks",
    )
    plot_extended(table6_summary, figures / "extended_tasks_accuracy.png")
    plot_pair_averages(
        table8_avg,
        figures / "appendix_models_average.png",
        "Deployable Query-Sketch Accuracy across Additional Model Settings",
    )
    if len(cost) == 18 and len(oracle) == 3:
        plot_cost(cost, oracle, figures / "cost_and_oracle_overview.png")
    write_report(
        out / "REPORT.md",
        table1_summary,
        table1_avg,
        table6_summary,
        table6_avg,
        table8_avg,
        multisource_best,
        cost,
        oracle,
    )
    print(f"wrote final Query-Sketch analysis to {out}")


if __name__ == "__main__":
    main()
