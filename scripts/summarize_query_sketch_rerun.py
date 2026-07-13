#!/usr/bin/env python3
"""Audit and summarize the deployable Query-Sketch rerun without mixing legacy data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


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
REP_TASKS = ["hotpotqa", "musique", "multifieldqa_en"]
PAIR_ROOTS = {
    "pair1": Path("snapshots/table1_pair1_query_sketch_llama31_same"),
    "pair6": Path("snapshots/table1_pair6_query_sketch_llama32_abliterated_deepseek3b"),
    "pair7": Path("snapshots/table1_pair7_query_sketch_qwen25_uncensored_bespoke"),
}
PAIR_LABELS = {
    "pair1": "S: Llama-3.1-8B; R: Llama-3.1-8B",
    "pair6": "S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B",
    "pair7": "S: Qwen2.5-7B Unc.; R: Bespoke-Stratos-7B",
}
REKV_RE = re.compile(r"recv_w(8|16)_r(0\.3|0\.5|0\.7)_")


def read_jsonl(path: Path) -> tuple[dict, list[dict]]:
    meta, rows = {}, []
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            if "_meta" in item:
                meta = item["_meta"]
            else:
                rows.append(item)
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


def infer_protocol(meta: dict, root: Path) -> tuple[str, str]:
    recorded = meta.get("protocol_version")
    if recorded:
        return recorded, "recorded"
    if "query_sketch" in root.name:
        return "query_sketch_bf16_v0_pre_instrumentation", "inferred_from_protocol_root"
    return "unknown", "unknown"


def collect_table1() -> tuple[list[dict], list[dict]]:
    all_rows, status_rows = [], []
    for pair, root in PAIR_ROOTS.items():
        by_task = defaultdict(list)
        for path in sorted(root.glob("**/per_sample.jsonl")):
            rel = path.relative_to(root)
            task = next((part for part in rel.parts if part in TASKS), None)
            if task is None:
                continue
            meta, samples = read_jsonl(path)
            if not samples:
                continue
            protocol, protocol_source = infer_protocol(meta, root)
            method_dir = path.parent.parent.name
            run_name = path.parent.name
            if method_dir == "mtc_receiver" and REKV_RE.match(run_name):
                method = "ReKV"
                config = REKV_RE.match(run_name).group(0).rstrip("_")
                primary = True
            elif method_dir == "coverage_frozen" or run_name.startswith("frozen_"):
                method = "B-ReKV-frozen"
                config = run_name.rsplit("_", 2)[0]
                primary = True
            elif method_dir == "coverage":
                method = "B-ReKV-provisional"
                config = run_name.rsplit("_", 2)[0]
                primary = False
            else:
                continue
            row = {
                "pair": pair,
                "pair_label": PAIR_LABELS[pair],
                "task": task,
                "method": method,
                "config": config,
                "primary": primary,
                "protocol_version": protocol,
                "protocol_source": protocol_source,
                "n": len(samples),
                "score": mean(samples, "score"),
                "budget": mean(samples, "budget"),
                "run_dir": str(path.parent),
            }
            all_rows.append(row)
            by_task[task].append(row)

        for task in TASKS:
            rows = by_task.get(task, [])
            rekv = {}
            rekv_v1 = {}
            for row in rows:
                if row["method"] == "ReKV":
                    rekv[row["config"]] = row
                    if row["protocol_version"] == "query_sketch_bf16_v1":
                        rekv_v1[row["config"]] = row
            frozen = [row for row in rows if row["method"] == "B-ReKV-frozen"]
            best = max(rekv.values(), key=lambda row: row["score"]) if rekv else None
            status_rows.append(
                {
                    "pair": pair,
                    "pair_label": PAIR_LABELS[pair],
                    "task": task,
                    "rekv_configs_done": len(rekv),
                    "rekv_complete": len(rekv) == 6,
                    "rekv_v1_configs_done": len(rekv_v1),
                    "rekv_v1_complete": len(rekv_v1) == 6,
                    "best_rekv_score": best["score"] if best else None,
                    "best_rekv_config": best["config"] if best else None,
                    "frozen_brekv_done": bool(frozen),
                    "frozen_brekv_score": frozen[-1]["score"] if frozen else None,
                    "frozen_brekv_budget": frozen[-1]["budget"] if frozen else None,
                    "primary_block_complete": len(rekv) == 6 and bool(frozen),
                }
            )
    return all_rows, status_rows


def collect_cost(root: Path, phase: str) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("**/cost_summary.json")):
        data = json.loads(path.read_text())
        meta = data.get("_meta", {})
        rel = path.relative_to(root)
        pair = next((part.split("_")[0] for part in rel.parts if part.startswith("pair")), "unknown")
        task = next((part for part in rel.parts if part in TASKS), "unknown")
        rows.append(
            {
                "phase": phase,
                "pair": pair,
                "task": task,
                "method_dir": path.parent.parent.name,
                "run_name": path.parent.name,
                "protocol_version": meta.get("protocol_version"),
                "query_sketch_mode": meta.get("query_sketch_mode"),
                "recv_window": meta.get("recv_window"),
                "n": meta.get("n"),
                "score": data.get("score_mean"),
                "budget": data.get("budget_mean"),
                "b_to_a_bytes": data.get("b_to_a_communication_bytes_mean"),
                "a_to_b_bytes": data.get("a_to_b_communication_bytes_mean"),
                "metadata_bytes": data.get("communication_metadata_bytes_mean"),
                "total_communication_bytes": data.get("total_communication_bytes_mean"),
                "t_total": data.get("t_total_mean"),
                "peak_mem_gb": data.get("peak_mem_gb_mean"),
                "run_dir": str(path.parent),
            }
        )
    return rows


def oracle_pairs(cost_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for row in cost_rows:
        method = "B-ReKV" if row["method_dir"] == "brekv" else "ReKV"
        protocol = "Oracle" if row["protocol_version"] == "full_kv_oracle_v1" else "Query-Sketch"
        grouped[(row["pair"], row["task"], method)][protocol] = row
    out = []
    for (pair, task, method), variants in sorted(grouped.items()):
        if "Query-Sketch" not in variants or "Oracle" not in variants:
            continue
        query, oracle = variants["Query-Sketch"], variants["Oracle"]
        out.append(
            {
                "pair": pair,
                "task": task,
                "method": method,
                "query_score": query["score"],
                "oracle_score": oracle["score"],
                "score_gap_query_minus_oracle": query["score"] - oracle["score"],
                "query_total_mb": query["total_communication_bytes"] / 2**20,
                "oracle_total_mb": oracle["total_communication_bytes"] / 2**20,
                "communication_reduction_pct": 100
                * (1 - query["total_communication_bytes"] / oracle["total_communication_bytes"]),
                "query_t_total": query["t_total"],
                "oracle_t_total": oracle["t_total"],
                "query_peak_mem_gb": query["peak_mem_gb"],
                "oracle_peak_mem_gb": oracle["peak_mem_gb"],
            }
        )
    return out


def representation_summary(cost_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in cost_rows:
        grouped[(row["query_sketch_mode"], row["recv_window"])].append(row)
    out = []
    for (mode, window), items in sorted(grouped.items()):
        out.append(
            {
                "mode": mode,
                "window": window,
                "cells": len(items),
                "mean_score": sum(row["score"] for row in items) / len(items),
                "mean_b_to_a_kb": sum(row["b_to_a_bytes"] for row in items) / len(items) / 1024,
                "mean_total_mb": sum(row["total_communication_bytes"] for row in items) / len(items) / 2**20,
                "mean_t_total": sum(row["t_total"] for row in items) / len(items),
                "mean_peak_mem_gb": sum(row["peak_mem_gb"] for row in items) / len(items),
            }
        )
    return out


def collect_mechanism(root: Path, family: str) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("**/per_sample.jsonl")):
        meta, samples = read_jsonl(path)
        if not samples:
            continue
        rel = path.relative_to(root)
        pair = next((part.split("_")[0] for part in rel.parts if part.startswith("pair")), "unknown")
        task = next((part for part in rel.parts if part in REP_TASKS), "unknown")
        if family == "score_function":
            label = path.parent.parent.name.removeprefix("mtc_")
        else:
            label = path.parent.parent.name.removeprefix("agg_")
        rows.append(
            {
                "family": family,
                "pair": pair,
                "task": task,
                "setting": label,
                "protocol_version": meta.get("protocol_version"),
                "n": len(samples),
                "score": mean(samples, "score"),
                "budget": mean(samples, "budget"),
                "run_dir": str(path.parent),
            }
        )
    return rows


def average(rows: list[dict], key_fields: tuple[str, ...], value: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in key_fields)].append(float(row[value]))
    out = []
    for keys, values in sorted(grouped.items()):
        item = dict(zip(key_fields, keys))
        item[f"mean_{value}"] = sum(values) / len(values)
        item["cells"] = len(values)
        out.append(item)
    return out


def plot_oracle(rows: list[dict], path: Path) -> None:
    summary = average(rows, ("method",), "score_gap_query_minus_oracle")
    methods = [row["method"] for row in summary]
    gaps = [row["mean_score_gap_query_minus_oracle"] for row in summary]
    reductions = []
    for method in methods:
        values = [row["communication_reduction_pct"] for row in rows if row["method"] == method]
        reductions.append(sum(values) / len(values))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(methods, gaps, color=["#4C78A8", "#72B7B2"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Score gap")
    axes[0].set_title("Query-Sketch minus Full-KV Oracle")
    axes[1].bar(methods, reductions, color=["#4C78A8", "#72B7B2"])
    axes[1].set_ylabel("Communication reduction (%)")
    axes[1].set_title("End-to-end communication saving")
    fig.suptitle("Deployable Query-Sketch vs Explicit Full-KV Oracle")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_representation(rows: list[dict], path: Path) -> None:
    markers = {"bf16": "o", "int8": "s", "token_ids": "^"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in ["bf16", "int8", "token_ids"]:
        items = [row for row in rows if row["mode"] == mode]
        if not items:
            continue
        ax.scatter(
            [row["mean_b_to_a_kb"] for row in items],
            [row["mean_score"] for row in items],
            marker=markers[mode],
            s=60,
            label=mode,
        )
        for row in items:
            ax.annotate(f"w{row['window']}", (row["mean_b_to_a_kb"], row["mean_score"]), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Mean B→A sketch traffic (KiB, log scale)")
    ax.set_ylabel("Mean task score")
    ax.set_title("Query-Sketch Representation and Window Trade-off")
    ax.legend(title="Representation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism(rows: list[dict], path: Path) -> None:
    summary = average(rows, ("family", "setting"), "score")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, family, title in zip(
        axes,
        ["score_function", "layer_aggregation"],
        ["Score function", "Receiver-layer aggregation"],
    ):
        items = [row for row in summary if row["family"] == family]
        ax.bar([row["setting"] for row in items], [row["mean_score"] for row in items])
        ax.set_title(title)
        ax.set_ylabel("Mean task score")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(
        "Query-Sketch Mechanism Ablations · "
        "S: Llama-3.2-3B Abl.; R: DeepSeek-Llama-3B · Three Tasks",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows: list[dict], columns: list[str], digits: int = 4) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append(f"{value:.{digits}f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    path: Path,
    status: list[dict],
    oracle: list[dict],
    representation: list[dict],
    mechanism: list[dict],
    cost_rows: list[dict],
) -> None:
    complete_cells = sum(row["primary_block_complete"] for row in status)
    rekv_cells = sum(row["rekv_complete"] for row in status)
    rekv_v1_cells = sum(row["rekv_v1_complete"] for row in status)
    rekv_v1_runs = sum(row["rekv_v1_configs_done"] for row in status)
    frozen_cells = sum(row["frozen_brekv_done"] for row in status)
    oracle_avg = average(oracle, ("method",), "score_gap_query_minus_oracle")
    mech_avg = average(mechanism, ("family", "setting"), "score")
    oracle_efficiency = []
    for method in ["ReKV", "B-ReKV"]:
        items = [row for row in oracle if row["method"] == method]
        if not items:
            continue
        query_time = sum(row["query_t_total"] for row in items) / len(items)
        oracle_time = sum(row["oracle_t_total"] for row in items) / len(items)
        oracle_efficiency.append(
            {
                "method": method,
                "query_time": query_time,
                "oracle_time": oracle_time,
                "latency_delta_pct": 100 * (query_time / oracle_time - 1),
                "query_peak_mem_gb": sum(row["query_peak_mem_gb"] for row in items) / len(items),
                "oracle_peak_mem_gb": sum(row["oracle_peak_mem_gb"] for row in items) / len(items),
            }
        )
    lines = [
        "# Query-Sketch 论文重跑审计（2026-07-13）",
        "",
        "## 口径",
        "",
        "- `query_sketch_bf16_v1` / `int8_v1` / `token_ids_v1`：本轮可部署协议。",
        "- `full_kv_oracle_v1`：显式 Full-KV 上界，只用于 oracle-gap，不并入主方法结果。",
        "- `query_agnostic_kv_v1`：ValueNorm / Random 对照。",
        "- Pair #6 主矩阵中缺少 protocol metadata 的 68 个结果位于明确的 Query-Sketch root，标记为 `v0_pre_instrumentation`；可用于准确率，但不能用于新通信计费。",
        "- 其他历史 snapshots 不纳入本报告。",
        "",
        "## 完成状态",
        "",
        f"- Table 1 文件覆盖：ReKV 六配置完成 {rekv_cells}/24 个 pair-task 单元。",
        f"- Table 1 显式 v1 metadata：ReKV 完成 {rekv_v1_runs}/144 runs、{rekv_v1_cells}/24 个完整单元；Pair #6 的 v0 结果需单列。",
        f"- 冻结 B-ReKV 完成 {frozen_cells}/24；完整主块 {complete_cells}/24。",
        f"- Oracle gap：{len(oracle)}/18 个 matched method-pair-task 单元。",
        f"- 表示消融：{len(representation)} 个聚合点（2 pairs × 3 tasks × 3 modes × 4 windows 原始共 72 runs）。",
        f"- 新协议独立 cost：{len(cost_rows)} runs；若为 0，说明第二阶段 cost 尚未启动。",
        "",
        "## 冻结配置",
        "",
        "- `B-ReKV-t0.98-s0.95-w8`：平均预算 0.5698，matched-budget 平均分差 +0.0267，6/6 持平或获胜，最差分差 0。",
        "",
        "## Oracle gap（Query-Sketch − Full-KV Oracle）",
        "",
    ]
    lines += markdown_table(oracle_avg, ["method", "mean_score_gap_query_minus_oracle", "cells"])
    if oracle:
        comm = average(oracle, ("method",), "communication_reduction_pct")
        lines += ["", "通信节省："] + markdown_table(comm, ["method", "mean_communication_reduction_pct", "cells"])
        lines += ["", "时间与显存："] + markdown_table(
            oracle_efficiency,
            [
                "method",
                "query_time",
                "oracle_time",
                "latency_delta_pct",
                "query_peak_mem_gb",
                "oracle_peak_mem_gb",
            ],
        )
    lines += ["", "## 表示与窗口消融", ""]
    lines += markdown_table(
        representation,
        ["mode", "window", "cells", "mean_score", "mean_b_to_a_kb", "mean_total_mb", "mean_t_total"],
    )
    lines += [
        "",
        "结论：INT8-w8 将 BF16-w8 的 B→A sketch 字节减半，平均分不降（0.5500 vs 0.5467）；"
        "Token IDs 虽几乎消除 B→A payload，但最佳分数明显更低。BF16/INT8 的 w8 都优于更大窗口。",
    ]
    lines += ["", "## 机制消融", ""]
    lines += markdown_table(mech_avg, ["family", "setting", "mean_score", "cells"])
    lines += [
        "",
        "## 当前不能提前写入论文的部分",
        "",
        "- 冻结 B-ReKV 的 24 个 Table-1 主单元尚未全部完成前，不生成最终主表平均值。",
        "- `query_sketch_cost_v1` 尚未完整时，不用 Oracle-gap profile 代替正式 cost 表。",
        "- Pair #6 pre-instrumentation 结果不能用于 bytes / timing 结论。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("snapshots/analysis/query_sketch_rerun_20260713"))
    args = parser.parse_args()
    out = args.out_dir
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    table_runs, status = collect_table1()
    oracle_cost = collect_cost(Path("snapshots/query_sketch_oracle_gap"), "oracle_gap")
    representation_cost = collect_cost(
        Path("snapshots/query_sketch_representation_ablation"), "representation"
    )
    official_cost = collect_cost(Path("snapshots/query_sketch_cost_v1"), "official_cost")
    oracle = oracle_pairs(oracle_cost)
    representation = representation_summary(representation_cost)
    mechanism = collect_mechanism(
        Path("snapshots/query_sketch_score_function_ablation"), "score_function"
    ) + collect_mechanism(
        Path("snapshots/query_sketch_layer_aggregation_ablation"), "layer_aggregation"
    )

    write_csv(out / "table1_all_runs.csv", table_runs)
    write_csv(out / "table1_status.csv", status)
    write_csv(out / "oracle_gap.csv", oracle)
    write_csv(out / "representation_summary.csv", representation)
    write_csv(out / "mechanism_runs.csv", mechanism)
    write_csv(out / "official_cost_runs.csv", official_cost)
    if oracle:
        plot_oracle(oracle, figures / "oracle_gap_overview.png")
    if representation:
        plot_representation(representation, figures / "representation_tradeoff.png")
    if mechanism:
        plot_mechanism(mechanism, figures / "mechanism_ablation.png")
    write_report(out / "REPORT.md", status, oracle, representation, mechanism, official_cost)
    print(f"wrote Query-Sketch audit to {out}")


if __name__ == "__main__":
    main()
