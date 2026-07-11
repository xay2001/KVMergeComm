#!/usr/bin/env python3
"""Build a compact NLD vs ReKV/B-ReKV cost comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = {"hotpotqa", "musique", "multifieldqa_en"}
TASK_ORDER = ["hotpotqa", "musique", "multifieldqa_en"]
METHOD_ORDER = ["NLD", "ReKV-w8 r=0.3", "B-ReKV"]
PAIR_LABELS = {
    "pair1_llama31_same": "S: Llama-3.1-8B; R: Llama-3.1-8B",
    "pair6_llama32_abliterated_deepseek3b": "S: Llama-3.2-3B-Abliterated; R: DeepSeek-R1-3B",
    "pair7_qwen25_uncensored_bespoke": "S: Qwen2.5-7B-Uncensored; R: Bespoke-Stratos-7B",
}
TASK_LABELS = {
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
    "multifieldqa_en": "MultiFieldQA-en",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def infer_pair(path: Path) -> str:
    for part in path.parts:
        if part.startswith("pair") or part.startswith("table1_pair"):
            pair = part.replace("table1_", "")
            if pair == "pair1_llama31_same_all8_full":
                return "pair1_llama31_same"
            if pair.endswith("_full"):
                pair = pair[:-5]
            return pair
    return "unknown"


def infer_task(path: Path) -> str:
    for part in path.parts:
        if part in TASKS:
            return part
    return "unknown"


def kv_method(run_dir: Path, summary: dict) -> str | None:
    name = run_dir.name
    if m := re.search(r"recv_w(\d+)_r([0-9.]+)", name):
        return f"ReKV-w{m.group(1)} r={m.group(2)}"
    if m := re.search(r"cov_t([0-9.]+)_s([0-9.]+)_w(\d+)", name):
        return f"B-ReKV t={m.group(1)} s={m.group(2)} w{m.group(3)}"
    meta = summary.get("_meta", {})
    if meta.get("score_mode") == "receiver" and meta.get("budget_mode") == "uniform":
        return f"ReKV-w{meta.get('recv_window')} r={meta.get('merge_ratio')}"
    if meta.get("budget_mode") == "coverage":
        return f"B-ReKV t={meta.get('coverage_tau')} s={meta.get('coverage_scale')} w{meta.get('recv_window')}"
    return None


def collect_kv_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("**/cost_summary.json")):
        task = infer_task(path)
        if task not in TASKS:
            continue
        summary = load_json(path)
        method = kv_method(path.parent, summary)
        if method is None:
            continue
        # Keep paper-focused points to avoid a very wide table.
        keep = (
            method in {"ReKV-w8 r=0.3", "ReKV-w16 r=0.3"}
            or method in {"B-ReKV t=0.95 s=0.75 w8", "B-ReKV t=0.95 s=0.85 w8", "B-ReKV t=0.95 s=0.90 w16"}
        )
        if not keep:
            continue
        rows.append(
            {
                "pair": infer_pair(path),
                "pair_label": PAIR_LABELS.get(infer_pair(path), infer_pair(path)),
                "task": task,
                "task_label": TASK_LABELS.get(task, task),
                "method": method,
                "score": summary.get("score_mean"),
                "payload_tokens": summary.get("kv_tokens_sent_mean"),
                "payload_bytes": summary.get("kv_bytes_sent_mean"),
                "payload_mb": round(summary["kv_bytes_sent_mean"] / (1024 ** 2), 4)
                if summary.get("kv_bytes_sent_mean") is not None
                else None,
                "t_total": summary.get("t_total_mean"),
                "peak_mem_gb": summary.get("peak_mem_gb_mean"),
                "compute_tokens_proxy": _sum_present(
                    summary.get("ctx_tokens_A_mean"),
                    summary.get("query_tokens_B_mean"),
                    summary.get("output_tokens_mean"),
                ),
                "run_dir": str(path.parent),
            }
        )
    return rows


def collect_nld_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("**/cost_summary.json")):
        task = infer_task(path)
        if task not in TASKS:
            continue
        summary = load_json(path)
        pair = infer_pair(path)
        rows.append(
            {
                "pair": pair,
                "pair_label": PAIR_LABELS.get(pair, pair),
                "task": task,
                "task_label": TASK_LABELS.get(task, task),
                "method": "NLD",
                "score": summary.get("score_mean"),
                "payload_tokens": summary.get("nld_text_payload_tokens_mean"),
                "payload_bytes": summary.get("nld_text_payload_bytes_mean"),
                "payload_mb": round(summary["nld_text_payload_bytes_mean"] / (1024 ** 2), 6)
                if summary.get("nld_text_payload_bytes_mean") is not None
                else None,
                "t_total": summary.get("t_total_mean"),
                "peak_mem_gb": summary.get("peak_mem_gb_mean"),
                "compute_tokens_proxy": _sum_present(
                    summary.get("ctx_tokens_A_mean"),
                    summary.get("query_tokens_B_mean"),
                    summary.get("nld_answer_tokens_A_mean"),
                    summary.get("nld_answer_tokens_B_initial_mean"),
                    summary.get("nld_refine_input_tokens_mean"),
                    summary.get("output_tokens_mean"),
                ),
                "run_dir": str(path.parent),
            }
        )
    return rows


def _sum_present(*values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals), 6) if vals else None


def method_family(method: str) -> str:
    if method == "NLD":
        return "NLD"
    if method.startswith("B-ReKV"):
        return "B-ReKV"
    if method.startswith("ReKV"):
        return method
    return method


def focused_rows(rows: list[dict]) -> list[dict]:
    """Keep NLD, ReKV-w8 r=0.3, and the best B-ReKV per pair/task."""
    selected = []
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["pair"], row["task"], method_family(row["method"]))].append(row)

    for pair in sorted({r["pair"] for r in rows}):
        for task in TASK_ORDER:
            for method in METHOD_ORDER:
                candidates = grouped.get((pair, task, method), [])
                if method == "ReKV-w8 r=0.3":
                    candidates = [r for r in rows if r["pair"] == pair and r["task"] == task and r["method"] == method]
                if not candidates:
                    continue
                best = max(candidates, key=lambda r: float(r["score"] or 0.0))
                row = dict(best)
                row["method_group"] = method
                selected.append(row)
    return selected


def average_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["pair"], row["method_group"])].append(row)

    out = []
    for (pair, method), items in sorted(grouped.items()):
        avg = {
            "pair": pair,
            "pair_label": PAIR_LABELS.get(pair, pair),
            "method_group": method,
            "n_tasks": len(items),
        }
        for key in ["score", "payload_tokens", "payload_mb", "t_total", "peak_mem_gb", "compute_tokens_proxy"]:
            vals = [float(r[key]) for r in items if r.get(key) not in (None, "")]
            avg[key] = round(sum(vals) / len(vals), 6) if vals else None
        out.append(avg)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(rows: list[dict], out: Path) -> None:
    pairs = [p for p in PAIR_LABELS if any(r["pair"] == p for r in rows)]
    metrics = [
        ("score", "Accuracy / Score", False),
        ("compute_tokens_proxy", "Token Consumption Proxy", True),
        ("t_total", "Total Time per Sample (s)", False),
        ("peak_mem_gb", "Peak Memory (GB)", False),
    ]
    colors = {"NLD": "#bab0ac", "ReKV-w8 r=0.3": "#4c78a8", "B-ReKV": "#f58518"}

    fig, axes = plt.subplots(len(pairs), len(metrics), figsize=(15, 3.4 * len(pairs)), squeeze=False)
    for row_i, pair in enumerate(pairs):
        pair_rows = [r for r in rows if r["pair"] == pair]
        for col_i, (metric, ylabel, logy) in enumerate(metrics):
            ax = axes[row_i][col_i]
            vals = []
            labels = []
            bar_colors = []
            for method in METHOD_ORDER:
                match = [r for r in pair_rows if r["method_group"] == method]
                if not match:
                    continue
                labels.append(method)
                vals.append(float(match[0][metric] or 0.0))
                bar_colors.append(colors.get(method, "#777777"))
            ax.bar(range(len(vals)), vals, color=bar_colors)
            ax.set_xticks(range(len(vals)), labels, rotation=25, ha="right")
            ax.set_ylabel(ylabel)
            if logy:
                ax.set_yscale("log")
            if col_i == 0:
                ax.set_title(PAIR_LABELS.get(pair, pair), fontsize=10)
            else:
                ax.set_title(ylabel, fontsize=10)
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Natural-Language Passing vs KV Communication", y=1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_task_scores(rows: list[dict], out: Path) -> None:
    pairs = [p for p in PAIR_LABELS if any(r["pair"] == p for r in rows)]
    colors = {"NLD": "#bab0ac", "ReKV-w8 r=0.3": "#4c78a8", "B-ReKV": "#f58518"}
    fig, axes = plt.subplots(len(pairs), len(TASK_ORDER), figsize=(13, 3.2 * len(pairs)), squeeze=False)
    for row_i, pair in enumerate(pairs):
        for col_i, task in enumerate(TASK_ORDER):
            ax = axes[row_i][col_i]
            vals, labels, bar_colors = [], [], []
            for method in METHOD_ORDER:
                match = [r for r in rows if r["pair"] == pair and r["task"] == task and r["method_group"] == method]
                if not match:
                    continue
                labels.append(method)
                vals.append(float(match[0]["score"] or 0.0))
                bar_colors.append(colors.get(method, "#777777"))
            ax.bar(range(len(vals)), vals, color=bar_colors)
            ax.set_xticks(range(len(vals)), labels, rotation=25, ha="right")
            ax.set_ylim(0, max(vals + [0.05]) * 1.25)
            ax.set_title(f"{PAIR_LABELS.get(pair, pair)}\n{TASK_LABELS.get(task, task)}", fontsize=9)
            if col_i == 0:
                ax.set_ylabel("Accuracy / Score")
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Accuracy Comparison by Task", y=1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-root", type=Path, default=Path("snapshots/cost_profile"))
    parser.add_argument("--nld-root", type=Path, default=Path("snapshots/nld_cost_profile"))
    parser.add_argument("--csv", type=Path, default=Path("snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_summary.csv"))
    parser.add_argument("--focused-csv", type=Path, default=Path("snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_focused.csv"))
    parser.add_argument("--average-csv", type=Path, default=Path("snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_average_by_pair.csv"))
    parser.add_argument("--figure-dir", type=Path, default=Path("snapshots/analysis/nld_vs_rekv/figures"))
    args = parser.parse_args()

    rows = collect_kv_rows(args.kv_root) + collect_nld_rows(args.nld_root)
    rows.sort(key=lambda r: (r["pair"], r["task"], r["method"]))
    write_csv(rows, args.csv)
    focused = focused_rows(rows)
    write_csv(focused, args.focused_csv)
    averaged = average_rows(focused)
    write_csv(averaged, args.average_csv)
    plot_overview(averaged, args.figure_dir / "nld_vs_rekv_cost_overview.png")
    plot_task_scores(focused, args.figure_dir / "nld_vs_rekv_accuracy_by_task.png")
    print(f"rows: {len(rows)}")
    print(f"output: {args.csv}")
    print(f"focused: {args.focused_csv}")
    print(f"average: {args.average_csv}")
    print(f"figures: {args.figure_dir}")


if __name__ == "__main__":
    main()
