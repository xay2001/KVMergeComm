#!/usr/bin/env python3
"""Summarize the latest ReKV/B-ReKV experiment batches.

This covers:
  - score-function ablation
  - receiver-layer aggregation ablation
  - Table 6 extended-task runs

The script reads existing per_sample.jsonl files, writes compact CSV summaries,
and creates lightweight overview figures for documentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


MAIN_TASKS = {
    "countries",
    "tipsheets",
    "hotpotqa",
    "qasper",
    "musique",
    "multifieldqa_en",
    "twowikimqa",
    "tmath",
}
TABLE6_TASKS = {"hotpotqa_full", "qasper_full", "musique_full", "samsum", "repobench"}
EXPECTED_TABLE6_METHODS = 9


def read_per_sample(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    rows: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if i == 0 and "_meta" in obj:
                meta = obj["_meta"]
                continue
            rows.append(obj)
    return meta, rows


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def infer_pair(path: Path) -> str:
    for part in path.parts:
        if part.startswith("pair") or part.startswith("table6_pair"):
            return part
    return "unknown"


def infer_task(path: Path) -> str:
    for part in path.parts:
        if part in MAIN_TASKS or part in TABLE6_TASKS:
            return part
    return "unknown"


def infer_method_from_run(run_dir: Path, meta: dict) -> tuple[str, str, str, str]:
    """Return method, method_family, window, ratio_or_scale."""
    name = run_dir.name
    parent = run_dir.parent.name
    score_mode = meta.get("score_mode") or ""
    budget_mode = meta.get("budget_mode") or ""
    window = str(meta.get("recv_window") or "")
    ratio_or_scale = ""

    if m := re.search(r"cov_t([0-9.]+)_s([0-9.]+)_w(\d+)", name):
        method = f"B-ReKV t={m.group(1)} s={m.group(2)} w{m.group(3)}"
        return method, "B-ReKV", m.group(3), m.group(2)
    if budget_mode == "coverage":
        method = f"B-ReKV t={meta.get('coverage_tau')} s={meta.get('coverage_scale')} w{window}"
        return method, "B-ReKV", window, str(meta.get("coverage_scale") or "")

    if m := re.search(r"(receiver(?:_x_value_norm|_recency)?)_w(\d+)_r([0-9.]+)", name):
        method = m.group(1)
        return method, method, m.group(2), m.group(3)
    if m := re.search(r"(value_norm|random)_r([0-9.]+)", name):
        method = m.group(1)
        return method, method, "", m.group(2)
    if m := re.search(r"recv_w(\d+)_r([0-9.]+)", name):
        return "ReKV", "ReKV", m.group(1), m.group(2)
    if parent.startswith("agg_"):
        agg = parent.replace("agg_", "")
        return f"agg_{agg}", "layer_agg", window, str(meta.get("merge_ratio") or "")
    if score_mode:
        return score_mode, score_mode, window, str(meta.get("merge_ratio") or "")
    return name, name, window, str(meta.get("merge_ratio") or "")


def summarize_per_sample(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("**/per_sample.jsonl")):
        meta, samples = read_per_sample(path)
        if not samples:
            continue
        run_dir = path.parent
        method, family, window, ratio = infer_method_from_run(run_dir, meta)
        scores = [float(r["score"]) for r in samples if r.get("score") is not None]
        budgets = [float(r["budget"]) for r in samples if r.get("budget") is not None]
        query_budgets = [float(r["query_budget"]) for r in samples if r.get("query_budget") is not None]
        rows.append(
            {
                "pair": infer_pair(path),
                "task": infer_task(path),
                "method": method,
                "method_family": family,
                "window": window,
                "ratio_or_scale": ratio,
                "n": len(samples),
                "score_mean": mean(scores),
                "budget_mean": mean(budgets),
                "query_budget_mean": mean(query_budgets),
                "run_dir": str(run_dir),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def best_rows(rows: list[dict], group_keys: list[str]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        score = row.get("score_mean")
        if score is None:
            continue
        if key not in best or score > best[key]["score_mean"]:
            best[key] = row
    return [best[k] for k in sorted(best)]


def plot_score_function(best: list[dict], out: Path) -> None:
    tasks = ["hotpotqa", "musique", "multifieldqa_en"]
    task_labels = {
        "hotpotqa": "HotpotQA",
        "musique": "MuSiQue",
        "multifieldqa_en": "MultiFieldQA-en",
    }
    methods = ["value_norm", "random", "receiver", "receiver_x_value_norm", "receiver_recency"]
    labels = ["ValueNorm", "Random", "Receiver", "Receiver x VNorm", "Receiver + Recency"]
    pairs = sorted({r["pair"] for r in best})
    setting_labels = {
        "pair1_llama31_same": "S: Llama-3.1-8B; R: Llama-3.1-8B",
        "pair6_llama32_abliterated_deepseek3b": "S: Llama-3.2-3B-Abliterated; R: DeepSeek-R1-3B",
        "pair7_qwen25_uncensored_bespoke": "S: Qwen2.5-7B-Uncensored; R: Bespoke-Stratos-7B",
    }
    fig, axes = plt.subplots(len(pairs), len(tasks), figsize=(13, 3.3 * len(pairs)), squeeze=False)
    for row_i, pair in enumerate(pairs):
        for col_i, task in enumerate(tasks):
            ax = axes[row_i][col_i]
            data = {(r["method_family"]): r["score_mean"] for r in best if r["pair"] == pair and r["task"] == task}
            vals = [data.get(m, 0.0) or 0.0 for m in methods]
            ax.bar(range(len(methods)), vals, color=["#f58518", "#bab0ac", "#4c78a8", "#72b7b2", "#54a24b"])
            ax.set_title(f"{setting_labels.get(pair, 'Model setting')}\n{task_labels.get(task, task)}", fontsize=10)
            ax.set_ylim(0, max(vals + [0.05]) * 1.25)
            ax.set_xticks(range(len(methods)), labels, rotation=35, ha="right")
            if col_i == 0:
                ax.set_ylabel("Score")
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Score Function Ablation", y=1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_layer_aggregation(best: list[dict], out: Path) -> None:
    tasks = ["countries", "tipsheets", "hotpotqa", "qasper", "musique", "multifieldqa_en", "twowikimqa", "tmath"]
    aggs = ["identity", "last", "mean", "top4", "last4"]
    matrix = []
    for agg in aggs:
        row_vals = []
        for task in tasks:
            vals = [r["score_mean"] for r in best if r["task"] == task and r["method"] == f"agg_{agg}"]
            row_vals.append(vals[0] if vals else 0.0)
        matrix.append(row_vals)

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set_yticks(range(len(aggs)), aggs)
    ax.set_xticks(range(len(tasks)), tasks, rotation=35, ha="right")
    ax.set_title("Layer Aggregation Ablation: Best Score by Task")
    for i, row_vals in enumerate(matrix):
        for j, val in enumerate(row_vals):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Score")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def table6_status(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        grouped[(row["pair"], row["task"])] += 1
    status = []
    for pair in sorted({r["pair"] for r in rows}):
        for task in sorted(TABLE6_TASKS):
            count = grouped.get((pair, task), 0)
            status.append(
                {
                    "pair": pair,
                    "task": task,
                    "completed_runs": count,
                    "expected_runs": EXPECTED_TABLE6_METHODS,
                    "status": "complete" if count >= EXPECTED_TABLE6_METHODS else ("partial" if count else "missing"),
                }
            )
    return status


def plot_table6_pair(best: list[dict], pair_key: str, title: str, out: Path) -> None:
    rows = [r for r in best if pair_key in r["pair"]]
    if not rows:
        return
    tasks = ["hotpotqa_full", "qasper_full", "musique_full", "samsum", "repobench"]
    task_labels = ["HotpotQA", "QASPER", "MuSiQue", "SAMSum", "RepoBench"]
    families = ["ReKV", "B-ReKV"]
    x = list(range(len(tasks)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for i, fam in enumerate(families):
        vals = []
        for task in tasks:
            vals.append(max([r["score_mean"] for r in rows if r["task"] == task and r["method_family"] == fam] or [0.0]))
        offs = [v + (i - 0.5) * width for v in x]
        ax.bar(offs, vals, width=width, label=fam)
    present_tasks = {r["task"] for r in rows}
    for i, task in enumerate(tasks):
        if task not in present_tasks:
            ax.text(i, 0.03, "missing\n(OOM)", ha="center", va="bottom", fontsize=9, color="#6b6b6b")
    ax.set_xticks(x, task_labels, rotation=25, ha="right")
    ax.set_ylabel("Best score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, default=Path("snapshots"))
    parser.add_argument("--out-dir", type=Path, default=Path("snapshots/analysis/latest_experiments"))
    args = parser.parse_args()

    out_dir = args.out_dir
    fig_dir = out_dir / "figures"

    score_rows = summarize_per_sample(args.snapshot_root / "score_function_ablation")
    write_csv(score_rows, out_dir / "score_function_summary.csv")
    score_best = best_rows(score_rows, ["pair", "task", "method_family"])
    write_csv(score_best, out_dir / "score_function_best_by_pair_task_method.csv")
    if score_best and plt is not None:
        plot_score_function(score_best, fig_dir / "score_function_ablation_best.png")

    layer_rows = summarize_per_sample(args.snapshot_root / "layer_aggregation_ablation")
    write_csv(layer_rows, out_dir / "layer_aggregation_summary.csv")
    layer_best = best_rows(layer_rows, ["task", "method"])
    write_csv(layer_best, out_dir / "layer_aggregation_best_by_task_method.csv")
    if layer_best and plt is not None:
        plot_layer_aggregation(layer_best, fig_dir / "layer_aggregation_heatmap.png")

    table6_rows = []
    for root in args.snapshot_root.glob("table6_pair*"):
        table6_rows.extend(summarize_per_sample(root))
    write_csv(table6_rows, out_dir / "table6_extended_summary.csv")
    write_csv(table6_status(table6_rows), out_dir / "table6_extended_status.csv")
    table6_best = best_rows(table6_rows, ["pair", "task", "method_family"])
    write_csv(table6_best, out_dir / "table6_extended_best_by_pair_task_family.csv")
    if table6_best and plt is not None:
        plot_table6_pair(
            table6_best,
            "table6_pair6",
            "Extended Tasks\nS: Llama-3.2-3B-Abliterated; R: DeepSeek-R1-3B",
            fig_dir / "table6_pair6_extended_best.png",
        )
        plot_table6_pair(
            table6_best,
            "table6_pair7",
            "Extended Tasks\nS: Qwen2.5-7B-Uncensored; R: Bespoke-Stratos-7B",
            fig_dir / "table6_pair7_extended_best.png",
        )

    print(f"score rows: {len(score_rows)}")
    print(f"layer rows: {len(layer_rows)}")
    print(f"table6 rows: {len(table6_rows)}")
    print(f"outputs: {out_dir}")


if __name__ == "__main__":
    main()
