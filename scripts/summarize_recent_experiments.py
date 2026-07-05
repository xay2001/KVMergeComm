#!/usr/bin/env python3
"""Summarize recent ReKV/B-ReKV experiment batches.

Outputs:
  - pair #6/#7 paper-focused cost subsets
  - pair #6/#7 B-ReKV robustness CSV + Pareto plots
  - pair #9 score-distribution diagnostics
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

FOCUS_COST_TASKS = {"hotpotqa", "musique", "multifieldqa_en"}
FOCUS_COST_METHODS = {
    "ReKV-w8 r=0.3",
    "ReKV-w16 r=0.3",
    "B-ReKV t=0.95 s=0.75 w8",
    "B-ReKV t=0.95 s=0.85 w8",
}

PAIR_ROOTS = {
    "pair6": Path("snapshots/table1_pair6_llama32_abliterated_deepseek3b"),
    "pair7": Path("snapshots/table1_pair7_qwen25_uncensored_bespoke"),
}

PAIR_COST_TABLES = {
    "pair6": Path("snapshots/cost_profile/table1_pair6_llama32_abliterated_deepseek3b_full/cost_table.csv"),
    "pair7": Path("snapshots/cost_profile/table1_pair7_qwen25_uncensored_bespoke_full/cost_table.csv"),
}

PAIR9_ROOT = Path("snapshots/table8_pair9_supernova_deepseek_llama8b")
OUT_ROOT = Path("snapshots/analysis")

RECV_RE = re.compile(r"recv_w(\d+)_r([0-9.]+)")
COV_RE = re.compile(r"cov_t([0-9.]+)_s([0-9.]+)_w(\d+)")


def read_per_sample(path: Path) -> tuple[dict, list[dict]]:
    meta = {}
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
            else:
                rows.append(row)
    return meta, rows


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No rows.\n")
        return
    cols = list(rows[0])
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def summarize_cost_subset() -> list[Path]:
    written = []
    combined = []
    for pair, path in PAIR_COST_TABLES.items():
        with path.open() as f:
            rows = list(csv.DictReader(f))
        selected = []
        for row in rows:
            if row["dataset"] not in FOCUS_COST_TASKS:
                continue
            if row["method"] not in FOCUS_COST_METHODS:
                continue
            out = {
                "pair": pair,
                "dataset": row["dataset"],
                "method": row["method"],
                "score": round(float(row["score"]), 4),
                "budget": round(float(row["budget"]), 4),
                "kv_mb_sent": round(float(row["kv_mb_sent"]), 3),
                "query_tokens_B": round(float(row["query_tokens_B"]), 2),
                "output_tokens": round(float(row["output_tokens"]), 2),
                "t_receiver_score": round(float(row["t_receiver_score"]), 4),
                "t_generate_total": round(float(row["t_generate_total"]), 4),
                "t_total": round(float(row["t_total"]), 4),
                "peak_mem_gb": round(float(row["peak_mem_gb"]), 3),
            }
            selected.append(out)
            combined.append(out)
        csv_path = OUT_ROOT / "cost" / f"{pair}_cost_focus_hotpotqa_musique_multifieldqa.csv"
        md_path = csv_path.with_suffix(".md")
        write_csv(selected, csv_path)
        write_markdown_table(selected, md_path)
        written.extend([csv_path, md_path])
    combined_path = OUT_ROOT / "cost" / "pair6_pair7_cost_focus_hotpotqa_musique_multifieldqa.csv"
    write_csv(combined, combined_path)
    write_markdown_table(combined, combined_path.with_suffix(".md"))
    written.extend([combined_path, combined_path.with_suffix(".md")])
    return written


def infer_dataset(path: Path) -> str:
    for part in path.parts:
        if part in MAIN_TASKS or part.endswith("_full") or part in {"samsum", "repobench"}:
            return part
    return "unknown"


def latest_by_key(paths: list[Path], key_fn) -> dict:
    latest = {}
    for path in paths:
        key = key_fn(path)
        if key is None:
            continue
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path
    return latest


def fixed_key(path: Path):
    m = RECV_RE.search(path.parent.name)
    if not m:
        return None
    return (int(m.group(1)), float(m.group(2)))


def cov_key(path: Path):
    m = COV_RE.search(path.parent.name)
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)), int(m.group(3)))


def summarize_robustness() -> list[Path]:
    written = []
    all_rows = []
    for pair, root in PAIR_ROOTS.items():
        for task in ["hotpotqa", "musique"]:
            fixed_paths = list((root / task / "mtc_receiver").glob("*/per_sample.jsonl"))
            cov_paths = list((root / task / "coverage").glob("*/per_sample.jsonl"))
            fixed = latest_by_key(fixed_paths, fixed_key)
            coverage = latest_by_key(cov_paths, cov_key)

            fixed_rows = []
            for (win, ratio), path in sorted(fixed.items()):
                _, rows = read_per_sample(path)
                scores = [float(r["score"]) for r in rows]
                fixed_rows.append({
                    "pair": pair,
                    "task": task,
                    "family": "fixed_rekv",
                    "win": win,
                    "ratio": ratio,
                    "coverage_tau": "",
                    "scale": "",
                    "n": len(rows),
                    "score_mean": mean(scores),
                    "acc_tau0.5": mean([1.0 if s >= 0.5 else 0.0 for s in scores]),
                    "avg_budget": ratio,
                })

            cov_rows = []
            for (tau, scale, win), path in sorted(coverage.items()):
                _, rows = read_per_sample(path)
                scores = [float(r["score"]) for r in rows]
                budgets = [float(r.get("budget", 0.0)) for r in rows]
                cov_rows.append({
                    "pair": pair,
                    "task": task,
                    "family": "brekv",
                    "win": win,
                    "ratio": "",
                    "coverage_tau": tau,
                    "scale": scale,
                    "n": len(rows),
                    "score_mean": mean(scores),
                    "acc_tau0.5": mean([1.0 if s >= 0.5 else 0.0 for s in scores]),
                    "avg_budget": mean(budgets),
                })

            rows = fixed_rows + cov_rows
            all_rows.extend(rows)
            csv_path = OUT_ROOT / "robustness" / f"{pair}_{task}_brekv_robustness_summary.csv"
            write_csv(rows, csv_path)
            written.append(csv_path)
            plot_robustness(pair, task, fixed_rows, cov_rows)
            written.append(OUT_ROOT / "robustness" / f"{pair}_{task}_brekv_pareto.png")

    combined = OUT_ROOT / "robustness" / "pair6_pair7_brekv_robustness_summary.csv"
    write_csv(all_rows, combined)
    written.append(combined)
    return written


def plot_robustness(pair: str, task: str, fixed_rows: list[dict], cov_rows: list[dict]) -> None:
    out = OUT_ROOT / "robustness" / f"{pair}_{task}_brekv_pareto.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 5.2))
    for win in sorted({r["win"] for r in fixed_rows}):
        pts = sorted([r for r in fixed_rows if r["win"] == win], key=lambda r: r["avg_budget"])
        plt.plot(
            [r["avg_budget"] for r in pts],
            [r["score_mean"] for r in pts],
            "o-",
            lw=1.8,
            label=f"Fixed ReKV w{win}",
        )
    markers = {8: "s", 16: "^"}
    for win in sorted({r["win"] for r in cov_rows}):
        pts = [r for r in cov_rows if r["win"] == win]
        plt.scatter(
            [r["avg_budget"] for r in pts],
            [r["score_mean"] for r in pts],
            marker=markers.get(win, "o"),
            s=55,
            alpha=0.82,
            label=f"B-ReKV w{win}",
        )
    plt.xlabel("Average KV budget")
    plt.ylabel("Mean score")
    plt.title(f"{pair} {task}: ReKV fixed budget vs B-ReKV")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def pair9_method_label(path: Path) -> str:
    name = path.parent.name
    if m := RECV_RE.search(name):
        return f"ReKV-w{m.group(1)} r={m.group(2)}"
    if m := COV_RE.search(name):
        return f"B-ReKV t={m.group(1)} s={m.group(2)} w{m.group(3)}"
    return name


def diagnose_pair9() -> list[Path]:
    rows = []
    examples = []
    for path in sorted(PAIR9_ROOT.glob("**/per_sample.jsonl")):
        meta, samples = read_per_sample(path)
        scores = [float(r["score"]) for r in samples]
        budgets = [float(r.get("budget", 0.0)) for r in samples if r.get("budget") is not None]
        nonzero = [s for s in scores if abs(s) > 1e-12]
        row = {
            "dataset": meta.get("dataset", infer_dataset(path)),
            "method": pair9_method_label(path),
            "n": len(samples),
            "score_mean": round(mean(scores), 6),
            "score_max": round(max(scores) if scores else 0.0, 6),
            "nonzero_count": len(nonzero),
            "nonzero_rate": round(len(nonzero) / max(len(samples), 1), 6),
            "avg_budget": round(mean(budgets), 6) if budgets else "",
            "run_dir": str(path.parent),
        }
        rows.append(row)
        if row["score_mean"] <= 0.01:
            bad = [r for r in samples if float(r["score"]) == 0.0][:3]
            for sample in bad:
                examples.append({
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "idx": sample.get("idx"),
                    "id": sample.get("id", ""),
                    "score": sample.get("score"),
                    "run_dir": str(path.parent),
                })

    csv_path = OUT_ROOT / "pair9" / "pair9_score_distribution.csv"
    write_csv(rows, csv_path)
    md_path = csv_path.with_suffix(".md")
    write_markdown_table(rows, md_path)
    ex_path = OUT_ROOT / "pair9" / "pair9_zero_score_examples.csv"
    write_csv(examples[:80], ex_path)

    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row["score_mean"])
    lines = [
        "# Pair #9 异常诊断",
        "",
        "基于现有 `per_sample.jsonl` 的分数分布诊断；这些文件不保存模型原始回答，因此输出文本诊断需要额外抽样重跑。",
        "",
        "## 数据集级摘要",
        "",
        "| Dataset | Runs | Mean of run means | Best run mean | Runs near zero (<=0.01) |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in sorted(by_dataset):
        vals = by_dataset[dataset]
        near_zero = sum(1 for v in vals if v <= 0.01)
        lines.append(
            f"| {dataset} | {len(vals)} | {mean(vals):.4f} | {max(vals):.4f} | {near_zero} |"
        )
    lines.extend([
        "",
        "## 结论",
        "",
        "- Pair #9 的异常不是单个 run 的偶发问题；多个非 `tmath` 数据集上大量 run 的均分接近 0。",
        "- 现有 per-sample 文件只能确认分数异常，不能确认是 prompt/template、chat special tokens、模型输出格式，还是模型能力/对齐问题。",
        "- 下一步应抽样重跑少量样本并保存 raw prompt / raw response / parsed answer，用于定位异常来源。",
    ])
    report_path = OUT_ROOT / "pair9" / "pair9_diagnostic_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    return [csv_path, md_path, ex_path, report_path]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    written = []
    written.extend(summarize_cost_subset())
    written.extend(summarize_robustness())
    written.extend(diagnose_pair9())
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
