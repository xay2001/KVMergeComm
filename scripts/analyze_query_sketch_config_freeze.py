#!/usr/bin/env python3
"""Select one global calibrated B-ReKV configuration at matched mean budget."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


FIXED_RE = re.compile(r"rekv_w(\d+)_r([0-9.]+)_")
COVERAGE_RE = re.compile(r"brekv_t([0-9.]+)_s([0-9.]+)_w(\d+)_")


def latest_runs(root: Path) -> list[Path]:
    grouped: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for path in root.glob("**/per_sample.jsonl"):
        name = path.parent.name
        match = FIXED_RE.match(name) or COVERAGE_RE.match(name)
        if match:
            logical_name = name.rsplit("_", 2)[0]
            grouped[(path.parent.parent, logical_name)].append(path)
    return [sorted(paths)[-1] for paths in grouped.values()]


def load_run(path: Path, root: Path) -> dict:
    rel = path.relative_to(root)
    pair, task, kind = rel.parts[:3]
    rows = []
    meta = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        if "_meta" in item:
            meta = item["_meta"]
        else:
            rows.append(item)
    if not rows:
        raise ValueError(f"empty run: {path}")
    name = path.parent.name
    if match := FIXED_RE.match(name):
        window, ratio = match.groups()
        config = f"ReKV-w{window}-r{ratio}"
        method = "fixed"
        tau = scale = None
    elif match := COVERAGE_RE.match(name):
        tau, scale, window = match.groups()
        config = f"B-ReKV-t{tau}-s{scale}-w{window}"
        method = "coverage"
        ratio = None
    else:
        raise ValueError(f"unknown run name: {name}")
    return {
        "pair": pair,
        "task": task,
        "method": method,
        "config": config,
        "window": int(window),
        "ratio": float(ratio) if ratio is not None else None,
        "tau": float(tau) if tau is not None else None,
        "scale": float(scale) if scale is not None else None,
        "n": len(rows),
        "score": sum(float(row["score"]) for row in rows) / len(rows),
        "budget": sum(float(row["budget"]) for row in rows) / len(rows),
        "protocol_version": meta.get("protocol_version"),
        "path": str(path.parent),
    }


def interpolate_fixed(curve: list[dict], budget: float) -> tuple[float, str]:
    curve = sorted(curve, key=lambda row: row["budget"])
    if budget <= curve[0]["budget"]:
        return curve[0]["score"], curve[0]["config"]
    if budget >= curve[-1]["budget"]:
        return curve[-1]["score"], curve[-1]["config"]
    for lower, upper in zip(curve, curve[1:]):
        if lower["budget"] <= budget <= upper["budget"]:
            span = upper["budget"] - lower["budget"]
            weight = 0.0 if span <= 0 else (budget - lower["budget"]) / span
            score = lower["score"] + weight * (upper["score"] - lower["score"])
            return score, f"{lower['config']}..{upper['config']}"
    raise AssertionError("unreachable")


def matched_rows(runs: list[dict]) -> list[dict]:
    fixed = defaultdict(list)
    coverage = []
    for row in runs:
        key = (row["pair"], row["task"], row["window"])
        if row["method"] == "fixed":
            fixed[key].append(row)
        else:
            coverage.append(row)

    out = []
    for row in coverage:
        key = (row["pair"], row["task"], row["window"])
        if not fixed[key]:
            continue
        matched_score, bracket = interpolate_fixed(fixed[key], row["budget"])
        out.append(
            {
                **row,
                "matched_fixed_score": matched_score,
                "score_delta": row["score"] - matched_score,
                "fixed_bracket": bracket,
            }
        )
    return out


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["tau"], row["scale"], row["window"])].append(row)
    summaries = []
    for (tau, scale, window), items in sorted(grouped.items()):
        deltas = [row["score_delta"] for row in items]
        complete = len({(row["pair"], row["task"]) for row in items}) == 6
        mean_delta = sum(deltas) / len(deltas)
        worst_delta = min(deltas)
        wins_or_ties = sum(delta >= -0.005 for delta in deltas)
        accepted = complete and mean_delta >= -0.01 and worst_delta >= -0.05 and wins_or_ties >= 4
        summaries.append(
            {
                "config": f"B-ReKV-t{tau:g}-s{scale:g}-w{window}",
                "tau": tau,
                "scale": scale,
                "window": window,
                "cells": len(items),
                "mean_score": sum(row["score"] for row in items) / len(items),
                "mean_budget": sum(row["budget"] for row in items) / len(items),
                "mean_matched_fixed_score": sum(row["matched_fixed_score"] for row in items) / len(items),
                "mean_delta": mean_delta,
                "worst_delta": worst_delta,
                "wins_or_ties": wins_or_ties,
                "accepted": accepted,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summaries: list[dict], details: list[dict]) -> None:
    accepted = [row for row in summaries if row["accepted"]]
    ranking = sorted(
        accepted or summaries,
        key=lambda row: (row["mean_delta"], -row["mean_budget"]),
        reverse=True,
    )
    selected = ranking[0] if ranking else None
    lines = [
        "# Query-Sketch B-ReKV 配置冻结报告",
        "",
        "验收标准：6 个 pair-task 单元齐全；matched-budget 平均分差不低于 -0.01；"
        "最差单元不低于 -0.05；至少 4/6 单元在 0.005 容差内持平或获胜。",
        "",
    ]
    if selected and selected["accepted"]:
        lines += [
            f"**冻结配置：`{selected['config']}`。**",
            "",
            f"- 平均实际预算：{selected['mean_budget']:.4f}",
            f"- 相对 matched-budget fixed ReKV 平均分差：{selected['mean_delta']:+.4f}",
            f"- 最差单元分差：{selected['worst_delta']:+.4f}",
            f"- 持平/获胜：{selected['wins_or_ties']}/6",
            "",
        ]
    elif selected:
        lines += [
            "**当前没有候选配置通过验收，因此不应冻结全局配置。**",
            "",
            f"当前最优候选为 `{selected['config']}`，平均分差 "
            f"{selected['mean_delta']:+.4f}，最差单元 {selected['worst_delta']:+.4f}。",
            "",
        ]
    else:
        lines += ["尚无完整结果。", ""]

    lines += [
        "## 候选汇总",
        "",
        "| 配置 | 单元 | 平均预算 | 平均分差 | 最差分差 | 持平/胜 | 通过 |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['config']} | {row['cells']} | {row['mean_budget']:.4f} | "
            f"{row['mean_delta']:+.4f} | {row['worst_delta']:+.4f} | "
            f"{row['wins_or_ties']}/6 | {'是' if row['accepted'] else '否'} |"
        )
    lines += ["", "## 逐单元结果", ""]
    for row in sorted(details, key=lambda item: (item["config"], item["pair"], item["task"])):
        lines.append(
            f"- {row['config']} / {row['pair']} / {row['task']}: "
            f"budget={row['budget']:.4f}, score={row['score']:.4f}, "
            f"matched={row['matched_fixed_score']:.4f}, delta={row['score_delta']:+.4f}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("snapshots/query_sketch_config_freeze"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else root / "analysis"

    runs = [load_run(path, root) for path in latest_runs(root)]
    details = matched_rows(runs)
    summaries = summarize(details)
    write_csv(out_dir / "all_runs.csv", runs)
    write_csv(out_dir / "matched_budget.csv", details)
    write_csv(out_dir / "candidate_summary.csv", summaries)
    write_report(out_dir / "REPORT.md", summaries, details)
    accepted = [row for row in summaries if row["accepted"]]
    ranked = sorted(
        accepted or summaries,
        key=lambda row: (row["mean_delta"], -row["mean_budget"]),
        reverse=True,
    )
    selection = ranked[0] if ranked else None
    (out_dir / "selection.json").write_text(
        json.dumps(
            {
                "accepted": bool(selection and selection["accepted"]),
                "selection": selection,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"runs={len(runs)} matched_cells={len(details)} report={out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
