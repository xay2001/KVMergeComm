#!/usr/bin/env python3
"""Summarize query-aware fairness ablations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RECV_RE = re.compile(r"recv_w(\d+)_r([0-9.]+)")
RATIO_RE = re.compile(r"(?:evict|random)_r([0-9.]+)")


class _TokenizerModelStub:
    def __init__(self, name: str):
        self.name = name
        self.device = "cpu"


def read_rows(path: Path) -> tuple[dict, list[dict]]:
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


def infer(path: Path, meta: dict) -> dict:
    run = path.parent.name
    method_dir = path.parent.parent.name
    if method_dir == "mtc_evict":
        method = "Evict/ValueNorm"
        win = None
        m = RATIO_RE.search(run)
        ratio = float(m.group(1)) if m else meta.get("merge_ratio")
        sketch = 0
    elif method_dir == "mtc_random":
        method = "Random-token"
        win = None
        m = RATIO_RE.search(run)
        ratio = float(m.group(1)) if m else meta.get("merge_ratio")
        sketch = 0
    else:
        method = "ReKV"
        m = RECV_RE.search(run)
        win = int(m.group(1)) if m else meta.get("recv_window")
        ratio = float(m.group(2)) if m else meta.get("merge_ratio")
        sketch = "all" if win == 0 else win
    return {"method": method, "recv_window": win, "ratio": ratio, "sketch_tokens": sketch}


def task_from_path(path: Path) -> str:
    parts = path.parts
    for task in ("hotpotqa", "musique", "multifieldqa_en"):
        if task in parts:
            return task
    return "unknown"


def query_token_stats(tasks: set[str], model_name: str, limit_by_task: dict[str, int]) -> dict[str, dict]:
    from transformers import AutoTokenizer

    from dataloader import get_evaluator
    from eval import (
        QA_INSTRUCTION,
        COMMUNICATION_QA_MSG_TEMPLATE_B,
        apply_chat_template,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_stub = _TokenizerModelStub(model_name)
    out = {}
    for task in sorted(tasks):
        if task == "unknown":
            continue
        evaluator = get_evaluator(task)
        lengths = []
        limit = limit_by_task.get(task)
        for i, item in enumerate(evaluator):
            if limit is not None and i >= limit:
                break
            msg = COMMUNICATION_QA_MSG_TEMPLATE_B.format(
                instruction=QA_INSTRUCTION,
                question=item["prompt_B"],
            )
            input_ids = apply_chat_template(evaluator, tokenizer, msg, model_stub)
            lengths.append(int(input_ids.shape[-1]))
        if lengths:
            out[task] = {
                "query_tokens_mean": sum(lengths) / len(lengths),
                "query_tokens_min": min(lengths),
                "query_tokens_max": max(lengths),
                "query_tokens_n": len(lengths),
            }
    return out


def summarize(root: Path, model_name: str, include_query_stats: bool = True, hidden_dim: int = 4096) -> list[dict]:
    out = []
    tasks = set()
    limit_by_task = {}
    for path in sorted(root.glob("**/per_sample.jsonl")):
        meta, rows = read_rows(path)
        if not rows:
            continue
        task = task_from_path(path)
        tasks.add(task)
        limit_by_task[task] = max(limit_by_task.get(task, 0), len(rows))
        scores = [float(r["score"]) for r in rows]
        budgets = [float(r["budget"]) for r in rows if "budget" in r]
        info = infer(path, meta)
        out.append({
            "task": task,
            **info,
            "n": len(rows),
            "score": round(sum(scores) / len(scores), 6),
            "budget": round(sum(budgets) / len(budgets), 6) if budgets else None,
            "run_dir": str(path.parent),
        })
    qstats = query_token_stats(tasks, model_name, limit_by_task) if include_query_stats else {}
    for row in out:
        qs = qstats.get(row["task"], {})
        row.update(qs)
        sketch_tokens = row.get("sketch_tokens")
        query_mean = qs.get("query_tokens_mean")
        if sketch_tokens == "all":
            sketch_mean = query_mean
        elif isinstance(sketch_tokens, int):
            sketch_mean = min(float(sketch_tokens), float(query_mean)) if query_mean is not None else float(sketch_tokens)
        else:
            sketch_mean = 0.0
        row["sketch_tokens_mean"] = round(sketch_mean, 6) if sketch_mean is not None else None
        row["sketch_query_ratio"] = round(sketch_mean / query_mean, 6) if query_mean else None
        # Token-id protocol: send compact token IDs as the query sketch.
        row["sketch_token_id_bytes"] = round(sketch_mean * 4, 3) if sketch_mean is not None else None
        # Hidden-state protocol: conservative upper bound if sending BF16 query states.
        row["sketch_hidden_bytes"] = round(sketch_mean * hidden_dim * 2, 3) if sketch_mean is not None else None
    out.sort(key=lambda r: (r["task"], str(r["ratio"]), r["method"], str(r["recv_window"])))
    return out


def print_markdown(rows: list[dict]) -> None:
    cols = [
        "task",
        "method",
        "ratio",
        "recv_window",
        "sketch_tokens",
        "query_tokens_mean",
        "sketch_query_ratio",
        "n",
        "score",
        "budget",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        print("| " + " | ".join("" if row.get(c) is None else str(row.get(c)) for c in cols) + " |")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("snapshots/query_fairness/pair1_llama31_same"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--model", default="/sharedspace/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--no_query_stats", action="store_true")
    args = parser.parse_args()

    rows = summarize(args.root, args.model, include_query_stats=not args.no_query_stats, hidden_dim=args.hidden_dim)
    print_markdown(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
