#!/usr/bin/env python3
"""Build a machine-readable manifest for KVComm/ReKV experiment runs.

The script is intentionally conservative: it only reads existing run artifacts
under snapshots/ and writes an index under snapshots/manifest/.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any


DATASETS = {
    "countries",
    "tipsheets",
    "hotpotqa",
    "qasper",
    "musique",
    "multifieldqa_en",
    "twowikimqa",
    "tmath",
}

PAIR_REGISTRY = {
    "flat_pair1": {
        "paper_table": "table8",
        "pair_id": 1,
        "pair_slug": "llama31_same",
        "model_a": "meta-llama/Llama-3.1-8B-Instruct",
        "model_b": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "table8_pair2_llama32_same": {
        "paper_table": "table8",
        "pair_id": 2,
        "pair_slug": "llama32_same",
        "model_a": "meta-llama/Llama-3.2-3B-Instruct",
        "model_b": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "table8_pair3_qwen25_7b_same": {
        "paper_table": "table8",
        "pair_id": 3,
        "pair_slug": "qwen25_7b_same",
        "model_a": "Qwen/Qwen2.5-7B-Instruct",
        "model_b": "Qwen/Qwen2.5-7B-Instruct",
    },
    "table1_pair6_llama32_abliterated_deepseek3b": {
        "paper_table": "table1",
        "pair_id": 6,
        "pair_slug": "llama32_abliterated_deepseek3b",
        "model_a": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "model_b": "suayptalha/DeepSeek-R1-Distill-Llama-3B",
    },
    "table1_pair7_qwen25_uncensored_bespoke": {
        "paper_table": "table1",
        "pair_id": 7,
        "pair_slug": "qwen25_uncensored_bespoke",
        "model_a": "Orion-zhen/Qwen2.5-7B-Instruct-Uncensored",
        "model_b": "bespokelabs/Bespoke-Stratos-7B",
    },
    "table1_pair8_falcon3_ultraset_abliterated": {
        "paper_table": "table1",
        "pair_id": 8,
        "pair_slug": "falcon3_ultraset_abliterated",
        "model_a": "ehristoforu/falcon3-ultraset",
        "model_b": "huihui-ai/Falcon3-7B-Instruct-abliterated",
    },
}

CONFIG_RE = re.compile(r"Configuration: AlignConfig\((.*)\)")
RESULT_RE = re.compile(r"communication result:\s*([-+0-9.eE]+)")
RUN_NAME_TS_RE = re.compile(r"_(\d{4}_\d{4})$")
TABLE_PAIR_RE = re.compile(r"table(\d+)_pair(\d+)_")
QUERY_SKETCH_ROOT_RE = re.compile(r"^table(\d+)_pair(\d+)_query_sketch_(.+)$")
RECV_RE = re.compile(r"(?:probe_)?(?:mtc_evict_)?recv_w(\d+)_r([0-9.]+)")
COV_RE = re.compile(r"cov_t([0-9.]+)_s([0-9.]+)_w(\d+)")
KVCOMM_RE = re.compile(r"kvcomm_top([0-9.]+)")
RATIO_RE = re.compile(r"(?:merge|evict|uniform|layer|query|querylayer)_r?([0-9.]+)")


def parse_align_config(text: str) -> dict[str, Any]:
    """Parse the repr emitted by dataclass AlignConfig enough for indexing."""
    match = CONFIG_RE.search(text)
    if not match:
        return {}
    body = match.group(1)
    out: dict[str, Any] = {}
    for key in (
        "model_A",
        "model_B",
        "top_layers",
        "merge",
        "merge_ratio",
        "merge_mode",
        "score_mode",
        "recv_window",
        "budget_mode",
        "budget_min",
        "budget_max",
        "coverage_tau",
        "coverage_scale",
        "test_task",
        "run_name",
    ):
        m = re.search(rf"{key}=((?:'[^']*')|(?:\"[^\"]*\")|(?:[^,)]*))", body)
        if not m:
            continue
        raw = m.group(1).strip()
        try:
            out[key] = ast.literal_eval(raw)
        except Exception:
            if raw in {"True", "False"}:
                out[key] = raw == "True"
            else:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw
    return out


def read_log(path: Path) -> tuple[dict[str, Any], float | None, str]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}, None, "missing_log"
    cfg = parse_align_config(text)
    results = RESULT_RE.findall(text)
    score = float(results[-1]) if results else None
    status = "done" if score is not None else ("failed" if "Traceback" in text or "Error" in text else "unknown")
    return cfg, score, status


def read_per_sample(path: Path) -> tuple[int | None, float | None, str | None]:
    if not path.exists():
        return None, None, None
    n = 0
    budgets: list[float] = []
    protocol = None
    try:
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if "_meta" in row:
                    protocol = (
                        protocol
                        or row["_meta"].get("protocol_version")
                        or row["_meta"].get("protocol")
                    )
                    continue
                n += 1
                if "budget" in row:
                    budgets.append(float(row["budget"]))
                elif "query_budget" in row:
                    budgets.append(float(row["query_budget"]))
    except Exception:
        return None, None, None
    avg_budget = sum(budgets) / len(budgets) if budgets else None
    return n, avg_budget, protocol


def read_cost_protocol(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text()).get("_meta", {})
        return meta.get("protocol_version") or meta.get("protocol")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def infer_protocol_version(
    run_dir: Path,
    cfg: dict[str, Any],
    recorded: str | None,
) -> str:
    """Classify legacy runs without retroactively changing their semantics."""
    if recorded:
        return recorded
    score_mode = str(cfg.get("score_mode") or "value_norm")
    if score_mode == "receiver_oracle":
        return "full_kv_oracle_v1"
    if any("query_sketch" in part for part in run_dir.parts):
        return "query_sketch_v1"
    if score_mode.startswith("receiver"):
        return "legacy_full_kv_oracle_v0"
    return "query_agnostic_kv_v1"


def query_sketch_pair_info(root_name: str) -> dict[str, Any]:
    """Map protocol-specific roots back to their canonical paper pair."""
    if "query_sketch" not in root_name:
        return {}
    table_pair = TABLE_PAIR_RE.search(root_name)
    if not table_pair:
        return {}
    table, pair_id = table_pair.groups()
    for pair in PAIR_REGISTRY.values():
        if pair["paper_table"] == f"table{table}" and pair["pair_id"] == int(pair_id):
            return dict(pair)
    match = QUERY_SKETCH_ROOT_RE.match(root_name)
    slug = match.group(3) if match else root_name
    return {
        "paper_table": f"table{table}",
        "pair_id": int(pair_id),
        "pair_slug": slug,
    }


def infer_pair_and_dataset(run_dir: Path, snapshots_dir: Path) -> dict[str, Any]:
    rel_parts = run_dir.relative_to(snapshots_dir).parts
    first = rel_parts[0]
    info: dict[str, Any] = {}
    if first in DATASETS:
        info.update(PAIR_REGISTRY["flat_pair1"])
        info["dataset"] = first
        info["root"] = str(snapshots_dir / first)
        return info

    registry = PAIR_REGISTRY.get(first, {}) or query_sketch_pair_info(first)
    info.update(registry)
    if not registry:
        m = TABLE_PAIR_RE.search(first)
        if m:
            info["paper_table"] = f"table{m.group(1)}"
            info["pair_id"] = int(m.group(2))
            info["pair_slug"] = first
        else:
            info["paper_table"] = "unknown"
            info["pair_id"] = None
            info["pair_slug"] = first
    info["root"] = str(snapshots_dir / first)
    info["dataset"] = next((p for p in rel_parts[1:] if p in DATASETS), None)
    return info


def infer_method(run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    name = run_dir.name
    parent = run_dir.parent.name
    method = parent
    window = cfg.get("recv_window")
    ratio = cfg.get("merge_ratio")
    coverage_tau = cfg.get("coverage_tau")
    coverage_scale = cfg.get("coverage_scale")
    budget_mode = cfg.get("budget_mode")

    if parent == "mtc_receiver" or "recv_w" in name:
        method = "ReKV"
        if name.startswith("probe_"):
            method = "ReKV-probe"
    elif parent == "coverage" or name.startswith("cov_t"):
        method = "B-ReKV"
    elif parent == "kvcomm" or name.startswith("kvcomm_top"):
        method = "KVComm"
    elif parent == "mtc_merge":
        method = "merge"
    elif parent == "mtc_evict":
        method = "evict"
    elif parent == "budget":
        method = f"budget-{budget_mode or name.split('_')[0]}"
    elif parent == "features":
        method = "pass1-features"
    elif parent == "progressive":
        method = "progressive"

    if m := RECV_RE.search(name):
        window = int(m.group(1))
        ratio = float(m.group(2))
    if m := COV_RE.search(name):
        coverage_tau = float(m.group(1))
        coverage_scale = float(m.group(2))
        window = int(m.group(3))
        ratio = None
    if m := KVCOMM_RE.search(name):
        ratio = float(m.group(1))
    if ratio is None and (m := RATIO_RE.search(name)):
        ratio = float(m.group(1))

    return {
        "method": method,
        "method_dir": parent,
        "run_name": name,
        "window": window,
        "ratio_or_budget": ratio,
        "budget_mode": budget_mode,
        "coverage_tau": coverage_tau if method == "B-ReKV" else None,
        "coverage_scale": coverage_scale if method == "B-ReKV" else None,
    }


def build_manifest(snapshots_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for log_path in sorted(snapshots_dir.glob("**/log.log")):
        run_dir = log_path.parent
        cfg, score, status = read_log(log_path)
        n_samples, avg_budget, per_sample_protocol = read_per_sample(run_dir / "per_sample.jsonl")
        recorded_protocol = per_sample_protocol or read_cost_protocol(run_dir / "cost_summary.json")
        protocol = infer_protocol_version(run_dir, cfg, recorded_protocol)
        pair_info = infer_pair_and_dataset(run_dir, snapshots_dir)
        method_info = infer_method(run_dir, cfg)
        ts_match = RUN_NAME_TS_RE.search(run_dir.name)

        row = {
            **pair_info,
            **method_info,
            "model_a": cfg.get("model_A") or pair_info.get("model_a"),
            "model_b": cfg.get("model_B") or pair_info.get("model_b"),
            "score": score,
            "protocol_version": protocol,
            "avg_budget": avg_budget,
            "n_samples": n_samples,
            "status": status,
            "timestamp": ts_match.group(1) if ts_match else None,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "per_sample_path": str(run_dir / "per_sample.jsonl") if (run_dir / "per_sample.jsonl").exists() else None,
        }
        rows.append(row)
    return rows


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "paper_table",
        "pair_id",
        "pair_slug",
        "dataset",
        "method",
        "method_dir",
        "run_name",
        "window",
        "ratio_or_budget",
        "budget_mode",
        "coverage_tau",
        "coverage_scale",
        "score",
        "protocol_version",
        "avg_budget",
        "n_samples",
        "status",
        "timestamp",
        "model_a",
        "model_b",
        "root",
        "run_dir",
        "log_path",
        "per_sample_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", default="snapshots", type=Path)
    parser.add_argument("--out-dir", default=Path("snapshots/manifest"), type=Path)
    args = parser.parse_args()

    snapshots_dir = args.snapshots.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_manifest(snapshots_dir)
    write_json(rows, out_dir / "experiments.json")
    write_csv(rows, out_dir / "experiments.csv")
    print(f"wrote {len(rows)} runs to {out_dir}")


if __name__ == "__main__":
    main()
