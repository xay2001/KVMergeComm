#!/usr/bin/env python3
"""Low-cost model-pair compatibility diagnostic for KV transfer.

The default ``summarize`` command is offline-only: it reuses existing
``per_sample.jsonl`` files, local checkpoint configs, and optional attention
calibration JSON files.  The ``calibrate-attention`` command is an explicit GPU
entry point; it never runs as part of offline summarization.

This is an n=8 diagnostic (seven established pairs plus the historical pair9
hard-negative), not a validated general-purpose predictor of KV compatibility.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PAIR_REGISTRY: dict[int, dict[str, Any]] = {
    1: {
        "slug": "pair1_llama31_same",
        "sender": "/sharedspace/models/Llama-3.1-8B-Instruct",
        "receiver": "/sharedspace/models/Llama-3.1-8B-Instruct",
        "alternates": ["/NAS/models/Llama-3.1-8B-Instruct"],
        "status": "successful",
    },
    2: {
        "slug": "pair2_llama32_same",
        "sender": "/NAS/models/Llama-3.2-3B-Instruct",
        "receiver": "/NAS/models/Llama-3.2-3B-Instruct",
        "status": "successful",
    },
    3: {
        "slug": "pair3_qwen25_7b_same",
        "sender": "/NAS/models/Qwen2.5-7B-Instruct",
        "receiver": "/NAS/models/Qwen2.5-7B-Instruct",
        "status": "successful",
    },
    4: {
        "slug": "pair4_falcon3_7b_same",
        "sender": "/NAS/models/Falcon3-7B-Instruct",
        "receiver": "/NAS/models/Falcon3-7B-Instruct",
        "status": "successful",
    },
    5: {
        "slug": "pair5_evolcodellama_toolace",
        "sender": "/NAS/models/EvolCodeLlama-3.1-8B-Instruct",
        "receiver": "/NAS/models/ToolACE-2-Llama-3.1-8B",
        "status": "successful",
    },
    6: {
        "slug": "pair6_llama32_abliterated_deepseek3b",
        "sender": "/NAS/models/Llama-3.2-3B-Instruct-abliterated",
        "receiver": "/NAS/models/DeepSeek-R1-Distill-Llama-3B",
        "status": "successful",
    },
    7: {
        "slug": "pair7_qwen25_uncensored_bespoke",
        "sender": "/NAS/models/Qwen2.5-7B-Instruct-Uncensored",
        "receiver": "/NAS/models/Bespoke-Stratos-7B",
        "status": "successful",
    },
    9: {
        "slug": "pair9_supernova_deepseek_llama8b",
        "sender": "/NAS/models/Llama-3.1-SuperNova-Lite",
        "receiver": "/NAS/models/DeepSeek-R1-Distill-Llama-8B",
        "status": "historical_hard_negative_corrected",
    },
}

DEFAULT_TASKS = ["hotpotqa", "musique", "multifieldqa_en"]
CONFIG_FIELDS = [
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "rope_theta",
]


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def read_per_sample(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                if "_meta" in obj:
                    meta.update(obj["_meta"])
                else:
                    rows.append(obj)
    except (OSError, json.JSONDecodeError):
        return {}, []
    return meta, rows


def score_file(path: Path) -> dict[str, Any] | None:
    meta, samples = read_per_sample(path)
    scores = [value for row in samples if (value := finite(row.get("score"))) is not None]
    if not scores:
        return None
    return {
        "score": mean(scores),
        "nonzero_rate": mean([value > 0 for value in scores]),
        "n": len(scores),
        "protocol": meta.get("protocol_version") or meta.get("protocol"),
        "path": str(path),
    }


def task_from_path(path: Path, tasks: Iterable[str]) -> str | None:
    task_set = set(tasks)
    return next((part for part in path.parts if part in task_set), None)


def latest_matching(paths: Iterable[Path], pattern: re.Pattern[str]) -> Path | None:
    matches = [path for path in paths if pattern.search(path.parent.name)]
    return sorted(matches, key=lambda path: (path.stat().st_mtime, str(path)))[-1] if matches else None


def find_rekv_runs(snapshot_root: Path, pair_id: int, tasks: list[str]) -> list[dict[str, Any]]:
    pair = PAIR_REGISTRY[pair_id]
    found: list[dict[str, Any]] = []
    if pair_id != 9:
        base = snapshot_root / "full_matched_budget_fairness_query_sketch" / pair["slug"]
        method_dir = "fairness_rekv"
        run_re = re.compile(r"^rekv_w8_r0[.]30_")
    else:
        base = snapshot_root / "table8_pair9_supernova_deepseek_llama8b_corrected"
        method_dir = "mtc_receiver"
        run_re = re.compile(r"^recv_w8_r0[.]3_")

    for task in tasks:
        candidates = (base / task / method_dir).glob("*/per_sample.jsonl")
        path = latest_matching(candidates, run_re)
        result = score_file(path) if path else None
        if result:
            result["task"] = task
            found.append(result)
    return found


def find_full_kv_runs(snapshot_root: Path, pair_id: int, tasks: list[str]) -> list[dict[str, Any]]:
    slug = PAIR_REGISTRY[pair_id]["slug"]
    found: list[dict[str, Any]] = []
    candidates = [
        path
        for path in snapshot_root.glob("**/full_kv/*/per_sample.jsonl")
        if slug in path.parts
    ]
    for task in tasks:
        task_paths = [path for path in candidates if task_from_path(path, tasks) == task]
        if not task_paths:
            continue
        path = sorted(task_paths, key=lambda item: (item.stat().st_mtime, str(item)))[-1]
        result = score_file(path)
        if result:
            result["task"] = task
            found.append(result)
    return found


def existing_model_path(configured: str, alternates: list[str] | None = None) -> Path | None:
    for raw in [configured, *(alternates or [])]:
        path = Path(raw)
        if (path / "config.json").exists():
            return path
    return None


def read_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return json.loads((path / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def config_relation(pair: dict[str, Any]) -> dict[str, Any]:
    alternates = pair.get("alternates", [])
    sender_path = existing_model_path(pair["sender"], alternates)
    receiver_path = existing_model_path(pair["receiver"], alternates)
    sender_cfg = read_config(sender_path)
    receiver_cfg = read_config(receiver_path)
    relation: dict[str, Any] = {
        "sender_config_path": str(sender_path / "config.json") if sender_path else None,
        "receiver_config_path": str(receiver_path / "config.json") if receiver_path else None,
        "same_configured_checkpoint": pair["sender"] == pair["receiver"],
        "config_available": sender_cfg is not None and receiver_cfg is not None,
    }
    if not relation["config_available"]:
        relation["structural_kv_compatible"] = None
        relation["matching_config_fields"] = None
        relation["config_field_count"] = 0
        return relation

    def config_value(config: dict[str, Any], field: str) -> Any:
        value = config.get(field)
        if field == "head_dim" and value is None:
            hidden = config.get("hidden_size")
            heads = config.get("num_attention_heads")
            if hidden is not None and heads:
                value = int(hidden) // int(heads)
        return value

    matches = {
        field: (
            config_value(sender_cfg, field) is not None
            and config_value(receiver_cfg, field) is not None
            and config_value(sender_cfg, field) == config_value(receiver_cfg, field)
        )
        for field in CONFIG_FIELDS
    }
    required = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ]
    relation["matching_config_fields"] = sum(matches.values())
    relation["config_field_count"] = len(matches)
    relation["config_match_fraction"] = rounded(mean(matches.values()))
    relation["structural_kv_compatible"] = all(matches[field] for field in required)
    relation["config_field_matches"] = matches
    return relation


def read_attention_calibration(
    calibration_root: Path, pair_id: int, tasks: list[str]
) -> dict[str, Any]:
    files = []
    pair_dir = calibration_root / PAIR_REGISTRY[pair_id]["slug"]
    for task in tasks:
        path = pair_dir / f"{task}.json"
        if path.exists():
            files.append(path)
    rows = []
    for path in files:
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "complete":
            rows.append(row)
    keys = ["attention_spearman", "attention_topk_jaccard", "qk_logit_cosine"]
    return {
        f"{key}_mean": rounded(mean(values)) if (values := [finite(row.get(key)) for row in rows if finite(row.get(key)) is not None]) else None
        for key in keys
    } | {
        "attention_calibration_tasks": len(rows),
        "attention_calibration_samples": sum(int(row.get("n_samples", 0)) for row in rows),
    }


def aggregate_pair(
    snapshot_root: Path,
    calibration_root: Path,
    pair_id: int,
    tasks: list[str],
) -> dict[str, Any]:
    pair = PAIR_REGISTRY[pair_id]
    full = find_full_kv_runs(snapshot_root, pair_id, tasks)
    rekv = find_rekv_runs(snapshot_root, pair_id, tasks)
    full_by_task = {row["task"]: row for row in full}
    rekv_by_task = {row["task"]: row for row in rekv}
    overlap = sorted(set(full_by_task) & set(rekv_by_task))
    recoveries = [
        rekv_by_task[task]["score"] / full_by_task[task]["score"]
        for task in overlap
        if full_by_task[task]["score"] > 0
    ]
    result = {
        "pair_id": pair_id,
        "pair": pair["slug"],
        "known_outcome": pair["status"],
        "diagnostic_n_pairs": 8,
        "full_kv_transfer_score": rounded(mean([row["score"] for row in full])) if full else None,
        "full_kv_tasks": len(full),
        "full_kv_samples": sum(row["n"] for row in full),
        "full_kv_status": "observed" if full else "missing_gpu_run",
        "rekv_w8_r0_3_score": rounded(mean([row["score"] for row in rekv])) if rekv else None,
        "rekv_nonzero_rate": rounded(
            sum(row["nonzero_rate"] * row["n"] for row in rekv) / sum(row["n"] for row in rekv)
        ) if rekv else None,
        "rekv_tasks": len(rekv),
        "rekv_samples": sum(row["n"] for row in rekv),
        "rekv_full_kv_recovery": rounded(mean(recoveries)) if recoveries else None,
        "recovery_tasks": len(recoveries),
        "full_kv_sources": [row["path"] for row in full],
        "rekv_sources": [row["path"] for row in rekv],
        **config_relation(pair),
        **read_attention_calibration(calibration_root, pair_id, tasks),
    }
    return result


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: csv_value(value) for key, value in row.items()} for row in rows])


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(rows: list[dict[str, Any]], path: Path, tasks: list[str]) -> None:
    observed_full = sum(row["full_kv_status"] == "observed" for row in rows)
    calibrated = sum(row["attention_calibration_tasks"] > 0 for row in rows)
    lines = [
        "# Model-pair compatibility diagnostic",
        "",
        "> **Scope warning:** this is an exploratory n=8 diagnostic (7 established pairs + "
        "the historical pair9 stress case). It must not be described as a validated or "
        "general predictor.",
        "",
        f"Tasks: `{', '.join(tasks)}`. Existing Full-KV coverage: {observed_full}/8 pairs. "
        f"Attention calibration coverage: {calibrated}/8 pairs.",
        "",
        "| Pair | Known outcome | Full-KV transfer | ReKV w8/r0.3 | Nonzero rate | "
        "ReKV/Full-KV | Config match | Attn Spearman | QK-logit cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {pair} | {outcome} | {full} | {rekv} | {nonzero} | {recovery} | "
            "{config} | {attn} | {qk} |".format(
                pair=row["pair"],
                outcome=row["known_outcome"],
                full=fmt(row["full_kv_transfer_score"]),
                rekv=fmt(row["rekv_w8_r0_3_score"]),
                nonzero=fmt(row["rekv_nonzero_rate"]),
                recovery=fmt(row["rekv_full_kv_recovery"]),
                config=fmt(row.get("config_match_fraction")),
                attn=fmt(row.get("attention_spearman_mean")),
                qk=fmt(row.get("qk_logit_cosine_mean")),
            )
        )
    pair9 = next((row for row in rows if row["pair_id"] == 9), None)
    if pair9 and pair9["rekv_w8_r0_3_score"] is not None:
        lines.extend(
            [
                "",
                "## Pair9 qualification",
                "",
                "Pair9's corrected ReKV macro is "
                f"`{pair9['rekv_w8_r0_3_score']:.4f}` with nonzero rate "
                f"`{pair9['rekv_nonzero_rate']:.4f}`. The old near-zero batch was affected "
                "by DeepSeek think-prefix handling. Therefore pair9 is a historical stress "
                "case, but the corrected artifacts do not establish a clean binary negative "
                "label for this diagnostic.",
            ]
        )
    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- **Full-KV transfer:** mean task score from an uncompressed sender-KV transfer run.",
            "- **ReKV availability target:** canonical `w=8, r=0.3` score and sample-level "
            "nonzero-score rate. Recovery is computed only on tasks with a matching Full-KV run.",
            "- **Config match:** fraction of structural config fields that match; absent local "
            "configs remain `NA` rather than being inferred from model names.",
            "- **Attention agreement:** mean layer/sample Spearman agreement and top-k Jaccard "
            "between receiver and sender last-query attention rankings.",
            "- **QK-logit cosine:** cosine between centered `log(attention)` vectors. This is a "
            "softmax-offset-invariant proxy for corresponding-layer query-key score stability, "
            "not direct Q/K weight CKA.",
            "",
            "Missing values mean the corresponding GPU calibration has not been run. No missing "
            "model-derived metric is imputed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def summarize(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        aggregate_pair(args.snapshot_root, args.calibration_root, pair_id, args.tasks)
        for pair_id in args.pairs
    ]
    (args.out_dir / "model_compatibility_diagnostic.json").write_text(
        json.dumps(
            {
                "_meta": {
                    "scope": "n=8 exploratory diagnostic; not a general predictor",
                    "pairs": args.pairs,
                    "tasks": args.tasks,
                },
                "pairs": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    write_csv(rows, args.out_dir / "model_compatibility_diagnostic.csv")
    write_report(rows, args.out_dir / "REPORT.md", args.tasks)
    print(f"wrote {len(rows)} pair rows to {args.out_dir}")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def cosine(left: list[float], right: list[float]) -> float | None:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return cosine(
        [value - mean(ranks(left)) for value in ranks(left)],
        [value - mean(ranks(right)) for value in ranks(right)],
    )


def attention_summary(attention: Any, topk: int) -> dict[str, Any]:
    # [batch, heads, query, key] -> head-mean distribution for last query.
    vector = attention[0, :, -1, :].float().mean(dim=0).cpu()
    values = vector.tolist()
    safe_logs = [math.log(max(value, 1e-12)) for value in values]
    center = mean(safe_logs)
    return {
        "attention": values,
        "centered_log_attention": [value - center for value in safe_logs],
        "topk": sorted(range(len(values)), key=values.__getitem__, reverse=True)[:topk],
    }


def collect_model_attentions(
    model_path: str,
    encoded_samples: list[list[int]],
    device: str,
    dtype_name: str,
    topk: int,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.output_attentions = True
    config.attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    collected = []
    with torch.inference_mode():
        for token_ids in encoded_samples:
            input_ids = torch.tensor([token_ids], device=device)
            output = model(input_ids=input_ids, use_cache=False, output_attentions=True)
            if output.attentions is None:
                raise RuntimeError(f"{model_path} did not return attentions")
            collected.append([attention_summary(layer, topk) for layer in output.attentions])
    metadata = {
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "vocab_size": getattr(config, "vocab_size", None),
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return collected, metadata


def calibration_prompts(task: str, limit: int) -> list[dict[str, str]]:
    from dataloader import get_evaluator

    evaluator = get_evaluator(task)
    prompts = []
    for index, item in enumerate(evaluator):
        if index >= limit:
            break
        prompts.append(
            {
                "id": str(item.get("id", index)),
                "text": f"Context:\n{item['prompt_A']}\n\nQuestion:\n{item['prompt_B']}",
            }
        )
    return prompts


def calibrate_attention(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    output = args.out_dir / PAIR_REGISTRY[args.pair]["slug"] / f"{args.task}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        print(f"skip existing {output}")
        return

    prompts = calibration_prompts(args.task, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.sender, trust_remote_code=True)
    encoded = [
        tokenizer(
            prompt["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_length,
        )["input_ids"]
        for prompt in prompts
    ]
    receiver_config = __import__("transformers").AutoConfig.from_pretrained(
        args.receiver, trust_remote_code=True
    )
    receiver_vocab = int(getattr(receiver_config, "vocab_size", 0))
    if any(token >= receiver_vocab for sample in encoded for token in sample):
        raise ValueError("sender token IDs exceed receiver vocabulary; shared-ID comparison is invalid")

    sender_attn, sender_meta = collect_model_attentions(
        args.sender, encoded, args.device, args.dtype, args.topk
    )
    receiver_attn, receiver_meta = collect_model_attentions(
        args.receiver, encoded, args.device, args.dtype, args.topk
    )

    spearmans: list[float] = []
    jaccards: list[float] = []
    qk_cosines: list[float] = []
    compared_layers = 0
    for sender_sample, receiver_sample in zip(sender_attn, receiver_attn):
        layer_count = min(len(sender_sample), len(receiver_sample))
        for layer_index in range(layer_count):
            left = sender_sample[layer_index]
            right = receiver_sample[layer_index]
            length = min(len(left["attention"]), len(right["attention"]))
            if length < 2:
                continue
            left_attention = left["attention"][-length:]
            right_attention = right["attention"][-length:]
            if (value := spearman(left_attention, right_attention)) is not None:
                spearmans.append(value)
            left_top = set(sorted(range(length), key=left_attention.__getitem__, reverse=True)[: args.topk])
            right_top = set(sorted(range(length), key=right_attention.__getitem__, reverse=True)[: args.topk])
            jaccards.append(len(left_top & right_top) / len(left_top | right_top))
            left_log = left["centered_log_attention"][-length:]
            right_log = right["centered_log_attention"][-length:]
            if (value := cosine(left_log, right_log)) is not None:
                qk_cosines.append(value)
            compared_layers += 1

    result = {
        "status": "complete",
        "pair_id": args.pair,
        "pair": PAIR_REGISTRY[args.pair]["slug"],
        "task": args.task,
        "n_samples": len(prompts),
        "max_length": args.max_length,
        "topk": args.topk,
        "shared_sender_token_ids": True,
        "attention_spearman": rounded(mean(spearmans)) if spearmans else None,
        "attention_topk_jaccard": rounded(mean(jaccards)) if jaccards else None,
        "qk_logit_cosine": rounded(mean(qk_cosines)) if qk_cosines else None,
        "compared_sample_layers": compared_layers,
        "sender": args.sender,
        "receiver": args.receiver,
        "sender_config": sender_meta,
        "receiver_config": receiver_meta,
        "scope_warning": "n=8 exploratory diagnostic; not a general predictor",
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summarize", help="offline artifact/config summary")
    summary.add_argument("--snapshot-root", type=Path, default=Path("snapshots"))
    summary.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("snapshots/model_compatibility_diagnostic/calibration"),
    )
    summary.add_argument(
        "--out-dir",
        type=Path,
        default=Path("snapshots/model_compatibility_diagnostic/analysis"),
    )
    summary.add_argument("--pairs", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 9])
    summary.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    summary.set_defaults(func=summarize)

    calibration = subparsers.add_parser(
        "calibrate-attention", help="run explicit small model attention calibration"
    )
    calibration.add_argument("--pair", type=int, choices=PAIR_REGISTRY, required=True)
    calibration.add_argument("--task", default="hotpotqa")
    calibration.add_argument("--sender", required=True)
    calibration.add_argument("--receiver", required=True)
    calibration.add_argument("--device", default="cuda:0")
    calibration.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    calibration.add_argument("--limit", type=int, default=4)
    calibration.add_argument("--max-length", type=int, default=512)
    calibration.add_argument("--topk", type=int, default=32)
    calibration.add_argument(
        "--out-dir",
        type=Path,
        default=Path("snapshots/model_compatibility_diagnostic/calibration"),
    )
    calibration.add_argument("--overwrite", action="store_true")
    calibration.set_defaults(func=calibrate_attention)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.pairs if args.command == "summarize" else [args.pair]) - PAIR_REGISTRY.keys())
    if unknown:
        raise SystemExit(f"unknown pair IDs: {unknown}")
    args.func(args)


if __name__ == "__main__":
    main()
