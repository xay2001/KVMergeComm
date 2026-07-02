#!/usr/bin/env python3
"""Deletion ablation for ReKV evidence tokens.

For each sample, this script:
1. Computes ReKV receiver-attention scores and query-agnostic value-norm scores.
2. Selects top content tokens for ReKV / Evict / Random.
3. Masks those tokens in A's context prompt.
4. Runs the same ReKV communication pipeline and measures answer-score drop.

This is intentionally small-scale (default 50 samples/task) because it performs
multiple generations per sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import get_evaluator
from eval import CommunicationEvaluator
from models import CVCommunicator
from scripts.dump_interpretability_examples import (
    aggregate_receiver_scores,
    aggregate_value_norm_scores,
)


TEMPLATE_TERMS = {
    "assistant",
    "system",
    "user",
    "instruction",
    "context",
    "date",
    "today",
    "knowledge",
    "cutting",
    "directly",
    "answer",
    "question",
    "needed",
    "explanation",
}


def norm_terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def is_template_token(tokenizer, token_id: int, window_text: str) -> bool:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    raw = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    stripped = text.strip()
    if not stripped:
        return True
    if "<|" in text or "<|" in raw:
        return True
    if not re.search(r"[A-Za-z0-9]", stripped):
        return True
    terms = norm_terms(stripped)
    if not terms:
        return True
    if all(t in TEMPLATE_TERMS for t in terms):
        return True
    low_window = window_text.lower()
    if "<|start_header_id|>" in low_window or "<|end_header_id|>" in low_window:
        return True
    return False


def content_indices(tokenizer, input_ids: torch.Tensor, scores: torch.Tensor, k: int) -> list[int]:
    order = torch.argsort(scores.float(), descending=True).tolist()
    ids = input_ids[0].detach().cpu()
    out = []
    for idx in order:
        left = max(0, idx - 5)
        right = min(ids.numel(), idx + 6)
        window_text = tokenizer.decode(ids[left:right], skip_special_tokens=False)
        if is_template_token(tokenizer, int(ids[idx]), window_text):
            continue
        out.append(int(idx))
        if len(out) >= k:
            break
    return out


def random_content_indices(tokenizer, input_ids: torch.Tensor, k: int, seed: int) -> list[int]:
    ids = input_ids[0].detach().cpu()
    candidates = []
    for idx in range(ids.numel()):
        left = max(0, idx - 5)
        right = min(ids.numel(), idx + 6)
        window_text = tokenizer.decode(ids[left:right], skip_special_tokens=False)
        if not is_template_token(tokenizer, int(ids[idx]), window_text):
            candidates.append(idx)
    rng = random.Random(seed)
    if len(candidates) <= k:
        return candidates
    return rng.sample(candidates, k)


def mask_input(input_ids: torch.Tensor, indices: list[int], mask_token_id: int) -> torch.Tensor:
    out = input_ids.clone()
    valid = [i for i in indices if 0 <= i < out.shape[-1]]
    if valid:
        out[0, valid] = mask_token_id
    return out


@torch.no_grad()
def generate_response(model_a, cv, comm_eval, input_ids_a, input_ids_b) -> tuple[str, float | None]:
    out_a = model_a(input_ids=input_ids_a, use_cache=True, return_dict=True)
    pkv = out_a.past_key_values
    if getattr(cv, "score_mode", None) == "receiver":
        cv.compute_receiver_importance(input_ids_b, pkv)
    output = cv.generate(
        input_ids_b,
        attention_mask=torch.ones_like(input_ids_b),
        out_A_past_key_values=pkv,
        **comm_eval.generate_args,
    )[0]
    response = comm_eval.get_response(output, input_ids_b.shape[-1])
    return response, getattr(cv, "last_kept_ratio", None)


def score_response(evaluator, item, response: str) -> float:
    prev_total = evaluator.f1_total
    prev_count = evaluator.f1_count
    evaluator.evaluate_item(item, response)
    score = float(evaluator.f1_total - prev_total)
    evaluator.f1_total = prev_total
    evaluator.f1_count = prev_count
    return score


@torch.no_grad()
def run_task(args, task: str, model_a, model_b, tokenizer) -> dict:
    evaluator = get_evaluator(task)
    comm_eval = CommunicationEvaluator(evaluator, tokenizer, use_wandb=False, max_input_length=args.max_input_length)
    cv = CVCommunicator(
        model_a,
        model_b,
        layer_from=0,
        layer_to=getattr(model_a.config, "num_hidden_layers", 32) - 1,
        layers_list=[-1],
        merge=True,
        merge_ratio=args.ratio,
        merge_mode="evict",
        score_mode="receiver",
        recv_window=args.recv_window,
        apply_attn_tracer=True,
        merge_sink=4,
        merge_recent=8,
        budget_mode="uniform",
    ).to(args.device)

    out_dir = args.out_dir / task
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"deletion_ablation_w{args.recv_window}_r{args.ratio}_k{args.top_k}.jsonl"
    mask_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id

    rows = []
    with jsonl_path.open("w") as f:
        pbar = tqdm(evaluator, desc=f"{task} deletion ablation")
        for i, item in enumerate(pbar):
            if i >= args.limit:
                break

            input_ids_a, input_ids_b = comm_eval.prepare_input_ids(item, cv.A, cv.B)

            # Compute original token rankings once.
            out_a = model_a(input_ids=input_ids_a, use_cache=True, return_dict=True)
            cv.compute_receiver_importance(input_ids_b, out_a.past_key_values)
            receiver_scores = aggregate_receiver_scores(cv.token_importance).cpu()
            value_scores = aggregate_value_norm_scores(out_a.past_key_values).cpu()
            score_len = min(receiver_scores.numel(), input_ids_a.shape[-1], value_scores.numel())
            receiver_scores = receiver_scores[:score_len]
            value_scores = value_scores[:score_len]
            usable_ids = input_ids_a[:, :score_len]

            selected = {
                "rekv": content_indices(tokenizer, usable_ids, receiver_scores, args.top_k),
                "evict": content_indices(tokenizer, usable_ids, value_scores, args.top_k),
                "random": random_content_indices(tokenizer, usable_ids, args.top_k, args.seed + i),
            }

            base_response, base_budget = generate_response(model_a, cv, comm_eval, input_ids_a, input_ids_b)
            base_score = score_response(evaluator, item, base_response)

            row = {
                "idx": i,
                "id": item.get("_id", item.get("id", None)) if hasattr(item, "get") else None,
                "base_score": round(base_score, 6),
                "base_budget": round(float(base_budget), 6) if base_budget is not None else None,
                "deletions": {},
            }

            for method, indices in selected.items():
                masked_a = mask_input(input_ids_a, indices, mask_token_id)
                response, budget = generate_response(model_a, cv, comm_eval, masked_a, input_ids_b)
                score = score_response(evaluator, item, response)
                row["deletions"][method] = {
                    "score": round(score, 6),
                    "drop": round(base_score - score, 6),
                    "budget": round(float(budget), 6) if budget is not None else None,
                    "n_deleted": len(indices),
                    "deleted_text": [
                        tokenizer.decode([int(usable_ids[0, j])], skip_special_tokens=False).strip()
                        for j in indices[: args.top_k]
                    ],
                }

            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            avg_drop = sum(r["deletions"]["rekv"]["drop"] for r in rows) / len(rows)
            pbar.set_description(f"{task} deletion ReKV-drop={avg_drop:.4f}")

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "task": task,
        "n": len(rows),
        "base_score": round(mean([r["base_score"] for r in rows]), 6),
    }
    for method in ("rekv", "evict", "random"):
        summary[f"{method}_deleted_score"] = round(mean([r["deletions"][method]["score"] for r in rows]), 6)
        summary[f"{method}_drop"] = round(mean([r["deletions"][method]["drop"] for r in rows]), 6)
    print(f"wrote {jsonl_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["hotpotqa", "musique", "multifieldqa_en"])
    ap.add_argument("--model", default="/sharedspace/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--recv_window", type=int, default=8)
    ap.add_argument("--max_input_length", type=int, default=64000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path, default=Path("snapshots/deletion_ablation/pair1_llama31_same"))
    args = ap.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": args.device}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model_b = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": args.device}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model_a.eval()
    model_b.eval()
    model_a.name = args.model
    model_b.name = args.model

    summaries = []
    for task in args.tasks:
        summaries.append(run_task(args, task, model_a, model_b, tokenizer))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"deletion_ablation_summary_w{args.recv_window}_r{args.ratio}_k{args.top_k}.csv"
    with summary_path.open("w", newline="") as f:
        fieldnames = list(summaries[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
