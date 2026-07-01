#!/usr/bin/env python3
"""Dump receiver-aware top-token examples for interpretability analysis.

This script does not generate answers. It runs A-prefill + receiver scoring,
then records which context tokens are selected by ReKV, ValueNorm, Random, and
a simple aggregate coverage rule. Plotting is intentionally left for later.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed

from dataloader import get_evaluator
from eval import CommunicationEvaluator
from models import CVCommunicator


def normalize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def answers_for_item(item) -> list[str]:
    if "answers" in item:
        return list(item["answers"])
    if "answer" in item:
        return [item["answer"]]
    return []


def token_record(tokenizer, input_ids, scores, idx: int, radius: int = 5) -> dict:
    left = max(0, idx - radius)
    right = min(input_ids.numel(), idx + radius + 1)
    token_id = int(input_ids[idx])
    return {
        "idx": int(idx),
        "token_id": token_id,
        "token": tokenizer.convert_ids_to_tokens([token_id])[0],
        "text": tokenizer.decode([token_id], skip_special_tokens=False),
        "score": round(float(scores[idx]), 8),
        "window_text": tokenizer.decode(input_ids[left:right], skip_special_tokens=False),
    }


def top_indices(scores: torch.Tensor, k: int) -> list[int]:
    k = min(k, scores.numel())
    return [int(i) for i in torch.topk(scores, k).indices.tolist()]


def aggregate_receiver_scores(token_importance: dict[int, torch.Tensor]) -> torch.Tensor:
    rows = []
    for layer_idx, s in sorted(token_importance.items()):
        if layer_idx == 0 or s.numel() < 2:
            continue
        s = s.float().clamp_min(0)
        total = s.sum()
        if total <= 0:
            continue
        rows.append(s / total)
    if not rows:
        raise RuntimeError("no receiver token importance available")
    min_len = min(r.numel() for r in rows)
    rows = [r[:min_len] for r in rows]
    return torch.stack(rows).mean(dim=0)


def aggregate_value_norm_scores(past_key_values) -> torch.Tensor:
    rows = []
    for layer_idx, value_cache in enumerate(past_key_values.value_cache):
        if layer_idx == 0:
            continue
        s = value_cache.float().norm(dim=-1).mean(dim=1)[0]
        total = s.sum()
        if total <= 0:
            continue
        rows.append(s / total)
    if not rows:
        raise RuntimeError("no value cache scores available")
    min_len = min(r.numel() for r in rows)
    rows = [r[:min_len] for r in rows]
    return torch.stack(rows).mean(dim=0)


def coverage_indices(scores: torch.Tensor, tau: float, scale: float, k_max: int) -> list[int]:
    p = scores.float().clamp_min(0)
    p = p / p.sum().clamp_min(1e-9)
    order = torch.argsort(p, descending=True)
    csum = torch.cumsum(p[order], dim=0)
    idx = int(torch.searchsorted(csum, torch.tensor(float(tau), device=csum.device)))
    k = min(max(1, int((idx + 1) * float(scale))), k_max, p.numel())
    return [int(i) for i in order[:k].tolist()]


def overlap_with_answers(token_texts: list[str], answers: list[str]) -> dict:
    answer_terms = set()
    for answer in answers:
        answer_terms.update(normalize_text(str(answer)))
    token_terms = []
    for text in token_texts:
        token_terms.extend(normalize_text(text))
    token_set = set(token_terms)
    hit_terms = sorted(answer_terms & token_set)
    return {
        "answer_terms": sorted(answer_terms),
        "hit_terms": hit_terms,
        "hit_count": len(hit_terms),
        "answer_term_count": len(answer_terms),
        "recall": round(len(hit_terms) / len(answer_terms), 6) if answer_terms else None,
    }


@torch.no_grad()
def dump_task(args, task: str, model_a, model_b, tokenizer) -> None:
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
    out_path = out_dir / f"top_tokens_w{args.recv_window}_r{args.ratio}.jsonl"
    summary_rows = []
    with out_path.open("w") as f:
        for i, item in enumerate(tqdm(evaluator, desc=f"{task} interpretability")):
            if i >= args.limit:
                break
            input_ids_A, input_ids_B = comm_eval.prepare_input_ids(item, cv.A, cv.B)
            query_tokens = int(input_ids_B.shape[-1])
            sketch_tokens = query_tokens if args.recv_window == 0 else min(int(args.recv_window), query_tokens)
            sketch_stats = {
                "query_tokens": query_tokens,
                "sketch_tokens": sketch_tokens,
                "sketch_query_ratio": round(sketch_tokens / query_tokens, 6) if query_tokens else None,
                "sketch_token_id_bytes": sketch_tokens * 4,
                "sketch_hidden_bytes": sketch_tokens * args.hidden_dim * 2,
            }
            out_A = model_a(input_ids=input_ids_A, use_cache=True, return_dict=True)
            cv.compute_receiver_importance(input_ids_B, out_A.past_key_values)

            receiver_scores = aggregate_receiver_scores(cv.token_importance).cpu()
            value_scores = aggregate_value_norm_scores(out_A.past_key_values).cpu()
            usable_ids = input_ids_A[0, : receiver_scores.numel()].cpu()
            k = min(args.top_k, receiver_scores.numel())

            rekv_idx = top_indices(receiver_scores, k)
            evict_idx = top_indices(value_scores[: receiver_scores.numel()], k)
            rng = random.Random(args.seed + i)
            random_idx = rng.sample(range(receiver_scores.numel()), k)
            cov_idx = coverage_indices(receiver_scores, args.coverage_tau, args.coverage_scale, k)

            answers = answers_for_item(item)
            rekv_texts = [tokenizer.decode([int(usable_ids[j])], skip_special_tokens=True) for j in rekv_idx]
            evict_texts = [tokenizer.decode([int(usable_ids[j])], skip_special_tokens=True) for j in evict_idx]
            random_texts = [tokenizer.decode([int(usable_ids[j])], skip_special_tokens=True) for j in random_idx]

            row = {
                "idx": i,
                "id": item.get("_id", item.get("id", None)) if hasattr(item, "get") else None,
                "task": task,
                "question": item.get("prompt_B", None) if hasattr(item, "get") else None,
                "answers": answers,
                "recv_window": args.recv_window,
                "ratio": args.ratio,
                "top_k": k,
                "query_sketch": sketch_stats,
                "rekv_top": [token_record(tokenizer, usable_ids, receiver_scores, j) for j in rekv_idx],
                "evict_top": [token_record(tokenizer, usable_ids, value_scores, j) for j in evict_idx],
                "random_top": [token_record(tokenizer, usable_ids, receiver_scores, j) for j in random_idx],
                "coverage_top": [token_record(tokenizer, usable_ids, receiver_scores, j) for j in cov_idx],
                "overlap": {
                    "rekv": overlap_with_answers(rekv_texts, answers),
                    "evict": overlap_with_answers(evict_texts, answers),
                    "random": overlap_with_answers(random_texts, answers),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary_rows.append({
                "idx": i,
                "rekv_recall": row["overlap"]["rekv"]["recall"],
                "evict_recall": row["overlap"]["evict"]["recall"],
                "random_recall": row["overlap"]["random"]["recall"],
                **sketch_stats,
            })

    summary_path = out_dir / f"answer_overlap_w{args.recv_window}_r{args.ratio}.csv"
    with summary_path.open("w") as f:
        f.write("idx,rekv_recall,evict_recall,random_recall,query_tokens,sketch_tokens,sketch_query_ratio,sketch_token_id_bytes,sketch_hidden_bytes\n")
        for row in summary_rows:
            f.write(
                f"{row['idx']},{row['rekv_recall']},{row['evict_recall']},{row['random_recall']},"
                f"{row['query_tokens']},{row['sketch_tokens']},{row['sketch_query_ratio']},"
                f"{row['sketch_token_id_bytes']},{row['sketch_hidden_bytes']}\n"
            )
    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["hotpotqa", "musique", "multifieldqa_en"])
    parser.add_argument("--model", default="/sharedspace/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--recv_window", type=int, default=8)
    parser.add_argument("--coverage_tau", type=float, default=0.95)
    parser.add_argument("--coverage_scale", type=float, default=0.75)
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--max_input_length", type=int, default=64000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, default=Path("snapshots/interpretability/pair1_llama31_same"))
    args = parser.parse_args()

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

    for task in args.tasks:
        dump_task(args, task, model_a, model_b, tokenizer)


if __name__ == "__main__":
    main()
