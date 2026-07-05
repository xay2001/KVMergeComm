#!/usr/bin/env python3
"""Measure whether selected HotpotQA context tokens fall in supporting facts.

Unlike the normal HotpotQA evaluator, this diagnostic uses the full HotpotQA
context, including distractor sentences. It then computes receiver-aware token
scores and measures how often the top selected context tokens overlap the gold
supporting sentences.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.hotpotqa import HotpotQAEvaluator  # noqa: E402
from eval import (  # noqa: E402
    COMMUNICATION_QA_MSG_TEMPLATE_A,
    COMMUNICATION_QA_MSG_TEMPLATE_B,
    QA_INSTRUCTION,
    apply_chat_template,
)
from models import CVCommunicator  # noqa: E402


def aggregate_receiver_scores(token_importance: dict[int, torch.Tensor]) -> torch.Tensor:
    rows = []
    for layer_idx, scores in sorted(token_importance.items()):
        if layer_idx == 0 or scores.numel() < 2:
            continue
        row = scores.float().clamp_min(0)
        total = row.sum()
        if total <= 0:
            continue
        rows.append(row / total)
    if not rows:
        raise RuntimeError("no receiver token importance available")
    min_len = min(row.numel() for row in rows)
    return torch.stack([row[:min_len] for row in rows]).mean(dim=0)


def aggregate_value_norm_scores(past_key_values) -> torch.Tensor:
    rows = []
    for layer_idx, value_cache in enumerate(past_key_values.value_cache):
        if layer_idx == 0:
            continue
        row = value_cache.float().norm(dim=-1).mean(dim=1)[0]
        total = row.sum()
        if total <= 0:
            continue
        rows.append(row / total)
    if not rows:
        raise RuntimeError("no value cache scores available")
    min_len = min(row.numel() for row in rows)
    return torch.stack([row[:min_len] for row in rows]).mean(dim=0)


def build_full_context(item) -> tuple[str, list[tuple[int, int]]]:
    support_pairs = set(zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"]))
    lines: list[str] = []
    support_spans: list[tuple[int, int]] = []
    cursor = 0

    for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
        for sent_idx, sent in enumerate(sentences):
            prefix = f"{title}: "
            line = prefix + sent
            sent_start = cursor + len(prefix)
            sent_end = sent_start + len(sent)
            if (title, sent_idx) in support_pairs:
                support_spans.append((sent_start, sent_end))
            lines.append(line)
            cursor += len(line) + 1

    return "\n".join(lines), support_spans


def overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    if end <= start:
        return False
    return any(start < sup_end and end > sup_start for sup_start, sup_end in spans)


def top_context_indices(scores: torch.Tensor, context_mask: torch.Tensor, k: int) -> list[int]:
    masked = scores.float().clone()
    masked[~context_mask] = float("-inf")
    k = min(k, int(context_mask.sum().item()))
    if k <= 0:
        return []
    return [int(i) for i in torch.topk(masked, k).indices.tolist()]


def random_context_indices(context_mask: torch.Tensor, k: int, seed: int) -> list[int]:
    candidates = [int(i) for i in torch.nonzero(context_mask, as_tuple=False).flatten().tolist()]
    rng = random.Random(seed)
    return rng.sample(candidates, min(k, len(candidates)))


def selection_records(tokenizer, input_ids, offsets, shifted_support_spans, indices: list[int]) -> list[dict]:
    records = []
    for idx in indices:
        start, end = offsets[idx]
        records.append(
            {
                "idx": idx,
                "span": [int(start), int(end)],
                "in_support": overlaps((int(start), int(end)), shifted_support_spans),
                "text": tokenizer.decode([int(input_ids[0, idx])], skip_special_tokens=False),
            }
        )
    return records


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/NAS/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--recv_window", type=int, default=8)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--max_input_length", type=int, default=64000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, default=Path("snapshots/supporting_overlap/hotpotqa_pair1_full_context"))
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": args.device}, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model_b = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": args.device}, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model_a.eval()
    model_b.eval()
    model_a.name = args.model
    model_b.name = args.model

    evaluator_stub = SimpleNamespace(name="hotpotqa")
    raw = HotpotQAEvaluator(n_samples=None).load_hotpotqa_dataset().shuffle(seed=args.seed)

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

    rows = []
    jsonl_path = args.out_dir / f"supporting_overlap_top{args.top_k}_w{args.recv_window}_r{args.ratio}.jsonl"
    with jsonl_path.open("w") as f:
        for idx, item in enumerate(raw):
            if idx >= args.limit:
                break
            full_context, support_spans = build_full_context(item)
            msg_a = COMMUNICATION_QA_MSG_TEMPLATE_A.format(instruction=QA_INSTRUCTION, context=full_context)
            chat_a = tokenizer.apply_chat_template(
                [{"role": "user", "content": msg_a}],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_start = chat_a.find(full_context)
            if context_start < 0:
                raise RuntimeError("failed to locate context in rendered chat template")
            shifted_support_spans = [(context_start + s, context_start + e) for s, e in support_spans]

            encoded_a = tokenizer(
                chat_a,
                add_special_tokens=False,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            input_ids_a = encoded_a["input_ids"].to(args.device)
            offsets = encoded_a["offset_mapping"][0].tolist()
            context_end = context_start + len(full_context)
            context_mask = torch.tensor(
                [overlaps((int(s), int(e)), [(context_start, context_end)]) for s, e in offsets],
                dtype=torch.bool,
            )

            msg_b = COMMUNICATION_QA_MSG_TEMPLATE_B.format(instruction=QA_INSTRUCTION, question=item["question"])
            input_ids_b = apply_chat_template(evaluator_stub, tokenizer, msg_b, model_b).to(args.device)
            if input_ids_a.shape[-1] + input_ids_b.shape[-1] > args.max_input_length:
                half = int((args.max_input_length - input_ids_b.shape[-1]) / 2)
                input_ids_a = torch.cat([input_ids_a[:, :half], input_ids_a[:, -half:]], dim=-1)
                offsets = offsets[:half] + offsets[-half:]
                context_mask = torch.cat([context_mask[:half], context_mask[-half:]], dim=0)

            out_a = model_a(input_ids=input_ids_a, use_cache=True, return_dict=True)
            cv.compute_receiver_importance(input_ids_b, out_a.past_key_values)
            receiver_scores = aggregate_receiver_scores(cv.token_importance).cpu()
            value_scores = aggregate_value_norm_scores(out_a.past_key_values).cpu()
            usable_len = min(receiver_scores.numel(), len(offsets), input_ids_a.shape[-1])
            receiver_scores = receiver_scores[:usable_len]
            value_scores = value_scores[:usable_len]
            context_mask = context_mask[:usable_len]
            offsets = offsets[:usable_len]

            rekv_idx = top_context_indices(receiver_scores, context_mask, args.top_k)
            evict_idx = top_context_indices(value_scores, context_mask, args.top_k)
            random_idx = random_context_indices(context_mask, args.top_k, args.seed + idx)

            def rate(indices: list[int]) -> float:
                if not indices:
                    return 0.0
                return sum(overlaps((int(offsets[i][0]), int(offsets[i][1])), shifted_support_spans) for i in indices) / len(indices)

            row = {
                "idx": idx,
                "id": item.get("id"),
                "question": item["question"],
                "answer": item["answer"],
                "supporting_fact_count": len(support_spans),
                "context_token_count": int(context_mask.sum().item()),
                "top_k": args.top_k,
                "rekv_support_rate": round(rate(rekv_idx), 6),
                "evict_support_rate": round(rate(evict_idx), 6),
                "random_support_rate": round(rate(random_idx), 6),
                "rekv_top": selection_records(tokenizer, input_ids_a.cpu(), offsets, shifted_support_spans, rekv_idx[:10]),
                "evict_top": selection_records(tokenizer, input_ids_a.cpu(), offsets, shifted_support_spans, evict_idx[:10]),
                "random_top": selection_records(tokenizer, input_ids_a.cpu(), offsets, shifted_support_spans, random_idx[:10]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    summary_path = args.out_dir / f"supporting_overlap_summary_top{args.top_k}_w{args.recv_window}_r{args.ratio}.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "top_k",
                "rekv_support_rate",
                "evict_support_rate",
                "random_support_rate",
            ],
        )
        writer.writeheader()
        n = len(rows)
        writer.writerow(
            {
                "n": n,
                "top_k": args.top_k,
                "rekv_support_rate": round(sum(r["rekv_support_rate"] for r in rows) / n, 6) if n else 0,
                "evict_support_rate": round(sum(r["evict_support_rate"] for r in rows) / n, 6) if n else 0,
                "random_support_rate": round(sum(r["random_support_rate"] for r in rows) / n, 6) if n else 0,
            }
        )
    print(f"wrote {jsonl_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
