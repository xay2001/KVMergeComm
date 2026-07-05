#!/usr/bin/env python3
"""Diagnose near-zero Table 8 pair #9 outputs with small sampled generations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import get_evaluator  # noqa: E402
from eval import CommunicationEvaluator  # noqa: E402
from models import CVCommunicator  # noqa: E402


DEEPSEEK_HF_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def answers_for(item) -> list[str]:
    if "answers" in item:
        return [str(x) for x in item["answers"]]
    if "answer" in item:
        return [str(item["answer"])]
    return []


def score_response(evaluator, response: str) -> float:
    prev_total = evaluator.f1_total
    evaluator.evaluate_item(evaluator._current_item, response)  # type: ignore[attr-defined]
    return float(evaluator.f1_total - prev_total)


@torch.no_grad()
def run_task(args, task: str, model_a, model_b, tokenizer, mode: str) -> list[dict]:
    evaluator = get_evaluator(task)
    comm_eval = CommunicationEvaluator(
        evaluator,
        tokenizer,
        use_wandb=False,
        max_input_length=args.max_input_length,
    )

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

    if mode == "forced_deepseek_think":
        model_b.name = DEEPSEEK_HF_ID
        cv.B.name = DEEPSEEK_HF_ID
    else:
        model_b.name = args.model_b
        cv.B.name = args.model_b

    rows = []
    for idx, item in enumerate(evaluator):
        if idx >= args.limit:
            break
        evaluator._current_item = item  # type: ignore[attr-defined]
        response = comm_eval.inference(model_a, cv, item)
        score = score_response(evaluator, response)
        rows.append(
            {
                "task": task,
                "mode": mode,
                "idx": idx,
                "id": item.get("_id", item.get("id", None)) if hasattr(item, "get") else None,
                "question": item.get("prompt_B", None) if hasattr(item, "get") else None,
                "answers": answers_for(item),
                "response": response,
                "response_len_chars": len(response),
                "response_preview": response[:500],
                "score": round(score, 6),
                "budget": round(float(getattr(cv, "last_kept_ratio", 0.0)), 6),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", default="/NAS/models/Llama-3.1-SuperNova-Lite")
    parser.add_argument("--model_b", default="/NAS/models/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+", default=["countries", "tipsheets", "hotpotqa", "musique"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--recv_window", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=64000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, default=Path("snapshots/diagnostics/pair9_outputs"))
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_b)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_a = AutoModelForCausalLM.from_pretrained(
        args.model_a,
        device_map={"": args.device},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model_b = AutoModelForCausalLM.from_pretrained(
        args.model_b,
        device_map={"": args.device},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model_a.eval()
    model_b.eval()
    model_a.name = args.model_a
    model_b.name = args.model_b

    all_rows = []
    for task in args.tasks:
        for mode in ("as_run_local_path", "forced_deepseek_think"):
            rows = run_task(args, task, model_a, model_b, tokenizer, mode)
            all_rows.extend(rows)
            out_path = args.out_dir / f"{task}_{mode}.jsonl"
            with out_path.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"wrote {out_path}")

    summary_path = args.out_dir / "summary.jsonl"
    with summary_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
