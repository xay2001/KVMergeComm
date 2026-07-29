#!/usr/bin/env python
"""Build controlled context-length QA data for the evidence-sparsity experiment.

For the first N hotpotqa samples (same seed-42 sampling as the standard
evaluator), keep the question + gold supporting sentences fixed and pad the
context with distractor paragraphs drawn from OTHER samples' contexts until a
target token length is reached. The gold evidence block is inserted at a
deterministic pseudo-random position, so evidence sparsity grows with length
while the answerable content stays constant.

Output: one JSONL per target length with fields
    prompt_A, prompt_B, answer, id, gold_char_start, gold_char_end
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from dataloader.hotpotqa import HotpotQAEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="/sharedspace/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_samples", type=int, default=40)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4000, 16000, 48000])
    parser.add_argument("--out_dir", default="snapshots/ctx_scaling_v1/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    evaluator = HotpotQAEvaluator()  # standard seed-42 sample of 500
    data = evaluator.data

    items = [data[i] for i in range(args.n_samples)]

    # Distractor pool: full paragraphs from samples OUTSIDE the evaluated set.
    pool = []
    for i in range(args.n_samples, min(len(data), args.n_samples + 300)):
        ctx = data[i]["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            paragraph = " ".join(sentences).strip()
            if paragraph:
                pool.append(paragraph)
    print(f"distractor pool: {len(pool)} paragraphs")

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    pool_order = list(range(len(pool)))

    for target in args.lengths:
        out_path = os.path.join(args.out_dir, f"hotpotqa_ctx{target}.jsonl")
        rows = []
        for sample_idx, item in enumerate(items):
            gold = str(item["prompt_A"]).strip()
            gold_tokens = len(tokenizer.encode(gold, add_special_tokens=False))
            local_rng = random.Random(args.seed * 100003 + sample_idx)
            order = pool_order[:]
            local_rng.shuffle(order)

            distractors = []
            used_tokens = gold_tokens
            for pool_idx in order:
                if used_tokens >= target:
                    break
                paragraph = pool[pool_idx]
                used_tokens += len(tokenizer.encode(paragraph, add_special_tokens=False)) + 1
                distractors.append(paragraph)

            insert_at = local_rng.randint(0, len(distractors))
            parts = distractors[:insert_at] + [gold] + distractors[insert_at:]
            context = "\n\n".join(parts)
            gold_char_start = context.find(gold)
            rows.append(
                {
                    "id": item.get("id", f"sample{sample_idx}"),
                    "prompt_A": context,
                    "prompt_B": item["prompt_B"],
                    "answer": item["answer"],
                    "gold_char_start": gold_char_start,
                    "gold_char_end": gold_char_start + len(gold),
                    "target_tokens": target,
                    "approx_tokens": used_tokens,
                }
            )
        with open(out_path, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        mean_tokens = sum(r["approx_tokens"] for r in rows) / len(rows)
        print(f"wrote {out_path}: n={len(rows)} mean_tokens={mean_tokens:.0f}")


if __name__ == "__main__":
    main()
