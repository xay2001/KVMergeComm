# Natural-Language Passing vs ReKV/B-ReKV Cost Comparison

## Scope

- Pairs:
  - `S: Llama-3.1-8B; R: Llama-3.1-8B`
  - `S: Llama-3.2-3B-Abliterated; R: DeepSeek-R1-3B`
  - `S: Qwen2.5-7B-Uncensored; R: Bespoke-Stratos-7B`
- Tasks: `hotpotqa`, `musique`, `multifieldqa_en`
- NLD setting: `--do_test_nld --profile_cost`, phase-1 answer cap `128` tokens.
- Main comparison rows: `NLD`, `ReKV-w8 r=0.3`, and best available `B-ReKV` per pair/task.

## Outputs

```text
snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_summary.csv
snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_focused.csv
snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_average_by_pair.csv
snapshots/analysis/nld_vs_rekv/figures/nld_vs_rekv_cost_overview.png
snapshots/analysis/nld_vs_rekv/figures/nld_vs_rekv_accuracy_by_task.png
```

## Pair-Averaged Results

| Model setting | Method | Avg score | Token proxy | Time / sample | Peak memory |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B / Llama-3.1-8B | NLD | 0.1865 | 2773 | 2.6416s | 30.50GB |
| Llama-3.1-8B / Llama-3.1-8B | ReKV-w8 r=0.3 | 0.6135 | 3734 | 1.3689s | 31.92GB |
| Llama-3.1-8B / Llama-3.1-8B | B-ReKV | 0.6236 | 3735 | 1.2558s | 31.85GB |
| Llama-3.2-3B-Abliterated / DeepSeek-R1-3B | NLD | 0.1453 | 2656 | 1.4188s | 12.41GB |
| Llama-3.2-3B-Abliterated / DeepSeek-R1-3B | ReKV-w8 r=0.3 | 0.4894 | 2518 | 0.6504s | 13.23GB |
| Llama-3.2-3B-Abliterated / DeepSeek-R1-3B | B-ReKV | 0.5147 | 2518 | 0.6602s | 13.24GB |
| Qwen2.5-7B-Uncensored / Bespoke-Stratos-7B | NLD | 0.0916 | 2923 | 3.7957s | 28.86GB |
| Qwen2.5-7B-Uncensored / Bespoke-Stratos-7B | ReKV-w8 r=0.3 | 0.3524 | 2662 | 1.7988s | 29.50GB |
| Qwen2.5-7B-Uncensored / Bespoke-Stratos-7B | B-ReKV | 0.3822 | 2661 | 1.7303s | 29.49GB |

## Key Takeaways

- Accuracy is the main advantage: ReKV/B-ReKV is consistently much stronger than NLD on all three model settings.
- Latency is also favorable: NLD needs A phase-1 generation, B phase-1 generation, and B refinement generation, so it is roughly 2x slower than ReKV/B-ReKV on average.
- Peak memory is similar: NLD is slightly lower in peak memory because it does not inject KV payload, but the gap is small relative to the accuracy loss.
- Text payload tokens are tiny for NLD, but this is not an apples-to-apples advantage: NLD sends a short generated text answer that often loses evidence or introduces sender-side errors, while ReKV/B-ReKV sends selected latent KV evidence for the receiver to reason over.

## Paper Framing

This comparison should be framed as:

- Natural-language passing is a simple communication protocol, but it requires the sender to generate an intermediate answer and relies on that answer being faithful.
- ReKV/B-ReKV keeps the sender as a context-side memory server: the sender does not answer the user, but sends task-relevant latent evidence selected with the receiver query.
- The experiment supports the paper's setting: receiver-aware KV communication is more accurate and faster than natural-language answer passing, while preserving the receiver as the final reasoning/alignment model.
