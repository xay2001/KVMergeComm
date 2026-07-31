# Model-pair compatibility diagnostic

> **Scope warning:** this is an exploratory n=8 diagnostic (7 established pairs + the historical pair9 stress case). It must not be described as a validated or general predictor.

Tasks: `hotpotqa, musique, multifieldqa_en`. Existing Full-KV coverage: 2/8 pairs. Attention calibration coverage: 0/8 pairs.

| Pair | Known outcome | Full-KV transfer | ReKV w8/r0.3 | Nonzero rate | ReKV/Full-KV | Config match | Attn Spearman | QK-logit cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pair1_llama31_same | successful | 0.6245 | 0.4944 | 0.4948 | 0.7870 | 1 | NA | NA |
| pair2_llama32_same | successful | NA | 0.4384 | 0.4217 | NA | NA | NA | NA |
| pair3_qwen25_7b_same | successful | NA | 0.4051 | 0.3843 | NA | NA | NA | NA |
| pair4_falcon3_7b_same | successful | NA | 0.3731 | 0.3730 | NA | NA | NA | NA |
| pair5_evolcodellama_toolace | successful | NA | 0.4144 | 0.3904 | NA | NA | NA | NA |
| pair6_llama32_abliterated_deepseek3b | successful | NA | 0.4967 | 0.4835 | NA | NA | NA | NA |
| pair7_qwen25_uncensored_bespoke | successful | 0.4189 | 0.3160 | 0.2965 | 0.7485 | NA | NA | NA |
| pair9_supernova_deepseek_llama8b | historical_hard_negative_corrected | NA | 0.3233 | 0.3365 | NA | NA | NA | NA |

## Pair9 qualification

Pair9's corrected ReKV macro is `0.3233` with nonzero rate `0.3365`. The old near-zero batch was affected by DeepSeek think-prefix handling. Therefore pair9 is a historical stress case, but the corrected artifacts do not establish a clean binary negative label for this diagnostic.

## Metric definitions

- **Full-KV transfer:** mean task score from an uncompressed sender-KV transfer run.
- **ReKV availability target:** canonical `w=8, r=0.3` score and sample-level nonzero-score rate. Recovery is computed only on tasks with a matching Full-KV run.
- **Config match:** fraction of structural config fields that match; absent local configs remain `NA` rather than being inferred from model names.
- **Attention agreement:** mean layer/sample Spearman agreement and top-k Jaccard between receiver and sender last-query attention rankings.
- **QK-logit cosine:** cosine between centered `log(attention)` vectors. This is a softmax-offset-invariant proxy for corresponding-layer query-key score stability, not direct Q/K weight CKA.

Missing values mean the corresponding GPU calibration has not been run. No missing model-derived metric is imputed.
