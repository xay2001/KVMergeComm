# Pair #6/#7 Query Fairness + B-ReKV Summary

## pair6_llama32_ablate_deepseek3b

### hotpotqa

| Method | Param | Score | Avg budget |
|---|---:|---:|---:|
| Evict/ValueNorm | r0.3 | 0.516 | 0.325 |
| Random-token | r0.3 | 0.406 | 0.325 |
| ReKV-w8 | r0.3 | 0.646 | 0.325 |
| ReKV-w16 | r0.3 | 0.668 | 0.325 |
| B-ReKV-w8 | t0.95_s0.75 | 0.642 | 0.272 |
| B-ReKV-w8 | t0.95_s0.85 | 0.640 | 0.300 |
| B-ReKV-w16 | t0.95_s0.90 | 0.674 | 0.368 |

Best overall: B-ReKV-w16 t0.95_s0.90 score=0.674.  \nBest ReKV - Evict = +0.152.  \nBest ReKV - Random = +0.262.  \nBest B-ReKV: B-ReKV-w16 t0.95_s0.90 score=0.674, budget=0.368.

### musique

| Method | Param | Score | Avg budget |
|---|---:|---:|---:|
| Evict/ValueNorm | r0.3 | 0.236 | 0.325 |
| Random-token | r0.3 | 0.182 | 0.325 |
| ReKV-w8 | r0.3 | 0.362 | 0.325 |
| ReKV-w16 | r0.3 | 0.354 | 0.325 |
| B-ReKV-w8 | t0.95_s0.75 | 0.384 | 0.251 |
| B-ReKV-w8 | t0.95_s0.85 | 0.370 | 0.279 |
| B-ReKV-w16 | t0.95_s0.90 | 0.368 | 0.360 |

Best overall: B-ReKV-w8 t0.95_s0.75 score=0.384.  \nBest ReKV - Evict = +0.126.  \nBest ReKV - Random = +0.180.  \nBest B-ReKV: B-ReKV-w8 t0.95_s0.75 score=0.384, budget=0.251.

### multifieldqa_en

| Method | Param | Score | Avg budget |
|---|---:|---:|---:|
| Evict/ValueNorm | r0.3 | 0.327 | 0.325 |
| Random-token | r0.3 | 0.320 | 0.325 |
| ReKV-w8 | r0.3 | 0.467 | 0.325 |
| ReKV-w16 | r0.3 | 0.467 | 0.325 |
| B-ReKV-w8 | t0.95_s0.75 | 0.467 | 0.231 |
| B-ReKV-w8 | t0.95_s0.85 | 0.487 | 0.257 |
| B-ReKV-w16 | t0.95_s0.90 | 0.493 | 0.328 |

Best overall: B-ReKV-w16 t0.95_s0.90 score=0.493.  \nBest ReKV - Evict = +0.140.  \nBest ReKV - Random = +0.147.  \nBest B-ReKV: B-ReKV-w16 t0.95_s0.90 score=0.493, budget=0.328.

## pair7_qwen25_uncensored_bespoke

### hotpotqa

| Method | Param | Score | Avg budget |
|---|---:|---:|---:|
| Evict/ValueNorm | r0.3 | 0.124 | 0.325 |
| Random-token | r0.3 | 0.118 | 0.325 |
| ReKV-w8 | r0.3 | 0.396 | 0.325 |
| ReKV-w16 | r0.3 | 0.376 | 0.325 |
| B-ReKV-w8 | t0.95_s0.75 | 0.414 | 0.331 |
| B-ReKV-w8 | t0.95_s0.85 | 0.446 | 0.369 |
| B-ReKV-w16 | t0.95_s0.90 | 0.402 | 0.436 |

Best overall: B-ReKV-w8 t0.95_s0.85 score=0.446.  \nBest ReKV - Evict = +0.272.  \nBest ReKV - Random = +0.278.  \nBest B-ReKV: B-ReKV-w8 t0.95_s0.85 score=0.446, budget=0.369.

### musique

| Method | Param | Score | Avg budget |
|---|---:|---:|---:|
| Evict/ValueNorm | r0.3 | 0.144 | 0.325 |
| Random-token | r0.3 | 0.080 | 0.325 |
| ReKV-w8 | r0.3 | 0.288 | 0.325 |
| ReKV-w16 | r0.3 | 0.298 | 0.325 |
| B-ReKV-w8 | t0.95_s0.75 | 0.300 | 0.292 |
| B-ReKV-w8 | t0.95_s0.85 | 0.308 | 0.326 |
| B-ReKV-w16 | t0.95_s0.90 | 0.300 | 0.412 |

Best overall: B-ReKV-w8 t0.95_s0.85 score=0.308.  \nBest ReKV - Evict = +0.154.  \nBest ReKV - Random = +0.218.  \nBest B-ReKV: B-ReKV-w8 t0.95_s0.85 score=0.308, budget=0.326.

### multifieldqa_en

| Method | Param | Score | Avg budget |
|---|---:|---:|---:|
| Evict/ValueNorm | r0.3 | 0.080 | 0.325 |
| Random-token | r0.3 | 0.140 | 0.325 |
| ReKV-w8 | r0.3 | 0.380 | 0.325 |
| ReKV-w16 | r0.3 | 0.393 | 0.325 |
| B-ReKV-w8 | t0.95_s0.75 | 0.393 | 0.209 |
| B-ReKV-w8 | t0.95_s0.85 | 0.380 | 0.232 |
| B-ReKV-w16 | t0.95_s0.90 | 0.373 | 0.309 |

Best overall: ReKV-w16 r0.3 score=0.393.  \nBest ReKV - Evict = +0.313.  \nBest ReKV - Random = +0.253.  \nBest B-ReKV: B-ReKV-w8 t0.95_s0.75 score=0.393, budget=0.209.

