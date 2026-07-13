# Stage 3 核心审稿证据

- 完成 runs：360
- 协议标识：query_agnostic_kv_v1, query_sketch_bf16_v1
- matched-budget 比较：27/27 可插值
- 超出 fixed-r 网格：0
- B-ReKV Pareto 点：93
- 预算分布单元：9
- Strict 对比单元：9
- PNG 图：未生成（缺少 matplotlib 或无数据）

## Matched-budget fairness

- pair1_llama31_same / hotpotqa / ReKV: budget=0.2754, delta=-0.0122, status=matched
- pair1_llama31_same / hotpotqa / ValueNorm/Evict: budget=0.2754, delta=+0.0367, status=matched
- pair1_llama31_same / hotpotqa / Random: budget=0.2754, delta=+0.1992, status=matched
- pair1_llama31_same / multifieldqa_en / ReKV: budget=0.3354, delta=+0.0067, status=matched
- pair1_llama31_same / multifieldqa_en / ValueNorm/Evict: budget=0.3354, delta=+0.1715, status=matched
- pair1_llama31_same / multifieldqa_en / Random: budget=0.3354, delta=+0.1003, status=matched
- pair1_llama31_same / musique / ReKV: budget=0.2745, delta=-0.0345, status=matched
- pair1_llama31_same / musique / ValueNorm/Evict: budget=0.2745, delta=+0.0234, status=matched
- pair1_llama31_same / musique / Random: budget=0.2745, delta=+0.1594, status=matched
- pair6_llama32_abliterated_deepseek3b / hotpotqa / ReKV: budget=0.2530, delta=-0.0472, status=matched
- pair6_llama32_abliterated_deepseek3b / hotpotqa / ValueNorm/Evict: budget=0.2530, delta=+0.0575, status=matched
- pair6_llama32_abliterated_deepseek3b / hotpotqa / Random: budget=0.2530, delta=+0.1478, status=matched
- pair6_llama32_abliterated_deepseek3b / multifieldqa_en / ReKV: budget=0.3948, delta=+0.0023, status=matched
- pair6_llama32_abliterated_deepseek3b / multifieldqa_en / ValueNorm/Evict: budget=0.3948, delta=+0.1131, status=matched
- pair6_llama32_abliterated_deepseek3b / multifieldqa_en / Random: budget=0.3948, delta=+0.1030, status=matched
- pair6_llama32_abliterated_deepseek3b / musique / ReKV: budget=0.2519, delta=+0.0263, status=matched
- pair6_llama32_abliterated_deepseek3b / musique / ValueNorm/Evict: budget=0.2519, delta=+0.1115, status=matched
- pair6_llama32_abliterated_deepseek3b / musique / Random: budget=0.2519, delta=+0.1725, status=matched
- pair7_qwen25_uncensored_bespoke / hotpotqa / ReKV: budget=0.3853, delta=-0.0326, status=matched
- pair7_qwen25_uncensored_bespoke / hotpotqa / ValueNorm/Evict: budget=0.3853, delta=+0.1827, status=matched
- pair7_qwen25_uncensored_bespoke / hotpotqa / Random: budget=0.3853, delta=+0.2059, status=matched
- pair7_qwen25_uncensored_bespoke / multifieldqa_en / ReKV: budget=0.3585, delta=-0.0759, status=matched
- pair7_qwen25_uncensored_bespoke / multifieldqa_en / ValueNorm/Evict: budget=0.3585, delta=+0.2189, status=matched
- pair7_qwen25_uncensored_bespoke / multifieldqa_en / Random: budget=0.3585, delta=+0.1569, status=matched
- pair7_qwen25_uncensored_bespoke / musique / ReKV: budget=0.3441, delta=-0.0187, status=matched
- pair7_qwen25_uncensored_bespoke / musique / ValueNorm/Evict: budget=0.3441, delta=+0.0701, status=matched
- pair7_qwen25_uncensored_bespoke / musique / Random: budget=0.3441, delta=+0.1296, status=matched

## Budget distribution

- pair1_llama31_same / hotpotqa: mean=0.2754, std=0.0202, p10-p90=0.2487-0.3000, unique=486
- pair1_llama31_same / multifieldqa_en: mean=0.3354, std=0.0331, p10-p90=0.2928-0.3737, unique=149
- pair1_llama31_same / musique: mean=0.2745, std=0.0243, p10-p90=0.2427-0.3028, unique=500
- pair6_llama32_abliterated_deepseek3b / hotpotqa: mean=0.2530, std=0.0211, p10-p90=0.2236-0.2791, unique=478
- pair6_llama32_abliterated_deepseek3b / multifieldqa_en: mean=0.3948, std=0.0466, p10-p90=0.3395-0.4457, unique=149
- pair6_llama32_abliterated_deepseek3b / musique: mean=0.2519, std=0.0259, p10-p90=0.2183-0.2823, unique=499
- pair7_qwen25_uncensored_bespoke / hotpotqa: mean=0.3853, std=0.0240, p10-p90=0.3540-0.4129, unique=488
- pair7_qwen25_uncensored_bespoke / multifieldqa_en: mean=0.3585, std=0.0420, p10-p90=0.3044-0.4018, unique=149
- pair7_qwen25_uncensored_bespoke / musique: mean=0.3441, std=0.0304, p10-p90=0.3094-0.3797, unique=496
