# Query-Sketch B-ReKV 配置冻结报告

验收标准：6 个 pair-task 单元齐全；matched-budget 平均分差不低于 -0.01；最差单元不低于 -0.05；至少 4/6 单元在 0.005 容差内持平或获胜。

**冻结配置：`B-ReKV-t0.98-s0.95-w8`。**

- 平均实际预算：0.5698
- 相对 matched-budget fixed ReKV 平均分差：+0.0267
- 最差单元分差：+0.0000
- 持平/获胜：6/6

## 候选汇总

| 配置 | 单元 | 平均预算 | 平均分差 | 最差分差 | 持平/胜 | 通过 |
|---|---:|---:|---:|---:|---:|:---:|
| B-ReKV-t0.9-s0.95-w8 | 6 | 0.2173 | -0.0654 | -0.1200 | 1/6 | 否 |
| B-ReKV-t0.95-s0.85-w8 | 6 | 0.3297 | +0.0197 | -0.0168 | 4/6 | 是 |
| B-ReKV-t0.95-s0.95-w8 | 6 | 0.3638 | +0.0099 | -0.0226 | 4/6 | 是 |
| B-ReKV-t0.95-s1-w8 | 6 | 0.3810 | -0.0097 | -0.0388 | 3/6 | 否 |
| B-ReKV-t0.98-s0.95-w8 | 6 | 0.5698 | +0.0267 | +0.0000 | 6/6 | 是 |
| B-ReKV-t0.98-s1-w8 | 6 | 0.5917 | +0.0217 | +0.0000 | 6/6 | 是 |

## 逐单元结果

- B-ReKV-t0.90-s0.95-w8 / pair1_llama31_same / hotpotqa: budget=0.2025, score=0.4100, matched=0.5000, delta=-0.0900
- B-ReKV-t0.90-s0.95-w8 / pair1_llama31_same / multifieldqa_en: budget=0.2477, score=0.4500, matched=0.4547, delta=-0.0047
- B-ReKV-t0.90-s0.95-w8 / pair1_llama31_same / musique: budget=0.1823, score=0.2500, matched=0.3200, delta=-0.0700
- B-ReKV-t0.90-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.1986, score=0.4100, matched=0.5300, delta=-0.1200
- B-ReKV-t0.90-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.3130, score=0.4600, matched=0.5175, delta=-0.0575
- B-ReKV-t0.90-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.1601, score=0.2300, matched=0.2800, delta=-0.0500
- B-ReKV-t0.95-s0.85-w8 / pair1_llama31_same / hotpotqa: budget=0.3045, score=0.5900, matched=0.5983, delta=-0.0083
- B-ReKV-t0.95-s0.85-w8 / pair1_llama31_same / multifieldqa_en: budget=0.3726, score=0.4700, matched=0.4614, delta=+0.0086
- B-ReKV-t0.95-s0.85-w8 / pair1_llama31_same / musique: budget=0.3059, score=0.3500, matched=0.3668, delta=-0.0168
- B-ReKV-t0.95-s0.85-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.2775, score=0.5900, matched=0.5615, delta=+0.0285
- B-ReKV-t0.95-s0.85-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.4392, score=0.5100, matched=0.4800, delta=+0.0300
- B-ReKV-t0.95-s0.85-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.2783, score=0.4100, matched=0.3335, delta=+0.0765
- B-ReKV-t0.95-s0.95-w8 / pair1_llama31_same / hotpotqa: budget=0.3358, score=0.6200, matched=0.6426, delta=-0.0226
- B-ReKV-t0.95-s0.95-w8 / pair1_llama31_same / multifieldqa_en: budget=0.4128, score=0.4700, matched=0.4863, delta=-0.0163
- B-ReKV-t0.95-s0.95-w8 / pair1_llama31_same / musique: budget=0.3380, score=0.3900, matched=0.3866, delta=+0.0034
- B-ReKV-t0.95-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.3030, score=0.6400, matched=0.6037, delta=+0.0363
- B-ReKV-t0.95-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.4867, score=0.5000, matched=0.4800, delta=+0.0200
- B-ReKV-t0.95-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.3063, score=0.4300, matched=0.3915, delta=+0.0385
- B-ReKV-t0.95-s1.00-w8 / pair1_llama31_same / hotpotqa: budget=0.3517, score=0.6300, matched=0.6688, delta=-0.0388
- B-ReKV-t0.95-s1.00-w8 / pair1_llama31_same / multifieldqa_en: budget=0.4328, score=0.4900, matched=0.4900, delta=+0.0000
- B-ReKV-t0.95-s1.00-w8 / pair1_llama31_same / musique: budget=0.3541, score=0.3600, matched=0.3933, delta=-0.0333
- B-ReKV-t0.95-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.3163, score=0.6200, matched=0.6257, delta=-0.0057
- B-ReKV-t0.95-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.5104, score=0.5000, matched=0.4800, delta=+0.0200
- B-ReKV-t0.95-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.3205, score=0.4200, matched=0.4207, delta=-0.0007
- B-ReKV-t0.98-s0.95-w8 / pair1_llama31_same / hotpotqa: budget=0.5487, score=0.7300, matched=0.7200, delta=+0.0100
- B-ReKV-t0.98-s0.95-w8 / pair1_llama31_same / multifieldqa_en: budget=0.6025, score=0.5500, matched=0.4900, delta=+0.0600
- B-ReKV-t0.98-s0.95-w8 / pair1_llama31_same / musique: budget=0.5554, score=0.4400, matched=0.4400, delta=+0.0000
- B-ReKV-t0.98-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.5193, score=0.7400, matched=0.7000, delta=+0.0400
- B-ReKV-t0.98-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.6562, score=0.4800, matched=0.4800, delta=+0.0000
- B-ReKV-t0.98-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.5366, score=0.4600, matched=0.4100, delta=+0.0500
- B-ReKV-t0.98-s1.00-w8 / pair1_llama31_same / hotpotqa: budget=0.5709, score=0.7300, matched=0.7200, delta=+0.0100
- B-ReKV-t0.98-s1.00-w8 / pair1_llama31_same / multifieldqa_en: budget=0.6263, score=0.5100, matched=0.4900, delta=+0.0200
- B-ReKV-t0.98-s1.00-w8 / pair1_llama31_same / musique: budget=0.5793, score=0.4400, matched=0.4400, delta=+0.0000
- B-ReKV-t0.98-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.5405, score=0.7700, matched=0.7000, delta=+0.0700
- B-ReKV-t0.98-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.6727, score=0.4800, matched=0.4800, delta=+0.0000
- B-ReKV-t0.98-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.5604, score=0.4400, matched=0.4100, delta=+0.0300
