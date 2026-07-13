# Query-Sketch B-ReKV 配置冻结报告

验收标准：6 个 pair-task 单元齐全；matched-budget 平均分差不低于 -0.01；最差单元不低于 -0.05；至少 4/6 单元在 0.005 容差内持平或获胜。

**冻结配置：`B-ReKV-t0.98-s1-w8`。**

- 平均实际预算：0.5919
- 相对 matched-budget fixed ReKV 平均分差：+0.0250
- 最差单元分差：+0.0100
- 持平/获胜：6/6

## 候选汇总

| 配置 | 单元 | 平均预算 | 平均分差 | 最差分差 | 持平/胜 | 通过 |
|---|---:|---:|---:|---:|---:|:---:|
| B-ReKV-t0.9-s0.95-w8 | 6 | 0.2175 | -0.0704 | -0.1200 | 1/6 | 否 |
| B-ReKV-t0.95-s0.85-w8 | 6 | 0.3298 | +0.0197 | -0.0074 | 5/6 | 是 |
| B-ReKV-t0.95-s0.95-w8 | 6 | 0.3640 | +0.0181 | -0.0068 | 5/6 | 是 |
| B-ReKV-t0.95-s1-w8 | 6 | 0.3812 | +0.0036 | -0.0200 | 4/6 | 是 |
| B-ReKV-t0.98-s0.95-w8 | 6 | 0.5700 | +0.0217 | -0.0200 | 4/6 | 是 |
| B-ReKV-t0.98-s1-w8 | 6 | 0.5919 | +0.0250 | +0.0100 | 6/6 | 是 |

## 逐单元结果

- B-ReKV-t0.90-s0.95-w8 / pair1_llama31_same / hotpotqa: budget=0.2028, score=0.3900, matched=0.5000, delta=-0.1100
- B-ReKV-t0.90-s0.95-w8 / pair1_llama31_same / multifieldqa_en: budget=0.2483, score=0.4500, matched=0.4548, delta=-0.0048
- B-ReKV-t0.90-s0.95-w8 / pair1_llama31_same / musique: budget=0.1827, score=0.2500, matched=0.3200, delta=-0.0700
- B-ReKV-t0.90-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.1986, score=0.4100, matched=0.5300, delta=-0.1200
- B-ReKV-t0.90-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.3125, score=0.4400, matched=0.5174, delta=-0.0774
- B-ReKV-t0.90-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.1598, score=0.2400, matched=0.2800, delta=-0.0400
- B-ReKV-t0.95-s0.85-w8 / pair1_llama31_same / hotpotqa: budget=0.3056, score=0.6000, matched=0.5995, delta=+0.0005
- B-ReKV-t0.95-s0.85-w8 / pair1_llama31_same / multifieldqa_en: budget=0.3733, score=0.4700, matched=0.4619, delta=+0.0081
- B-ReKV-t0.95-s0.85-w8 / pair1_llama31_same / musique: budget=0.3066, score=0.3600, matched=0.3674, delta=-0.0074
- B-ReKV-t0.95-s0.85-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.2770, score=0.5900, matched=0.5606, delta=+0.0294
- B-ReKV-t0.95-s0.85-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.4388, score=0.5100, matched=0.4800, delta=+0.0300
- B-ReKV-t0.95-s0.85-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.2777, score=0.3900, matched=0.3323, delta=+0.0577
- B-ReKV-t0.95-s0.95-w8 / pair1_llama31_same / hotpotqa: budget=0.3370, score=0.6500, matched=0.6446, delta=+0.0054
- B-ReKV-t0.95-s0.95-w8 / pair1_llama31_same / multifieldqa_en: budget=0.4136, score=0.4800, matched=0.4868, delta=-0.0068
- B-ReKV-t0.95-s0.95-w8 / pair1_llama31_same / musique: budget=0.3388, score=0.4100, matched=0.3870, delta=+0.0230
- B-ReKV-t0.95-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.3024, score=0.6200, matched=0.6027, delta=+0.0173
- B-ReKV-t0.95-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.4863, score=0.5100, matched=0.4800, delta=+0.0300
- B-ReKV-t0.95-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.3057, score=0.4300, matched=0.3902, delta=+0.0398
- B-ReKV-t0.95-s1.00-w8 / pair1_llama31_same / hotpotqa: budget=0.3530, score=0.6700, matched=0.6709, delta=-0.0009
- B-ReKV-t0.95-s1.00-w8 / pair1_llama31_same / multifieldqa_en: budget=0.4337, score=0.4700, matched=0.4900, delta=-0.0200
- B-ReKV-t0.95-s1.00-w8 / pair1_llama31_same / musique: budget=0.3550, score=0.4000, matched=0.3936, delta=+0.0064
- B-ReKV-t0.95-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.3156, score=0.6300, matched=0.6246, delta=+0.0054
- B-ReKV-t0.95-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.5100, score=0.5200, matched=0.4800, delta=+0.0400
- B-ReKV-t0.95-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.3198, score=0.4100, matched=0.4193, delta=-0.0093
- B-ReKV-t0.98-s0.95-w8 / pair1_llama31_same / hotpotqa: budget=0.5498, score=0.7100, matched=0.7200, delta=-0.0100
- B-ReKV-t0.98-s0.95-w8 / pair1_llama31_same / multifieldqa_en: budget=0.6032, score=0.5500, matched=0.4900, delta=+0.0600
- B-ReKV-t0.98-s0.95-w8 / pair1_llama31_same / musique: budget=0.5563, score=0.4200, matched=0.4400, delta=-0.0200
- B-ReKV-t0.98-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.5185, score=0.7500, matched=0.7000, delta=+0.0500
- B-ReKV-t0.98-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.6560, score=0.4900, matched=0.4800, delta=+0.0100
- B-ReKV-t0.98-s0.95-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.5360, score=0.4500, matched=0.4100, delta=+0.0400
- B-ReKV-t0.98-s1.00-w8 / pair1_llama31_same / hotpotqa: budget=0.5721, score=0.7300, matched=0.7200, delta=+0.0100
- B-ReKV-t0.98-s1.00-w8 / pair1_llama31_same / multifieldqa_en: budget=0.6270, score=0.5200, matched=0.4900, delta=+0.0300
- B-ReKV-t0.98-s1.00-w8 / pair1_llama31_same / musique: budget=0.5802, score=0.4500, matched=0.4400, delta=+0.0100
- B-ReKV-t0.98-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / hotpotqa: budget=0.5398, score=0.7700, matched=0.7000, delta=+0.0700
- B-ReKV-t0.98-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / multifieldqa_en: budget=0.6725, score=0.4900, matched=0.4800, delta=+0.0100
- B-ReKV-t0.98-s1.00-w8 / pair6_llama32_abliterated_deepseek3b / musique: budget=0.5597, score=0.4300, matched=0.4100, delta=+0.0200
