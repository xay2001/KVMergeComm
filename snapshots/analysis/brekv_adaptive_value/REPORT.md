# B-ReKV 自适应预算价值分析

## 口径

- 输入：完整 1568-run matched-budget sweep；分析仅加载 canonical B-ReKV 与 9 档 fixed ReKV 的 per-sample 数据。
- fixed budget matching：在实际预算轴上对相邻 fixed ReKV 档位做样本级线性插值。
- `per_cell_best_fixed_unmatched` 是额外的强静态上界：每个 cell 取均分最高的 fixed 档，不保证预算匹配，预算差单独报告。
- Oracle solved：9 档中存在 `score >= 1.0`；需要预算取所有成功档中的最小实际预算。
- Capped sensitivity：未解样本的未知需求封顶为该样本 9 档中最大的实际预算；这是保守的可观测上限敏感性分析。
- CI：样本严格配对 bootstrap；task/macro 在 cell 内重采样后等权分层聚合，percentile 95% CI。
- Spearman task/macro：先按 cell 计算 rho，再对 cell 等权宏平均；不把 pair/task 的预算尺度差异混入 pooled rho。
- 上下文长度：不加载 tokenizer/model，使用实际 dataloader `prompt_A` 的字符数与空白分词数。

## Fixed-policy 对比

- 全局 budget-matched fixed：Δscore=+0.005825，95% CI [+0.001050, +0.010581]，P(Δ>0)=0.9923
- 每任务 budget-matched fixed：Δscore=+0.007577，95% CI [+0.003087, +0.012043]，P(Δ>0)=0.9992
- 每 cell budget-matched fixed：Δscore=+0.003587，95% CI [-0.000655, +0.007805]，P(Δ>0)=0.9501
- 每 cell best fixed（非预算匹配）：Δscore=-0.088923，95% CI [-0.094296, -0.083540]，P(Δ>0)=0.0000

## Oracle 与相关性

- 总样本-cell 观测：19950；未解 8932 (44.77%)。
- Oracle needed budget（仅已解）均值：0.251790；封顶敏感性均值：0.413852。
- 每-cell best fixed 的平均预算差（B-ReKV - fixed）：-0.224185。
- solved_only：rho=+0.0853，95% CI [+0.0625, +0.1069]。
- unsolved_capped_at_observed_max：rho=+0.0527，95% CI [+0.0369, +0.0681]。

## 聚焦元数据

- hotpotqa：已提取。
- musique：已提取。
- multifieldqa_en：已提取。
- HotpotQA 可提取 supporting-fact/evidence count，但源元数据没有显式 hop count，故 hop 为 NA。
- MuSiQue 从 question decomposition 与 supporting paragraphs 提取 hop/evidence count。
- MultiFieldQA-en 源元数据无 hop/evidence 字段，二者均为 NA。

## 图

- `/home/xay/KVMergeComm/snapshots/analysis/brekv_adaptive_value/brekv_adaptive_value.png`
- `/home/xay/KVMergeComm/snapshots/analysis/brekv_adaptive_value/brekv_adaptive_value.pdf`
