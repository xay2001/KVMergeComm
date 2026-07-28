# B-ReKV held-out 与 shuffled 分析

## 口径

- Development cells 固定为 pair1/hotpotqa 与 pair1/musique；其余 54 cells 为 held-out。
- Development 选择的 fixed ratio：`0.6`（从原 sweep 的 9 档选择；均分优先，实际预算与 ratio 依次破同分）。
- Canonical B-ReKV：t=.95/s=.75/w8；所有汇总均按 cell 等权。
- Per-task Best Fixed 在 held-out 上按任务选择，是 oracle 上界，不是可部署选择。
- Oracle 使用 exact grid .1/.2/.3/.4/.5/.6/.7，成功定义为 score>=1，取成功点中最小实际预算。
- Easy=oracle ratio<=.3，hard>=.5，.4 为 medium；预算 bins 按实际预算 .2/.3/.4/.5/.6 切分。
- CI 为 cell 内严格配对样本 bootstrap 后的 54-cell 等权 percentile 95% CI。

## Held-out policies

- B-ReKV: score=0.399699, budget=0.323240, regret=0.161827, B-ReKV wins=0/8, Δ(B-ReKV-policy)=+0.000000 [+0.000000, +0.000000]
- Fixed 0.2: score=0.330248, budget=0.231308, regret=0.282323, B-ReKV wins=7/8, Δ(B-ReKV-policy)=+0.069451 [+0.063938, +0.075066]
- Fixed 0.3: score=0.393204, budget=0.324504, regret=0.163962, B-ReKV wins=2/8, Δ(B-ReKV-policy)=+0.006496 [+0.001642, +0.011357]
- Fixed 0.4: score=0.446100, budget=0.420764, regret=0.066472, B-ReKV wins=0/8, Δ(B-ReKV-policy)=-0.046400 [-0.051444, -0.041626]
- Fixed 0.5: score=0.467735, budget=0.517268, regret=0.022706, B-ReKV wins=0/8, Δ(B-ReKV-policy)=-0.068036 [-0.073366, -0.062698]
- Fixed 0.6: score=0.479829, budget=0.613793, regret=0.000243, B-ReKV wins=0/8, Δ(B-ReKV-policy)=-0.080130 [-0.085763, -0.074564]
- Dev-selected Fixed: score=0.479829, budget=0.613793, regret=0.000243, B-ReKV wins=0/8, Δ(B-ReKV-policy)=-0.080130 [-0.085840, -0.074550]
- Per-task Best Fixed: score=0.479952, budget=0.588761, regret=0.000000, B-ReKV wins=0/8, Δ(B-ReKV-policy)=-0.080253 [-0.085605, -0.074705]

## Exact oracle

- Solved=10420/18950，Spearman=0.0351，oracle budget=0.283124。
- Easy/medium/hard/unsolved=0.4140/0.0643/0.0716/0.4501。

## Shuffled budget

- Δscore=+0.007084 [+0.002373, +0.011764]，P(Δ>0)=0.9987。
- 54 cells 预算 multiset 校验通过；最大排序后误差 0。
