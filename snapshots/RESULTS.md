# KVComm / ReKV 实验方案、结果与后续计划总文档

> **2026-07-13 协议区分说明：** 本文件包含大量历史实验记录，其中早期
> `score_mode=receiver` 结果属于 `legacy_implicit_receiver_v0`，不能自动
> 解释为当前可部署 Query-Sketch。最新完成度、冻结配置、Stage 3 结果及
> Table 6/8/10 新旧协议边界统一以
> `snapshots/analysis/experiment_inventory_20260713.md` 为准。

> 当前维护入口（2026-06-30 新增）：
> - `snapshots/INDEX.md`：按论文 Table / model pair / dataset / method 检查当前结果与缺口。
> - `snapshots/PAIR_REGISTRY.md`：KVComm 论文 pair 与本地 snapshot root 的映射。
> - `snapshots/manifest/experiments.csv` / `experiments.json`：由 `scripts/build_experiment_manifest.py` 从 `snapshots/**/log.log` 自动生成的机器可读 run 索引。
>
> 本文件继续作为叙事型总记录与历史结果汇总；新增实验优先更新 manifest / INDEX，再把关键结论写回本文。

> 本文档由以下四份 Markdown 合并整理而成，并作为 `snapshots/` 下唯一保留的实验总文档：
> `RESULTS.md`、`KVCOMM_PAPER_EXPERIMENT_PLAN.md`、`EXPERIMENT_RUN_STATUS_2026-06-25.md`、`EXPERIMENT_MAP_AND_TODO.md`。
>
> 合并原则：实验数据、表格数值、运行状态、脚本名、路径和 TODO 原样保留；只新增统一目录和分区说明，方便按“方案 → 结果 → 运行状态 → 后续计划”的顺序阅读。

## 阅读顺序

1. **第一部分：研究主线、方法与完整实验结果**  
   汇总 ReKV / B-ReKV 的方法定义、主要对照表、各数据集结果、Budget-aware 正负实验链路、局限与脚本说明。
2. **第二部分：KVComm 原论文实验版图与 ReKV 跟跑计划**  
   对齐 KVComm 原论文 Table / Figure / model pair / dataset 的实验矩阵，明确哪些 baseline 直接引用、哪些 ReKV 实验需要补跑。
3. **第三部分：实验运行状态（2026-06-25）**  
   记录当日 GPU 队列、已完成任务、正在运行任务和下一步检查命令。
4. **第四部分：实验地图与后续 TODO**  
   用更工程化的视角列出当前模型、数据集、方法清单、论文表格规划、脚本目录和后续 TODO。

---

## 合并来源 1：第一部分：研究主线、方法与完整实验结果

> 原文件：`snapshots/RESULTS.md`。以下为该文件正文内容，实验数值未改动。

# KVComm → ReKV 实验汇总

> 方法:**ReKV(Receiver-aware KV Communication)** —— 在 LLM 间用 KV cache 直接通信的场景下,
> 由发送方 A 依据**接收方 B 的问题注意力**,对每个 token 的 KV 打分,只挑高分 token 传输。
> 模型:Llama-3.1-8B-Instruct → Llama-3.1-8B-Instruct(同模型)。指标:F1 / EM(各数据集自带)。

---

## 0. 研究主线与创新点定位(论文骨架)

### 0.1 问题定位

在 **LLM 间 latent communication** 中,KV-cache 是信息最丰富、但 payload 最大、架构依赖最强的通道(survey 三轴:**WHAT**=传什么 / **WHICH**=传哪些层 / **HOW**=怎么注入)。survey 明确把 **KV-cache 通信的压缩/量化** 列为 open problem。本工作落在"**WHAT=KV、压缩**"这一格,且强调**通信场景特有**的杠杆(接收方已知 query),而非通用单模型 cache eviction。

- 基线 **KVComm**(ICLR):**WHICH** 轴做文章——选部分**层**整层传,校准一次写死,query 无关。
- 本工作把战场从"选哪些层"换到"**按接收方需求,选哪些 token、给多少预算**"。

### 0.2 研究主线(两阶段)

```
阶段 A:ReKV —— 把"层选择"升级为"接收方感知的 token 选择"        [§1–§4, 已验证 ✓]
   KVComm(层级丢弃) → evict(token 级,value-norm) → ReKV(token 级,receiver 打分 + 观测窗口)
   结论:同预算下 ReKV > evict > merge;难任务上 token 级 ≫ 层级;merge 无益("选对" > "合并")

阶段 B:Budget-aware —— 把"固定保留比"升级为"按 query 自适应预算"  [§5–§10, 路线修正后成立 ✓]
   Step 0  验证前提:每条 query 的最优预算大幅波动,固定 r 严重过供(理论 64–67% headroom)  → 前提成立
   Step 1  开环预测:发送方侧统计量(熵/层重要性)预测不出 per-query 预算            → 证伪
   Step 2a 离线上界:oracle-stop 渐进理论可省 35–47% 预算(但这是完美停止信号的上界)   → 仅上界
   Step 2b 在线渐进:learned controller 实测,只有 hotpotqa 中段微正,musique/2wiki 负   → 证伪 + 多轮开销
   牌2    单发预测:Pass-1 特征单发预测最优预算,LODO AUC 0.585、等精度全程负节省      → 证伪
   Step 3  B-ReKV:不预测 r,改用 receiver-attention coverage/fidelity 阈值自动定预算 → 在线实测正收益

  → 关键转折:直接预测 query difficulty / budget 不稳定,但把预算定义为"接收方证据覆盖率"后,
    budget control 从不可靠预测问题变成可解释 fidelity constraint。MuSiQue 的 w8 coverage 形成强 Pareto:
    15.8% 平均预算即可达到 0.442(+40.7% 等精度省预算),27.9% 平均预算达到 0.482,
    超过 fixed-r 最高精度。
```

### 0.3 创新点定位(随实验修正)

1. **Receiver-Aware KV Communication(ReKV,主)**:利用接收方 B 的 query sketch(问题末 N 词的注意力)指导发送方 A **在 token 粒度**选择性传输 KV。定位为 **selection**(选对 token),并实验证伪 cache-merging 的必要性。这是已坐实的核心贡献。
2. **机制分析(merge vs evict)**:系统对比表明在通信压缩场景"**选对 token**" > "合并被丢的 token",纠正了 CaM 式 cache-merging 的迁移假设。
3. **B-ReKV(阶段 B 正向抓手,§10)**:直接预测 query 难度失败后,提出 training-free 的 **receiver evidence coverage** 预算控制:保留最少 KV token,使其覆盖接收方 query attention mass 的目标比例(90/95%)。这把预算从"手动固定 r"改为"可解释的证据保真度阈值",在线实测在 MuSiQue/HotpotQA 均有正收益。
4. **Budget-aware negative-to-positive 机制链**:先系统证伪 entropy/layer/predictor/progressive 四条不稳定路线(§6–§9),再用 DBudgetKV/GVote 启发的 coverage/fidelity threshold 找到可行路线(§10)。论文叙事从"预算预测难校准"自然转到"接收方证据保真度约束"。

---

## 1. 方法与对照

| 方法 | 压缩粒度 | 选择判据 | 是否 query 自适应 | 备注 |
|---|---|---|---|---|
| **KVComm**(基线) | 层级(丢整层) | B 注意力汇总成**每层一个标量** | 否,校准一次写死 | 原 ICLR 工作 |
| **merge** | token 级 + 融合 | value 向量 L2 范数(选)+ key 相似度(融) | 否(query 无关) | CaM 风格 |
| **evict** | token 级(只丢) | value 向量 L2 范数 | 否(query 无关) | SnapKV/H2O 风格 |
| **receiver (ReKV)** | token 级(只丢) | **B 问题末 N 词对每个 token 的注意力** | **是,每条 query 重算** | 本工作 |
| **B-ReKV** | token 级(动态预算) | receiver attention coverage:保留最少 token 覆盖 θ 比例的 B-query 注意力质量 | **是,每条 query / 每层动态预算** | 阶段 B 新方案 |

- 记号:token 方法的 `r` = **保留比例**(r=0.2 → 只留 20% token,压得最狠);KVComm 的 `top` = **保留层比例**。
- `recv_wN`:观测窗口 = 只用问题最后 N 个 token 的 query 打分(N∈{8,16})。
- 固定开销:第 0 层全量保留;每层至少保留 sink(4)+ recent(8)。
- B-ReKV 记号:`cov_t0.95_s0.8_w8` = 目标覆盖 95% receiver attention,预算乘 scale 0.8,观测窗口 8。

## 1.1 主对比表(各方法 @ 0.3 / 0.5 / 0.7 预算)

> 预算口径:KVComm(x)=保留 x 比例的**层**;merge/evict/ReKV(x)=保留 x 比例的**token**。两者都≈传输 x 比例的全量 KV,可近似等带宽对比。
> `–` 表示未跑:QASPER(进行中)、2WikiMQA / TMATH(未跑);HotpotQA 的 ReKV 仅跑到 r0.5。基线行(Baseline…KVComm)转录自既有总表。

| Method | Countries | Tipsheets | HotpotQA | QASPER | MuSiQue | MultiField-QA-en | 2WikiM-QA | TMATH |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.00 | 0.05 | 0.19 | 0.02 | 0.01 | 0.07 | 0.06 | 0.35 |
| Skyline | 0.62 | 0.92 | 0.74 | 0.35 | 0.54 | 0.56 | 0.52 | 0.36 |
| NLD | 0.58 | 0.87 | 0.52 | 0.13 | 0.25 | 0.17 | 0.10 | 0.36 |
| CIPHER | 0.57 | 0.84 | 0.57 | 0.13 | 0.15 | 0.15 | 0.10 | 0.36 |
| AC (mean) | 0.00 | 0.12 | 0.19 | 0.02 | 0.01 | 0.08 | 0.03 | 0.35 |
| AC (replace) | 0.00 | 0.36 | 0.15 | 0.02 | 0.01 | 0.07 | 0.05 | 0.35 |
| AC (sum) | 0.00 | 0.09 | 0.20 | 0.02 | 0.01 | 0.09 | 0.04 | 0.35 |
| KVComm (0.3) | 0.51 | 0.93 | 0.33 | 0.07 | 0.11 | 0.21 | 0.29 | 0.37 |
| merge (0.3) | 0.62 | 0.66 | 0.48 | – | 0.23 | 0.34 | – | – |
| evict (0.3) | 0.57 | 0.68 | 0.57 | – | 0.32 | 0.33 | – | – |
| **ReKV-w8 (0.3)** | 0.60 | 0.87 | 0.70 | – | **0.48** | 0.51 | – | – |
| **ReKV-w16 (0.3)** | 0.61 | 0.87 | **0.70** | – | 0.46 | 0.51 | – | – |
| KVComm (0.5) | 0.62 | 0.95 | 0.60 | 0.29 | 0.34 | 0.50 | 0.37 | 0.37 |
| merge (0.5) | 0.57 | 0.78 | 0.67 | – | 0.40 | 0.40 | – | – |
| evict (0.5) | 0.59 | 0.78 | 0.68 | – | 0.41 | 0.39 | – | – |
| **ReKV-w8 (0.5)** | 0.60 | 0.88 | 0.73 | – | **0.48** | 0.53 | – | – |
| **ReKV-w16 (0.5)** | 0.59 | 0.89 | **0.75** | – | 0.47 | 0.52 | – | – |
| KVComm (0.7) | 0.62 | 0.96 | 0.69 | 0.29 | 0.39 | 0.53 | 0.38 | 0.38 |
| merge (0.7) | 0.61 | 0.82 | 0.71 | – | 0.47 | 0.55 | – | – |
| evict (0.7) | 0.62 | 0.83 | 0.71 | – | 0.49 | 0.49 | – | – |
| **ReKV-w8 (0.7)** | 0.60 | 0.89 | – | – | **0.48** | 0.53 | – | – |
| **ReKV-w16 (0.7)** | 0.61 | 0.89 | – | – | 0.48 | 0.54 | – | – |

要点:同 token 预算下 **ReKV > evict > merge**(中压尤显);难任务(HotpotQA/MuSiQue)ReKV 在更低带宽即反超 KVComm;简单任务(Countries/Tipsheets)KVComm 丢层鲁棒、ReKV 不占优。

## 2. 各数据集结果(F1/EM,按保留预算对齐)

### 2.1 hotpotqa(多跳问答,难)

| 预算 | KVComm | merge | evict | recv_w8 | recv_w16 |
|---|---|---|---|---|---|
| 0.1 | – | 0.060 | 0.138 | 0.292 | 0.230 |
| 0.2 | – | 0.330 | 0.448 | **0.634** | 0.614 |
| 0.3 | 0.334 | 0.478 | 0.568 | 0.696 | **0.700** |
| 0.4 | – | 0.586 | 0.646 | 0.718 | **0.726** |
| 0.5 | 0.604 | 0.672 | 0.680 | 0.726 | **0.746** |
| 0.7 | 0.692 | 0.708 | 0.714 | – | – |
| 1.0 | 0.760 | – | – | – | – |

> 等预算 0.3:receiver 0.70 vs KVComm 0.334(2× 以上);留 50% token 即逼近全量 0.76。

### 2.2 musique(多跳问答,最难)

| 预算 | KVComm | merge | evict | recv_w8 | recv_w16 |
|---|---|---|---|---|---|
| 0.1 | – | 0.040 | 0.118 | **0.362** | 0.294 |
| 0.2 | – | 0.102 | 0.246 | **0.434** | 0.404 |
| 0.3 | 0.112 | 0.228 | 0.322 | **0.480** | 0.462 |
| 0.4 | – | 0.298 | 0.380 | **0.480** | 0.470 |
| 0.5 | 0.340 | 0.402 | 0.406 | **0.484** | 0.474 |
| 0.7 | 0.390 | 0.468 | 0.492 | 0.480 | 0.478 |
| 1.0 | 0.552 | – | – | – | – |

> receiver 在 r=0.3 即饱和(0.48,≈88% 全量);等预算 0.3 是 KVComm 的 **4.3×**。merge 全程低于 evict(合并有害)。

### 2.3 multifieldqa_en(单文档 QA,中等)

| 预算 | KVComm | merge | evict | recv_w8 | recv_w16 |
|---|---|---|---|---|---|
| 0.1 | – | 0.160 | 0.180 | 0.520 | **0.533** |
| 0.2 | – | 0.260 | 0.313 | 0.507 | 0.507 |
| 0.3 | 0.200 | 0.340 | 0.333 | 0.507 | 0.513 |
| 0.4 | – | 0.400 | 0.407 | 0.520 | 0.520 |
| 0.5 | 0.493 | 0.400 | 0.387 | 0.527 | 0.520 |
| 0.7 | 0.540 | 0.553 | 0.493 | 0.533 | 0.540 |
| 1.0 | 0.573 | – | – | – | – |

> receiver **在 r=0.1 就达到 ~0.53**(≈93% 全量 0.573),token 效率最夸张;evict/merge 在低压区远低于它。

### 2.4 tipsheets(简单,基线饱和)

| 预算 | KVComm | merge | evict | recv_w8 | recv_w16 |
|---|---|---|---|---|---|
| 0.1 | – | 0.400 | 0.374 | 0.416 | 0.400 |
| 0.2 | – | 0.556 | 0.580 | 0.858 | **0.870** |
| 0.3 | 0.920 | 0.656 | 0.680 | 0.868 | 0.874 |
| 0.4 | – | 0.756 | 0.772 | 0.890 | **0.896** |
| 0.5 | 0.946 | 0.784 | 0.776 | 0.876 | 0.888 |
| 0.7 | 0.952 | 0.818 | 0.826 | 0.892 | 0.894 |
| 1.0 | 0.942 | – | – | – | – |

> receiver ≫ merge/evict(r0.2:0.87 vs 0.58,+0.29);但 KVComm 丢层在此任务异常鲁棒(top0.3=0.92),等预算下 token 方法不占优 → 仅作 receiver-aware 消融。

### 2.5 countries(简单,基线饱和)

| 预算 | KVComm | merge | evict | recv_w8 | recv_w16 |
|---|---|---|---|---|---|
| 0.1 | – | 0.000 | 0.030 | 0.030 | 0.030 |
| 0.2 | – | 0.120 | 0.335 | 0.500 | **0.505** |
| 0.3 | 0.505 | 0.620 | 0.570 | 0.600 | 0.605 |
| 0.5 | 0.620 | 0.570 | 0.585 | 0.600 | 0.590 |
| 0.7 | 0.620 | 0.605 | 0.615 | 0.605 | 0.610 |
| 1.0 | 0.620 | – | – | – | – |

> r0.2 receiver 0.50 vs evict 0.335 vs merge 0.12;任务小且饱和(top0.3 已 0.505),区分度低。

### 2.6 qasper(科学论文 QA)

- 数据源已从废弃的 `tau/scrolls`(脚本式)切到 **LongBench qasper 配置**(parquet,200 条)。
- 实验**进行中**,暂无最终结果。

## 3. 跨数据集核心结论

1. **receiver-aware 是主杠杆**:所有数据集上 `receiver ≫ evict ≈ merge`,中压区(r0.2–0.4)增益最大(+0.1~0.3)。
2. **难任务上 token 级 ≫ 层级**:hotpotqa(0.3 预算 2×)、musique(4.3×)、multifieldqa(低压区大幅领先)。层级丢弃会连带丢掉散落在具体 token 的证据。
3. **merge 无稳定增益、难任务上有害**:musique 上 merge 全程低于 evict → 核心是"**选对 token**",而非"合并被丢的"。方法定位为 **selection** 而非 cache-merging。
4. **简单任务(tipsheets/countries)基线饱和**:KVComm 丢层也鲁棒,token 方法不占优 → 这两个数据集仅用于 receiver-aware 消融,不用于论证"token>层"。
5. **token 效率极高**:receiver 普遍在 r=0.1~0.3 即饱和(multifieldqa r0.1≈93% 全量,musique r0.3≈88%),说明大部分 KV token 对回答是冗余的。

## 4. 消融观察

- **观测窗口 N(8 vs 16)是任务相关超参**:
  - 难/精确检索任务(musique):**w8 > w16**(越窄越聚焦,r0.3:0.480 vs 0.462)。
  - 简单任务(tipsheets):w16 略好。
  - 建议补 {4,8,16,32,全部} 完整扫描曲线。
- **query 自适应 vs 静态**:KVComm 选择在校准阶段算一次后写死(query 无关);ReKV 每条 query 实时重算(query 自适应)。这是除粒度外的第二条区别轴。

---

# 阶段 B:Budget-aware 升级研究线

> 动机:ReKV 仍对每条 query、每一层用**同一个固定保留比 r**。但不同 query 的信息需求差异巨大(简单 fact lookup vs 多跳推理)。能否**按 query 自适应分配通信预算**?以下三步依次回答:(Step 0)前提是否成立 →(Step 1)发送方能否开环预测 →(Step 2)接收方能否闭环索取。

## 5. Step 0 — 预算 headroom 验证(oracle 分析)

**做法**:在 ReKV(receiver-w16)上对每条样本扫密集预算档 `r∈{0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7}`,逐样本落盘得分(`eval.py` → `per_sample.jsonl`)。定义样本的 **oracle 最小预算** = 能解出它(F1≥τ=0.5)的最小 r。脚本:`scripts/analyze_oracle.py`。

| 数据集 | N | 可解率@max | **avg oracle 预算** | 最优固定 r | **等精度省预算潜力** | 过供比例 | 覆盖@0.2 | 覆盖@0.3 |
|---|---|---|---|---|---|---|---|---|
| hotpotqa | 500 | 80.8% | **0.179** | 0.5 | **64.1%** | 96.8% | 80.9% | 93.1% |
| musique | 500 | 58.0% | **0.167** | 0.5 | **66.5%** | 95.2% | 80.0% | 91.0% |
| twowikimqa | 200 | 52.0% | 0.086 | 0.1 | 14.4% | 76.9% | 94.2% | 96.2% |

- **oracle 最小预算分布极散**(hotpotqa 在 r=0.15 处达峰、向两侧铺开;7.2% 样本 r=0.05 即可,9.8% 需 r≥0.3)。
- 固定 r 严重过度供给:hotpotqa/musique 约 **95–97% 的可解样本**被分配了多于其实际所需的预算,平均 regret 0.32。
- **覆盖曲线**(oracle≤r_high 的比例):2 档 `{0.2,0.3}` 即覆盖 ~90% 可解样本 → 天然支持 2–3 轮渐进。

> **结论**:在高方差多跳任务(hotpotqa/musique)上,"每条 query 该给多少预算"差异巨大,固定 r 的浪费潜力 **64–67%**,**前提稳健成立**。twowikimqa 是低方差异类(多在 r≤0.1 解出,headroom 小),不作为主论据。

## 6. Step 1 — 开环预算自适应(negative result)

**做法**(`models.py` 新增 `budget_mode`,脚本 `scripts/run_budget.sh`,分析 `scripts/analyze_budget.py`):

- `query`:每条 query 按 receiver 重要性的**熵**定总预算 `B(Q)=r_min+(r_max−r_min)·mean_l(H_l/log L_l)`(越分散→给更多);
- `layer`:总预算固定,按层重要性 `softmax(I_l/τ)·n·B` 分配(均值锁定为 B);
- `query+layer`:两者叠加。全部基于已验证的 receiver 打分,**无需训练**。

### 6.1 层间分配:等预算下无收益、低预算翻车

`layer_r` vs `uniform_r` 同预算 score 差(Δ):

| 数据集 | r0.2 Δ | r0.3 Δ | r0.5 Δ |
|---|---|---|---|
| hotpotqa | −0.016 | +0.000 | −0.006 |
| musique | +0.002 | −0.002 | −0.002 |
| countries | **−0.195** ⚠️ | −0.010 | +0.005 |
| tipsheets | −0.010 | +0.004 | −0.004 |
| multifieldqa_en | 0.000 | 0.000 | +0.007 |
| twowikimqa | +0.010 | +0.010 | +0.005 |
| tmath | −0.001 | +0.004 | +0.001 |

→ 全部落在 ±0.01(n=200–500 即 1–2 个样本)噪声带;countries r0.2 因 softmax+floor 把覆盖层饿死而**崩到 0.31(vs uniform 0.505)**。**层间分配无系统增益、低预算有风险**。

### 6.2 query 自适应:低预算崩、高预算微涨,且预测器根本没在自适应

等预算 gain(query vs uniform 插值)随区间变化:低区间 `[0.05,0.3]` 普遍崩(hotpotqa −0.13、countries −0.475、tipsheets −0.15),高区间 `[0.15,0.7]` 仅微弱正向(hotpotqa +0.017、tipsheets +0.015、twowiki +0.010)。

**病根证据**——逐样本 `query_budget`(`[0.1,0.5]` 档)的方差与相关性:

| 数据集 | qbudget 均值 | **标准差** | corr(qbudget, score) |
|---|---|---|---|
| hotpotqa | 0.256 | **0.009** | +0.003 |
| countries | 0.241 | **0.001** | −0.045 |
| tipsheets | 0.236 | **0.001** | −0.062 |
| twowikimqa | 0.252 | **0.008** | −0.107 |

→ 预测预算**样本间几乎不变(std≈0)**、与"是否解出"**零相关**。熵把所有样本压到 ~0.25,等于换种方式重选全局 r,**没有真正按 query 自适应**。

> **结论**:**发送方侧的开环统计量(熵、层重要性)无法预测每条 query 的通信预算**。这是一个干净、可写进论文的 negative ablation(创新点 2),并直接反证出方向:让**接收方反馈闭环**去索取,而非发送方开环猜测。
>
> ⚠️ 注意:`analyze_budget.py` 报的"等精度省 36%/45%"(tipsheets/multifield)是饱和段假象——曲线平坦处一个 +0.009 的微小纵向增益被换算成大横向省预算,且在噪声量级,**不可作为卖点**;"等预算 gain"才是稳信号。

## 7. Step 2a — 闭环渐进式通信(离线 oracle 上界)

**做法**(`scripts/sim_progressive.py`,**零额外 GPU**):复用 Step 0 的多预算逐样本得分,模拟"低预算起步 → 不确定就请求下一档 → 解出即停"。离线没有运行时不确定性信号,故用两条策略夹出可达区间:

- **ORACLE-stop(上界)**:解出才升档——任何不确定性触发器的最好情形;
- **FIXED-r(下界)**:无反馈、单一固定预算 = 当前 ReKV 基线。
- **成本模型**:高 r 的 top-k 是低 r 的(近)超集 → 每轮只传**增量**,总传输≈最终到达档(incremental)。非嵌套重传(restart)成本会反超固定 r,故**协议必须增量传输**。

### 7.1 实用 4 档梯子 `[0.1,0.2,0.3,0.5]`(oracle-stop)

| 数据集 | 渐进精度 | 最佳固定 r 精度 | avg 轮数 | **avg 预算(增量)** | **等精度省预算 vs 固定0.5** |
|---|---|---|---|---|---|
| hotpotqa | **0.798** | 0.746 | 2.39 | 0.265 | **+47%** |
| musique | **0.554** | 0.474 | 2.75 | 0.323 | **+35%** |
| twowikimqa | **0.490** | 0.440 | 2.63 | 0.317 | **+37%** |

### 7.2 极简 2 档 `[0.1,0.3]`, cap=2 轮

| 数据集 | 渐进精度 | avg 轮数 | avg 预算 | 等精度省预算 vs 固定0.3 |
|---|---|---|---|---|
| hotpotqa | 0.722 | 1.77 | 0.254 | +15% |
| musique | 0.494 | 1.71 | 0.241 | +20% |
| twowikimqa | 0.460 | 1.56 | 0.212 | +29% |

> **结论**:闭环渐进在三个难任务上**等精度省 35–47% 预算、≤2.4 轮**(oracle 上界)。
> - 渐进精度**反超**最佳固定 r(0.798>0.746):它给每条样本挑到能解的预算,连"低预算解出、高预算反崩"的样本也吃到——但这是 **oracle 上界**(完美停止信号),含部分非单调噪声红利,真实触发器落其下;**稳的主张是"等精度省预算",不是"涨精度"**。
> - oracle 上界与固定 r 下界之间的 gap,就是 Step 2b 在线触发器要赚的钱——而 §8 表明它基本赚不到。

## 8. Step 2b — 在线渐进式通信(实测,learned controller)

**做法**(`eval.py` 加 `_test_progressive`/`_generate_uncertainty`/`compute_context_attention`,脚本 `scripts/run_progressive.sh`):对每档 r 真实生成,记录 6 个运行时不确定性信号(answer entropy 首/均、top-2 logit margin 首/均、B 对压缩 KV 的注意力质量 `ctx_mass` / 集中度 `ctx_conc`)。先用单信号扫阈值(`analyze_progressive_online.py`),再训一个 logistic 回归把全部信号融合成停止策略(`learn_stop_policy.py`,GroupKFold 防泄漏)。

**musique(N=500,梯子 [0.1,0.2,0.3,0.5])实测**:

| | 结果 |
|---|---|
| OOF AUC P(solved) | **0.751**(单信号更低) |
| 主导信号 | `margin_mean +0.74`、`ent_first −0.71`;`ctx_mass/ctx_conc ≈ 0`(KV 充分性信号几乎无用) |
| oracle 上界 | acc 0.554 @ b0.344 |
| learned 策略 等精度省预算 | 中段 **−10% ~ −25%**(更差),仅极端档微正(+4~5%) |

- 单信号(熵 / margin / ctx_*)阈值法在三个数据集上**均无法稳定跑赢固定 r**。
- 融合 learned controller:**只有 hotpotqa 中段拿到约 +6~16%**;musique / twowikimqa **负节省**。
- 即便偶有正节省,代价是 **avg rounds 升到 2.7–4.0**——多轮 = 多次 RTT + 重复 prefill/生成,在分布式 MAS 下吃掉所有带宽收益。

> **结论**:在线多轮渐进**不构成稳定、可泛化的增益**,且与"用 KV 一次性替代 context"的初衷(省往返)冲突。放弃多轮路线。

## 9. 牌2 — 单发 Pass-1 预算预测(LODO 证伪)

**动机**:多轮太贵 → 能否**单发**?即只用 Pass-1(receiver 打分后、生成前)可得的特征,一次性预测该 query 的最优预算,只传一次。**关键检验是跨数据集泛化(LODO)**——因为 oracle 标注成本高,逐任务标注不现实。

**做法**:`models.py` 加 `compute_pass1_features()`(从重要度分布抽 `rcap{50,90,95}`=覆盖 X% 注意力所需 token 比例、熵、Gini、top10/20 质量、recency/sink 偏置、log 上下文长,层间取均值/方差,**全部量纲无关**);`eval.py` 加 `--dump_pass1_features`(只前向打分、不生成,落 `per_sample_feat.jsonl`);`scripts/learn_budget_predictor.py` 把特征与 §5 的 oracle 标签按 idx 对齐,训 P(solved | features, r) 分类器,**WITHIN**(同任务 K 折,上界)对比 **LODO**(留一任务,真实泛化)。

**结果(hotpotqa / musique / twowikimqa,τ=0.5)**:

| 指标 | WITHIN(上界) | LODO(泛化) |
|---|---|---|
| AUC P(solved) | 0.697 | **0.585**(≈随机) |
| 等精度省预算 | **全程负**(−25%~−190%) | **全程负**(−40%~−113%) |

- 判别特征符合直觉(`top10_mean +2.10`、`gini_mean −1.35`、`log_ctx_len −1.70`:越集中/越短越省预算),但**跨任务一换就失效**(AUC 0.70→0.585)。
- 即便同任务训练(WITHIN 上界),逐样本预测器在等精度下仍**比固定 r 更费预算**——固定 r 曲线本身是极强 baseline,逐样本方差带来的噪声 > 挖到的信号(与 §6 开环、§8 在线同病)。

> **结论**:**单发 Pass-1 预算预测证伪**,且不满足跨数据集泛化的发表门槛(否决线:LODO 在 ≥2 留出任务正节省 ≥10%,实测三任务全负)。这说明"直接预测 query difficulty / budget r"不是可行路线。阶段 B 随后转向 §10 的 **coverage/fidelity threshold** 范式。

## 10. Step 3 — B-ReKV(接收方证据覆盖率预算,正结果)

**动机**:DBudgetKV/GVote 相关工作提示,动态预算不一定要预测一个 `r`。更稳的做法是定义一个**保真度阈值**:DBudgetKV 是"剪到 attention norm 快坏为止";GVote 是"保留 future queries 需要的 key"。在 inter-LLM communication 中,接收方 B 的 query 是已知的,因此可直接用 **receiver-query attention coverage** 定义预算。

**方法**(`models.py` 新增 `budget_mode=coverage`,脚本 `scripts/run_coverage.sh` / `run_coverage_stage1.sh`,分析 `scripts/analyze_coverage.py`):

1. 仍先用 ReKV 计算每层 token 重要性 `s_i = Attn_B(q_tail -> token_i)`。
2. 归一化 `p_i = s_i / sum_j s_j`。
3. 按 `p_i` 从大到小排序,找最小 `k_l(Q)` 使 `sum_{i in Top-k} p_i >= coverage_tau`。
4. 每层动态预算 `r_l(Q)=clamp(k_l/L_l * coverage_scale, budget_min, budget_max)`。
5. 压缩仍为 evict-only(top-k + sink/recent),不训练、不多轮、不需要 oracle labels。

**关键区别**:

- 失败路线(§6/§9):`features/query difficulty -> predict r`。
- B-ReKV:`receiver evidence coverage target -> derive r`。
- `coverage_tau=0.90/0.95` 有直接语义:保留 90%/95% 的接收方注意力证据,比固定 `r=0.3` 更可解释。

### 10.1 零 GPU 离线预检(rcap + probe 查表)

**做法**:`scripts/sim_coverage_budget.py` 复用 `per_sample_feat.jsonl` 中的 `rcap90/95` 和 Step 0 的 probe scores,将 `rcapXX * scale` 映射到已有预算档,查表模拟 accuracy / avg budget。

| 数据集 | 策略 | acc | avg budget | 等精度省预算 |
|---|---|---:|---:|---:|
| hotpotqa | rcap90 x0.7 | 0.632 | 0.212 | +4.1% |
| hotpotqa | rcap95 x0.9 | 0.738 | 0.442 | +3.9% |
| musique | rcap90 x0.7 | 0.424 | 0.206 | +12.2% |
| musique | rcap90 x0.75 | 0.438 | 0.226 | +12.5% |
| musique | rcap90 x0.8 | 0.452 | 0.247 | +12.7% |
| musique | rcap95 x0.8 | 0.472 | 0.395 | +12.3% |
| twowikimqa | rcap90 x0.4 | 0.435 | 0.085 | +43.3% |

> **离线结论**:musique 信号最稳;hotpotqa 局部微正;twowikimqa fixed-r 曲线非单调,节省率易被插值放大,仅作辅助。

### 10.2 在线实测(Stage-1 B-ReKV)

**做法**:真实运行 `budget_mode=coverage`,每条样本 Pass-1 后按 coverage 动态设每层 `r_l`,再生成答案。与 fixed-r ReKV probe 曲线做等精度预算对比。

MuSiQue accuracy–budget Pareto 图见: `snapshots/musique/coverage_pareto.png`。

#### MuSiQue(主卖点)

fixed-r 曲线:

| fixed r | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acc | 0.088 | 0.294 | 0.374 | 0.404 | 0.462 | 0.470 | 0.474 | 0.478 |

B-ReKV 在线结果:

| coverage_tau | scale | window | acc | avg budget | 等精度省预算 |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.70 | 16 | 0.406 | 0.192 | +5.6% |
| 0.90 | 0.75 | 16 | 0.418 | 0.203 | +9.4% |
| 0.90 | 0.80 | 16 | 0.424 | 0.214 | +8.6% |
| 0.90 | 0.90 | 16 | 0.442 | 0.236 | +11.0% |
| 0.95 | 0.75 | 16 | 0.470 | 0.336 | +16.0% |
| 0.95 | 0.80 | 16 | 0.472 | 0.356 | +20.8% |
| 0.95 | 0.85 | 16 | 0.474 | 0.377 | +24.7% |
| 0.95 | 0.90 | 16 | 0.480 | 0.397 | acc>fixed-ceiling |
| **0.90** | **0.75** | **8** | **0.442** | **0.158** | **+40.7%** |
| 0.90 | 0.80 | 8 | 0.440 | 0.165 | +36.9% |
| 0.90 | 0.90 | 8 | 0.428 | 0.181 | +25.0% |
| 0.95 | 0.65 | 8 | 0.464 | 0.246 | +24.2% |
| **0.95** | **0.70** | **8** | **0.470** | **0.263** | **+34.3%** |
| **0.95** | **0.75** | **8** | **0.482** | **0.279** | **acc>fixed-ceiling** |
| 0.95 | 0.80 | 8 | 0.470 | 0.296 | +26.0% |
| **0.95** | **0.85** | **8** | **0.490** | **0.313** | **acc>fixed-ceiling** |
| 0.95 | 0.90 | 8 | 0.480 | 0.329 | acc>fixed-ceiling |

**MuSiQue 结论**:

- `w8` 明显强于 `w16`,和原始 ReKV 中 "MuSiQue 更偏好窄观测窗口" 的观察一致。
- `cov_t0.95_s0.75_w8`:**acc=0.482, avg budget=0.279**,超过 fixed-r 最高点 0.478,且只用 27.9% 平均 KV。
- `cov_t0.95_s0.85_w8`:**acc=0.490, avg budget=0.313**,当前 MuSiQue 最高 coverage 精度,仍显著低于 fixed r=0.5 的预算。
- `cov_t0.95_s0.70_w8`:**acc=0.470, avg budget=0.263**,在接近 fixed r=0.4/0.5 的精度区间取得 **+34.3%** 等精度省预算。
- `cov_t0.90_s0.75_w8`:**acc=0.442, avg budget=0.158**,等精度省 **+40.7%**。
- `w8 tau=0.95` 从 `scale=0.65→0.85` 形成清晰 Pareto:0.246/0.464 → 0.263/0.470 → 0.279/0.482 → 0.313/0.490。
- 这是目前 budget-aware 线最强正结果,足以作为主图/主表。

#### HotpotQA(辅助正结果)

fixed-r 曲线:

| fixed r | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acc | 0.072 | 0.230 | 0.488 | 0.614 | 0.700 | 0.726 | 0.746 | 0.748 |

B-ReKV 在线结果:

| coverage_tau | scale | window | acc | avg budget | 等精度省预算 |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.70 | 16 | 0.580 | 0.206 | -10.4% |
| 0.90 | 0.90 | 16 | 0.660 | 0.251 | +0.9% |
| 0.95 | 0.80 | 16 | 0.720 | 0.360 | +4.6% |
| **0.95** | **0.90** | **16** | **0.732** | **0.400** | **+6.9%** |
| 0.95 | 1.00 | 16 | 0.738 | 0.441 | +4.1% |
| 0.98 | 0.80 | 16 | 0.734 | 0.530 | -20.4% |
| 0.98 | 0.90 | 16 | 0.744 | 0.592 | -20.8% |
| 0.98 | 1.00 | 16 | 0.746 | 0.643 | -28.6% |

**HotpotQA 结论**:

- aggressive `0.90/0.70` 欠分配,会明显负收益。
- 保守 `coverage_tau=0.95` 有稳定小正收益,最佳 `scale=0.9`:**+6.9%**。
- `coverage_tau=0.98` 过度保守,预算太高,不适合。

#### 2WikiMQA(仅辅助观察)

fixed-r 曲线非单调:

| fixed r | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acc | 0.400 | 0.440 | 0.435 | 0.425 | 0.410 | 0.410 | 0.410 | 0.425 |

B-ReKV 在线结果:

| coverage_tau | scale | window | acc | avg budget | 等精度省预算 |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.40 | 16 | 0.400 | 0.090 | -79.4% |
| 0.90 | 0.50 | 16 | 0.405 | 0.098 | +43.9% |
| 0.90 | 0.60 | 16 | 0.420 | 0.108 | +64.1% |
| 0.90 | 0.70 | 16 | 0.420 | 0.118 | +60.7% |

**2WikiMQA 结论**:coverage 能用极低预算保持 0.40–0.42,但 fixed-r 曲线在 `r=0.1` 最高、非单调严重,等精度 saving 容易被插值放大。作为"coverage 避免高预算反崩"观察,不作主论据。

### 10.3 阶段 B 当前结论

> 直接预测 query difficulty / budget r 的方法(熵、层分配、learned Pass-1、在线多轮 controller)均不稳定或不泛化;但 **receiver-attention coverage** 把预算控制转成可解释 fidelity constraint 后,在线实测在 MuSiQue 获得强正收益、HotpotQA 获得稳健小正收益。Budget-aware 主线从 "predictive budget" 修正为 **B-ReKV**。

### 10.4 审稿风险补强实验汇总(2026-07-01)

> 目的:把 ReKV 从一个"query attention trick"升级为 **Receiver-Initiated KV Communication** 协议,并补齐审稿人最可能追问的 cost、robustness、fairness、interpretability 证据。

#### 10.4.1 Cost / efficiency profiling

脚本与产物:

```text
scripts/run_cost_profile_all8_gpu2.sh
scripts/analyze_cost_profile.py
snapshots/cost_profile/pair1_llama31_same_all8_full/cost_table.csv
```

结论:

- receiver scoring overhead 可控:典型难任务上 `t_receiver_score` 约 0.06-0.15s,不是主要瓶颈。
- B-ReKV 在长上下文任务上显著降低 KV payload:
  - `multifieldqa_en`: Coverage `170-190MB` vs ReKV-r0.3 `279MB` vs KVComm-top0.3 `244MB`,且 Coverage score `0.517/0.524` 远高于 KVComm-top0.3 `0.193`。
  - `qasper`: Coverage `137-153MB` vs ReKV-r0.3 `196MB`,score `0.333/0.327` vs KVComm-top0.3 `0.073`。
  - `musique`: Coverage `12.9-14.4MB`,接近 ReKV-r0.3 `15.2MB`,但 score `0.479/0.489` 远高于 KVComm-top0.3 `0.109`。
- 简单任务(`countries/tipsheets`)KVComm 已饱和,不作为 cost 主卖点;`tmath` 由超长生成(`output_tokens≈254`)主导,不适合作为通信 latency 主论据。

#### 10.4.2 Coverage robustness / Pareto

脚本与产物:

```text
scripts/run_coverage_robustness_gpu7.sh
scripts/analyze_coverage.py
scripts/plot_coverage_pareto.py
snapshots/coverage_robustness_summary.txt
snapshots/musique/coverage_pareto.png
snapshots/hotpotqa/coverage_pareto.png
snapshots/multifieldqa_en/coverage_pareto.png
snapshots/brekv_budget_distribution.png
snapshots/brekv_budget_distribution_summary.csv
```

结论:

- MuSiQue 是主卖点,形成清晰 Pareto 区域,不是单点 cherry-pick:
  - `tau=0.90 scale=0.65 w8`: acc `0.428`, budget `0.142`, saving `+41.1%`。
  - `tau=0.90 scale=0.75 w8`: acc `0.442`, budget `0.158`, saving `+40.7%`。
  - `tau=0.95 scale=0.70 w8`: acc `0.470`, budget `0.263`, saving `+34.3%`。
  - `tau=0.95 scale=0.75 w8`: acc `0.482`, budget `0.279`,超过 fixed-r ceiling。
  - `tau=0.95 scale=0.85 w8`: acc `0.490`, budget `0.313`,超过 fixed-r ceiling。
- HotpotQA 是稳定小正收益辅助结果,`tau=0.95,w16` 较稳:
  - `scale=0.75`: acc `0.718`, budget `0.339`, saving `+8.1%`。
  - `scale=0.85`: acc `0.730`, budget `0.380`, saving `+9.5%`。
  - `scale=0.90`: acc `0.732`, budget `0.400`, saving `+6.9%`。
- MultiFieldQA-en 的 fixed-r 曲线平坦/非单调,不强调插值 saving;但 Coverage `tau=0.90 scale=0.60 w8` 用 `0.099` 平均预算达到 fixed-r 最高 acc `0.540`,可作为低预算辅助观察。
- B-ReKV 的 per-sample budget distribution 证明它不是换了一个固定 `r`:
  - `musique`: mean `0.279`, p25/p50/p75=`0.259/0.279/0.301`, min/max=`0.170/0.368`。
  - `hotpotqa`: mean `0.306`, p25/p50/p75=`0.292/0.307/0.323`, min/max=`0.216/0.367`。
  - `multifieldqa_en`: mean `0.198`, p25/p50/p75=`0.177/0.199/0.220`, min/max=`0.129/0.272`,全部样本预算低于 fixed `r=0.3`。
  - 这些分布在同一全局 `tau=0.95, scale=0.75, w=8` 下产生,可作为 Figure 3: **query-adaptive B-ReKV budget distribution**。

#### 10.4.3 Receiver-initiated / query-aware fairness

脚本与产物:

```text
scripts/run_query_fairness_gpu2_7.sh
scripts/analyze_query_fairness.py
snapshots/query_fairness/pair1_llama31_same/query_fairness.csv
scripts/run_pair6_pair7_query_fairness_brekv_gpu7.sh
snapshots/query_fairness/pair6_pair7_query_fairness_brekv_summary.csv
```

同预算 `r=0.3` 下,query-aware receiver signal 是主要增益来源:

| Dataset | Random-token | Evict / ValueNorm | ReKV best | Query sketch |
|---|---:|---:|---:|---|
| HotpotQA | 0.432 | 0.568 | **0.700** | `w16`,约 21.3% query tokens |
| MuSiQue | 0.166 | 0.322 | **0.480** | `w8`,约 10.4% query tokens |
| MultiFieldQA-en | 0.387 | 0.333 | **0.513** | `w4/w16/all`,`w4` 仅约 5.7% query tokens |

关键论点:

- ReKV 不需要 full query;`w8/w16` 轻量 query sketch 即可达到或超过 full-query attention。
- 这回应了"改变 KVComm query-blind setting 不公平"的审稿风险:本文研究的是 **Receiver-Initiated KV Communication**,不是声称在原 KVComm 完全 query-blind sender setting 下同设定取胜。
- B→A sketch overhead 可计入通信量:
  - token-id 口径:`w8` 仅 32 bytes,`w16` 仅 64 bytes。
  - hidden-state BF16 口径:`w8` 约 64KB,`w16` 约 128KB,相对 A→B selected KV MB 级 payload 仍很小。

2026-07-02 补充:pair #6/#7 fine-tuned model pairs 上的 fairness extension 已完成,并加入 B-ReKV canonical 点:

- Pair #6:
  - HotpotQA: Evict `0.516`, Random `0.406`, best ReKV `0.668`, best B-ReKV `0.674`。
  - MuSiQue: Evict `0.236`, Random `0.182`, best ReKV `0.362`, best B-ReKV `0.384`。
  - MultiFieldQA-en: Evict `0.327`, Random `0.320`, best ReKV `0.467`, best B-ReKV `0.493`。
- Pair #7:
  - HotpotQA: Evict `0.124`, Random `0.118`, best ReKV `0.396`, best B-ReKV `0.446`。
  - MuSiQue: Evict `0.144`, Random `0.080`, best ReKV `0.298`, best B-ReKV `0.308`。
  - MultiFieldQA-en: Evict `0.080`, Random `0.140`, best ReKV `0.393`, best B-ReKV `0.393`。

结论:在异构 fine-tuned pair 上,ReKV/B-ReKV 仍显著优于 query-agnostic Evict/Random,说明 receiver-aware query sketch 不是 pair #1 same-model 的偶然收益。

#### 10.4.4 Interpretability / evidence proxy

脚本与产物:

```text
scripts/run_interpretability_dump_gpu2_7.sh
scripts/dump_interpretability_examples.py
snapshots/interpretability/pair1_llama31_same/interpretability_overlap_summary.csv
snapshots/interpretability/pair1_llama31_same/interpretability_examples.md
snapshots/interpretability/pair1_llama31_same/cleaned/clean_interpretability_examples.md
snapshots/interpretability/pair1_llama31_same/cleaned/*_clean_top_tokens.png
scripts/run_deletion_ablation_gpu2.sh
scripts/run_deletion_ablation.py
snapshots/deletion_ablation/pair1_llama31_same/deletion_ablation_summary_w8_r0.3_k20.csv
```

答案词 overlap(50 samples/task, top-20 tokens):

| Dataset | ReKV answer-term recall | Evict recall | Random recall |
|---|---:|---:|---:|
| HotpotQA | **0.325** | 0.133 | 0.147 |
| MuSiQue | **0.353** | 0.142 | 0.143 |
| MultiFieldQA-en | **0.293** | 0.017 | 0.099 |

结论:ReKV top tokens 更频繁覆盖答案词/证据相关词,支持 receiver attention mass 作为 interpretable receiver-evidence proxy。当前 overlap 是粗粒度 lexical proxy,后续可从 `interpretability_examples.md` 挑选 qualitative case 并画 heatmap。

清洗后的 qualitative examples 已补齐,过滤 `<|...|>` special tokens、`system/user/assistant`、Instruction/Context/Date 等模板词后,每个任务保留 1 个可放论文的 case:

- HotpotQA idx=35:`D1NZ` 问题中 ReKV clean top tokens 命中 `drifting`,Evict/Random recall 为 0。
- MuSiQue idx=40:ReKV clean top tokens 命中 `000/nearly/Zurich/25`,answer-term recall `0.667`,Evict 仅 `0.167`。
- MultiFieldQA-en idx=27:ReKV clean top tokens 命中 `wearable/sensors`,answer-term recall `1.0`,Evict/Random 为 0。

这些 examples 对应 `hotpotqa_clean_top_tokens.png`,`musique_clean_top_tokens.png`,`multifieldqa_en_clean_top_tokens.png`,适合作为论文中 token bar / qualitative evidence figure 的候选。

2026-07-02 补充:deletion ablation 已完成。设置为 pair #1, `recv_window=8`, `r=0.3`,每任务 50 samples,删除/屏蔽 ReKV、Evict、Random 各自 top-20 content tokens 后重新生成答案。

| Dataset | Base score | Delete ReKV score / drop | Delete Evict score / drop | Delete Random score / drop |
|---|---:|---:|---:|---:|
| HotpotQA | 0.78 | 0.42 / **0.36** | 0.62 / 0.16 | 0.64 / 0.14 |
| MuSiQue | 0.58 | 0.22 / **0.36** | 0.44 / 0.14 | 0.54 / 0.04 |
| MultiFieldQA-en | 0.48 | 0.24 / **0.24** | 0.48 / 0.00 | 0.50 / -0.02 |

样本级统计也支持同一结论:删除 ReKV tokens 后的平均 drop 最大,且 ReKV deletion 在 HotpotQA/MuSiQue/MultiFieldQA-en 上分别有 `44/50`, `47/50`, `47/50` 个样本达到或并列达到最大 drop。该结果把 interpretability 从 lexical overlap 推进到 causal-ish evidence:ReKV-selected tokens 被移除后,答案质量下降明显更大。

---

## 11. 时间 / 开销(待严格测量)

- 当前 log 的 `communication time` 是**整轮 500 样本墙钟**,受"压坏→生成变长"和并行抢卡污染,**非干净延迟**。
- receiver 比 evict 多一遍 question 打分前向,均摊每样本 ~0.05s,可忽略。
- 单卡上压缩不省"传输时间"(无真实传输),省的是 B 的 prefill 计算;真实通信收益须在分布式下测。
- **TODO**:补受控延迟实验(单样本分别计时 A-prefill / 打分 / B-prefill / 生成)。

## 12. 后续路线图

| 步骤 | 内容 | 机器成本 | 价值 | 状态 |
|---|---|---|---|---|
| **B-ReKV 收尾(当前主攻)** | MuSiQue `w8 tau=0.95` 细扫已完成;下一步补 `multifieldqa_en`,画 accuracy–budget 帕累托图,整理 LaTeX 表 | 小-中 | **Budget-aware 正向抓手** | 进行中 |
| Receiver-Initiated KV Communication 协议化 | 把 ReKV 从"query attention trick"升级为 receiver-initiated / query-aware KV communication setting;优先补 `recv_window={4,8,16,32,all}` sketch-size ablation、query-sketch overhead accounting、ReKV vs Evict/Random-token | 小-中 | **核心叙事升级 / 审稿风险补强** | 待开 |
| Head-wise B-ReKV | 借鉴 Ada-KV:每个 head 独立 coverage,避免 head 平均抹平检索头差异 | 中 | 放大创新点 | 待开 |
| 牌 1 | **Cross-model ReKV**:A、B 异构(不同权重 / tokenizer / 层数)时如何对齐打分。需排查 `compute_receiver_importance` 的同深度假设、跨 tokenizer 的 token 对齐、层数不等时的映射 | 中(GPU) | 真实 MAS 扩展 | 待开 |
| Step 3 | 受控延迟实验:单样本分别计时 A-prefill / 打分 / B-prefill / 生成,分布式下测真实通信收益 | 小 | rebuttal | 待开 |
| 收尾 | 补 qasper;难任务 receiver 补 r0.6–0.9;窗口 {4,8,16,32,all} 完整曲线;生成 LaTeX 表 | 小 | 完整性 | 待开 |

> Budget-aware 路线已从"预测预算"修正为"证据覆盖率预算"。§6–§9 作为 failed predictive-budget ablations 保留;§10 是当前正向方法。

## 13. 局限

- **同模型限定**:A、B 同权重时打分可在 A 端精确复现;异构模型需 B 传 query 向量或近似(牌 1,§12,future work)。
- **打分为启发式**:注意力之和;可升级为失真最优判据(注意力 × ‖value‖ / 输出分布变化),给 rate-distortion 论证。
- **Predictive budget 已证伪**(§6–§9):前提虽成立(headroom 真实),但开环熵/层分配、在线多轮、单发 Pass-1 预测都无法稳定泛化。B-ReKV(§10)说明可行路线应是 receiver-evidence fidelity/coverage,而不是直接预测 query difficulty。
- **B-ReKV 仍需更多任务验证**:当前强正结果主要来自 MuSiQue,HotpotQA 为中等正收益,2WikiMQA 因 fixed-r 非单调只能辅助。需补 MultiFieldQA/QASPER 与更多窗口/阈值。
- **渐进上界含非单调噪声红利**:§7 的精度反超部分来自 oracle 挑到"低预算偶然解出"的样本,真实触发器(§8)无法复现。
- **Problem setting 需讲清楚**:ReKV 不是在原 KVComm 的完全 query-blind sender 设定下声称同设定取胜,而是研究更实际的 receiver-initiated / query-aware communication。接收方发送轻量 query sketch 指导 context holder 选择 KV,应把 sketch 开销计入通信量,并用 Evict/Random-token/sketch-size 消融证明 receiver-aware 信号本身有效。

## 附 A:数据集目录结构

```
snapshots/<dataset>/
  ├── kvcomm/        kvcomm_top{0.3,0.5,0.7,1.0}_*
  ├── mtc_merge/     merge_r{0.1..0.9}_*
  ├── mtc_evict/     evict_r{0.1..0.9}_*
  ├── mtc_receiver/  recv_w{8,16}_r{0.1..0.9}_*  +  probe_recv_w16_r{0.05..0.7}_*(Step 0 密集探针 / 牌2 oracle 标签)
  ├── budget/        {uniform,layer,query,querylayer}_*（Step 1 预算分配）
  ├── coverage/      cov_t{0.90,0.95,0.98}_s*_w{8,16}_*（B-ReKV 在线实验）
  ├── progressive/   per_sample_prog.jsonl（Step 2b 在线渐进的逐档信号）
  └── features/      feat_w16_*/per_sample_feat.jsonl（牌2 Pass-1 单发特征）
```
每个 run 的最终指标在 `*/log.log` 最后一行 `communication result:`(前面的 1.0000 是单样本校准值,需取 tail -1);逐样本得分在 `*/per_sample.jsonl`(含 `score`、实际 `budget`、`query_budget`)。

## 附 B:分析脚本

| 脚本 | 用途 |
|---|---|
| `scripts/run_dataset.sh` / `run_budget.sh` | 单数据集跑全方法 / 跑 budget 四模式 |
| `scripts/run_budget_all.sh` | 7 数据集分 GPU0/1 并行跑 budget 全套 |
| `scripts/run_probe.sh` | Step 0 密集预算探针(生成 per_sample) |
| `scripts/analyze_oracle.py` | Step 0:oracle 最小预算分布 / 省预算 / regret / coverage |
| `scripts/analyze_budget.py` | Step 1:budget 模式 vs uniform 的等预算增益 / 等精度省预算 |
| `scripts/sim_progressive.py` | Step 2a:离线 oracle 渐进模拟(accuracy / rounds / budget) |
| `scripts/run_progressive.sh` / `analyze_progressive_online.py` / `learn_stop_policy.py` | Step 2b:在线渐进生成 + 单信号扫阈值 + learned controller |
| `scripts/run_features.sh` / `learn_budget_predictor.py` | 牌2:Pass-1 单发特征落盘 + WITHIN/LODO 预算预测器 |
| `scripts/sim_coverage_budget.py` | B-ReKV 离线预检:rcap90/95 + probe scores 查表模拟 |
| `scripts/run_coverage.sh` / `run_coverage_stage1.sh` / `analyze_coverage.py` | B-ReKV 在线运行与等精度预算分析 |

---

## 合并来源 2：第二部分：KVComm 原论文实验版图与 ReKV 跟跑计划

> 原文件：`snapshots/KVCOMM_PAPER_EXPERIMENT_PLAN.md`。以下为该文件正文内容，实验数值未改动。

# KVComm 原论文实验版图与 ReKV 跟跑计划

> 目标:先完整拆解 KVComm 原论文做了哪些实验、哪些表格、哪些模型对、哪些数据集。原论文已有的 `Baseline / Skyline / NLD / CIPHER / AC / KVComm` 结果原则上不重复复现,直接作为 paper baseline 引用或转录。我们要做的是:在**同样数据集、同样模型对、同样预算点**上跑 `ReKV / B-ReKV`,形成一一对比。

---

## 0. 重要更正

我们现在截图里的同模型 Llama-3.1-8B 表格,不是 KVComm 原论文主文 Table 1,而是 **Appendix Table 8** 里的 model pair #1:

```text
M_s = meta-llama/Llama-3.1-8B-Instruct
M_r = meta-llama/Llama-3.1-8B-Instruct
```

KVComm 原论文主文 Table 1 报的是 3 个 fine-tuned model pairs:

```text
pair #6: Llama-3.2-3B fine-tuned pair
pair #7: Qwen2.5-7B fine-tuned pair
pair #8: Falcon3-7B fine-tuned pair
```

所以我们对齐 KVComm 原文时应分两层:

- **主文对齐**:优先跟跑 KVComm Table 1 的 3 个主模型对。
- **完整对齐**:再跟跑 Appendix Table 8 的 9 个模型对,其中包含我们已经在跑的 Llama-3.1-8B 同模型。

---

## 1. KVComm 原论文实验设置

### 1.1 任务定义

KVComm 是 two-agent contextual communication:

- Sender `M_s` 只看 context `C`。
- Receiver `M_r` 只看 query `Q`。
- 通信协议把 `M_s` 从 `C` 中提取的信息传给 `M_r`。
- `M_r` 结合 `Q` 和通信信息生成答案。

原 KVComm 传 selected layer KV cache:

```text
k_r^l <- [k_s^l ; k_r^l]
v_r^l <- [v_s^l ; v_r^l]
```

原 KVComm 的选择粒度和预算:

- 按 **layer** 选 KV。
- `KVComm(0.3/0.5/0.7)` 表示传 `30%/50%/70%` 的层。
- 层选择分数 = attention importance score + Gaussian prior。
- `mu = L/2`, `sigma = 10`。
- `alpha = 1` for Llama, `alpha = 0.8` for Qwen/Falcon。
- calibration size = 1。

### 1.2 数据集

KVComm 原论文主数据集 8 个:

| Dataset | Size in paper | Metric | 说明 |
|---|---:|---|---|
| Countries | 200 | F1 | 合成地理事实 QA |
| Tipsheets | 500 | F1 | 合成投资选择 QA |
| HotpotQA | 500 | F1 | 多跳 QA |
| QASPER | 500 | F1 | scientific paper QA |
| MuSiQuest / MuSiQue | 500 | F1 | 多跳组合推理 |
| MultiFieldQA-en | 150 | F1 | LongBench 子集 |
| 2WikiMQA | 200 | F1 | LongBench 子集 |
| TMATH | 300 | ROUGE-L Recall | 数学 hint / reasoning |

Appendix E 扩展数据集:

| Dataset | Size in paper | 用途 |
|---|---:|---|
| HotpotQA-E | 7,405 | full/extended HotpotQA |
| QASPER-E | 1,726 | full/extended QASPER |
| MuSiQuest-E | 2,417 | full/extended MuSiQuest |
| SAMSum | 819 | summarization |

### 1.3 模型对

KVComm 原论文共 9 个 model pairs:

| Pair | Sender `M_s` | Receiver `M_r` | 类型 | 我们当前状态 |
|---:|---|---|---|---|
| 1 | `meta-llama/Llama-3.1-8B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` | Same model | 已在跑/已有大量结果 |
| 2 | `meta-llama/Llama-3.2-3B-Instruct` | `meta-llama/Llama-3.2-3B-Instruct` | Same model | Table 8 ReKV/B-ReKV 已完成 |
| 3 | `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | Same model | Table 8 ReKV/B-ReKV 已完成 |
| 4 | `tiiuae/Falcon3-7B-Instruct` | `tiiuae/Falcon3-7B-Instruct` | Same model | Table 8 ReKV/B-ReKV 已完成 |
| 5 | `yuvraj17/EvolCodeLlama-3.1-8B-Instruct` | `Team-ACE/ToolACE-2-Llama-3.1-8B` | fine-tuned from pair #1 base | Table 8 ReKV/B-ReKV 已完成 |
| 6 | `huihui-ai/Llama-3.2-3B-Instruct-abliterated` | `suayptalha/DeepSeek-R1-Distill-Llama-3B` | fine-tuned from pair #2 base | Table 1 ReKV/B-ReKV 已完成 |
| 7 | `Orion-zhen/Qwen2.5-7B-Instruct-Uncensored` | `bespokelabs/Bespoke-Stratos-7B` | fine-tuned from pair #3 base | Table 1 ReKV/B-ReKV 已完成 |
| 8 | `ehristoforu/falcon3-ultraset` | `huihui-ai/Falcon3-7B-Instruct-abliterated` | fine-tuned from pair #4 base | 暂缓；receiver checkpoint 不可获得或目录不完整 |
| 9 | `arcee-ai/Llama-3.1-SuperNova-Lite` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | fine-tuned from pair #1 base | Table 8 ReKV/B-ReKV 已完成但暂缓作为正向对比；KVComm probe 在 QA/multi-hop 上也很差 |

注意:

- KVComm 原文没有直接跨完全不同 architecture 传 KV;它限制在 same model 或 same-base fine-tuned models。
- 我们 ReKV 当前也最适合同结构/同 tokenizer/同层数设置。
- pair #6/#7/#8 是最优先的“对齐 KVComm 主文”模型对。

---

## 2. KVComm 原论文表格 / 图清单

### Table 1: 主文通信结果

目的:比较 `KVComm` 与 `Baseline / Skyline / NLD / CIPHER / AC`。

模型对:

- pair #6: Llama-3.2-3B fine-tuned pair
- pair #7: Qwen2.5-7B fine-tuned pair
- pair #8: Falcon3-7B fine-tuned pair

数据集:8 个主数据集全跑。

原论文方法:

- `Baseline`
- `Skyline`
- `NLD`
- `CIPHER`
- `AC(mean)`
- `AC(replace)`
- `AC(sum)`
- `KVComm(0.3)`
- `KVComm(0.5)`
- `KVComm(0.7)`

我们要补:

- `ReKV-w8(0.3/0.5/0.7)`
- `ReKV-w16(0.3/0.5/0.7)`
- `B-ReKV` 代表点:
  - `w8 tau=0.95 scale=0.75`
  - `w8 tau=0.95 scale=0.85`
  - `w16 tau=0.95 scale=0.90`

这是最应该优先跟跑的表。

### Table 2: 主文 selection strategy ablation

目的:证明 KVComm 的 layer selection 比 random layer selection 好。

模型对:

- pair #6

数据集:

- 8 个主数据集。

原论文方法:

- `Random(0.3/0.5/0.7)`
- `KVComm(0.3/0.5/0.7)`

我们对应做:

- `Random-token(0.3/0.5/0.7)`:随机 token 选择。
- `ValueNorm / Evict(0.3/0.5/0.7)`:query-agnostic token baseline。
- `ReKV-w8/w16(0.3/0.5/0.7)`:receiver-aware token selection。

这张表对应我们的核心消融:不是随机 token,不是 value-norm,而是 receiver-aware token selection 起作用。

### Table 3: Countries / Tipsheets prompt examples

只展示合成数据格式,不需要跑实验。我们可在 appendix 说明任务格式。

### Table 4: 主数据集统计

列 8 个数据集的 size。我们论文中需要对齐这个表,并标注是否使用同样样本数。

当前注意:

- 我们 QASPER 数据源/样本数需要最终确认。
- 如果样本数不同,必须在表格或实验设置里注明。

### Table 5: 9 个 model pairs

列出所有模型对。我们需要复用成自己的 ReKV model-pair evaluation plan。

### Table 6: extended tasks 通信结果

目的:在 extended datasets 上验证 robustness。

数据集:

- HotpotQA-E
- QASPER-E
- MuSiQuest-E
- SAMSum

模型对:

- pair #6
- pair #7
- pair #8

方法:

- `Baseline / Skyline / NLD / CIPHER / AC / KVComm(0.3/0.5/0.7)`

我们后续可补:

- `ReKV-w8/w16(0.3/0.5/0.7)`
- `B-ReKV` 代表点

优先级低于 Table 1 / Table 8,因为 full datasets 成本高。

### Table 7: extended datasets 统计

只需要记录,不需要跑。

### Table 8: Appendix 完整 communication results

目的:给 9 个 model pairs 的完整通信结果。

模型对:

- pair #1 - #9 全部。

数据集:

- 8 个主数据集。

方法:

- `Baseline`
- `Skyline`
- `NLD`
- `CIPHER`
- `AC(mean/replace/sum)`
- `KVComm(0.3/0.5/0.7)`

我们当前截图的 Llama-3.1-8B 同模型表就是 Table 8 中的 pair #1。

我们要补:

- 对每个 pair 跑 `ReKV-w8/w16 r=0.3/0.5/0.7`。
- 对重点 pair 跑 dense curve `r=0.1/0.2/0.3/0.4/0.5/0.7`。
- 对 pair #1/#6/#7/#8 跑 `B-ReKV`。

### Table 9: Appendix random selection 更多模型对

目的:补充 Table 2,证明 KVComm selection strategy 在更多 model pair 上好于 random。

涉及 model pairs:

- pair #1/#2/#3/#4/#5/#6/#7/#8

我们对应补:

- `Random-token` vs `ValueNorm/Evict` vs `ReKV`
- 优先在 pair #1/#6/#7/#8 做。

### Table 10: multi-source KVComm

目的:两个 sender + 一个 receiver,证明多源 KV 能融合。

任务:

- HotpotQA
- MuSiQuest
- 2WikiMQA

我们后续可以做:

- Multi-source ReKV:两个 sender 的 token KV 一起按 receiver attention score 选择。
- 这是很好的后续创新点,但不是当前第一优先级。

### Table 11: positional embedding coherence

目的:比较 `KVComm` 和 `KVComm-S`,其中 `KVComm-S` 表示 non-selected layers 不保持 position shift。

我们后续可以做:

- ReKV 的 `shift_back` / positional coherence ablation。
- 当前不是主表优先级。

### Figures 1-3: 方法与 hidden-state 动机

- Figure 1: KVComm framework。
- Figure 2: last token hidden state 最重要。
- Figure 3: prepending hidden states 只有 early-to-early 有效。

我们不需要完全复现,可以作为动机引用:KVComm 已证明 hidden state 不是理想通信载体,所以我们沿用 KV 通道,只改选择粒度和预算。

### Figures 4-6: contiguous chunk vs non-contiguous layer selection

目的:证明 KVComm 的 non-contiguous layer selection 优于一个连续 layer chunk。

我们可对应做:

- token-level contiguous span vs ReKV top-k token。
- 按 token 位置连续保留 recent/middle span,与 receiver top-k 比。

优先级中等。

### Figure 7: attention importance level

目的:选 attention importance score 高的层更好。

我们可对应做:

- receiver attention score 高的 token 更好。
- 按 score 分桶:top / middle / bottom token groups 传输,比较效果。

这很适合作为 ReKV mechanism figure。

### Figure 8: system efficiency

目的:比较 `AC / Skyline / KVComm` 的 FLOPs 和 memory。

原论文模型/数据集:

- Llama-3.2-3B pair
- Tipsheets
- MultiFieldQA-en

我们必须补:

- `ReKV / B-ReKV` 的 FLOPs 或 wall-clock / memory。
- 指标:communicated KV budget, A-prefill, receiver scoring, B-prefill, generation, peak memory。

这是后续论文 rebuttal 级别必需实验。

### Figure 11: calibration set size

目的:证明 KVComm 只需 1 个 calibration sample。

我们可以转化为优势:

- ReKV 不需要 calibration,每条 query 直接 receiver-aware 打分。
- 不一定需要复现。

### Figure 12: NLD transmitted token length

分析 NLD token 长度对结果影响。我们一般不需要复现。

### Figures 13-14: online calibration / layer ranking similarity

- Figure 13: mixed-task online calibration interval。
- Figure 14: dataset 间 layer ranking Kendall Tau。

我们可对应做 query-level budget/coverage distribution,但不是第一优先级。

---

## 3. 我们要跑的对齐实验矩阵

### Priority A: 必跑,对齐主文 Table 1

目标:在 KVComm 主文 3 个 fine-tuned model pairs 上跑我们的 ReKV / B-ReKV。

模型对:

- pair #6
- pair #7
- pair #8

数据集:

- Countries
- Tipsheets
- HotpotQA
- QASPER
- MuSiQue
- MultiFieldQA-en
- 2WikiMQA
- TMATH

方法:

- `ReKV-w8 r=0.3/0.5/0.7`
- `ReKV-w16 r=0.3/0.5/0.7`
- `B-ReKV`:
  - `w8 tau=0.95 scale=0.75`
  - `w8 tau=0.95 scale=0.85`
  - `w16 tau=0.95 scale=0.90`

输出表:

```text
Table A: KVComm Table 1 + Ours
```

### Priority B: 必跑,对齐 Appendix Table 8 pair #1

目标:补完 Llama-3.1-8B same-model 的完整表。

模型对:

- pair #1

当前状态:

- 大部分 ReKV / Coverage 正在跑或已有。
- `scripts/run_table_completion_2x5.sh` 正在补 B-ReKV 表格三行。

输出表:

```text
Table B: Llama-3.1-8B same-model KVComm Table 8 block + Ours
```

这是我们当前最接近完成的一张表。

### Priority C: Selection ablation,对齐 Table 2 / Table 9

目标:把 KVComm 的 random layer selection 消融,对应成我们的 token selection 消融。

模型对:

- 先 pair #1。
- 再 pair #6/#7/#8。

数据集:

- HotpotQA
- MuSiQue
- MultiFieldQA-en
- Countries/Tipsheets 作为简单任务对照。

方法:

- `Random-token`
- `ValueNorm / Evict`
- `ReKV-w8`
- `ReKV-w16`

预算:

- `r=0.3/0.5/0.7`

输出表:

```text
Table C: Token selection ablation
```

### Priority D: Efficiency,对齐 Figure 8

目标:证明 ReKV/B-ReKV 不只是精度高,通信和计算也有优势。

模型:

- 先 pair #1 Llama-3.1-8B。
- 如成本允许,再 pair #2 Llama-3.2-3B,与原 Figure 8 更一致。

数据集:

- Tipsheets
- MultiFieldQA-en
- MuSiQue 或 HotpotQA 加一个难任务。

方法:

- Skyline
- KVComm(0.3/0.5/0.7),可引用或少量重跑。
- ReKV-w8/w16
- B-ReKV

指标:

- communicated KV budget
- A-prefill time
- receiver scoring time
- B-prefill time
- generation time
- peak memory

输出图:

```text
Figure Efficiency: Accuracy vs budget/time/memory
```

### Priority E: Appendix full coverage,对齐 Table 8 全 9 模型对

目标:把 ReKV 方法扩展到 KVComm 论文所有 9 个模型对。

建议分阶段:

1. same-model pairs:
   - #1 Llama-3.1-8B
   - #2 Llama-3.2-3B
   - #3 Qwen2.5-7B
   - #4 Falcon3-7B
2. fine-tuned main pairs:
   - #6/#7/#8
3. 其它:
   - #5/#9

最小跑法:

- 每个 pair 先只跑:
  - `ReKV-w16 r=0.3/0.5/0.7`
  - `ReKV-w8 r=0.3/0.5/0.7`
- coverage 只对 pair #1/#6/#7/#8 跑。

---

## 4. 当前我们已经做了什么

### 已完成或正在完成:pair #1 Llama-3.1-8B same-model

模型:

```text
M_s = /sharedspace/models/Llama-3.1-8B-Instruct
M_r = /sharedspace/models/Llama-3.1-8B-Instruct
```

已做:

- KVComm 原论文 Table 8 pair #1 的 baseline 结果已转录到 `snapshots/RESULTS.md`。
- Merge / Evict / ReKV-w8 / ReKV-w16 在多个预算点已跑。
- Budget-aware 失败路线已完成:
  - Step 0 oracle headroom
  - Step 1 open-loop
  - Step 2a offline oracle progressive
  - Step 2b online progressive
  - Pass-1 predictor
- B-ReKV 已在 MuSiQue / HotpotQA / MultiFieldQA-en 有正结果。
- Table 1 当前缺口正在用 GPU2/GPU5 补:
  - Countries
  - Tipsheets
  - QASPER
  - 2WikiMQA
  - TMATH

### 当前未做:KVComm 主文 Table 1 的 3 个 fine-tuned pairs

还没有系统跑:

- pair #6
- pair #7
- pair #8

这是下一阶段最重要的实验。

---

## 5. 后续 TODO

### TODO 1: 先完成当前 pair #1 表

- [ ] 等 `scripts/run_table_completion_2x5.sh` 跑完。
- [ ] 汇总 `analyze_coverage.py`。
- [ ] 更新 `snapshots/RESULTS.md` 和论文 Table B。
- [ ] 标明这是 KVComm Appendix Table 8 pair #1 对齐表。

### TODO 2: 准备模型下载 / 路径检查

需要确认本机是否已有以下模型:

- [ ] `meta-llama/Llama-3.2-3B-Instruct`
- [ ] `Qwen/Qwen2.5-7B-Instruct`
- [ ] `tiiuae/Falcon3-7B-Instruct`
- [ ] `huihui-ai/Llama-3.2-3B-Instruct-abliterated`
- [ ] `suayptalha/DeepSeek-R1-Distill-Llama-3B`
- [ ] `Orion-zhen/Qwen2.5-7B-Instruct-Uncensored`
- [ ] `bespokelabs/Bespoke-Stratos-7B`
- [ ] `ehristoforu/falcon3-ultraset`
- [ ] `huihui-ai/Falcon3-7B-Instruct-abliterated`

### TODO 3: 跑 KVComm 主文 Table 1 对齐实验

对 pair #6/#7/#8 跑:

- [ ] ReKV-w8 `r=0.3/0.5/0.7`
- [ ] ReKV-w16 `r=0.3/0.5/0.7`
- [ ] B-ReKV 三个代表点

数据集:

- [ ] Countries
- [ ] Tipsheets
- [ ] HotpotQA
- [ ] QASPER
- [ ] MuSiQue
- [ ] MultiFieldQA-en
- [ ] 2WikiMQA
- [ ] TMATH

### TODO 4: Selection ablation

- [ ] 实现/确认 random-token selection。
- [ ] 对 pair #1 跑 Random-token vs Evict vs ReKV。
- [ ] 对 pair #6/#7/#8 选 2-3 个任务补。

### TODO 5: Efficiency figure

- [ ] 加单样本 timing logger。
- [ ] 记录 A-prefill / receiver scoring / B-prefill / generation。
- [ ] 记录 peak memory。
- [ ] 画 accuracy-budget-time/memory 图。

### TODO 6: Receiver-Initiated KV Communication / reviewer-risk ablations

目的:把 ReKV 从一个"query attention trick"升级为完整的 receiver-initiated KV communication protocol,避免被误解为在 KVComm query-blind sender 设定下不公平对比。

- [ ] Problem setting:定义 **Receiver-Initiated KV Communication**: receiver sends a lightweight query sketch to the context holder; sender returns query-relevant KV, not an answer。
- [ ] RAG/Agent 对齐:说明该 setting 类似 query-aware retrieval / memory server,区别是返回 selected KV 而不是 text chunks。
- [ ] Query sketch size ablation:优先跑 `recv_window={4,8,16,32,all}`;报告 accuracy、A→B selected KV bytes、B→A sketch token/bytes、total communication。
- [ ] Communication accounting:把 B→A 的 query sketch token/bytes 计入总通信量,与 A→B selected KV bytes 一起报;证明 sketch overhead 相对 context KV 很小。
- [ ] ReKV vs Evict:二者都是 token-level evict-only,区别只有 receiver query signal,用于证明 query-aware selection 有效。
- [ ] ReKV vs Random-token:证明不是任意 token 子集都能工作。
- [ ] 可选 Sender-only baseline:如实现成本低,补 sender-only/value-norm/sender-attention 与 receiver-query attention 对照。
- [ ] Evidence sparsity analysis:证明证据主要稀疏分布在 token 上而非 layer 上,可复用 attention concentration / evidence overlap / top-token case study。
- [ ] "Why not let sender answer?" 写入 discussion:sender 是 context holder / memory server; receiver 是 user-facing reasoning model,拥有 instruction-following、tool-use、safety 或最终回答职责。
- [ ] 论文措辞:避免说 "same setting as KVComm";改写为 "we extend KVComm from query-agnostic layer-wise KV sharing to receiver-aware token-wise KV communication"。

优先数据集:

```text
hotpotqa, musique, multifieldqa_en
```

优先模型:

```text
pair #1 Llama-3.1 same-model first; then pair #6/#7 representative fine-tuned pairs if needed.
```

最小执行顺序:

```text
1. recv_window={4,8,16,32,all} on hotpotqa/musique/multifieldqa_en.
2. Query sketch overhead accounting table.
3. ReKV vs Evict vs Random-token table.
4. Evidence concentration / top-token examples.
```

### TODO 7: Appendix expansion

- [ ] 对 same-model #2/#3/#4 跑 ReKV。
- [ ] 对 fine-tuned #5/#9 跑 ReKV。
- [ ] 只在主模型对跑 B-ReKV,避免实验量爆炸。

---

## 6. 实验量估算

### 对齐主文 Table 1 的最小量

3 个 model pairs × 8 个 datasets × 2 windows × 3 fixed budgets:

```text
3 * 8 * 2 * 3 = 144 runs
```

Coverage 三点:

```text
3 model pairs * 8 datasets * 3 coverage settings = 72 runs
```

合计最小:

```text
216 runs
```

这还不含模型下载、OOM 重试、QASPER fixed curve 等。

### 推荐执行顺序

先不要一次性全跑 216 个。建议:

1. pair #1 当前表收尾。
2. pair #6 先跑 HotpotQA / MuSiQue / MultiFieldQA-en 三个任务。
3. 如果 pair #6 正向,再补 pair #6 全 8 任务。
4. 再跑 pair #7/#8 的三任务小集。
5. 最后扩展到完整 8 任务。

---

## 7. 论文叙事建议

不要说“复现 KVComm 全部实验”。更准确:

```text
We use the original KVComm experimental protocol and reported baselines as the comparison frame.
On the same datasets and model-pair settings, we evaluate our receiver-aware token-level KV selection and dynamic coverage budgeting.
```

中文:

```text
我们沿用 KVComm 原论文的实验版图:同样的数据集、同样的模型对、同样的预算点。
原论文已有的 KVComm / Baseline / Skyline / AC / NLD / CIPHER 结果作为对照。
我们新增运行 ReKV 和 B-ReKV,看在同一设置下 token 级 receiver-aware 选择是否超过层级 KVComm。
```

最关键的对比句:

```text
KVComm shows that KV cache is an effective communication medium.
ReKV shows that, under the same KV communication frame, receiver-aware token selection is more bandwidth-efficient than static layer selection.
B-ReKV further replaces fixed communication ratios with receiver-evidence coverage constraints.
```

---

## 合并来源 3：第三部分：实验运行状态（2026-06-25）

> 原文件：`snapshots/EXPERIMENT_RUN_STATUS_2026-06-25.md`。以下为该文件正文内容，实验数值未改动。

# Experiment Run Status - 2026-06-25 13:08

This note records the current queue status parsed from `logs/` and the Table 8 model-pair snapshot logs.

## Queue Summary

| Queue / Log | Model Pair | GPU | Status | Notes |
|---|---|---:|---|---|
| `logs/gpu2_table_queue.log` | Llama-3.1-8B-Instruct same-model | 2 | Done | B-ReKV table points for `countries`, `tipsheets`, `twowikimqa` finished. |
| `logs/gpu5_table_queue_resume.log` | Llama-3.1-8B-Instruct same-model | 5 | Done | QASPER fixed-r tail, QASPER coverage, TMATH coverage, and HotpotQA extra finished. |
| `snapshots/table8_pair2_llama32_same/logs/gpu2_pair2_0624_2358.log` | Llama-3.2-3B-Instruct same-model | 2 | Done | All 8 datasets finished for ReKV and B-ReKV. |
| `snapshots/table8_pair3_qwen25_7b_same/logs/gpu6_pair3_0625_0001.log` | Qwen2.5-7B-Instruct same-model | 6 | Running | Finished through TMATH ReKV. Currently running TMATH coverage `w8 tau=0.95 scale=0.75`. |

Current active KVComm process:

```text
GPU6: python com.py --test_task tmath ... --budget_mode coverage --coverage_tau 0.95 --coverage_scale 0.75 --recv_window 8
```

GPU5 has another non-KVComm Python process, not from these experiment queues.

## Llama-3.1 Same-Model Table Completion

### `logs/gpu2_table_queue.log` - Done

B-ReKV completed:

| Dataset | Setting | Result |
|---|---|---:|
| `countries` | `w8 tau=0.95 scale=0.75` | 0.6150 |
| `countries` | `w8 tau=0.95 scale=0.85` | 0.6150 |
| `countries` | `w16 tau=0.95 scale=0.90` | 0.6000 |
| `tipsheets` | `w8 tau=0.95 scale=0.75` | 0.8780 |
| `tipsheets` | `w8 tau=0.95 scale=0.85` | 0.8780 |
| `tipsheets` | `w16 tau=0.95 scale=0.90` | 0.9060 |
| `twowikimqa` | `w8 tau=0.95 scale=0.75` | 0.4100 |
| `twowikimqa` | `w8 tau=0.95 scale=0.85` | 0.4100 |
| `twowikimqa` | `w16 tau=0.95 scale=0.90` | 0.4100 |

Finished at `2026-06-24 22:22:42`.

### `logs/gpu5_table_queue_resume.log` - Done

QASPER fixed-r tail and remaining coverage points completed:

| Dataset | Method / Setting | Result |
|---|---|---:|
| `qasper` | ReKV `w16 r=0.8` | 0.3480 |
| `qasper` | ReKV `w16 r=0.9` | 0.3420 |
| `qasper` | Coverage `w8 tau=0.95 scale=0.75` | 0.3320 |
| `qasper` | Coverage `w8 tau=0.95 scale=0.85` | 0.3260 |
| `qasper` | Coverage `w16 tau=0.95 scale=0.90` | 0.3380 |
| `tmath` | Coverage `w8 tau=0.95 scale=0.75` | 0.3481 |
| `tmath` | Coverage `w8 tau=0.95 scale=0.85` | 0.3523 |
| `tmath` | Coverage `w16 tau=0.95 scale=0.90` | 0.3525 |
| `hotpotqa` | Coverage `w8 tau=0.95 scale=0.85` | 0.7160 |

Finished at `2026-06-25 04:32:06`.

## Table 8 Pair #2: Llama-3.2-3B-Instruct Same-Model

Log: `snapshots/table8_pair2_llama32_same/logs/gpu2_pair2_0624_2358.log`

Status: done at `2026-06-25 12:18:54`.

This queue finished all 8 datasets:

```text
countries
tipsheets
hotpotqa
musique
multifieldqa_en
twowikimqa
qasper
tmath
```

For each dataset it ran:

```text
ReKV: w8/w16 x r in {0.3, 0.5, 0.7}
B-ReKV: w8 tau=0.95 scale=0.75
B-ReKV: w8 tau=0.95 scale=0.85
B-ReKV: w16 tau=0.95 scale=0.90
```

No `Traceback`, `RuntimeError`, `ImportError`, or OOM marker was found in this log.

## Table 8 Pair #3: Qwen2.5-7B-Instruct Same-Model

Log: `snapshots/table8_pair3_qwen25_7b_same/logs/gpu6_pair3_0625_0001.log`

Status: running.

Completed datasets so far:

```text
countries
tipsheets
hotpotqa
musique
multifieldqa_en
twowikimqa
qasper
```

TMATH completed fixed-r ReKV:

| Dataset | Method / Setting | Result |
|---|---|---:|
| `tmath` | ReKV `w8 r=0.3` | 0.3112 |
| `tmath` | ReKV `w8 r=0.5` | 0.3111 |
| `tmath` | ReKV `w8 r=0.7` | 0.3172 |
| `tmath` | ReKV `w16 r=0.3` | 0.3099 |
| `tmath` | ReKV `w16 r=0.5` | 0.3164 |
| `tmath` | ReKV `w16 r=0.7` | 0.3162 |

Currently running:

```text
tmath B-ReKV: w8 tau=0.95 scale=0.75
```

Remaining in this queue after the current run:

```text
tmath B-ReKV: w8 tau=0.95 scale=0.85
tmath B-ReKV: w16 tau=0.95 scale=0.90
```

No `Traceback`, `RuntimeError`, `ImportError`, or OOM marker has appeared in this log so far.

## Immediate Next Check

After GPU6 finishes, check:

```bash
rg "DONE|Traceback|RuntimeError|ImportError|CUDA out of memory|communication result: [0-9]\\.[0-9]{4}, communication time" snapshots/table8_pair3_qwen25_7b_same/logs/gpu6_pair3_0625_0001.log
```

The queue is complete when the log contains:

```text
######## [GPU6 pair3 qwen25_7b_same] DONE ...
```

## 2026-06-25 晚间最新补充:Table 8 pair #2/#3 与 Table 1 pair #6

> 更新时间:2026-06-25 18:48。以下结果来自最新 `snapshots/table*/logs/*.log`。  
> 记号:`ReKV-w8` / `ReKV-w16` 列中三个数依次为 `r=0.3/0.5/0.7`;`Coverage` 列中三个数依次为 `w8-t0.95-s0.75 / w8-t0.95-s0.85 / w16-t0.95-s0.90`。

### Table 8 pair #2:Llama-3.2-3B-Instruct same-model(已完成)

模型对:

```text
M_s = /sharedspace/models/Llama-3.2-3B-Instruct
M_r = /sharedspace/models/Llama-3.2-3B-Instruct
```

日志:`snapshots/table8_pair2_llama32_same/logs/gpu2_pair2_0624_2358.log`。状态:已完成,`2026-06-25 12:18:54`。

| Dataset | ReKV-w8 r=.3/.5/.7 | ReKV-w16 r=.3/.5/.7 | B-ReKV 3 pts |
|---|---|---|---|
| `countries` | 0.505 / 0.565 / 0.565 | 0.500 / 0.565 / 0.570 | 0.530 / 0.525 / 0.565 |
| `tipsheets` | 0.420 / 0.610 / 0.782 | 0.554 / 0.666 / 0.792 | 0.434 / 0.446 / 0.552 |
| `hotpotqa` | 0.654 / 0.702 / 0.692 | 0.666 / 0.696 / 0.696 | 0.642 / 0.652 / 0.676 |
| `musique` | 0.350 / 0.356 / 0.376 | 0.338 / 0.370 / 0.384 | 0.340 / 0.342 / 0.344 |
| `multifieldqa_en` | 0.513 / 0.500 / 0.507 | 0.500 / 0.513 / 0.493 | 0.567 / 0.560 / 0.533 |
| `twowikimqa` | 0.395 / 0.415 / 0.415 | 0.395 / 0.415 / 0.415 | 0.380 / 0.380 / 0.410 |
| `qasper` | 0.320 / 0.324 / 0.330 | 0.328 / 0.330 / 0.318 | 0.316 / 0.330 / 0.330 |
| `tmath` | 0.341 / 0.350 / 0.353 | 0.344 / 0.346 / 0.353 | 0.340 / 0.340 / 0.345 |

初步观察:pair #2 上 B-ReKV 在 `multifieldqa_en` 明显强于固定 ReKV;`hotpotqa/qasper/tmath` 基本接近固定 ReKV;`tipsheets/musique/twowikimqa` 的三个 coverage 代表点偏保守或偏低,后续若要主打 pair #2 需要补更合适的 `tau/scale`。

### Table 8 pair #3:Qwen2.5-7B-Instruct same-model(已完成)

模型对:

```text
M_s = /sharedspace/models/Qwen2.5-7B-Instruct
M_r = /sharedspace/models/Qwen2.5-7B-Instruct
```

日志:`snapshots/table8_pair3_qwen25_7b_same/logs/gpu6_pair3_0625_0001.log`。状态:已完成,`2026-06-25 15:55:53`。

| Dataset | ReKV-w8 r=.3/.5/.7 | ReKV-w16 r=.3/.5/.7 | B-ReKV 3 pts |
|---|---|---|---|
| `countries` | 0.025 / 0.210 / 0.360 | 0.020 / 0.265 / 0.360 | 0.270 / 0.260 / 0.295 |
| `tipsheets` | 0.890 / 0.912 / 0.948 | 0.916 / 0.932 / 0.940 | 0.848 / 0.872 / 0.866 |
| `hotpotqa` | 0.532 / 0.634 / 0.698 | 0.602 / 0.658 / 0.668 | 0.580 / 0.606 / 0.666 |
| `musique` | 0.340 / 0.434 / 0.458 | 0.370 / 0.454 / 0.446 | 0.354 / 0.356 / 0.410 |
| `multifieldqa_en` | 0.460 / 0.493 / 0.487 | 0.480 / 0.460 / 0.487 | 0.440 / 0.460 / 0.473 |
| `twowikimqa` | 0.440 / 0.445 / 0.450 | 0.435 / 0.465 / 0.460 | 0.420 / 0.395 / 0.440 |
| `qasper` | 0.332 / 0.340 / 0.342 | 0.338 / 0.340 / 0.340 | 0.296 / 0.296 / 0.312 |
| `tmath` | 0.311 / 0.311 / 0.317 | 0.310 / 0.316 / 0.316 | 0.312 / 0.318 / 0.311 |

初步观察:pair #3 的固定 ReKV 在 `tipsheets/hotpotqa/musique/twowikimqa/qasper` 上整体稳定;B-ReKV 在 `tmath` 的 `w8-s0.85` 略高于固定 ReKV 最优,但在 `qasper/tipsheets/musique` 上代表点偏保守,需要依赖后续平均预算分析判断是否有 Pareto 优势。

### Table 8 pair #4:Falcon3-7B-Instruct same-model(已完成)

模型对:

```text
M_s = /NAS/models/Falcon3-7B-Instruct
M_r = /NAS/models/Falcon3-7B-Instruct
```

日志:

- 初始 GPU1 队列 `snapshots/table8_pair4_falcon3_7b_same/logs/gpu1_pair4_0702_1507.log` 在本地数据集缺失时停在 `hotpotqa`。
- 续跑 GPU2 队列 `snapshots/table8_pair4_falcon3_7b_same/logs/gpu2_pair4_0702_1558.log` 已完成,`2026-07-02 21:44:01`。

状态:8 datasets x 9 paper-table runs 完整。

| Dataset | ReKV-w8 r=.3/.5/.7 | ReKV-w16 r=.3/.5/.7 | B-ReKV 3 pts |
|---|---|---|---|
| `countries` | 0.045 / 0.360 / 0.445 | 0.040 / 0.395 / 0.425 | 0.270 / 0.315 / 0.365 |
| `tipsheets` | 0.804 / 0.930 / 0.942 | 0.664 / 0.922 / 0.944 | 0.908 / 0.922 / 0.900 |
| `hotpotqa` | 0.534 / 0.616 / 0.606 | 0.554 / 0.604 / 0.624 | 0.556 / 0.570 / 0.598 |
| `musique` | 0.382 / 0.394 / 0.416 | 0.372 / 0.402 / 0.418 | 0.374 / 0.366 / 0.386 |
| `multifieldqa_en` | 0.467 / 0.453 / 0.473 | 0.433 / 0.447 / 0.480 | 0.413 / 0.400 / 0.440 |
| `twowikimqa` | 0.380 / 0.405 / 0.410 | 0.395 / 0.385 / 0.405 | 0.385 / 0.400 / 0.395 |
| `qasper` | 0.236 / 0.242 / 0.248 | 0.230 / 0.236 / 0.252 | 0.226 / 0.214 / 0.226 |
| `tmath` | 0.300 / 0.305 / 0.307 | 0.299 / 0.304 / 0.304 | 0.302 / 0.300 / 0.301 |

初步观察:Falcon3 same-model 上固定 ReKV 在 `tipsheets/hotpotqa/musique/multifieldqa_en` 表现稳定;B-ReKV 代表点整体偏保守,但在 `tipsheets` 和 `countries` 上仍有可用信号。

### Table 8 pair #5:EvolCodeLlama -> ToolACE(已完成)

模型对:

```text
M_s = /NAS/models/EvolCodeLlama-3.1-8B-Instruct
M_r = /NAS/models/ToolACE-2-Llama-3.1-8B
```

日志:`snapshots/table8_pair5_then_pair9_gpu3_0702_1601.log`。状态:已完成,`2026-07-02 22:22:45`。

实现注意:本地 `EvolCodeLlama-3.1-8B-Instruct` 目录包含 full weights 和 PEFT adapter metadata。运行脚本使用 no-adapter symlink view 避免 `AutoModelForCausalLM` 联网加载 `meta-llama/Meta-Llama-3.1-8B-Instruct` base。

| Dataset | ReKV-w8 r=.3/.5/.7 | ReKV-w16 r=.3/.5/.7 | B-ReKV 3 pts |
|---|---|---|---|
| `countries` | 0.330 / 0.590 / 0.580 | 0.380 / 0.605 / 0.595 | 0.165 / 0.290 / 0.320 |
| `tipsheets` | 0.852 / 0.908 / 0.906 | 0.906 / 0.906 / 0.902 | 0.848 / 0.868 / 0.914 |
| `hotpotqa` | 0.554 / 0.580 / 0.614 | 0.538 / 0.614 / 0.618 | 0.552 / 0.552 / 0.542 |
| `musique` | 0.320 / 0.334 / 0.348 | 0.314 / 0.334 / 0.340 | 0.310 / 0.308 / 0.322 |
| `multifieldqa_en` | 0.567 / 0.560 / 0.567 | 0.553 / 0.560 / 0.560 | 0.553 / 0.547 / 0.547 |
| `twowikimqa` | 0.350 / 0.360 / 0.385 | 0.360 / 0.380 / 0.390 | 0.320 / 0.330 / 0.365 |
| `qasper` | 0.244 / 0.256 / 0.262 | 0.234 / 0.252 / 0.258 | 0.232 / 0.242 / 0.236 |
| `tmath` | 0.361 / 0.368 / 0.372 | 0.357 / 0.364 / 0.375 | 0.361 / 0.363 / 0.358 |

初步观察:pair #5 是较干净的 same-base fine-tuned pair 结果;固定 ReKV 在多数任务上随预算增加稳定提升。B-ReKV 代表点在 `tipsheets` 可达到或超过固定 ReKV,但在 `countries/hotpotqa/musique/twowikimqa/qasper` 上偏保守。

### Table 8 pair #9:SuperNova -> DeepSeek-R1-Distill-Llama-8B(已完成)

模型对:

```text
M_s = /NAS/models/Llama-3.1-SuperNova-Lite
M_r = /NAS/models/DeepSeek-R1-Distill-Llama-8B
```

日志:`snapshots/table8_pair5_then_pair9_gpu3_0702_1601.log`。状态:已完成,`2026-07-03 09:04:02`。

| Dataset | ReKV-w8 r=.3/.5/.7 | ReKV-w16 r=.3/.5/.7 | B-ReKV 3 pts |
|---|---|---|---|
| `countries` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| `tipsheets` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| `hotpotqa` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| `musique` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| `multifieldqa_en` | 0.020 / 0.013 / 0.013 | 0.013 / 0.013 / 0.013 | 0.020 / 0.013 / 0.013 |
| `twowikimqa` | 0.005 / 0.005 / 0.005 | 0.005 / 0.005 / 0.005 | 0.005 / 0.005 / 0.005 |
| `qasper` | 0.002 / 0.002 / 0.002 | 0.000 / 0.002 / 0.002 | 0.000 / 0.000 / 0.000 |
| `tmath` | 0.325 / 0.322 / 0.327 | 0.325 / 0.323 / 0.328 | 0.321 / 0.325 / 0.324 |

最终处理:pair #9 已完整跑完,但除 `tmath` 外结果几乎全为 0。2026-07-05 额外做了 raw-output probe 和 KVComm top=0.3 limit=50 probe。raw-output 显示模型并非乱码或模板污染,但经常选错 evidence 或只完成中间 hop；KVComm probe 也很差: `countries=0.120`, `tipsheets=0.560`, `hotpotqa=0.060`, `musique=0.020`, `qasper=0.000`。因此该异常不是 ReKV/B-ReKV 独有问题,更像 SuperNova -> DeepSeek-Llama-8B 这个 hard heterogeneous pair 的 KV 兼容/长上下文 grounding 问题。论文中暂缓把 pair #9 作为正向泛化对比,最多作为 hard negative / limitation 附录说明。

### Table 1 pair #6:Llama-3.2 abliterated -> DeepSeek-R1-Distill-Llama-3B(已完成)

模型对:

```text
M_s = /sharedspace/models/Llama-3.2-3B-Instruct-abliterated
M_r = /sharedspace/models/DeepSeek-R1-Distill-Llama-3B
```

日志:`snapshots/table1_pair6_llama32_abliterated_deepseek3b/logs/gpu7_pair6_0625_1339.log`。状态:已完成,`2026-06-26 01:23:39`。

| Dataset | ReKV-w8 r=.3/.5/.7 | ReKV-w16 r=.3/.5/.7 | B-ReKV 3 pts |
|---|---|---|---|
| `countries` | 0.375 / 0.520 / 0.575 | 0.370 / 0.515 / 0.575 | 0.490 / 0.505 / 0.500 |
| `tipsheets` | 0.486 / 0.594 / 0.736 | 0.484 / 0.604 / 0.716 | 0.410 / 0.412 / 0.470 |
| `hotpotqa` | 0.646 / 0.702 / 0.718 | 0.668 / 0.694 / 0.716 | 0.642 / 0.640 / 0.674 |
| `musique` | 0.362 / 0.384 / 0.384 | 0.354 / 0.390 / 0.396 | 0.384 / 0.370 / 0.368 |
| `multifieldqa_en` | 0.467 / 0.480 / 0.460 | 0.467 / 0.487 / 0.460 | 0.467 / 0.487 / 0.493 |
| `twowikimqa` | 0.395 / 0.425 / 0.420 | 0.410 / 0.420 / 0.420 | 0.400 / 0.395 / 0.410 |
| `qasper` | 0.306 / 0.302 / 0.310 | 0.316 / 0.326 / 0.306 | 0.300 / 0.306 / 0.324 |
| `tmath` | 0.318 / 0.321 / 0.323 | 0.317 / 0.321 / 0.320 | 0.317 / 0.319 / 0.317 |

初步观察:这是 KVComm 主文 Table 1 的优先 pair #6。固定 ReKV 在 `hotpotqa/musique/twowikimqa/qasper` 有较稳曲线;B-ReKV 在 `multifieldqa_en` 接近或略高于固定 ReKV,在 `hotpotqa/qasper` 接近固定 ReKV,在 `tipsheets/tmath` 偏低。

### KVComm 对比与 Coverage 平均预算收益

> 下表把每个模型对的最佳 ReKV / B-ReKV 与原论文 KVComm 最佳值对比。  
> `best ReKV` 和 `best Coverage` 使用原始 `communication result`/平均分数。`avg budget` 来自每条样本的实际 `budget` 字段,包含 sink/recent 等实际保留比例。  
> `equal-acc saving` 沿用 `scripts/analyze_coverage.py` 的口径:以 `score >= 0.5` 作为答对阈值,在 fixed ReKV-w16 曲线上插值得到同等 accuracy 所需固定预算,再计算 `(1 - avg_budget / fixed_budget)`. 这对 F1 型 QA 合理;对 `tmath` 的 ROUGE-L Recall 仅作粗略参考。

#### Table 8 pair #1:Llama-3.1-8B-Instruct same-model

| Dataset | KVComm best | best ReKV | best B-ReKV | Coverage budget / saving |
|---|---:|---:|---:|---|
| `countries` | 0.62 | 0.61 | 0.615 | avg budget 0.327; saving n/a |
| `tipsheets` | 0.96 | 0.89 | 0.906 | avg budget 0.370; saving n/a |
| `hotpotqa` | 0.69 | 0.748 | 0.746 | avg budget 0.643; best saving +7.4% |
| `qasper` | 0.29 | 0.348 | 0.338 | avg budget 0.329; best saving +27.2% |
| `musique` | 0.39 | 0.478 | 0.490 | avg budget 0.313; best saving +40.7% |
| `multifieldqa_en` | 0.54 | 0.540 | 0.540 | avg budget 0.099; saving n/a |
| `twowikimqa` | 0.38 | 0.440 | 0.420 | avg budget 0.108; best saving +64.1% |
| `tmath` | 0.38 | n/a | 0.167 | avg budget 0.300; saving n/a |

结论:pair #1 是 ReKV 主正结果来源,`hotpotqa/musique/qasper/twowikimqa` 均超过 KVComm;B-ReKV 在 `musique/twowikimqa/qasper` 显示出明显预算节省潜力。

#### Table 8 pair #2:Llama-3.2-3B-Instruct same-model

| Dataset | KVComm best | best ReKV | best B-ReKV | Coverage budget / saving |
|---|---:|---:|---:|---|
| `countries` | 0.57 | 0.570 | 0.565 | avg budget 0.329; best saving +36.5% |
| `tipsheets` | 0.80 | 0.792 | 0.552 | avg budget 0.295; best saving +33.1% |
| `hotpotqa` | 0.65 | 0.702 | 0.676 | avg budget 0.354; best saving +15.4% |
| `qasper` | 0.27 | 0.330 | 0.330 | avg budget 0.287; best saving +63.8% |
| `musique` | 0.29 | 0.384 | 0.344 | avg budget 0.362; best saving +22.6% |
| `multifieldqa_en` | 0.48 | 0.513 | 0.567 | avg budget 0.237; acc above fixed curve |
| `twowikimqa` | 0.35 | 0.415 | 0.410 | avg budget 0.302; best saving +35.7% |
| `tmath` | 0.37 | 0.353 | 0.345 | avg budget 0.236; best saving +27.3%* |

结论:pair #2 上 ReKV 在 5/8 个任务超过 KVComm 最佳值,尤其 `hotpotqa/qasper/musique/multifieldqa_en/twowikimqa`;B-ReKV 的平均预算通常落在 0.24-0.36,明显低于固定 `r=0.5/0.7`。

#### Table 8 pair #3:Qwen2.5-7B-Instruct same-model

| Dataset | KVComm best | best ReKV | best B-ReKV | Coverage budget / saving |
|---|---:|---:|---:|---|
| `countries` | 0.57 | 0.360 | 0.295 | avg budget 0.379; best saving +35.4% |
| `tipsheets` | 0.98 | 0.948 | 0.872 | avg budget 0.305; best saving +15.2% |
| `hotpotqa` | 0.72 | 0.698 | 0.666 | avg budget 0.430; best saving +36.0% |
| `qasper` | 0.29 | 0.342 | 0.312 | avg budget 0.349; best saving +23.3% |
| `musique` | 0.48 | 0.458 | 0.410 | avg budget 0.411; best saving +22.2% |
| `multifieldqa_en` | 0.45 | 0.493 | 0.473 | avg budget 0.307; best saving +60.7% |
| `twowikimqa` | 0.35 | 0.465 | 0.440 | avg budget 0.278; best saving +43.2% |
| `tmath` | 0.33 | 0.318 | 0.318 | avg budget 0.299; best saving +39.8%* |

结论:Qwen same-model 上 ReKV 不如 Llama 稳,但 `qasper/multifieldqa_en/twowikimqa` 明确超过 KVComm;`hotpotqa/musique/tipsheets` 接近但未超过 KVComm 最佳值。

#### Table 1 pair #6:Llama-3.2 abliterated -> DeepSeek-R1-Distill-Llama-3B

| Dataset | KVComm best | best ReKV | best B-ReKV | Coverage budget / saving |
|---|---:|---:|---:|---|
| `countries` | 0.57 | 0.575 | 0.505 | avg budget 0.374; best saving +26.9% |
| `tipsheets` | 0.81 | 0.736 | 0.470 | avg budget 0.276; best saving +39.6% |
| `hotpotqa` | 0.65 | 0.718 | 0.674 | avg budget 0.368; best saving +16.4% |
| `qasper` | 0.29 | 0.326 | 0.324 | avg budget 0.343; best saving +64.4% |
| `musique` | 0.36 | 0.396 | 0.384 | avg budget 0.251; best saving +48.4% |
| `multifieldqa_en` | 0.51 | 0.487 | 0.493 | avg budget 0.328; best saving +50.4% |
| `twowikimqa` | 0.37 | 0.425 | 0.410 | avg budget 0.302; best saving +32.3% |
| `tmath` | 0.35 | 0.323 | 0.319 | avg budget 0.218; best saving +43.9%* |

结论:pair #6 是对齐 KVComm 主文 Table 1 的关键正结果。ReKV 在 `hotpotqa/qasper/musique/twowikimqa` 超过 KVComm,`countries` 打平略高;`tipsheets/tmath` 低于 KVComm。B-ReKV 虽然多数任务 accuracy 略低于 best fixed ReKV,但平均预算只有约 0.22-0.37,在 `qasper/musique/multifieldqa_en` 上给出 48%-64% 的等精度预算节省信号。

带 `*` 的 TMATH saving 只作参考,因为该任务原指标是 ROUGE-L Recall,`score >= 0.5` 的二值化不如 F1 QA 任务稳定。

### 2026-07-01 补充:Table 1 pair #7 TMATH fixed ReKV(已完成)

模型对:

```text
M_s = /sharedspace/models/Qwen2.5-7B-Instruct-Uncensored
M_r = /sharedspace/models/Bespoke-Stratos-7B
```

日志目录:`snapshots/table1_pair7_qwen25_uncensored_bespoke/tmath/mtc_receiver/`。状态:固定 ReKV 的 `w8/w16 × r=0.3/0.5/0.7` 六个点均已完成,每个 run 写出 `per_sample.jsonl`,样本数 301。

| Dataset | ReKV-w8 r=.3/.5/.7 | ReKV-w16 r=.3/.5/.7 |
|---|---|---|
| `tmath` | 0.3274 / 0.3301 / 0.3342 | 0.3311 / 0.3356 / 0.3309 |

运行时间:

| Run | communication time |
|---|---:|
| `recv_w8_r0.3_0630_1505` | 5138.49s |
| `recv_w8_r0.5_0630_1632` | 5017.60s |
| `recv_w8_r0.7_0630_1756` | 5826.77s |
| `recv_w16_r0.3_0630_1934` | 4984.52s |
| `recv_w16_r0.5_0630_2058` | 4806.94s |
| `recv_w16_r0.7_0630_2219` | 5957.69s |

初步观察:`w16 r=0.5` 是当前 `pair #7 / tmath` 固定 ReKV 最优点(0.3356),略高于 `w8 r=0.7`(0.3342);整体曲线很平,预算从 0.3 增到 0.7 没有带来稳定收益。

---

## 合并来源 4：第四部分：实验地图与后续 TODO

> 原文件：`snapshots/EXPERIMENT_MAP_AND_TODO.md`。以下为该文件正文内容，实验数值未改动。

# KVComm / ReKV 实验地图与后续 TODO

> 用途:这份文档回答“我们基于 KVComm 论文做了哪些模型、哪些数据集、哪些实验,论文里 Table 1/2/3/4 分别是什么,以及后续还要补什么”。更详细的数值结果见 `snapshots/RESULTS.md`。

---

## 1. 对照论文与我们当前工作的定位

原论文: **KVComm: Enabling Efficient LLM Communication through Selective KV Sharing**(ICLR 2026 Poster, OpenReview: `https://openreview.net/forum?id=F7rUng23nw`)。

原论文核心思想:

- 在 multi-agent / inter-LLM communication 中,不传自然语言回复,也不传 hidden states,而是让发送方模型把部分 KV cache 传给接收方模型。
- 原 KVComm 的压缩粒度是**层级选择**:先用 attention importance + Gaussian prior 选重要层,然后传这些层的 KV。
- 原论文结论是:只传约 30% 层的 KV,可以接近 Skyline / full-context 上界,同时降低通信成本。

我们当前工作的核心改动:

- 从“选哪些层”升级到“在每一层里选哪些 token”。
- 从“发送方/校准集静态选择”升级到“接收方 query-aware 选择”。
- 从“固定预算 r”进一步尝试“query-adaptive / budget-aware 动态预算”。

当前最清晰的论文主线:

```text
KVComm(layer-wise selective KV sharing)
  -> token-level KV compression(Merge / Evict)
  -> ReKV(receiver-aware KV communication)
  -> B-ReKV(receiver-attention coverage budget)
```

---

## 2. 当前实验模型

### 2.1 主实验模型

目前所有主要表格实验都使用同模型通信:

- Sender model `M_s`: `/sharedspace/models/Llama-3.1-8B-Instruct`
- Receiver model `M_r`: `/sharedspace/models/Llama-3.1-8B-Instruct`
- 精度: `torch.bfloat16`
- attention implementation: `sdpa`
- 最大输入长度: `64000`
- 运行方式:单卡加载两个同模型实例,用 `CUDA_VISIBLE_DEVICES=<gpu>` 指定物理 GPU。

表格 caption 可以写:

```text
M_s: meta-llama/Llama-3.1-8B-Instruct;
M_r: meta-llama/Llama-3.1-8B-Instruct.
```

### 2.2 尚未完成的模型扩展

当前还没有正式完成 cross-model 表:

- Llama-3.1-8B -> Llama-3.1-8B:已完成主线。
- Llama -> Qwen / Mistral / Gemma:未完成,属于后续 Cross-model ReKV。
- 不同 tokenizer / 不同层数模型:未完成,需要额外处理 token 对齐与层映射。

---

## 3. 当前实验数据集

本项目当前围绕 8 个任务做表:

| 数据集 | 脚本名 / task name | 类型 | 当前用途 |
|---|---|---|---|
| Countries | `countries` | 简单事实 / 地理 QA | 简单任务对照,看低压下是否饱和 |
| Tipsheets | `tipsheets` | 自定义/合成 QA | 简单任务对照,KVComm 层级方法很强 |
| HotpotQA | `hotpotqa` | 多跳 QA | 主任务之一,ReKV 与 B-ReKV 有正结果 |
| QASPER | `qasper` | 科学论文 QA | 正在补表,需要 fixed ReKV + coverage |
| MuSiQue | `musique` | 多跳组合推理 | 当前最强主任务,B-ReKV 主卖点 |
| MultiField-QA-en | `multifieldqa_en` | 长文本/多领域 QA | ReKV 低预算效率很强,Coverage 已补较多 |
| 2WikiM-QA | `twowikimqa` | 多跳桥接 QA | oracle / budget 线已做,coverage 表格三行正在补 |
| TMATH | `tmath` | 数学题 | 原 KVComm 表已有,coverage 表格三行正在补 |

指标口径:

- `score`:各 evaluator 产出的任务分数,通常是 F1 / EM 或任务自带匹配分。
- `analyze_coverage.py --tau 0.5`:把 `score >= 0.5` 视为正确,用于计算 accuracy / equal-accuracy budget saving。
- 注意:这个 `--tau 0.5` 是“答对阈值”,不是 B-ReKV 的 `coverage_tau=0.95`。

---

## 4. 做过的方法清单

### 4.1 原 KVComm 论文/代码自带对照

| 方法 | 在表里名字 | 做了什么 | 是否我们新增 |
|---|---|---|---|
| Baseline | `Baseline` | 不通信/普通回答 | 否 |
| Skyline | `Skyline` | 上界:把完整上下文给一个模型 | 否 |
| NLD | `NLD` | Natural Language Debate,模型间传自然语言 | 否 |
| CIPHER | `CIPHER` | learned embedding / cipher 通信 | 否 |
| Activation Communication | `AC(mean/replace/sum)` | 传 hidden activation 并注入 | 否 |
| KVComm | `KVComm(0.3/0.5/0.7)` | 原论文方法,按层选择 KV | 否,复现/对照 |

### 4.2 我们新增的 token-level / ReKV 方法

| 方法 | 在表里名字 | 做了什么 | 结论 |
|---|---|---|---|
| Merge-then-Communicate | `Merge(0.3/0.5/0.7)` | 每层 token 级压缩,被丢 token 合并到保留 token | 整体不稳定,难任务上常弱于 evict |
| Evict-only | `Evict(0.3/0.5/0.7)` | 每层 token 级只保留 top token,不合并 | 比 merge 更稳,证明“选对 token”比“合并”重要 |
| ReKV-w8 | `ReKV-w8(r)` | 用接收方 query 最后 8 tokens 注意力给 A-context token 打分 | 主方法之一,MuSiQue 很强 |
| ReKV-w16 | `ReKV-w16(r)` | 用接收方 query 最后 16 tokens 注意力打分 | 主方法之一,HotpotQA 较强 |
| B-ReKV | `B-ReKV-w8/w16` | 不固定 r,按 receiver attention coverage 自动得到每条 query 的预算 | 当前 budget-aware 正结果 |

### 4.3 做过但证伪的 Budget-aware 方法

| 阶段 | 名字 | 方法 | 结论 |
|---|---|---|---|
| Step 0 | oracle 最小预算分析 | 密集扫 fixed-r ReKV,找每条样本最小可解预算 | 前提成立:query 间预算需求差异很大 |
| Step 1 | 开环预算预测 | 用 receiver 重要度熵 / 层重要性猜预算 | 证伪:预算 std≈0,corr≈0 |
| Step 2a | 离线 oracle 渐进 | 用完美 stop signal 模拟多轮补传 | 只是上界:理论省 35-47% |
| Step 2b | 在线渐进 + learned controller | 真实生成多轮,测 entropy/margin/context attention 信号 | 证伪:不稳且平均轮数 2.7-4.0 |
| 牌2 | Pass-1 单发预算预测 | 抽 rcap/entropy/gini 等特征,训练分类器预测预算 | 证伪:LODO AUC 0.585,等精度全负 |
| Step 3 | B-ReKV | 不预测 query difficulty,改用 receiver evidence coverage 定预算 | 成立:MuSiQue 强正,HotpotQA 小正 |

---

## 5. 论文表格规划

### Table 1: 主对比总表

目的:回答“我们的 ReKV / B-ReKV 相比原 KVComm 和其它通信方式怎么样?”

包含方法:

- 原论文/复现对照:`Baseline`, `Skyline`, `NLD`, `CIPHER`, `AC(mean/replace/sum)`, `KVComm(0.3/0.5/0.7)`。
- 我们新增:`Merge`, `Evict`, `ReKV-w8`, `ReKV-w16`。
- B-ReKV 三个代表点:
  - `B-ReKV-w8 (tau=0.95, scale=0.75)`
  - `B-ReKV-w8 (tau=0.95, scale=0.85)`
  - `B-ReKV-w16 (tau=0.95, scale=0.90)`

当前状态:

- `MuSiQue`, `MultiField-QA-en`, `HotpotQA` 的 B-ReKV 关键点基本已有。
- `Countries`, `Tipsheets`, `QASPER`, `2WikiM-QA`, `TMATH` 正在用 `scripts/run_table_completion_2x5.sh` 补齐。
- QASPER 还需要先补 fixed ReKV-w16 曲线,因为 `analyze_coverage.py` 要用 fixed-r 曲线做 equal-accuracy 对比。

建议 Table 1 caption:

```text
Comparison between KVComm, token-level compression baselines, ReKV, and B-ReKV under different communication budgets.
For B-ReKV, the subscript denotes the actual average KV budget used by the dynamic coverage policy.
```

### Table 2: ReKV dense budget / window ablation

目的:回答“ReKV 的提升来自 receiver-aware token selection,而不是某个单点偶然结果。”

应包含:

- `fixed-r` 曲线:`r = 0.1,0.2,0.3,0.4,0.5,0.7`。
- 方法:`KVComm`, `Merge`, `Evict`, `ReKV-w8`, `ReKV-w16`。
- 数据集重点:
  - HotpotQA:难多跳,ReKV-w16 强。
  - MuSiQue:最强主任务,ReKV-w8 强。
  - MultiField-QA-en:低预算 token 效率很强。
  - Countries/Tipsheets:简单任务,说明 ReKV 不是所有任务都赢 KVComm,更诚实。

当前结论:

- MuSiQue: `ReKV-w8 r=0.3` 达到约 0.480,明显高于 `KVComm(0.3)=0.112`。
- HotpotQA: `ReKV-w16 r=0.5` 达到约 0.746,接近 full / Skyline。
- MultiField-QA-en: `ReKV` 在 `r=0.1` 已接近饱和。
- Merge 通常弱于 Evict,说明“合并被丢 KV”不如“直接保留最相关 token”。

### Table 3: Budget-aware negative ablations

目的:回答“我们不是只试了一个 coverage,而是系统验证过哪些 budget-aware 方案不行。”

包含 4 组:

- Step 0 oracle headroom:
  - 扫 `r={0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7}`。
  - 结论:HotpotQA/MuSiQue 固定 r 理论浪费很大,headroom 真实存在。
- Step 1 open-loop:
  - `uniform`, `layer`, `query`, `query+layer`。
  - 结论:query budget 标准差接近 0,与正确性相关性接近 0。
- Step 2 online progressive:
  - 多轮 ladder `[0.1,0.2,0.3,0.5]`。
  - 信号:entropy, margin, context attention mass/concentration。
  - 结论:在线 controller 不稳,轮数太多。
- 牌2 Pass-1 predictor:
  - 特征:rcap50/90/95, entropy, gini, top mass, recency/sink, context length。
  - 结论:LODO AUC 0.585,跨数据集几乎随机。

建议 Table 3 不放所有细节,只放压缩摘要:

```text
Step / method / signal used / key metric / conclusion
```

详细数值放 appendix。

### Table 4: B-ReKV 动态预算主表

目的:回答“Budget-aware 最终可行方案是什么,实际省多少预算?”

核心展示:

- 固定 ReKV 曲线 vs B-ReKV 点。
- 每个点报告:
  - `coverage_tau`
  - `coverage_scale`
  - `recv_window`
  - `accuracy`
  - `average budget`
  - `equal-accuracy saving`

当前最强结果:

- MuSiQue:
  - `tau=0.95, scale=0.75, w8`:acc 0.482, avg budget 0.279,超过 fixed-r 最高点。
  - `tau=0.95, scale=0.85, w8`:acc 0.490, avg budget 0.313,当前最高 coverage 精度。
  - `tau=0.90, scale=0.75, w8`:acc 0.442, avg budget 0.158,等精度约 +40.7% 省预算。
- HotpotQA:
  - `tau=0.95, scale=0.90, w16`:acc 0.732, avg budget 0.400,约 +6.9% 省预算。

建议主图:

- Figure 1/2:MuSiQue accuracy-budget Pareto。
- Figure 3:B-ReKV per-sample budget distribution,证明不是固定 r 的变体,而是 query-adaptive。

---

## 6. 当前脚本与结果目录对应关系

### 6.1 运行脚本

| 脚本 | 对应实验 | 输出目录 |
|---|---|---|
| `scripts/run_baseline.sh` | Baseline | `snapshots/<task>/baseline` |
| `scripts/run_dataset.sh` | 原 KVComm / 多方法批量 | `snapshots/<task>/...` |
| `scripts/run_merge.sh` | token-level merge | `snapshots/<task>/mtc_merge` |
| `scripts/run_evict.sh` | token-level evict | `snapshots/<task>/mtc_evict` |
| `scripts/run_receiver.sh` | fixed-r ReKV | `snapshots/<task>/mtc_receiver` |
| `scripts/run_probe.sh` | Step 0 dense fixed-r probe | `snapshots/<task>/mtc_receiver/probe_recv_w16_r*` |
| `scripts/run_budget.sh` | Step 1 open-loop budget modes | `snapshots/<task>/budget` |
| `scripts/run_progressive.sh` | Step 2b online progressive | `snapshots/<task>/progressive` |
| `scripts/run_features.sh` | Pass-1 feature dump | `snapshots/<task>/features` |
| `scripts/run_coverage.sh` | B-ReKV 单任务扫描 | `snapshots/<task>/coverage` |
| `scripts/run_table_completion_2x5.sh` | 当前补 Table 1 缺失 coverage 点 | `logs/gpu2_table_queue.log`, `logs/gpu5_table_queue.log` |

### 6.2 分析脚本

| 脚本 | 用途 |
|---|---|
| `scripts/analyze_oracle.py` | Step 0 oracle 最小预算分布 |
| `scripts/analyze_budget.py` | Step 1 budget mode 对比 |
| `scripts/sim_progressive.py` | Step 2a 离线 oracle 渐进上界 |
| `scripts/analyze_progressive_online.py` | Step 2b 在线多轮阈值扫描 |
| `scripts/learn_stop_policy.py` | Step 2b learned controller |
| `scripts/learn_budget_predictor.py` | 牌2 Pass-1 预算预测器 |
| `scripts/sim_coverage_budget.py` | B-ReKV 零 GPU 离线预检 |
| `scripts/analyze_coverage.py` | B-ReKV 在线结果分析 |
| `scripts/plot_coverage_pareto.py` | Coverage Pareto 图 |
| `scripts/plot_budget_distribution.py` | 动态预算分布图 |

---

## 7. 当前正在跑 / 正在补的实验

当前补表脚本:

```bash
bash scripts/run_table_completion_2x5.sh
```

只使用 GPU 2 和 GPU 5:

- GPU2 队列:
  - `countries`: `w8 tau=0.95 scale=0.75/0.85`, `w16 tau=0.95 scale=0.90`
  - `tipsheets`: `w8 tau=0.95 scale=0.75/0.85`, `w16 tau=0.95 scale=0.90`
  - `twowikimqa`: `w8 tau=0.95 scale=0.75/0.85`, `w16 tau=0.95 scale=0.90`
- GPU5 队列:
  - `qasper`:先跑 fixed ReKV-w16 `r=0.1..0.9`
  - `qasper`:再跑 coverage 三点
  - `tmath`:coverage 三点
  - `hotpotqa`:补 `w8 tau=0.95 scale=0.85`

进度检查:

```bash
tail -f logs/gpu2_table_queue.log
tail -f logs/gpu5_table_queue.log
```

跑完后统一分析:

```bash
python scripts/analyze_coverage.py \
  --tasks countries tipsheets hotpotqa qasper musique multifieldqa_en twowikimqa tmath \
  --tau 0.5
```

---

## 8. 后续 TODO List

### 8.1 立即完成:补齐 Table 1

- [ ] 等 `scripts/run_table_completion_2x5.sh` 全部跑完。
- [ ] 跑 `scripts/analyze_coverage.py` 汇总所有 coverage 结果。
- [ ] 把 `Countries/Tipsheets/QASPER/2WikiM-QA/TMATH` 的 B-ReKV 三行填进 Table 1。
- [ ] 若 QASPER fixed ReKV-w16 跑完但没有 w8 fixed 曲线,先确认 Table 1 是否只需要 coverage 表格值,还是还要 equal-accuracy saving。
- [ ] 检查 coverage 每个点的 `avg budget`,在表格中用下标标注,例如 `0.482_{0.279}`。

### 8.2 主结果收尾

- [ ] 更新 `snapshots/RESULTS.md` 的 Table 1,保证和最终论文表一致。
- [ ] 重新生成 MuSiQue Pareto 图,必要时加入 HotpotQA / MultiField-QA-en 小图。
- [ ] 生成 per-sample budget distribution 图,证明 B-ReKV 的 budget 是 query-adaptive。
- [ ] 写一段英文方法描述,把 B-ReKV 定义为 receiver evidence fidelity / coverage constraint。

### 8.3 论文需要的补充实验

- [ ] Window scan:在主任务上补 `recv_window={4,8,16,32,all}`。
- [ ] Coverage threshold scan:主任务补 `coverage_tau={0.90,0.95,0.98}` 与 `scale` 网格,但论文主表只放 2-3 个代表点。
- [ ] Latency breakdown:单样本拆开测 A-prefill / receiver scoring / B-prefill / generation。
- [ ] Budget distribution by dataset:看哪些数据集天然预算低、哪些预算高。
- [ ] Failure case analysis:挑 MuSiQue 上 coverage 成功/失败样本,分析 receiver attention 是否聚焦证据 token。

### 8.4 进一步创新点

- [ ] Head-wise B-ReKV:每个 attention head 单独算 coverage,避免 head 平均抹掉检索头。
- [ ] Cross-model ReKV:不同模型之间通信,例如 Llama -> Qwen / Mistral / Gemma。
- [ ] Tokenizer mismatch 处理:当 A/B tokenizer 不同,需要字符级 span 对齐或让 B 传 query embedding。
- [ ] Layer mapping:当 A/B 层数不同,需要线性映射、最近层映射或 learned mapping。
- [ ] Rate-distortion scoring:把 token score 从纯 attention 升级为 `attention * value_norm` 或输出分布扰动。

---

## 9. 当前结论摘要

已经能讲的结论:

- 原 KVComm 证明 KV 是好的通信载体,但它主要是层级选择。
- 我们证明 token-level selection 更适合压缩通信,尤其是多跳 QA。
- Merge 不稳定,Evict 更稳,说明关键不是“合并压缩”,而是“选对 token”。
- ReKV 的关键增益来自 receiver-aware:用接收方 query 注意力选择 A-context token。
- 直接预测 query difficulty / budget 的路线失败了:开环熵、层分配、在线多轮、Pass-1 predictor 都不稳定或不泛化。
- B-ReKV 是当前 budget-aware 正路线:不预测 r,而是保留最少 token 覆盖 receiver attention mass,从而自然得到 per-query dynamic budget。

最适合论文主张的结果:

- MuSiQue 上 B-ReKV `w8, tau=0.95, scale=0.75/0.85` 形成强 Pareto,能在低平均预算下超过 fixed-r ReKV 最高精度。
- HotpotQA 上 `w16, tau=0.95, scale=0.90` 有稳定小正收益。
- MultiField-QA-en 用于证明 ReKV 低预算效率很高。
- Countries/Tipsheets/TMath/2Wiki/QASPER 用于补完整主表,但不一定都作为主卖点。

---

## 10. 2026-07-05 实验状态审计

这次审计基于 git 状态、`snapshots/` 产物和队列日志。审计开始时 git 工作区无可见未提交改动；本次审计只更新记录文档。由于大量实验产物可能被 `.gitignore` 忽略，实验状态以 `snapshots/` 下的实际产物为准。

已完成：

- **Table 1 / Table 8 主表覆盖**：Table 1 pair #6/#7 已完整；Table 8 pair #1/#2/#3/#4/#5 已完整并可作为正向附录证据。pair #9 的 8 datasets x 9 paper-table runs 虽完成，但已暂缓作为正向对比；KVComm probe 同样显示 QA/multi-hop 任务很差，倾向解释为 hard heterogeneous pair / KV compatibility issue。
- **Pair #6/#7 full cost profile**：已完成。每个 pair 覆盖 8 个主任务 x 10 个 method block，产物为 `snapshots/cost_profile/table1_pair6_llama32_abliterated_deepseek3b_full/cost_table.csv` 和 `snapshots/cost_profile/table1_pair7_qwen25_uncensored_bespoke_full/cost_table.csv`。
- **B-ReKV robustness**：pair #1 的 HotpotQA / MuSiQue / MultiFieldQA-en Pareto 与 budget distribution 已完成；pair #6/#7 的 HotpotQA / MuSiQue 小网格也已完成。
- **Fairness / interpretability**：pair #1/#6/#7 的 ReKV vs Evict/Random fairness 已完成；answer overlap、cleaned examples、token bar、deletion ablation 已完成。
- **Sink/recent ablation**：已完成。目录为 `snapshots/mechanism/pair1_llama31_same/sink_recent/`，覆盖 8 个主任务，每个任务 ReKV/B-ReKV x 4 个 sink/recent 组合。
- **2026-07-05 收尾分析产物**：pair #6/#7 cost 论文子表已生成到 `snapshots/analysis/cost/pair6_pair7_cost_focus_hotpotqa_musique_multifieldqa.csv`；pair #6/#7 robustness summary 与 Pareto 图已生成到 `snapshots/analysis/robustness/`；pair #9 分数异常、raw-output 和 KVComm probe 诊断已记录到 `snapshots/analysis/pair9/pair9_diagnostic_report.md`。
- **Table 11 positional coherence 可跑部分**：2026-07-05 GPU2 队列已完成 8 个主任务的 ReKV normal / ReKV-S / B-ReKV normal，汇总见 `snapshots/analysis/mechanism/positional_coherence_summary.md`。ReKV-S 在 HotpotQA/MuSiQue/MultiFieldQA-en/2Wiki/QASPER 上均明显低于 normal，支持 positional coherence 很重要这一机制结论。

部分完成 / 中断：

- **B-ReKV-S positional coherence**：不再作为当前必跑项。`--shift_back` + coverage budget 会触发 `models.py:get_short_past_key_values` 的 `assert len(lengths) <= 2`，根因是 coverage budget 造成多层 KV 长度档位超过 shift-back 当前实现假设；诊断报告见 `snapshots/analysis/mechanism/brekv_shiftback_diagnosis.md`。当前论文写法建议只报告 ReKV-S 和 B-ReKV normal，B-ReKV-S 标为实现限制。

尚未完成：

- **Table 6 extended tasks**：脚本已准备；当前已可通过 `RUN_SINK_RECENT=0 RUN_POSITIONAL=1 RUN_BREKV_SHIFT=0 RUN_TABLE6=1 GPU=7 bash scripts/run_gpu7_mechanism_extended_full_queue.sh` 绕开 B-ReKV-S 并继续后续队列，但尚未开始生成 `snapshots/table6_pair*` 结果目录。
- **Pair #9 处理决定**：raw-output probe 和 KVComm top=0.3 limit=50 probe 已完成。KVComm 在 HotpotQA/MuSiQue/QASPER 上同样很差，因此 pair #9 不再作为正向对比补跑，暂缓/作为 hard negative 或 limitation 记录。
- **Score function ablation / Layer aggregation ablation**：脚本和代码已准备，队列已启动或待完成；结果完成后需要汇总成机制表/图。
- **Table 10 multi-source / Head-wise B-ReKV / Falcon pair #8**：未做或暂缓。Falcon pair #8 仍受 checkpoint 不可用/不完整限制。

下一步建议优先级：

1. 继续 Table 6 extended tasks。
2. 等 score/layer ablation 队列完成后汇总机制表/图。
3. 再考虑 Table 10 multi-source / Head-wise B-ReKV。
4. pair #9 和 B-ReKV-S 暂缓，不再投入 GPU 作为正向对比补跑。

### 10.1 2026-07-06 Table 11 positional coherence 汇总

产物：

```text
snapshots/analysis/mechanism/positional_coherence_summary.csv
snapshots/analysis/mechanism/positional_coherence_summary.md
snapshots/mechanism/logs/gpu2_mechanism_extended_full_0705_1217.log
```

核心数值：

| Task | ReKV normal | ReKV-S | B-ReKV normal | B-ReKV-S |
|---|---:|---:|---:|---|
| countries | 0.6000 | 0.6000 | 0.6150 | skipped |
| tipsheets | 0.8680 | 0.8620 | 0.8780 | skipped |
| hotpotqa | 0.6960 | 0.6120 | 0.7080 | skipped |
| qasper | 0.3440 | 0.2900 | 0.3320 | skipped |
| musique | 0.4800 | 0.3440 | 0.4820 | skipped |
| multifieldqa_en | 0.5067 | 0.4267 | 0.5200 | skipped |
| twowikimqa | 0.4050 | 0.2600 | 0.4100 | skipped |
| tmath | 0.3408 | 0.3494 | 0.3481 | skipped |

结论：除 countries / tmath 这类较简单或形式特殊任务外，ReKV-S 相比 ReKV normal 普遍下降，尤其 HotpotQA、MuSiQue、2Wiki、QASPER、MultiFieldQA-en 更明显。这可以作为 Table 11 的主要机制结论：KV 通信不是随便拼接 cache，位置一致性/positional coherence 对长文和多跳任务很重要。B-ReKV-S 因当前 shift-back 实现不支持 coverage budget 的多层不规则长度，暂不报告为正向实验项。

### 10.2 2026-07-08 Evidence / Failure / Task-Type 分析

GPU4 串行分析已完成，日志：

```text
snapshots/analysis/logs/gpu4_analysis_suite_0708_1915.log
```

产物：

```text
snapshots/supporting_overlap/hotpotqa_pair1_full_context/supporting_overlap_summary_top20_w8_r0.3.csv
snapshots/supporting_overlap/hotpotqa_pair1_full_context/supporting_overlap_top20_w8_r0.3.jsonl
snapshots/analysis/failure_cases/failure_case_summary.csv
snapshots/analysis/failure_cases/failure_case_examples.csv
snapshots/analysis/task_type_sensitivity/task_type_family_summary.csv
snapshots/analysis/task_type_sensitivity/task_type_run_summary.csv
snapshots/analysis/figures/supporting_overlap_bar.png
snapshots/analysis/figures/failure_rate_heatmap.png
snapshots/analysis/figures/task_type_sensitivity_bar.png
```

HotpotQA supporting-facts overlap 结果：

| Method | Top-20 selected tokens in supporting facts |
|---|---:|
| ReKV | **0.5148** |
| Evict / ValueNorm | 0.0638 |
| Random-token | 0.0555 |

解释：ReKV 选中的 top tokens 有超过一半落在 gold supporting facts 中，而 query-agnostic Evict/Random 只有约 6%。这比 answer-term overlap 更直接说明 ReKV 的 receiver-aware attention 确实在定位证据句。

Task-type sensitivity 摘要：

| Task type | ReKV mean | B-ReKV mean |
|---|---:|---:|
| simple fact | 0.4233 | 0.4525 |
| simple synthetic | 0.7607 | 0.6560 |
| multi-hop | 0.4031 | 0.4040 |
| long document | 0.3501 | 0.3444 |
| math / reasoning | 0.3258 | 0.3237 |

解释：ReKV/B-ReKV 的主卖点仍应放在 evidence-heavy 的 multi-hop / long-document 任务上；simple synthetic 任务更容易饱和，不适合作为主要贡献叙事。

Failure-case analysis 已生成 failure rate 汇总和样例索引。当前 failure examples 只基于 `per_sample.jsonl`，不含 raw response；主要用途是挑选后续人工检查样本，区分 budget 不足、证据没选到和 receiver 推理失败。图 `failure_rate_heatmap.png` 可用于汇报哪些任务/方法 family 失败率最高。

### 10.3 2026-07-09 最新实验审计：Table 6 / Score Function / Layer Aggregation / NLD 对比

当前运行状态：

- GPU7 正在运行 `scripts/run_gpu7_mechanism_extended_full_queue.sh`。
- 当前子任务是 Table 6 pair #7 `hotpotqa_full` 的 ReKV-w16 `r=0.5`。
- Pair #6 Table 6 已完成 5 个 extended tasks x 9 paper-style runs。
- Pair #7 已完成 `hotpotqa_full` 的 4/9 runs，后续还需继续 `hotpotqa_full` 剩余 5 runs 和其他 4 个 extended tasks。

新增汇总产物：

```text
snapshots/analysis/latest_experiments/score_function_summary.csv
snapshots/analysis/latest_experiments/score_function_best_by_pair_task_method.csv
snapshots/analysis/latest_experiments/figures/score_function_ablation_best.png
snapshots/analysis/latest_experiments/layer_aggregation_summary.csv
snapshots/analysis/latest_experiments/layer_aggregation_best_by_task_method.csv
snapshots/analysis/latest_experiments/figures/layer_aggregation_heatmap.png
snapshots/analysis/latest_experiments/table6_extended_summary.csv
snapshots/analysis/latest_experiments/table6_extended_status.csv
snapshots/analysis/latest_experiments/table6_extended_best_by_pair_task_family.csv
snapshots/analysis/latest_experiments/figures/table6_pair6_extended_best.png
```

Table 6 extended tasks 当前结论：

| Pair | Task | Best ReKV | Best B-ReKV | Note |
|---|---|---:|---:|---|
| #6 | hotpotqa_full | **0.7579** | 0.7153 | B-ReKV avg budget ≈ 0.3685 |
| #6 | musique_full | **0.4344** | 0.4096 | B-ReKV avg budget ≈ 0.2501 |
| #6 | qasper_full | **0.3459** | 0.3384 | B-ReKV 接近 ReKV |
| #6 | samsum | **0.2886** | 0.2609 | summarization 上仍可用 |
| #6 | repobench | **0.3549** | 0.3402 | B-ReKV avg budget ≈ 0.1089 |
| #7 | hotpotqa_full | 0.5754 | pending | 已完成 4/9 runs，当前 best 为 ReKV-w8 r=0.7 |

解释：pair #6 已足以支持“ReKV/B-ReKV 不只在主任务有效”的 appendix robustness；pair #7 跑完后再形成最终 Table 6 跨 pair 表。

Score-function ablation 当前结论：

- 已完成 pair #1/#6/#7 x `hotpotqa` / `musique` / `multifieldqa_en`，共 144 个 runs。
- 原始 `receiver` attention 是稳定强 baseline：pair #1 HotpotQA `0.746`、pair #6 HotpotQA `0.702`。
- `receiver_x_value_norm` 在 MuSiQue 和 MultiFieldQA-en 上有小幅增益，例如 pair #1 MuSiQue `0.494` vs receiver `0.484`，pair #7 MuSiQue `0.342` vs receiver `0.338`。
- `receiver_recency` 在 pair #7 HotpotQA 最好：`0.474`。
- 结论：提升主要来自 receiver-aware attention；value norm / random 无法解释 ReKV 的收益。混合 score 可作为附录增强，但正文保留原始 receiver scoring 更简洁。

Layer aggregation ablation 当前结论：

- 已完成 pair #1 的 8 个主任务，`identity / last / mean / top4 / last4` x `w8/w16`，共 80 个 runs。
- HotpotQA：`identity 0.700` 最强，`top4 0.682`、`mean 0.662`，`last 0.516`。
- MuSiQue：`identity 0.480` 最强，`top4 0.474`、`mean 0.466`，`last 0.290`。
- 结论：原始 paired-layer aggregation 是稳健默认；只用最后层 attention 明显不稳定。

Natural-language passing 对比准备：

- 已新增 `--do_test_nld --profile_cost` 路径，用于统计 NLD 的准确率、自然语言 payload tokens/bytes、三阶段时间和峰值显存。
- 运行脚本：`scripts/run_nld_vs_rekv_cost_gpu7.sh`。
- 汇总脚本：`scripts/summarize_nld_vs_rekv_cost.py`。
- 当前对比表：`snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_summary.csv`，目前包含已有 ReKV/B-ReKV cost 行；NLD 跑完后重新执行汇总脚本会自动加入 NLD 行。
- 建议等 GPU7 Table 6 队列结束后运行：

```bash
GPU=7 PAIRS="1 6 7" TASKS="hotpotqa musique multifieldqa_en" LIMIT=500 \
  bash scripts/run_nld_vs_rekv_cost_gpu7.sh

python scripts/summarize_nld_vs_rekv_cost.py
```

论文叙事建议：NLD 是“sender 先生成自然语言，再由 receiver 二次生成”，会引入额外生成 latency 和 sender 答案错误；ReKV/B-ReKV 则让 sender 只作为 context-side KV memory server，传 selected latent evidence，由 receiver 负责最终推理和对齐。这组实验能把“为什么不是直接文本传递”讲清楚。

### 10.4 2026-07-09 Natural-Language Passing 对比完成

GPU6 的 NLD cost profile 已完成，日志：

```text
snapshots/nld_cost_profile/logs/gpu6_nld_cost_0709_1756.log
```

覆盖：

- Pair #1: `S: Llama-3.1-8B; R: Llama-3.1-8B`
- Pair #6: `S: Llama-3.2-3B-Abliterated; R: DeepSeek-R1-3B`
- Pair #7: `S: Qwen2.5-7B-Uncensored; R: Bespoke-Stratos-7B`
- Tasks: `hotpotqa`, `musique`, `multifieldqa_en`
- NLD phase-1 answer cap: `128` tokens

产物：

```text
snapshots/analysis/nld_vs_rekv/nld_vs_rekv_report.md
snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_summary.csv
snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_focused.csv
snapshots/analysis/nld_vs_rekv/nld_vs_rekv_cost_average_by_pair.csv
snapshots/analysis/nld_vs_rekv/figures/nld_vs_rekv_cost_overview.png
snapshots/analysis/nld_vs_rekv/figures/nld_vs_rekv_accuracy_by_task.png
```

Pair-averaged 结果：

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

结论：

- NLD 的自然语言 payload 很小，但这不是同单位优势：它传的是 sender 生成的短文本答案，常常丢证据或引入中间答案错误。
- ReKV/B-ReKV 在所有 pair 上准确率显著高于 NLD，尤其 multi-hop / long-document 任务差距很大。
- NLD 需要 A 生成、B 初答、B refinement 三段生成，平均 latency 明显高于 ReKV/B-ReKV。
- 显存方面 NLD 略低或接近，但不足以抵消准确率和延迟劣势。
- 论文中可以把该实验作为“为什么不直接用自然语言传递”的回答：ReKV/B-ReKV 不是让 sender 代答，而是让 sender 作为 context-side KV memory server，向 receiver 传 selected latent evidence。

### 10.5 2026-07-11 最新实验审计：Table 6 拆卡队列完成情况

当前运行状态：

- 未检测到仍在运行的 `python com.py` / Table 6 / analysis suite 实验进程。
- Table 6 pair #6 已完成 5 个 extended tasks x 9 paper-style runs，共 45 runs。
- Table 6 pair #7 已完成 `hotpotqa_full` / `qasper_full` / `musique_full` / `samsum`，每个任务 9/9 runs，共 36 runs。
- Table 6 pair #7 的 `repobench` 为 0/9。日志 `snapshots/table6_pair7_qwen25_uncensored_bespoke/logs/gpu7_table6_pair7_remaining_0710_1237.log` 显示第一项 `repobench ReKV w8 r=0.3` 在约 379/1000 样本处 CUDA OOM，发生在 receiver-attention scoring 的 `softmax` 阶段。

重新生成的汇总产物：

```text
snapshots/analysis/latest_experiments/score_function_summary.csv
snapshots/analysis/latest_experiments/score_function_best_by_pair_task_method.csv
snapshots/analysis/latest_experiments/layer_aggregation_summary.csv
snapshots/analysis/latest_experiments/layer_aggregation_best_by_task_method.csv
snapshots/analysis/latest_experiments/table6_extended_summary.csv
snapshots/analysis/latest_experiments/table6_extended_status.csv
snapshots/analysis/latest_experiments/table6_extended_best_by_pair_task_family.csv
snapshots/analysis/latest_experiments/figures/score_function_ablation_best.png
snapshots/analysis/latest_experiments/figures/layer_aggregation_heatmap.png
snapshots/analysis/latest_experiments/figures/table6_pair6_extended_best.png
snapshots/analysis/latest_experiments/figures/table6_pair7_extended_best.png
```

Table 6 extended tasks 最新结果：

| Pair | Task | Best ReKV | Best B-ReKV | Status |
|---|---|---:|---:|---|
| #6 | hotpotqa_full | **0.7579** | 0.7153 | complete |
| #6 | qasper_full | **0.3459** | 0.3384 | complete |
| #6 | musique_full | **0.4344** | 0.4096 | complete |
| #6 | samsum | **0.2886** | 0.2609 | complete |
| #6 | repobench | **0.3549** | 0.3402 | complete |
| #7 | hotpotqa_full | **0.5754** | 0.4917 | complete |
| #7 | qasper_full | **0.2329** | 0.2109 | complete |
| #7 | musique_full | **0.3736** | 0.3103 | complete |
| #7 | samsum | **0.3387** | 0.3150 | complete |
| #7 | repobench | missing | missing | OOM |

结论：Table 6 现在已经有 pair #6 的完整 extended-task 证据，以及 pair #7 的 4/5 extended-task 补充证据。整体上 ReKV 在所有完成的 extended tasks 上都是最强或明显强于 B-ReKV；B-ReKV 虽略低于 best fixed-budget ReKV，但使用显著更低的 adaptive budget，仍可作为“budget-aware compression”附录证据。Pair #7 RepoBench 不建议继续占用主线队列，后续若要补齐可单独用更保守的显存设置重跑，或作为 OOM limitation 记录。

### 10.6 2026-07-12 更新：Table 6 Pair #7 RepoBench 补跑完成

7 月 11 日记录的 RepoBench OOM 缺口已经补齐。数据集下载到
`datasets/RepoBench` 后，使用 98 GB RTX PRO 6000（GPU 0–3）拆分运行，
9/9 配置全部完成，每个配置包含 1000 条样本。

完整产物：

```text
snapshots/table6_pair7_qwen25_uncensored_bespoke/repobench/
snapshots/analysis/latest_experiments/table6_pair7_repobench_summary.md
snapshots/analysis/latest_experiments/table6_extended_summary.csv
snapshots/analysis/latest_experiments/table6_extended_status.csv
snapshots/analysis/latest_experiments/table6_extended_best_by_pair_task_family.csv
```

Pair #7 RepoBench 结果：

| Method | Window | Ratio / scale | Score | Actual KV budget |
|---|---:|---:|---:|---:|
| ReKV | 8 | r=0.3 | 0.3485 | 0.3250 |
| ReKV | 8 | r=0.5 | 0.3511 | 0.5178 |
| **ReKV** | **8** | **r=0.7** | **0.3530** | 0.7107 |
| ReKV | 16 | r=0.3 | 0.3487 | 0.3250 |
| ReKV | 16 | r=0.5 | 0.3474 | 0.5178 |
| ReKV | 16 | r=0.7 | 0.3507 | 0.7107 |
| **B-ReKV** | **8** | **t=0.95, s=0.75** | **0.3400** | **0.1594** |
| B-ReKV | 8 | t=0.95, s=0.85 | 0.3357 | 0.1734 |
| B-ReKV | 16 | t=0.95, s=0.90 | 0.3213 | 0.1825 |

更新后的 Table 6 状态：pair #6 与 pair #7 均完成 5 个 extended tasks ×
9 个配置，各 45 runs。Pair #7 RepoBench 的固定 ReKV 对预算不敏感：
w8 从 r=0.3 增至 r=0.7，实际 KV 预算增加 2.19×，分数仅增加 0.0045。
最佳 B-ReKV 以 0.1594 实际预算取得 0.3400；相对 ReKV-w8 r=0.3，
约使用 49% KV，绝对分数仅下降 0.0085。

显存限制仍需记录：当前 receiver scoring 在应用 `recv_window` 前构造完整
QK/softmax 矩阵，长代码样本的单进程显存约 81 GB。后续应把 query window
裁剪前移到 QK 矩阵乘法之前。

### 10.7 2026-07-13 Query-Sketch 论文重跑审计（严格区分协议）

本节只统计新协议 roots，不把历史隐式 Full-KV Oracle 结果混入正文结论。
完整动态报告与机器可读数据：

```text
snapshots/analysis/query_sketch_rerun_20260713/REPORT.md
snapshots/analysis/query_sketch_rerun_20260713/table1_all_runs.csv
snapshots/analysis/query_sketch_rerun_20260713/table1_status.csv
snapshots/analysis/query_sketch_rerun_20260713/oracle_gap.csv
snapshots/analysis/query_sketch_rerun_20260713/representation_summary.csv
snapshots/analysis/query_sketch_rerun_20260713/mechanism_runs.csv
snapshots/analysis/query_sketch_rerun_20260713/figures/
```

协议口径：

- `query_sketch_bf16_v1` / `query_sketch_int8_v1` /
  `query_sketch_token_ids_v1`：本轮可部署协议。
- `full_kv_oracle_v1`：显式 Full-KV 上界，只进入 oracle-gap。
- `query_agnostic_kv_v1`：ValueNorm / Random 对照。
- Pair #6 root 中 68 个缺少 metadata 的结果包括 47 个 fixed ReKV 和 21 个
  provisional B-ReKV；统一标记为 `query_sketch_bf16_v0_pre_instrumentation`。
  它们位于明确的 Query-Sketch root，但没有可审计的 `protocol_version`，
  因此仅用于准确率参考，不能用于新 bytes / timing 结论，也不冒充显式 v1。
- 其他历史 snapshots 不进入本轮主结论。

配置选择审计：

- 原分析器选择的高预算配置不应作为全局主配置。`t=0.98` 候选的
  6 个实际预算全部超过 fixed-r 校准网格上界，原 matched-budget
  胜/平统计来自端点截断。
- 正文低预算主 operating point 继续使用 `B-ReKV t=0.95, s=0.75, w=8`；
  `t=0.95, s=0.85, w=8` 作为中预算 Pareto 点。

Query-Sketch vs 显式 Full-KV Oracle（3 pairs × 3 tasks，profile n=50）：

- ReKV：平均 score gap `-0.0378`，端到端通信降低 `61.0%`，平均 latency
  降低约 `4.1%`。
- 高预算 B-ReKV 消融：平均 score gap `+0.0067`，端到端通信降低 `35.1%`，
  平均 latency 降低约 `1.6%`。
- 两种方法峰值显存均与 Oracle 基本持平并略低。该结果支持“部署协议闭环后，
  性能没有系统性崩塌，通信收益也没有被 scoring latency 抵消”。

Query-sketch 表示消融（2 pairs × 3 tasks × 4 windows，共 72 runs）：

- BF16 w8：平均 score `0.5467`，B→A sketch `1696 KiB`。
- INT8 w8：平均 score `0.5500`，B→A sketch `849 KiB`；通信减半且没有精度损失，
  是当前最有价值的轻量化结果。
- Token IDs：B→A 仅 `0.023–0.133 KiB`，但最佳 w16 平均 score 仅 `0.4467`；
  适合作为超轻量下界，不应替代 receiver Q sketch 主协议。
- w8 对 BF16 / INT8 都是最稳窗口；w16/w32 没有继续提升。

新协议机制消融（Pair #6，HotpotQA / MuSiQue / MultiFieldQA-en）：

- Score function 三任务均值：Random `0.3027`，ValueNorm `0.3596`，
  Receiver×ValueNorm `0.4771`，Receiver `0.4896`，
  Receiver+Recency `0.4936`。
- Receiver-aware scoring 明显强于 query-agnostic 对照；recency 只有很小增益，
  正文仍可保留原始 Receiver score 以保持方法简洁。
- Layer aggregation 三任务均值：Last `0.3580`，Mean `0.4004`，
  Last4 `0.4198`，Top4 `0.4400`，Identity `0.4896`。
- 原始 paired-layer identity 明显最稳，不建议改成全层平均或只取最后层。

## 2026-07-14 本机 Query-Sketch fast-node 队列完成

完整报告：`snapshots/analysis/fast_node_completion_20260714.md`。

- 本机目标 123/123 unique cells 完成：GPU0 54/54、GPU3 69/69。
- Table 10 真 Multi-Source：18/18；best score 为 HotpotQA 0.668、
  MuSiQue 0.450、2WikiMQA 0.440。
- Table 6 Pair #7：35/35；main B-ReKV 宏平均 0.2899 @ budget 0.3475，
  相对 best ReKV 平均少 0.0793 分、预算低 51.09%。
- Table 8 Pair #5 前四任务：28/28；main B-ReKV 宏平均
  0.4315 @ budget 0.3142。
- Table 1 Pair #1/#7 各 56/56；Pair #1 main B-ReKV
  0.4734 @ 0.3019，Pair #7 为 0.3702 @ 0.3734。
- 正式 cost v1：12/18，Pair #1/#7 完成。
- 当前 main B-ReKV Oracle gap：12/18；六单元平均 score gap -0.1300，
  通信节省 59.62%。旧报告的 B-ReKV `+0.0067` 来自旧高预算配置，
  不再作为主配置结论。

更新后的未完成项：

- Table 1：113/168，Pair #6 尚缺 47 fixed ReKV + 8 main B-ReKV。
- Table 6：35/70，Pair #6 尚缺 35。
- Table 8：28/224，尚缺 196。
- Cost v1：12/18，Pair #6 尚缺 6。
- Main B-ReKV Oracle gap：12/18，Pair #6 尚缺 6。
