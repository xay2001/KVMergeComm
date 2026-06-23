# KVComm → RASC 实验汇总

> 方法:**RASC(Receiver-Aware Selective KV Communication)** —— 在 LLM 间用 KV cache 直接通信的场景下,
> 由发送方 A 依据**接收方 B 的问题注意力**,对每个 token 的 KV 打分,只挑高分 token 传输。
> 模型:Llama-3.1-8B-Instruct → Llama-3.1-8B-Instruct(同模型)。指标:F1 / EM(各数据集自带)。

---

## 0. 研究主线与三个创新点(论文骨架)

### 0.1 问题定位

在 **LLM 间 latent communication** 中,KV-cache 是信息最丰富、但 payload 最大、架构依赖最强的通道(survey 三轴:**WHAT**=传什么 / **WHICH**=传哪些层 / **HOW**=怎么注入)。survey 明确把 **KV-cache 通信的压缩/量化** 列为 open problem。本工作落在"**WHAT=KV、压缩**"这一格,且强调**通信场景特有**的杠杆(接收方已知 query),而非通用单模型 cache eviction。

- 基线 **KVComm**(ICLR):**WHICH** 轴做文章——选部分**层**整层传,校准一次写死,query 无关。
- 本工作把战场从"选哪些层"换到"**按接收方需求,选哪些 token、给多少预算**"。

### 0.2 研究主线(两阶段)

```
阶段 A:RASC —— 把"层选择"升级为"接收方感知的 token 选择"        [§1–§4, 已验证 ✓]
   KVComm(层级丢弃) → evict(token 级,value-norm) → RASC(token 级,receiver 打分 + 观测窗口)
   结论:同预算下 RASC > evict > merge;难任务上 token 级 ≫ 层级;merge 无益("选对" > "合并")

阶段 B:Budget-aware —— 把"固定保留比"升级为"按 query 自适应预算"  [§5–§9, 全面证伪 ✗]
   Step 0  验证前提:每条 query 的最优预算大幅波动,固定 r 严重过供(理论 64–67% headroom)  → 前提成立
   Step 1  开环预测:发送方侧统计量(熵/层重要性)预测不出 per-query 预算            → 证伪
   Step 2a 离线上界:oracle-stop 渐进理论可省 35–47% 预算(但这是完美停止信号的上界)   → 仅上界
   Step 2b 在线渐进:learned controller 实测,只有 hotpotqa 中段微正,musique/2wiki 负   → 证伪 + 多轮开销
   牌2    单发预测:Pass-1 特征单发预测最优预算,LODO AUC 0.585、等精度全程负节省      → 证伪

  → 阶段 B 的全部变体(开环 / 离线 / 在线多轮 / 单发)在等精度下都打不过"RASC + 固定预算",
    且不跨数据集泛化。这与 BAGEN("预算可训练但难校准、不泛化")一致。整条线作为强 negative result 收编。
```

### 0.3 创新点定位(随实验修正)

1. **Receiver-Aware KV Communication(RASC,主)**:利用接收方 B 的 query sketch(问题末 N 词的注意力)指导发送方 A **在 token 粒度**选择性传输 KV。定位为 **selection**(选对 token),并实验证伪 cache-merging 的必要性。这是已坐实的核心贡献。
2. **机制分析(merge vs evict)**:系统对比表明在通信压缩场景"**选对 token**" > "合并被丢的 token",纠正了 CaM 式 cache-merging 的迁移假设。
3. **Budget-aware 的系统性 negative result(§5–§9)**:从前提验证(headroom 真实存在)到四类预算自适应方案(开环 / 离线 oracle / 在线多轮 / 单发预测)逐一证伪,得出"**在 receiver-aware selection 之上,query 自适应预算无可开环/单发利用的信号、且不跨任务泛化**"——这是一个干净、可写半页的反面结论,反衬主方法的简洁有效。
4. **(进行中)牌 1 — Cross-model RASC**:A、B 异构(不同权重/tokenizer/层数)时的打分对齐,是真实 MAS 刚需、KVComm 未解决的问题,拟作为第三个正向抓手。详见 §11。

---

## 1. 方法与对照

| 方法 | 压缩粒度 | 选择判据 | 是否 query 自适应 | 备注 |
|---|---|---|---|---|
| **KVComm**(基线) | 层级(丢整层) | B 注意力汇总成**每层一个标量** | 否,校准一次写死 | 原 ICLR 工作 |
| **merge** | token 级 + 融合 | value 向量 L2 范数(选)+ key 相似度(融) | 否(query 无关) | CaM 风格 |
| **evict** | token 级(只丢) | value 向量 L2 范数 | 否(query 无关) | SnapKV/H2O 风格 |
| **receiver (RASC)** | token 级(只丢) | **B 问题末 N 词对每个 token 的注意力** | **是,每条 query 重算** | 本工作 |

- 记号:token 方法的 `r` = **保留比例**(r=0.2 → 只留 20% token,压得最狠);KVComm 的 `top` = **保留层比例**。
- `recv_wN`:观测窗口 = 只用问题最后 N 个 token 的 query 打分(N∈{8,16})。
- 固定开销:第 0 层全量保留;每层至少保留 sink(4)+ recent(8)。

## 1.1 主对比表(各方法 @ 0.3 / 0.5 / 0.7 预算)

> 预算口径:KVComm(x)=保留 x 比例的**层**;merge/evict/RASC(x)=保留 x 比例的**token**。两者都≈传输 x 比例的全量 KV,可近似等带宽对比。
> `–` 表示未跑:QASPER(进行中)、2WikiMQA / TMATH(未跑);HotpotQA 的 RASC 仅跑到 r0.5。基线行(Baseline…KVComm)转录自既有总表。

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
| **RASC-w8 (0.3)** | 0.60 | 0.87 | 0.70 | – | **0.48** | 0.51 | – | – |
| **RASC-w16 (0.3)** | 0.61 | 0.87 | **0.70** | – | 0.46 | 0.51 | – | – |
| KVComm (0.5) | 0.62 | 0.95 | 0.60 | 0.29 | 0.34 | 0.50 | 0.37 | 0.37 |
| merge (0.5) | 0.57 | 0.78 | 0.67 | – | 0.40 | 0.40 | – | – |
| evict (0.5) | 0.59 | 0.78 | 0.68 | – | 0.41 | 0.39 | – | – |
| **RASC-w8 (0.5)** | 0.60 | 0.88 | 0.73 | – | **0.48** | 0.53 | – | – |
| **RASC-w16 (0.5)** | 0.59 | 0.89 | **0.75** | – | 0.47 | 0.52 | – | – |
| KVComm (0.7) | 0.62 | 0.96 | 0.69 | 0.29 | 0.39 | 0.53 | 0.38 | 0.38 |
| merge (0.7) | 0.61 | 0.82 | 0.71 | – | 0.47 | 0.55 | – | – |
| evict (0.7) | 0.62 | 0.83 | 0.71 | – | 0.49 | 0.49 | – | – |
| **RASC-w8 (0.7)** | 0.60 | 0.89 | – | – | **0.48** | 0.53 | – | – |
| **RASC-w16 (0.7)** | 0.61 | 0.89 | – | – | 0.48 | 0.54 | – | – |

要点:同 token 预算下 **RASC > evict > merge**(中压尤显);难任务(HotpotQA/MuSiQue)RASC 在更低带宽即反超 KVComm;简单任务(Countries/Tipsheets)KVComm 丢层鲁棒、RASC 不占优。

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
- **query 自适应 vs 静态**:KVComm 选择在校准阶段算一次后写死(query 无关);RASC 每条 query 实时重算(query 自适应)。这是除粒度外的第二条区别轴。

---

# 阶段 B:Budget-aware 升级研究线

> 动机:RASC 仍对每条 query、每一层用**同一个固定保留比 r**。但不同 query 的信息需求差异巨大(简单 fact lookup vs 多跳推理)。能否**按 query 自适应分配通信预算**?以下三步依次回答:(Step 0)前提是否成立 →(Step 1)发送方能否开环预测 →(Step 2)接收方能否闭环索取。

## 5. Step 0 — 预算 headroom 验证(oracle 分析)

**做法**:在 RASC(receiver-w16)上对每条样本扫密集预算档 `r∈{0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7}`,逐样本落盘得分(`eval.py` → `per_sample.jsonl`)。定义样本的 **oracle 最小预算** = 能解出它(F1≥τ=0.5)的最小 r。脚本:`scripts/analyze_oracle.py`。

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
- **FIXED-r(下界)**:无反馈、单一固定预算 = 当前 RASC 基线。
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

> **结论**:**单发 Pass-1 预算预测证伪**,且不满足跨数据集泛化的发表门槛(否决线:LODO 在 ≥2 留出任务正节省 ≥10%,实测三任务全负)。至此 **budget-aware 整条线(§5–§9)系统性证伪**,作为 negative result 收编,第三个正向抓手转向 §11 的牌 1。

---

## 10. 时间 / 开销(待严格测量)

- 当前 log 的 `communication time` 是**整轮 500 样本墙钟**,受"压坏→生成变长"和并行抢卡污染,**非干净延迟**。
- receiver 比 evict 多一遍 question 打分前向,均摊每样本 ~0.05s,可忽略。
- 单卡上压缩不省"传输时间"(无真实传输),省的是 B 的 prefill 计算;真实通信收益须在分布式下测。
- **TODO**:补受控延迟实验(单样本分别计时 A-prefill / 打分 / B-prefill / 生成)。

## 11. 后续路线图

| 步骤 | 内容 | 机器成本 | 价值 | 状态 |
|---|---|---|---|---|
| **牌 1(下一步,主攻)** | **Cross-model RASC**:A、B 异构(不同权重 / tokenizer / 层数)时如何对齐打分。需排查 `compute_receiver_importance` 的同深度假设、跨 tokenizer 的 token 对齐、层数不等时的映射。这是真实 MAS 刚需、KVComm 未解决,作为第三个正向创新点 | 中(GPU) | **顶会抓手** | 待开 |
| Step 3 | 受控延迟实验:单样本分别计时 A-prefill / 打分 / B-prefill / 生成,分布式下测真实通信收益 | 小 | rebuttal | 待开 |
| 收尾 | 补 qasper;难任务 receiver 补 r0.6–0.9;窗口 {4,8,16,32,all} 完整曲线;生成 LaTeX 表 | 小 | 完整性 | 待开 |

> Budget-aware 路线(原 Step 2b / 2c)已在 §8–§9 证伪并归档为 negative result,不再投入。

## 12. 局限

- **同模型限定**:A、B 同权重时打分可在 A 端精确复现;异构模型需 B 传 query 向量或近似(牌 1,§11,future work)。
- **打分为启发式**:注意力之和;可升级为失真最优判据(注意力 × ‖value‖ / 输出分布变化),给 rate-distortion 论证。
- **Budget-aware 整条线已证伪**(§5–§9):前提虽成立(headroom 真实),但开环(§6)、离线 oracle(§7,仅上界)、在线多轮(§8)、单发 Pass-1 预测(§9,LODO AUC 0.585)都无法在等精度下稳定、可泛化地跑赢固定预算 RASC。论文中作为 negative result 收编,主张回到"RASC + 固定预算"的简洁形态。
- **渐进上界含非单调噪声红利**:§7 的精度反超部分来自 oracle 挑到"低预算偶然解出"的样本,真实触发器(§8)无法复现。

## 附 A:数据集目录结构

```
snapshots/<dataset>/
  ├── kvcomm/        kvcomm_top{0.3,0.5,0.7,1.0}_*
  ├── mtc_merge/     merge_r{0.1..0.9}_*
  ├── mtc_evict/     evict_r{0.1..0.9}_*
  ├── mtc_receiver/  recv_w{8,16}_r{0.1..0.9}_*  +  probe_recv_w16_r{0.05..0.7}_*(Step 0 密集探针 / 牌2 oracle 标签)
  ├── budget/        {uniform,layer,query,querylayer}_*（Step 1 预算分配）
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
