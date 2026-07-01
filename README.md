# KVComm

论文 [KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](https://openreview.net/forum?id=F7rUng23nw)（ICLR 2026）的官方实现。

一个面向大语言模型（LLM）之间通信的框架，研究模型如何高效共享信息，以提升协作推理与问答性能。

## 安装

```bash
pip install -r requirements.txt
```

注意：需要指定版本 `transformers==4.53.3`。

## 数据集

| 数据集 | 任务类型 | 说明 | 数据路径 |
|-------------------|-----------------------|---------------------------|-----------------------------------|
| `hotpotqa`        | 多跳问答              | 基于维基百科的推理        | HuggingFace                       |
| `qasper`          | 科学问答              | 基于论文的问题            | HuggingFace                       |
| `musique`         | 多跳问答              | 组合式推理                | HuggingFace                       |
| `multifieldqa_en` | 多领域问答            | 跨领域知识                | HuggingFace                       |
| `twowikimqa`      | 多跳问答              | 维基百科桥接实体          | HuggingFace                       |
| `tipsheets`       | 自定义问答            | 合成推理任务              | `dataloader/data/tipsheets.jsonl` |
| `countries`       | 地理问答              | 基于国家的问题            | `dataloader/data/countries.jsonl` |
| `tmath`           | 数学                  | 数学题求解                | `dataloader/data/TMATH`           |

## 快速开始

### Baseline 测试
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_baseline \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct
```

### Baseline 测试（本地模型 + 本地数据集）
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_baseline \
    --model_A /sharedspace/models/Llama-3.1-8B-Instruct \
    --model_B /sharedspace/models/Llama-3.1-8B-Instruct
```

### Skyline 测试

使用 HuggingFace 路径的模型：
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_skyline \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct
```

使用本地路径的模型：
```bash
CUDA_VISIBLE_DEVICES=2 python com.py \
    --test_task hotpotqa \
    --do_test_skyline \
    --model_A /sharedspace/models/Llama-3.1-8B-Instruct \
    --model_B /sharedspace/models/Llama-3.1-8B-Instruct
```

### KVComm 通信

使用 HuggingFace 路径的模型：
```bash
python com.py \
    --test_task hotpotqa \
    --do_test \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --top_layers 0.3
```

使用本地路径的模型（指定 0 号卡）：
```bash
CUDA_VISIBLE_DEVICES=0 python com.py \
    --test_task hotpotqa \
    --do_test \
    --model_A /sharedspace/models/Llama-3.1-8B-Instruct \
    --model_B /sharedspace/models/Llama-3.1-8B-Instruct \
    --top_layers 0.3
```

### 激活通信（Activation Communication, AC）
```bash
python com.py \
    --test_task tipsheets \
    --do_test_ac \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --layer_k 26 \
    --layer_j 26 \
    --f replace
```

### 自然语言辩论（Natural Language Debate, NLD）

使用 HuggingFace 路径的模型：
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_nld \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

使用本地路径的模型（指定 2 号卡）：
```bash
CUDA_VISIBLE_DEVICES=2 python com.py \
    --test_task hotpotqa \
    --do_test_nld \
    --model_A /sharedspace/models/Llama-3.1-8B-Instruct \
    --model_B /sharedspace/models/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

### CIPHER 通信

使用 HuggingFace 路径的模型：
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_cipher \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

使用本地路径的模型（指定 3 号卡）：
```bash
CUDA_VISIBLE_DEVICES=3 python com.py \
    --test_task hotpotqa \
    --do_test_cipher \
    --model_A /sharedspace/models/Llama-3.1-8B-Instruct \
    --model_B /sharedspace/models/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

## 通信方法

### 1. KVComm（跨视图通信）
- **机制**：将模型 A 指定层的 key-value 缓存共享给模型 B
- **参数**：`--layers_list`、`--layer_from`、`--layer_to`、`--top_layers`
- **适用场景**：以极小的计算开销实现高效信息传递

### 2. 激活通信（AC）
- **机制**：将模型 A 的隐藏激活在指定层注入模型 B
- **参数**：`--layer_k`（源层）、`--layer_j`（目标层）、`--f`（融合方式）
- **融合方式**：`replace`、`sum`、`mean`

### 3. 自然语言辩论（NLD）
- **机制**：模型间交换自然语言回复并迭代修正答案
- **参数**：`--nld_max_tokens_model_A_and_B_phase1`、`--sender_aware`
- **流程**：初始回复 → 交换 → 修正

### 4. CIPHER 通信
- **机制**：模型通过学习到的 embedding 表示进行通信
- **特性**：温度可控生成、最近邻解码

## 配置项

### 模型配置
- `--model_A`、`--model_B`：HuggingFace 模型标识或本地路径
- `--device`：CUDA 设备（默认 `cuda:0`）
- `--max_input_length`：最大输入 token 长度（默认 64000）

### 通信参数
- `--layers_list`：KVComm 通信使用的具体层
- `--top_layers`：使用的高重要性层比例
- `--layer_k`、`--layer_j`：AC 的源层与目标层
- `--f`：AC 的融合函数（`replace`、`sum`、`mean`）

### 评测设置
- `--test_task`：评测数据集
- `--limit`：限制评测样本数量
- `--calib_size`：层重要性的校准集大小

### 实验跟踪
- `--use_wandb`：启用 Weights & Biases 日志
- `--wandb_project`：W&B 项目名
- `--wandb_entity`：W&B 实体
- `--run_name`：自定义实验名

## 层重要性分析

框架内置自动层重要性检测：

```bash
python com.py \
    --test_task hotpotqa \
    --do_test \
    --top_layers 0.3
```

它会自动识别对通信最重要的层，并选取它们用于主评测。

# KVMergeComm

Token 级 KV 压缩通信：不再像 KVComm 那样丢弃整层，而是保留全部层、在每层内选择重要的
**token**。接收方感知打分（ReKV）按*接收方*模型的问题对*发送方*模型 KV 的注意力来选 token。

## 新增参数

| 参数 | 取值 | 含义 |
|-----------|--------|------|
| `--merge` | flag | 启用 token 级压缩（保留全部层） |
| `--merge_ratio` | 0.0–1.0 | 每层保留的 token 比例（≈ 带宽） |
| `--merge_mode` | `merge` / `evict` | `merge`：归一化 value 合并；`evict`：只丢弃（效果最好） |
| `--score_mode` | `value_norm` / `receiver` | token 重要性：value L2-范数（与 query 无关）vs 接收方感知 |
| `--recv_window` | 整数（如 8/16） | 接收方打分只用问题最后 N 个 token（0 = 全部） |
| `--merge_sink` | 整数 | 始终保留的前 N 个 token（attention sink） |
| `--merge_recent` | 整数 | 始终保留的后 N 个 token（recency） |

示例（ReKV，接收方感知，窗口 8，每层保留 30% token）：
```bash
CUDA_VISIBLE_DEVICES=0 python com.py \
    --test_task hotpotqa --do_test \
    --model_A /sharedspace/models/Llama-3.1-8B-Instruct \
    --model_B /sharedspace/models/Llama-3.1-8B-Instruct \
    --merge --merge_mode evict --score_mode receiver --recv_window 8 \
    --merge_ratio 0.3 --merge_sink 4 --merge_recent 8
```

## 脚本（`scripts/`）

所有脚本都用环境变量 `TASK`（数据集）和 `GPU`，结果写入 `snapshots/<TASK>/<method>/`。

| 脚本 | 作用 |
|--------|------|
| `download_datasets.sh` | 通过 `hfd.sh` + hf-mirror 把 QA 数据集下载到 `datasets/` |
| `run_baseline.sh` | KVComm 层丢弃（top 0.3/0.5/0.7）+ Full 上界 |
| `run_merge.sh` | 归一化 value 合并扫描（r=0.1..0.9） |
| `run_evict.sh` | token 选择、只丢弃、value-norm 打分（r=0.1..0.9） |
| `run_receiver.sh` | ReKV：接收方感知 + 观测窗口（`WIN=8/16`） |
| `run_dataset.sh` | 对单个数据集跑完整套方法 |

### 第一步 —— 下载数据集（一次性）
`countries` / `tipsheets` / `tmath` / `hotpotqa` 已是本地数据，只需下载 3 个 HF 仓库：
```bash
bash scripts/download_datasets.sh
# 等价于：
# export HF_ENDPOINT=https://hf-mirror.com
# ./datasets/hfd.sh Xnhyacinth/LongBench --dataset --local-dir datasets/LongBench
# ./datasets/hfd.sh dgslibisey/MuSiQue   --dataset --local-dir datasets/MuSiQue
# ./datasets/hfd.sh tau/scrolls          --dataset --local-dir datasets/scrolls
```
加载器会优先使用 `datasets/<name>`，没有则回退到 HuggingFace hub
（见 `dataloader/local_loader.py`）。

### 第二步 —— 跑单个数据集（完整套方法）
```bash
TASK=hotpotqa GPU=0 bash scripts/run_dataset.sh
```
会依次跑 baseline + merge + evict + receiver(w8/w16)，结果保存到 `snapshots/hotpotqa/`。

### 第三步 —— 跑全部 8 个数据集（分卡并行）
```bash
# 本地数据，可直接跑
TASK=countries  GPU=0 bash scripts/run_dataset.sh
TASK=tipsheets  GPU=1 bash scripts/run_dataset.sh
TASK=tmath      GPU=4 bash scripts/run_dataset.sh
# 需先执行 download_datasets.sh
TASK=twowikimqa      GPU=0 bash scripts/run_dataset.sh
TASK=multifieldqa_en GPU=1 bash scripts/run_dataset.sh
TASK=musique         GPU=4 bash scripts/run_dataset.sh
TASK=qasper          GPU=5 bash scripts/run_dataset.sh
```

### 只跑单个方法（而非完整套）
```bash
TASK=musique GPU=0 WIN=8 bash scripts/run_receiver.sh
```

> 提示：在原始 `com.py` 命令上加 `--limit 5` 可在全量运行前快速冒烟测试数据集加载。
