# Fine-tuning 指南

> 使用多提供商 API 对光学领域基准数据集进行微调。

---

## 工作流概览

采用**两阶段显式流水线**，数据文件（JSONL）是两阶段唯一契约：

```
阶段 A：数据准备（独立步骤）
  gold_answer + prompt + dataset ──build_finetune_dataset.py──▶ train.jsonl / val.jsonl

阶段 B：微调执行（配置驱动）
  YAML(training_file 指向 JSONL) ──finetune.py──▶ 上传 → 创建任务 → 轮询/管理
  结果: ft:gpt-4o-mini:... ◀──────────────────────────────────────┘
  ↓ 填回 configs/llm/GPT_OpenAI.yaml 的 model.name
  复用现有 llm_pred.py + eval.py 完成微调前后效果对比
```

---

## 支持的提供商

| 提供商 | 类型 | 事件查询 | 说明 |
|--------|------|----------|------|
| OpenAI | `openai` | ✅ | 支持 GPT-4o mini/4o/4.1 mini 等模型 |
| Mistral | `mistral` | ❌ | 支持 Mistral 模型微调 |
| Together AI | `together` | ❌ | 支持 DeepSeek、Qwen 等开源模型 |
| Bedrock | `bedrock` | ❌ | 支持 Claude Haiku、Titan 等 AWS 模型 |
| DashScope | `dashscope` | ✅ | 支持 Qwen 系列模型（阿里云百炼） |

> ⚠️ Bedrock 需要额外安装 `boto3`：`pip install boto3`

---

## 阶段 A：数据准备

### 运行命令

```bash
python utils/build_finetune_dataset.py \
  -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \
  -p prompts/paper_info_extract/zero-shot_v1.0.txt \
  -d dataset/paper_info_extract/dataset_json/dataset_v1.json \
  -o results/finetune/train.jsonl \
  --val-ratio 0.2 \
  --seed 42
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-g / --gold` | **必需**，gold answer JSON 路径（`[{id, data}]`格式） | — |
| `-p / --prompt` | prompt 模板文件（作为 system 消息，跳过前两行注释） | 无 system 消息 |
| `-d / --dataset` | 数据集 JSON（含 PDF 路径等字段），按 title 与 gold 匹配 | 仅使用标题 |
| `-o / --output` | 训练集输出 JSONL 路径 | `results/finetune/train.jsonl` |
| `--val-output` | 验证集输出 JSONL 路径 | `<output>_val.jsonl` |
| `--val-ratio` | 验证集比例 `[0, 1)` | `0.2` |
| `--seed` | 切分随机种子 | `42` |
| `--user-field` | 取数据集记录的哪个字段作为 user 消息 | `title` |
| `--max-chars` | 用户内容最大字符数 | `100000` |
| `--encoding-model` | 用于 token 计数的 tiktoken 模型 | `gpt-4o-mini` |

### 输出格式（每行一个 JSON）

```json
{"messages": [
  {"role": "system", "content": "<prompt 模板内容>"},
  {"role": "user", "content": "<论文标题或提取全文>"},
  {"role": "assistant", "content": "<gold answer 的 JSON 字符串>"}
]}
```

### 自动告警

- 样本数 < 50 时：警告 OpenAI 建议最少 50 样本
- 单样本 token > 60,000 时：警告可能超出 gpt-4o-mini 训练限制
- 转换过程会输出 token 统计用于估算费用

---

## 阶段 B：微调执行

### 配置文件

#### OpenAI 配置示例

```yaml
llm:
  provider:
    type: "openai"
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"

fine_tuning:
  training_file: "results/finetune/train.jsonl"
  validation_file: null
  base_model: "gpt-4o-mini-2024-07-18"
  method: "supervised"
  suffix: "optis-bench"
  seed: null
  hyperparameters:
    n_epochs: 3
    batch_size: "auto"
    learning_rate_multiplier: "auto"

execution:
  poll_interval: 30
  poll_timeout: 86400
  status_output_path: "results/finetune/job_status.json"
```

#### Mistral 配置示例

```yaml
llm:
  provider:
    type: "mistral"
    api_key: "${MISTRAL_API_KEY}"

fine_tuning:
  training_file: "results/finetune/train.jsonl"
  base_model: "mistral-7b-v0.1"
  method: "supervised"
  suffix: "optis-mistral"
  hyperparameters:
    training_steps: 100
    learning_rate: 0.0001

execution:
  poll_interval: 30
  poll_timeout: 86400
  status_output_path: "results/finetune/job_status.json"
```

#### Together AI 配置示例

```yaml
llm:
  provider:
    type: "together"
    api_key: "${TOGETHER_API_KEY}"
    base_url: "https://api.together.xyz"

fine_tuning:
  training_file: "results/finetune/train.jsonl"
  base_model: "meta-llama/Llama-3-8b-hf"
  method: "supervised"
  suffix: "optis-together"

execution:
  poll_interval: 30
  poll_timeout: 86400
  status_output_path: "results/finetune/job_status.json"
```

#### Bedrock 配置示例

```yaml
llm:
  provider:
    type: "bedrock"
    region: "us-east-1"
    aws_key: "${AWS_ACCESS_KEY_ID}"
    aws_secret: "${AWS_SECRET_ACCESS_KEY}"

fine_tuning:
  training_file: "s3://my-bucket/train.jsonl"
  base_model: "anthropic.claude-3-haiku-20240307-v1:0"
  method: "supervised"

execution:
  poll_interval: 60
  poll_timeout: 86400
  status_output_path: "results/finetune/job_status.json"
```

#### DashScope (阿里云百炼) 配置示例

```yaml
llm:
  provider:
    type: "dashscope"
    api_key: "${DASHSCOPE_API_KEY}"
    base_url: "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

fine_tuning:
  training_file: "results/finetune/train.jsonl"
  base_model: "qwen3-14b"
  method: "supervised"
  suffix: "optis-dashscope"
  hyperparameters:
    n_epochs: 3
    batch_size: 16
    max_length: 8192
    learning_rate: "1.6e-5"

execution:
  poll_interval: 60
  poll_timeout: 86400
  status_output_path: "results/finetune/job_status.json"
```

### CLI 命令

```bash
# 创建并等待完成
python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml --wait

# 仅创建（打印 job id）
python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml

# 校验配置与数据（不触网）
python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml --dry-run

# 查询状态
python src/finetune.py --status ftjob-abc123

# 列出历史任务
python src/finetune.py --list

# 导出事件日志
python src/finetune.py --events ftjob-abc123 -o results/finetune/events.json

# 取消任务
python src/finetune.py --cancel ftjob-abc123
```

### 任务状态流转

```
creating → validating_files → queued → running → succeeded
                                         ↓     → failed
                                         ↓     → cancelled
```

`--wait` 模式会实时打印训练事件（step / loss / checkpoint），任务完成后状态落盘至 `status_output_path`。

---

## 微调完成后：效果对比评测

1. 将 `results/finetune/job_status.json` 中的 `fine_tuned_model`（形如 `ft:gpt-4o-mini:org:suffix:id`）填入 `configs/llm/GPT_OpenAI.yaml` 的 `llm.model.name`
2. 运行标准推理与评测管线，与微调前的 baseline 对比：

```bash
python src/llm_pred.py -c configs/llm/GPT_OpenAI.yaml
python src/eval.py -i results/llm_outputs.jsonl \
  -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \
  -e configs/evaluations/paper_info_extract.yaml
```

---

## 支持的微调方法

| 方法 | `method` | 说明 |
|------|----------|------|
| Supervised Fine-tuning (SFT) | `supervised` | 基于 JSONL 对话格式的标准监督微调 |
| Direct Preference Optimization | `dpo` | 直接偏好优化，训练数据需含 `rejected` 内容 |

当前版本主要演示 SFT；DPO 训练数据格式遵循各提供商官方规范。

---

## 支持的基座模型

### OpenAI

| 模型 | 建议快照 |
|------|----------|
| GPT-4o mini | `gpt-4o-mini-2024-07-18` |
| GPT-4o | `gpt-4o-2024-08-06` |
| GPT-4.1 mini | `gpt-4.1-mini-2025-04-14` |

完整列表见：[OpenAI Fine-tuning Models](https://platform.openai.com/docs/guides/fine-tuning#which-models-can-be-fine-tuned)

### Mistral

支持的模型见：[Mistral Fine-tuning](https://docs.mistral.ai/capabilities/fine-tuning/)

### Together AI

支持 100+ 开源模型，完整列表见：[Together AI Models](https://docs.together.ai/docs/chat-fine-tuning)

### Bedrock

| 模型 | 说明 |
|------|------|
| Claude 3 Haiku | 通过 Bedrock 微调 |
| Amazon Titan | AWS 自研模型 |
| Meta Llama 2 | 开源模型 |

完整列表见：[Bedrock Custom Model Jobs](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html)

---

## 费用说明

- **训练费用**：按输入 + 输出 token 总量计费，转换器会输出 `Total tokens` 用于估算
- **文件上传**：免费，但长期闲置的上传文件可能计费
- **推理费用**：微调后模型的推理按对应模型定价计费

建议在转换完成后查看 token 统计，通过各提供商定价页面预估费用。

---

## 架构设计

### 提供商适配器模式

```
BaseFineTuneProvider (ABC)
├── OpenAIFineTuneProvider      # OpenAI + 兼容 API
├── MistralFineTuneProvider     # Mistral AI
├── TogetherAIFineTuneProvider  # Together AI
└── BedrockFineTuneProvider     # AWS Bedrock
```

工厂函数 `create_finetune_provider()` 根据配置自动创建对应的适配器实例。

### 扩展新提供商

1. 继承 `BaseFineTuneProvider` ABC
2. 实现所有抽象方法（upload_file, create_job, retrieve_job 等）
3. 在 `finetune_provider_factory.py` 中注册新提供商

---

## 关键设计决策

| 决策 | 理由 |
|------|------|
| YAML 不含 `task` 字段 | 数据来源是转换器的职责，JSONL 文件即两阶段契约，避免两处真相源 |
| `llm` 节仅含 `provider` | 微调基座模型属于 `fine_tuning` 参数，不属于推理 `llm` 段 |
| `execution` 节仅含轮询参数 | 轮询间隔/超时是客户端行为，非 API 请求参数 |
| 状态文件单条记录 | 简化 `--status` / `--wait` 的恢复逻辑，历史任务用 `--list` 查看 |
| 适配器模式 | 统一接口，支持多提供商，易于扩展 |
