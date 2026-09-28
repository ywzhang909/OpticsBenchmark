# Fine-tuning Configs

**Path:** `configs/fine_tuning/` — 微调任务配置文件目录。

包含用于实际微调任务的配置文件，支持 OpenAI、DashScope（阿里云通义千问）、Together AI（Llama）和 Mistral AI 四个提供商。

## 文件说明

| 文件 | 提供商 | 模型 | 描述 |
|------|--------|------|------|
| `GPT_OpenAI_finetune.yaml` | OpenAI | GPT-4o-mini | OpenAI 微调任务配置 |
| `qwen_dashscope.yaml` | DashScope | Qwen3.7-plus | 阿里云通义千问微调任务配置 |
| `llama_together.yaml` | Together AI | Meta-Llama-3.1-8B-Instruct | Together AI 微调任务配置 |
| `ministral_mistral.yaml` | Mistral AI | Ministral-3b | Mistral AI 微调任务配置 |

## 配置结构

每个配置文件包含三个主要部分：

```yaml
llm:
  provider:
    type: "openai"  # 或 "dashscope"、"together"、"mistral"
    api_key: "${ENV_VAR}"
    base_url: "https://api.openai.com/v1"

fine_tuning:
  training_file: "dataset/fine_tune/paper_info_extract/test.jsonl"
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

> **注意**：不同提供商的 `base_url`、`hyperparameters` 字段及 API 参数不同。例如 Together AI 将超参数作为请求体顶层字段发送（`n_epochs`/`batch_size`/`learning_rate`/...），且使用官方 SDK (together-py)，`base_url` 需带 `/v1`；Mistral AI 使用原生字段（`training_steps`/`learning_rate`/`weight_decay`/`warmup_fraction`），且无需 `base_url`（使用官方 SDK）。

## 工作流

```bash
# 阶段 A：生成训练数据
python utils/build_finetune_dataset.py \
  -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \
  -p prompts/paper_info_extract/zero-shot_v1.0.txt \
  -d dataset/paper_info_extract/dataset_json/dataset_v1.json \
  -o results/finetune/train.jsonl --val-ratio 0.2

# 阶段 B：运行微调任务
python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml --wait
# 或使用 Together AI（Llama）
python src/finetune.py -c configs/fine_tuning/llama_together.yaml --wait
# 或使用 Mistral AI（Ministral）
python src/finetune.py -c configs/fine_tuning/ministral_mistral.yaml --wait

# 使用微调后的模型
# 将 status_output_path 中记录的模型名填入对应推理配置的 model.name
# (如 configs/llm/GPT_OpenAI.yaml 的 ft:gpt-4o-mini:...，
#  或 configs/llm/llama_openai.yaml 的 together/...，
#  或 configs/llm/mistral_official.yaml 的 ft:ministral-3b-latest:...)
```

## 官方文档

- OpenAI 微调: https://platform.openai.com/docs/guides/fine-tuning
- DashScope 微调: https://help.aliyun.com/zh/model-studio/developer-reference/finetune-overview
- Together AI 微调: https://docs.together.ai/docs/fine-tuning/supervised
- Mistral AI 微调: https://docs.mistral.ai/capabilities/finetuning/
