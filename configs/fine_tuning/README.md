# Fine-tuning Configs

**Path:** `configs/fine_tuning/` — Configuration files for fine-tuning jobs.

Holds the configuration files for real fine-tuning jobs, supporting four
providers: OpenAI, DashScope (Alibaba Cloud Tongyi Qianwen), Together AI (Llama)
and Mistral AI.

## Files

| File | Provider | Model | Description |
|------|----------|-------|-------------|
| `GPT_OpenAI_finetune.yaml` | OpenAI | GPT-4o-mini | OpenAI fine-tuning job configuration |
| `qwen_dashscope.yaml` | DashScope | Qwen3.7-plus | Alibaba Cloud Tongyi Qianwen fine-tuning job configuration |
| `llama_together.yaml` | Together AI | Meta-Llama-3.1-8B-Instruct | Together AI fine-tuning job configuration |
| `ministral_mistral.yaml` | Mistral AI | Ministral-3b | Mistral AI fine-tuning job configuration |

## Configuration Structure

Each configuration file contains three main sections:

```yaml
llm:
  provider:
    type: "openai"  # or "dashscope", "together", "mistral"
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

> **Note**: `base_url`, the `hyperparameters` fields and the API parameters
> differ between providers. For example, Together AI sends hyperparameters as
> top-level request body fields (`n_epochs`/`batch_size`/`learning_rate`/...) and
> uses the official SDK (together-py), where `base_url` must include `/v1`; Mistral
> AI uses native fields (`training_steps`/`learning_rate`/`weight_decay`/`warmup_fraction`)
> and needs no `base_url` (it uses the official SDK).

## Workflow

```bash
# Stage A: generate the training data
python utils/build_finetune_dataset.py \
  -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \
  -p prompts/paper_info_extract/zero-shot_v1.0.txt \
  -d dataset/paper_info_extract/dataset_json/dataset_v1.json \
  -o results/finetune/train.jsonl --val-ratio 0.2

# Stage B: run the fine-tuning job
python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml --wait
# or use Together AI (Llama)
python src/finetune.py -c configs/fine_tuning/llama_together.yaml --wait
# or use Mistral AI (Ministral)
python src/finetune.py -c configs/fine_tuning/ministral_mistral.yaml --wait

# Use the fine-tuned model
# Put the model name recorded in status_output_path into model.name of the
# matching inference config
# (e.g. ft:gpt-4o-mini:... in configs/llm/GPT_OpenAI.yaml,
#  together/... in configs/llm/llama_openai.yaml,
#  or ft:ministral-3b-latest:... in configs/llm/mistral_official.yaml)
```

## Official Documentation

- OpenAI fine-tuning: https://platform.openai.com/docs/guides/fine-tuning
- DashScope fine-tuning: https://help.aliyun.com/zh/model-studio/developer-reference/finetune-overview
- Together AI fine-tuning: https://docs.together.ai/docs/fine-tuning/supervised
- Mistral AI fine-tuning: https://docs.mistral.ai/capabilities/finetuning/
