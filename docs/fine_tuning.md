# Fine-tuning Guide

> Fine-tune optics-domain benchmark datasets using multi-provider APIs.

---

## Workflow Overview

The pipeline is a **two-stage explicit pipeline**, where the data files (JSONL)
are the only contract between the two stages:

```
Stage A: data preparation (a standalone step)
  gold_answer + prompt + dataset ──build_finetune_dataset.py──▶ train.jsonl / val.jsonl

Stage B: fine-tuning execution (config driven)
  YAML(training_file points at the JSONL) ──finetune.py──▶ upload → create job → poll/manage
  Result: ft:gpt-4o-mini:... ◀──────────────────────────────────────┘
  ↓ put it back into model.name in configs/llm/GPT_OpenAI.yaml
  Reuse the existing llm_pred.py + eval.py to compare pre- and post-fine-tuning results
```

---

## Supported Providers

| Provider | Type | Event query | Description |
|----------|------|-------------|-------------|
| OpenAI | `openai` | ✅ | Supports GPT-4o mini/4o/4.1 mini and similar models |
| Mistral | `mistral` | ❌ | Supports fine-tuning Mistral models |
| Together AI | `together` | ❌ | Supports open-source models such as DeepSeek and Qwen |
| Bedrock | `bedrock` | ❌ | Supports AWS models such as Claude Haiku and Titan |
| DashScope | `dashscope` | ✅ | Supports the Qwen model family (Alibaba Cloud Bailian) |

> ⚠️ Bedrock requires `boto3` to be installed separately: `pip install boto3`

---

## Stage A: Data Preparation

### Command

```bash
python utils/build_finetune_dataset.py \
  -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \
  -p prompts/paper_info_extract/zero-shot_v1.0.txt \
  -d dataset/paper_info_extract/dataset_json/dataset_v1.json \
  -o results/finetune/train.jsonl \
  --val-ratio 0.2 \
  --seed 42
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-g / --gold` | **Required**, path to the gold answer JSON (format `[{id, data}]`) | — |
| `-p / --prompt` | Prompt template file (used as the system message, skipping the first two comment lines) | No system message |
| `-d / --dataset` | Dataset JSON (with fields such as the PDF path), matched to the gold answers by `title` | Titles only |
| `-o / --output` | Output JSONL path for the training set | `results/finetune/train.jsonl` |
| `--val-output` | Output JSONL path for the validation set | `<output>_val.jsonl` |
| `--val-ratio` | Validation ratio `[0, 1)` | `0.2` |
| `--seed` | Random seed for the split | `42` |
| `--user-field` | Which field of the dataset record to use as the user message | `title` |
| `--max-chars` | Maximum number of characters in the user content | `100000` |
| `--encoding-model` | The tiktoken model used for token counting | `gpt-4o-mini` |

### Output Format (one JSON object per line)

```json
{"messages": [
  {"role": "system", "content": "<prompt template content>"},
  {"role": "user", "content": "<paper title or the full extracted text>"},
  {"role": "assistant", "content": "<the gold answer as a JSON string>"}
]}
```

### Automatic Warnings

- Fewer than 50 samples: warns that OpenAI recommends a minimum of 50 samples
- A single sample above 60,000 tokens: warns that the gpt-4o-mini training limit may be exceeded
- The conversion prints token statistics used to estimate the cost

---

## Stage B: Fine-tuning Execution

### Configuration Files

#### OpenAI Configuration Example

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

#### Mistral Configuration Example

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

#### Together AI Configuration Example

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

#### Bedrock Configuration Example

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

#### DashScope (Alibaba Cloud Bailian) Configuration Example

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

### CLI Commands

```bash
# Create and wait for completion
python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml --wait

# Create only (prints the job id)
python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml

# Validate the config and the data (no network calls)
python src/finetune.py -c configs/fine_tuning/GPT_OpenAI_finetune.yaml --dry-run

# Query the status
python src/finetune.py --status ftjob-abc123

# List past jobs
python src/finetune.py --list

# Export the event log
python src/finetune.py --events ftjob-abc123 -o results/finetune/events.json

# Cancel a job
python src/finetune.py --cancel ftjob-abc123
```

### Job Status Transitions

```
creating → validating_files → queued → running → succeeded
                                         ↓     → failed
                                         ↓     → cancelled
```

`--wait` mode prints the training events (step / loss / checkpoint) in real time;
once the job completes, the status is written to `status_output_path`.

---

## After Fine-tuning: Comparative Evaluation

1. Put the `fine_tuned_model` from `results/finetune/job_status.json` (of the form
   `ft:gpt-4o-mini:org:suffix:id`) into `llm.model.name` in `configs/llm/GPT_OpenAI.yaml`
2. Run the standard inference and evaluation pipeline and compare against the
   pre-fine-tuning baseline:

```bash
python src/llm_pred.py -c configs/llm/GPT_OpenAI.yaml
python src/eval.py -i results/llm_outputs.jsonl \
  -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \
  -e configs/evaluations/paper_info_extract.yaml
```

---

## Supported Fine-tuning Methods

| Method | `method` | Description |
|--------|----------|-------------|
| Supervised Fine-tuning (SFT) | `supervised` | Standard supervised fine-tuning based on the JSONL conversation format |
| Direct Preference Optimization | `dpo` | Direct preference optimization; the training data must contain `rejected` content |

The current version mainly demonstrates SFT; the DPO training data format
follows each provider's official specification.

---

## Supported Base Models

### OpenAI

| Model | Recommended snapshot |
|-------|----------------------|
| GPT-4o mini | `gpt-4o-mini-2024-07-18` |
| GPT-4o | `gpt-4o-2024-08-06` |
| GPT-4.1 mini | `gpt-4.1-mini-2025-04-14` |

For the full list see: [OpenAI Fine-tuning Models](https://platform.openai.com/docs/guides/fine-tuning#which-models-can-be-fine-tuned)

### Mistral

Supported models see: [Mistral Fine-tuning](https://docs.mistral.ai/capabilities/fine-tuning/)

### Together AI

Supports 100+ open-source models. For the full list see: [Together AI Models](https://docs.together.ai/docs/chat-fine-tuning)

### Bedrock

| Model | Description |
|-------|-------------|
| Claude 3 Haiku | Fine-tuned via Bedrock |
| Amazon Titan | An AWS in-house model |
| Meta Llama 2 | An open-source model |

For the full list see: [Bedrock Custom Model Jobs](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html)

---

## Cost Notes

- **Training cost**: billed on the total number of input + output tokens; the
  converter prints `Total tokens` for estimation
- **File upload**: free, but long-idle uploaded files may be billed
- **Inference cost**: inference with the fine-tuned model is billed at the
  corresponding model's pricing

After the conversion, check the token statistics and estimate the cost using
each provider's pricing page.

---

## Architecture Design

### Provider Adapter Pattern

```
BaseFineTuneProvider (ABC)
├── OpenAIFineTuneProvider      # OpenAI + compatible APIs
├── MistralFineTuneProvider     # Mistral AI
├── TogetherAIFineTuneProvider  # Together AI
└── BedrockFineTuneProvider     # AWS Bedrock
```

The factory function `create_finetune_provider()` automatically creates the
matching adapter instance from the configuration.

### Adding a New Provider

1. Subclass the `BaseFineTuneProvider` ABC
2. Implement all abstract methods (upload_file, create_job, retrieve_job, etc.)
3. Register the new provider in `finetune_provider_factory.py`

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| The YAML has no `task` field | Sourcing the data is the converter's job; the JSONL file is the contract between the two stages, which avoids two sources of truth |
| The `llm` section only holds `provider` | The fine-tuning base model belongs to the `fine_tuning` parameters, not to the inference `llm` section |
| The `execution` section only holds polling parameters | The polling interval/timeout are client behaviour, not API request parameters |
| A single record in the status file | Simplifies the resume logic of `--status` / `--wait`; past jobs are listed with `--list` |
| Adapter pattern | A unified interface, multi-provider support, easy to extend |
