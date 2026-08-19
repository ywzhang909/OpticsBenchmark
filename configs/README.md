# Configuration

**Path:** `configs/` — YAML-driven configuration for Optis Benchmark.

A three-part config system: LLM provider configs (9 YAML files), evaluation configs, and a global system config template. All fields use `snake_case`; secrets use `${ENV_VAR}` syntax expanded at load time.

---

## Directory Structure

```
configs/
├── llm/                       # LLM provider configs (9 YAML files)
│   ├── GPT_OpenAI.yaml        # OpenAI GPT-4/4o
│   ├── claude_anthropic.yaml  # Anthropic Claude 3.5 Sonnet
│   ├── gemini_google.yaml     # Google Gemini 1.5 Pro
│   ├── kimi_openai.yaml       # Moonshot Kimi (K3/K2.x)
│   ├── glm_openai.yaml        # Zhipu GLM-4/GLM-Z1
│   ├── deepseek_openai.yaml   # DeepSeek V4 Pro
│   ├── qwen_openai.yaml       # Alibaba Qwen 3.5/3.7
│   ├── llama_openai.yaml      # Meta Llama 4 via Together AI
│   └── mistral_official.yaml  # Mistral via official SDK
├── evaluations/               # Evaluation configs
│   ├── paper_info_extract.yaml
│   └── template.yaml
├── system/
│   └── template.yaml          # Global settings template: logging, parallel, sandbox, rate-limit
└── README.md                  # This file
```

---

## 1. LLM Configuration (`configs/llm/`)

Each YAML file defines a complete LLM provider configuration with model, generation parameters, and optional tools.

### Common Fields

| Field | Description |
|-------|-------------|
| `model.provider` | Provider type (`openai`, `anthropic`, `google`, `kimi`, `glm`, `mistral`, `groq`, `bedrock`, `ollama`, `together`) |
| `model.name` | Model identifier (e.g. `gpt-4o`, `claude-3-5-sonnet-20241022`) |
| `model.api_key` | `${ENV_VAR}` placeholder for API key |
| `model.base_url` | API endpoint URL (for compatible providers) |
| `model.setup` | Generation parameters (temperature, max_tokens, top_p, etc.) |
| `tools` | Enabled capabilities (web_search, file_search, etc.) |

### Provider-Specific Features

| Provider | Special Features |
|----------|-----------------|
| `openai` | Chat Completions + Responses API (`api_method`), structured output (`response_format`) |
| `anthropic` | Thinking budget, MCP server, web search/fetch, tool search (regex/BM25) |
| `google` | Interactions API (`client.interactions.create()`), thinking levels, cached content |
| `kimi` | K3 uses `reasoning_effort`, K2.x uses `thinking`, moonshot-v1-* skips both |
| `glm` | OpenAI SDK compatible, web_search/file_search/tool_search support |
| `mistral` | Official `mistralai` SDK, web search support |
| `groq` | Fast inference, short timeout |
| `bedrock` | AWS region & credentials |
| `ollama` | Local endpoint, custom model names |
| `together` | Together AI API (OpenAI-compatible) |

### Example: OpenAI GPT-4o

```yaml
# configs/llm/GPT_OpenAI.yaml
model:
  provider: "openai"
  name: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"
  setup:
    temperature: 0.0
    max_tokens: 4096
    top_p: 1.0
```

---

## 2. Task Configuration

Task configurations are defined within each LLM config file (`configs/llm/*.yaml`) under the `task` section. Each LLM config includes task-specific settings like dataset path, prompt file, and evaluation parameters.

### Available Tasks

| Task ID | Category | Difficulty | Environment | Scoring Method |
|---------|----------|-----------|-------------|----------------|
| `paper_info_extract` | paper_info_extract | 2 | Optical sandbox | exact_match + ROUGE + BERTScore |
| `paper_review` | paper_review | 3 | Local | metric_based |
| `optics_question_answer` | optics_question_answer | 2 | Optical sandbox | exact_match + ROUGE + BERTScore |

Additional task configs are planned: `lens_design`, `system_analysis`, `paper_retrieval_eval`, `multi_doc_summary`, `research_overview`.

### Example: Paper Info Extract (in LLM config)

```yaml
# configs/llm/GPT_OpenAI.yaml (task section)
task:
  dataset_path: "dataset/paper_info_extract/dataset_json/dataset_v1.json"
  prompt_file: "prompts/paper_info_extract/zero-shot_v1.0.txt"
  file_input: true
  max_samples: 100
  shuffle: false
  structured_output: true
  gold_answer_path: "dataset/paper_info_extract/dataset_json/gold_answer_v1.json"
```

---

## 3. Evaluation Configuration (`configs/evaluations/*.yaml`)

Evaluation configs specify which metric evaluators run on agent outputs. The factory in `src/evaluators/factory.py` reads the `eval_metrics` key.

### Supported Evaluator Types

| Evaluator | Method | Use Case |
|-----------|--------|----------|
| `exact_match` | Normalized string equality per JSON field | Structured info extraction (title, DOI, authors, etc.) |
| `rouge` | ROUGE-1/2/L precision/recall/F1 | Text summarization, paper reviews |
| `bert_score` | BERTScore via transformer embeddings | Semantic similarity (abstracts, methods) |
| `citation` | Precision/recall/F1 for retrieved papers | Paper retrieval evaluation |

### Example: Paper Info Extract

```yaml
# configs/evaluations/paper_info_extract.yaml
eval_metrics:
  exact_match:
    info_names:
      - "title"
      - "publication year"
      - "doi"
      - "journal"
      - "authors"
  rouge:
    info_names:
      - "objective"
      - "method"
      - "performance metrics"
    metrics:
      - "rouge1"
      - "rouge2"
      - "rougeL"
  bert_score:
    info_names:
      - "objective"
      - "method"
```

---

## 4. System Configuration (`system/template.yaml`)

Global runtime settings for the benchmark runner.

| Section | Key Settings |
|---------|-------------|
| `logging` | Level (DEBUG–CRITICAL), file path, rotation (100 MB), retention (30 days), compression (zip) |
| `parallel` | `max_workers` (4), `batch_size` (1), `retry_attempts` (3), `retry_delay` (5s) |
| `sandbox` | `timeout` (300s), `max_steps` (50), `memory_limit` (4GB); Docker & local backends |
| `rate_limit` | Global (60 rpm, 1000 rph); per-provider limits (OpenAI 500, Anthropic 50, Groq 100) |
| `evaluation` | Intermediate & final result directories, metrics to compute |
| `export` | Format (jsonl/json/csv), compression, metadata inclusion |
| `security` | File ops, network, subprocess permissions; max file size |
| `development` | Debug mode, API tracing, mock API |
| `paths` | Dataset, prompts, scripts, docs, website directories |

---

## Usage

```bash
# Run evaluation with a specific LLM + task
uv run python src/llm_pred.py \
  -a configs/llm/GPT_OpenAI.yaml \
  -t paper_info_extract
```

---

## Known Issues

- `system/template.yaml` is a minimal template with most settings commented out — not a full config.
