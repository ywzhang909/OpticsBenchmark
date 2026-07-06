# Configuration

**Path:** `configs/` — YAML-driven configuration for OptiS Benchmark.

A four-part config system: pluggable agent configs (7 LLM providers, 12 model configs), task configs (9 task types), evaluation configs, composite scoring, and a global system config. All fields use `snake_case`; secrets use `${ENV_VAR}` syntax expanded at load time.

---

## Directory Structure

```
configs/
├── agents/                    # Agent model configurations (7 provider dirs)
│   ├── anthropic/             # Claude 3.5 Sonnet
│   │   ├── claude-3.yaml
│   │   └── template.yaml
│   ├── bedrock/               # AWS Bedrock
│   │   ├── bedrock.yaml
│   │   └── template.yaml
│   ├── google/                # Gemini 1.5 Pro (Vertex AI)
│   │   ├── gemini.yaml
│   │   └── template.yaml
│   ├── groq/                  # LLaMA 3.1 70B (fast inference)
│   │   ├── groq.yaml
│   │   └── template.yaml
│   ├── ollama/                # Local Ollama models
│   │   ├── ollama.yaml
│   │   └── template.yaml
│   ├── openai/                # 5 OpenAI-compatible models
│   │   ├── deepseek-v4-pro.yaml
│   │   ├── gpt-4.yaml
│   │   ├── llama4-scout.yaml
│   │   ├── mistral-medium-3.5.yaml
│   │   ├── qwen3.7-max.yaml
│   │   └── template.yaml
│   └── together/              # Together AI
│       ├── together.yaml
│       └── template.yaml
├── tasks/                     # Task definitions (9 YAML files)
│   ├── lens_design.yaml       # Lens design & optimization (Zemax)
│   ├── multi_doc_summary.yaml # Multi-document summarization
│   ├── optics_question_answer.yaml  # QA over optics papers
│   ├── paper_info_extract.yaml      # Structured info extraction from papers
│   ├── paper_retrieval_eval.yaml    # Citation/retrieval evaluation
│   ├── paper_review.yaml           # Academic paper review
│   ├── research_overview.yaml      # Research field survey
│   ├── system_analysis.yaml        # Optical system analysis
│   └── template.yaml               # Template for new tasks
├── evaluations/               # Evaluation configs
│   ├── paper_info_extract.yaml
│   └── template.yaml
├── eval_scoring.yaml          # Composite scoring: dimensions, weights, anti-patterns
├── system.yaml                # Global settings: logging, parallel, sandbox, rate-limit
├── AGENTS.md                  # Detailed configuration reference & conventions
└── README.md                  # This file
```

---

## 1. Agent Configuration (`configs/agents/*/`)

Each agent directory contains one or more model YAML files plus a `template.yaml` for that provider.

### Common Fields

| Field | Description |
|-------|-------------|
| `agent.name` | Unique identifier for the agent |
| `agent.version` | Semantic version |
| `agent.description` | Human-readable description |
| `model.provider` | Provider name (`openai`, `anthropic`, `google`, `groq`, `bedrock`, `ollama`, `together`) |
| `model.name` | Model identifier (e.g. `gpt-4-turbo`, `claude-3-5-sonnet-20241022`) |
| `model.api_key` | `${ENV_VAR}` placeholder for API key |
| `model.api_base` | API endpoint URL |
| `model.setup` | Generation parameters (temperature, max_tokens, top_p, etc.) |
| `system_prompt_file` | Path to system prompt template |
| `tools` | Enabled capabilities (file I/O, bash, python, Zemax, web search, MCP, etc.) |
| `execution` | Retry, timeout, cache settings |
| `cost_tracking` | API usage logging toggle |

### Provider-Specific Features

| Provider | Special Features |
|----------|-----------------|
| `anthropic` | Thinking budget, MCP server, web search/fetch, tool search (regex/BM25) |
| `openai` | API params (`n`, `store`), file search, Zemax ZOS-API integration |
| `google` | Vertex AI support, project/location config, thinking levels |
| `groq` | Short timeout (120s) for fast inference |
| `bedrock` | AWS region & credentials |
| `ollama` | Local endpoint, custom model names |
| `together` | Together AI API parameters |

### Example: OpenAI GPT-4

```yaml
# configs/agents/openai/gpt-4.yaml
agent:
  name: "optis-gpt4"
  version: "1.0.0"
  description: "GPT-4 Turbo Optical Design Agent"
model:
  provider: "openai"
  name: "gpt-4-turbo"
  api_base: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  setup:
    temperature: 0.0
    max_completion_tokens: 4096
    top_p: 1.0
    frequency_penalty: 0.0
    presence_penalty: 0.0
execution:
  max_retries: 3
  timeout: 300
  cache_enabled: true
  cache_ttl: 3600
```

---

## 2. Task Configuration (`configs/tasks/*.yaml`)

Each task YAML defines the full evaluation pipeline: dataset, environment, metrics, prompts, and cost.

### Schema

| Section | Sub-fields | Description |
|---------|-----------|-------------|
| `task` | `id`, `name`, `description`, `category`, `difficulty` (1-5), `estimated_time`, `tags` | Task metadata |
| `dataset` | `path`, `num_samples`, `shuffle`, `format` (input/output/metadata fields) | Data source config |
| `environment` | `type`, `software` (required/optional), `sandbox` (timeout/steps/memory) | Execution sandbox |
| `evaluation` | `scoring_method`, `metrics`, `success_criteria` | How to score agent outputs |
| `prompt` | `system_file`, `template_file`, `variables` | Prompt template paths |
| `cost` | `max_cost_per_task`, `budget_per_task` | Budget constraints |

### Available Tasks

| Task ID | Category | Difficulty | Environment | Scoring Method |
|---------|----------|-----------|-------------|----------------|
| `lens_design` | lens_design | 3 | Zemax sandbox (600s, 8GB) | metric_based (MTF, spot_size, distortion, etc.) |
| `system_analysis` | system_analysis | 2 | Zemax sandbox | metric_based + LLM judge |
| `paper_info_extract` | paper_info_extract | 2 | Optical sandbox | exact_match + ROUGE + BERTScore |
| `paper_retrieval_eval` | paper_retrieval | 2 | Local | metric_based (recall, precision, citation_accuracy) |
| `paper_review` | paper_review | 3 | Local | metric_based |
| `multi_doc_summary` | summarization | 3 | Local | metric_based (ROUGE-L, coherence, factuality) |
| `optics_question_answer` | optics_question_answer | 2 | Optical sandbox | exact_match + ROUGE + BERTScore |
| `research_overview` | research_overview | 4 | Local (600s) | metric_based (coverage, categorization, timeline) |

### Example: Lens Design

```yaml
# configs/tasks/lens_design.yaml
task:
  id: "lens_design"
  name: "Lens Design Optimization"
  category: "lens_design"
  difficulty: 3
dataset:
  path: "dataset/processed/lens_design.jsonl"
  num_samples: 50
environment:
  type: "optical_sandbox"
  sandbox:
    timeout: 600
    max_steps: 100
    memory_limit: "8GB"
evaluation:
  scoring_method: "metric_based"
  metrics:
    - name: "mtf_performance"
      type: "numeric"
    - name: "spot_size"
      type: "numeric"
  success_criteria:
    - metric: "mtf_performance"
      operator: ">="
      value: 0.7
prompt:
  system_file: "prompts/system/optical_agent.txt"
  template_file: "prompts/templates/lens_design.txt"
cost:
  max_cost_per_task: 10.0
```

---

## 3. Evaluation Configuration (`configs/evaluations/*.yaml`)

Evaluation configs specify which metric evaluators run on agent outputs. The factory in `src/core/evaluator.py` reads the `eval_metrics` key.

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

## 4. Composite Scoring (`eval_scoring.yaml`)

Multi-dimensional scoring with configurable weights, layer blending, and anti-pattern penalties.

### Layers

| Layer | Weight | Source |
|-------|--------|--------|
| Static | 0.7 | Automated metrics (exact match, ROUGE, BERTScore) |
| LLM Judge | 0.3 | LLM-based evaluation with structured rubrics |

### Dimensions (weights sum to 1.0)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| `optical_accuracy` | 0.25 | Accuracy of optical design outputs (MTF, spot size) |
| `metric_correctness` | 0.20 | Correct computation of evaluation metrics |
| `output_completeness` | 0.15 | All required fields present |
| `citation_accuracy` | 0.12 | Correctness of citations |
| `reasoning_quality` | 0.10 | Clarity of explanations |
| `robustness` | 0.08 | Handling of edge cases |
| `efficiency` | 0.05 | Computational efficiency |
| `reproducibility` | 0.05 | Deterministic results |

### Anti-Pattern Penalties

Penalties apply multiplicatively to the raw composite score (worst penalty wins):

| Pattern | Penalty | Effect |
|---------|---------|--------|
| `empty_output` | 0.6 | −40% |
| `hallucinated_citation` | 0.7 | −30% |
| `incorrect_calculation` | 0.75 | −25% |
| `parse_failure` | 0.5 | −50% |

---

## 5. System Configuration (`system.yaml`)

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
# Run evaluation with a specific agent + task
uv run python src/main.py \
  -a configs/agents/openai/gpt-4.yaml \
  -t lens_design
```

---

## Known Issues

- `system.yaml` references Docker sandbox (`image: optis_benchmark/sandbox:latest`) but no Dockerfile exists.
- `system.yaml` `save_api_keys: false` has a cautionary comment but no runtime enforcement.
- Agent `provider` field is hardcoded in factory — no plugin/discovery mechanism.
- `eval_scoring.yaml` overlaps with composite scoring config that could be part of `configs/evaluations/`.
