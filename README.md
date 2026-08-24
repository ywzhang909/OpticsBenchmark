# Optis Benchmark

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Open-Source Automated Benchmark for LLM Academic Capabilities in Optics**

</div>

---

## Overview

Optis Benchmark is a modular, extensible evaluation framework designed to assess the performance of Large Language Models (LLMs) in tasks related to optical science papers, such as **information extraction, academic Q&A, and paper review**. The framework provides standardized test benchmarks, configurable LLM backends with multi-vendor support, multi-dimensional evaluation metrics, and asynchronous parallel execution capabilities.

### Key Features

- **Optics-focused environments** — Zemax OpticStudio ZOS-API integration for ray tracing, lens design, tolerance analysis
- **Multi-Provider Support** — Compatible with OpenAI, Anthropic, Google Gemini, Moonshot (Kimi), Zhipu (GLM), Mistral, Groq, Ollama, AWS Bedrock, and Together AI
- **Structured output support** — JSON schema generation from gold-answer files for OpenAI Responses API
- **Multi-dimensional evaluation** — 6 metric modules: Exact Match, ROUGE, BLEU, BERTScore, Sentence Similarity, Citation F1
- **Composite scoring** — PluginEval-inspired weighted scoring with LLM judge and anti-pattern penalties
- **Configuration-driven** — YAML-based configuration for LLMs and tasks; switching LLMs or tasks requires no code changes
- **Parallel execution** — Async concurrency with semaphore-based task control
- **Report generation** — Automatic HTML/Markdown reports with statistics and model comparison

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ywzhang909/OpticsBenchmark.git
cd OpticsBenchmark

# Using uv (recommended)
uv sync

# Or using pip
python -m venv .venv
# .venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### PyTorch / CUDA Auto-Configuration

This project uses **BERTScore** which requires PyTorch. `uv sync` installs the **CPU version** by default.

If you have an **NVIDIA GPU**, install the GPU-accelerated version:

```bash
# Manually specify a CUDA version
uv pip install torch --torch-backend=cu130 --upgrade
```

> Supported backends: `auto`, `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`, `cu130`

---

### Environment Setup

```bash
# Create .env file with your API keys (Linux/Mac/Git Bash)
cat > .env << EOF
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=sk-ant-your-key
GOOGLE_API_KEY=your-gemini-key
MOONSHOT_API_KEY=your-moonshot-key
ZHIPUAI_API_KEY=your-zhipuai-key
QWEN_API_KEY=your-qwen-key
DEEPSEEK_API_KEY=your-deepseek-key
MISTRAL_API_KEY=your-mistral-key
GROQ_API_KEY=your-groq-key
TOGETHER_API_KEY=your-together-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
EOF
```

### Download Datasets

```bash
# Dataset directories are excluded from version control via .gitignore
# You need to prepare them manually:
# - dataset/paper_info_extract/    (PDF papers + JSON dataset/gold-answer files)
# - dataset/info_extraction/       (AO paper analysis files)
# - dataset/paper_review/          (Paper review dataset)
# - dataset/optics_question_answer/ (Q&A dataset)

ls dataset/paper_info_extract/  # Verify PDF papers are present
```

### Run an Evaluation (Two-Phase Pipeline)

> **Note:** The `optis` CLI entry point is currently unavailable (`src/main.py` was removed). Use `python src/llm_pred.py` directly.

**Phase 1** — Run model inference to generate outputs (via `src/llm_pred.py`):

```bash
# Run inference with a config file
python src/llm_pred.py -c configs/llm/GPT_OpenAI.yaml

# Override output path
python src/llm_pred.py -c configs/llm/GPT_OpenAI.yaml -o results/my_outputs.jsonl

# Limit sample count
python src/llm_pred.py -c configs/llm/GPT_OpenAI.yaml -n 10

# Set concurrency
python src/llm_pred.py -c configs/llm/GPT_OpenAI.yaml --concurrency 4

# Dry run (show config only, no API calls)
python src/llm_pred.py -c configs/llm/GPT_OpenAI.yaml --dry-run
```

**Phase 2** — Evaluate model outputs with scoring metrics:

```bash
# Evaluate using eval config and gold answers
python src/eval.py -i results/agent_outputs.jsonl -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json -e configs/evaluations/paper_info_extract.yaml

# Specify output path for evaluation results
python src/eval.py -i results/agent_outputs.jsonl -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json -e configs/evaluations/paper_info_extract.yaml -o results/eval_results.json
```

---

## Project Structure

```
OpticsBenchmark/
├── configs/                                # Configuration center
│   ├── system/template.yaml               # Global runtime settings template
│   ├── evaluations/                       # Evaluation configs
│   │   ├── paper_info_extract.yaml        # Paper info extraction evaluator config
│   │   └── template.yaml                  # Generic evaluator config template
│   ├── llm/                               # LLM provider configs (9 YAML files)
│   │   ├── GPT_OpenAI.yaml                # OpenAI GPT-4/4o
│   │   ├── claude_anthropic.yaml          # Anthropic Claude 3.5 Sonnet
│   │   ├── gemini_google.yaml             # Google Gemini 1.5 Pro
│   │   ├── kimi_openai.yaml               # Moonshot Kimi (K3/K2.x)
│   │   ├── glm_openai.yaml                # Zhipu GLM-4/GLM-Z1
│   │   ├── deepseek_openai.yaml           # DeepSeek V4 Pro
│   │   ├── qwen_openai.yaml               # Alibaba Qwen 3.5/3.7
│   │   ├── llama_openai.yaml              # Meta Llama 4 via Together AI
│   │   └── mistral_official.yaml          # Mistral via official SDK
│   └── README.md                          # Configuration system documentation
├── src/                                   # Core source package
│   ├── __init__.py                        # Package root exporting core modules, environments, utils
│   ├── llm_pred.py                        # Phase 1: LLM inference entry point
│   ├── eval.py                            # Phase 2: Evaluation engine
│   ├── core/
│   │   ├── config.py                      # Shared TaskConfig dataclass
│   │   ├── llm_judge.py                   # LLM-as-judge with anchored rubrics
│   │   ├── llm_runner.py                  # LLMPredRunner for prediction
│   │   └── runner.py                      # Async parallel AgentRunner (255 lines)
│   ├── evaluators/                        # Metric evaluators
│   │   ├── base.py                        # BaseEvaluator ABC
│   │   ├── factory.py                     # create_evaluator() config-driven factory
│   │   ├── helpers.py                     # JSON parsing, sentence matching, dict normalization
│   │   ├── exact_match_evaluator.py       # Normalized string equality evaluator
│   │   ├── rouge_evaluator.py             # ROUGE-1/2/L evaluator with Hungarian alignment
│   │   ├── bert_score_evaluator.py        # BERTScore precision/recall/F1 evaluator
│   │   ├── citation_evaluator.py          # Citation precision/recall/F1 via NLI
│   │   └── scorer/                        # Thin wrappers → algorithm/* functions
│   ├── algorithm/                         # Pure-math evaluation algorithms (8 modules)
│   │   ├── em_eval_utils.py               # Text normalization and exact-match utilities
│   │   ├── rouge_eval_utils.py            # ROUGE score computation with multi-reference support
│   │   ├── bleu_eval_utils.py             # BLEU score computation with smoothing
│   │   ├── bert_score_eval_utils.py        # BERTScore computation with batch support
│   │   ├── citation_eval_utils.py         # Citation F1 via AutoAIS NLI model
│   │   ├── hungarian_algorithm_utils.py   # Optimal sentence assignment via Hungarian algorithm
│   │   ├── sentence_similarity_utils.py   # Transformer-based sentence embedding similarity
│   │   └── model_registry.py              # Thread-safe GPU model registry and caching
│   ├── llm/                               # LLM abstraction layer (11 model classes, 7 providers)
│   │   ├── __init__.py                    # Provider/LLM registry + factory functions
│   │   ├── base.py                        # BaseLLM ABC
│   │   ├── models/                        # Model-specific LLM implementations
│   │   │   ├── claude_llm.py               # Anthropic Claude (official SDK)
│   │   │   ├── deepseek_llm.py             # DeepSeek (OpenAI-compatible)
│   │   │   ├── gemini_llm.py               # Google Gemini (Interactions API)
│   │   │   ├── glm_llm.py                  # Zhipu GLM (OpenAI SDK)
│   │   │   ├── gpt_llm.py                  # OpenAI GPT (Chat Completions + Responses)
│   │   │   ├── kimi_llm.py                 # Moonshot Kimi (OpenAI-compatible)
│   │   │   ├── llama_llm.py                # Meta Llama (Together AI + OpenAI SDK)
│   │   │   ├── mistral_llm.py              # Mistral (official mistralai SDK)
│   │   │   ├── ollama_llm.py               # Ollama (local)
│   │   │   └── qwen_llm.py                 # Alibaba Qwen (OpenAI-compatible)
│   │   └── providers/                     # Provider-specific API clients
│   │       ├── anthropic_provider.py        # Async Anthropic SDK wrapper
│   │       ├── bedrock_provider.py          # Async AWS Bedrock boto3 wrapper
│   │       ├── google_provider.py           # Async Google GenAI client wrapper
│   │       ├── mistral_provider.py          # Async Mistral SDK wrapper
│   │       ├── ollama_provider.py           # Async Ollama HTTP API wrapper
│   │       ├── openai_provider.py           # Async OpenAI-compatible SDK wrapper
│   │       └── together_ai_provider.py       # Async Together AI HTTP API wrapper
│   ├── environments/
│   │   ├── base_env.py                    # BaseEnvironment ABC + LocalEnvironment
│   │   └── zos_env.py                     # Zemax ZOS-API integration (stub)
│   ├── module/
│   │   └── result.py                      # EvaluationResult + AggregatedResults dataclasses
│   ├── utils/
│   │   ├── logger.py                      # Loguru-based logging (console + file + rotation)
│   │   ├── parser.py                      # YAML/JSONL/config parser with env-var expansion
│   │   ├── generate_report.py             # HTML/Markdown report generator
│   │   └── general.py                     # Standalone utilities (_dict_to_response_format)
├── dataset/                                # Evaluation datasets (excluded from git, prepare locally)
│   ├── paper_info_extract/                # 15 PDF papers + JSON dataset/gold-answer files
│   ├── info_extraction/                   # AO paper analysis files (274 texts)
│   ├── paper_review/                      # Paper review dataset
│   └── optics_question_answer/            # Q&A dataset
├── prompts/                                # LLM prompt templates
│   ├── system/                            # System prompts (optical_agent, research_agent)
│   ├── templates/                         # Task-specific templates
│   ├── paper_info_extract/                # Paper info extraction prompt
│   ├── paper_review/                      # Paper review prompt
│   └── optics_question_answers/           # Optics Q&A prompt
├── utils/                                  # Standalone utility scripts
│   ├── list_openai_support_models.py      # List available OpenAI-compatible models
│   └── paper_data_to_dateset.py           # Convert paper files to JSON dataset
├── self_test/                              # Self-test datasets and scripts (excluded from git)
├── tests/                                  # Pytest test suite (18 test files)
├── docs/                                   # Chinese technical documentation
│   ├── foundation/                        # Optical basics, agent theory, evaluation methodology
│   ├── theory.md                          # Evaluation theory
│   ├── contribution.md                    # Contribution guide
│   └── ...                                # Design docs
├── AGENTS.md                               # AI agent instructions
├── STANDARDS.md                            # Code standards and conventions
├── TODO.md                                 # Development roadmap and known issues
├── pyproject.toml                          # Single-source config (build, deps, tools)
├── requirements.txt                        # Python dependency list
└── environment.yml                         # Conda environment specification
```

---

## LLM Config Format

Each YAML config file in `configs/llm/` follows a three-section structure:

```yaml
llm:
  provider:
    type: openai          # Maps to Provider registry (openai, anthropic, google, etc.)
    api_key: ${OPENAI_API_KEY}  # Supports ${ENV_VAR} expansion
    base_url: https://api.openai.com/v1
  model:
    type: gpt             # Maps to LLM registry (gpt, claude, qwen, etc.)
    name: gpt-4o          # Actual API model name
  setup:                  # Request parameters
    max_completion_tokens: 4096
    temperature: 0.7
    # ... model-specific params (thinking_budget, tools, etc.)

task:
  dataset_path: dataset/paper_info_extract/dataset_json/gold_answer_v1.json
  prompt_template: prompts/paper_info_extract/
  file_input: true        # Read files (PDF) instead of text
  max_samples: null       # null = all samples
  structured_output: false
  gold_answer_path: dataset/paper_info_extract/dataset_json/gold_answer_v1.json

execution:
  concurrency: 1
  timeout: 300
  output_path: results/pred.jsonl
```

See [`configs/README.md`](configs/README.md) for detailed field documentation.

---

## Supported LLM Providers

| Provider | Config(s) | Client Library |
|----------|-----------|----------------|
| OpenAI (GPT-4/4o) | `configs/llm/GPT_OpenAI.yaml` | `openai` |
| Anthropic (Claude 3.5 Sonnet) | `configs/llm/claude_anthropic.yaml` | `anthropic` |
| Google Gemini (1.5 Pro) | `configs/llm/gemini_google.yaml` | `google-genai` |
| Moonshot Kimi (K3/K2.x) | `configs/llm/kimi_openai.yaml` | `openai` (compatible) |
| Zhipu GLM (GLM-4/GLM-Z1) | `configs/llm/glm_openai.yaml` | `openai` (compatible) |
| DeepSeek (V4 Pro) | `configs/llm/deepseek_openai.yaml` | `openai` (compatible) |
| Alibaba Qwen (3.5/3.7) | `configs/llm/qwen_openai.yaml` | `openai` (compatible) |
| Meta Llama (4 Scout) | `configs/llm/llama_openai.yaml` | `openai` (Together AI) |
| Mistral | `configs/llm/mistral_official.yaml` | `mistralai` |
| Ollama (local) | via `src/llm/models/ollama_llm.py` | `httpx` |
| AWS Bedrock | via `src/llm/providers/bedrock_provider.py` | `boto3` |
| Together AI | via `src/llm/providers/together_ai_provider.py` | `openai` (compatible) |

> **Note:** Groq is not currently supported — it is not registered in the provider/LLM maps (`src/llm/__init__.py`).

Per-provider features:
- **OpenAI**: Chat Completions API + Responses API (selectable via `api_method`); structured output via `response_format`
- **Anthropic**: Configurable `thinking_budget` for extended thinking
- **Google Gemini**: Interactions API (`client.interactions.create()`), thinking levels, cached content
- **Moonshot Kimi**: K3 uses `reasoning_effort`, K2.x uses `thinking`, moonshot-v1-* skips both
- **Zhipu GLM**: OpenAI SDK compatible, web_search/file_search/tool_search support
- **Mistral**: Official `mistralai` SDK, web search support
- **Llama**: Dual provider support — TogetherAIProvider (httpx) + OpenAIProvider (OpenAI SDK)

---

## Supported Tasks

| Task ID | Description | Difficulty | Environment | Evaluation Metrics |
|---------|-------------|------------|-------------|-------------------|
| `paper_info_extract` | 13-field structured extraction from optical science papers | 2 | Optical sandbox | Exact Match, ROUGE, BERTScore |
| `paper_review` | Academic paper review | 3 | Local | ROUGE-L, content coverage |
| `optics_question_answer` | Optics question answering | 2 | Optical sandbox | Exact Match, ROUGE-L |

The `paper_info_extract` task supports structured output (`structured_output: true`), generating a JSON Schema from the gold-answer file at runtime. It also uses `file_input: true` to read PDF files directly.

Additional task configs are planned: `lens_design`, `system_analysis`, `paper_retrieval_eval`, `multi_doc_summary`, `research_overview`.

---

## Evaluation Types

### Core Evaluators (`src/evaluators/`)

| Evaluator | Scoring Method | Description |
|-----------|----------------|-------------|
| ExactMatchEvaluator | `exact_match` | Normalized string equality per JSON field |
| RougeEvaluator | `rouge` | ROUGE-1/2/L with Hungarian sentence alignment |
| BertScoreEvaluator | `bert_score` | BERTScore P/R/F1 with transformer embeddings |
| CitationEvaluator | `citation` | Precision, recall, F1 for citation accuracy |

Additional components:
- **LLMJudge** (`src/core/llm_judge.py`) — LLM-as-judge with structured rubrics (PluginEval Layer 2)
- **ReportGenerator** (`src/utils/generate_report.py`) — HTML/Markdown report generation from evaluation results
- **ResultAnalyzer** — Per-task result analysis and aggregation
- **ErrorAnalyzer** — Error classification and pattern detection

### Standalone Metric Modules (`src/algorithm/`)

These 8 independent evaluation utility modules are self-contained with consistent interfaces:

| Module | Metric | Description |
|--------|--------|-------------|
| `em_eval_utils.py` | Exact Match | Normalized text equality |
| `rouge_eval_utils.py` | ROUGE-L F1 | Recall-oriented n-gram overlap |
| `bleu_eval_utils.py` | BLEU | N-gram precision |
| `bert_score_eval_utils.py` | BERTScore | Semantic similarity via BERT embeddings |
| `citation_eval_utils.py` | Citation F1 | Citation verification via NLI |
| `hungarian_algorithm_utils.py` | Optimal Assignment | Minimum-cost matching |
| `sentence_similarity_utils.py` | Sentence Embedding | Semantic similarity + Hungarian matching |
| `model_registry.py` | Model Registry | Centralized model configurations |

---

## Evaluation Methodology

Optis Benchmark implements a **three-layer evaluation architecture** inspired by [Vercel Labs benchmark-agents / PluginEval](https://www.skills.sh/vercel-labs/vercel-plugin/benchmark-agents):

### Composite Weighted Scoring

The composite scoring engine blends scores across 8 optical-design dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| `optical_accuracy` | 0.25 | Accuracy of optical design outputs (MTF, spot size, etc.) |
| `metric_correctness` | 0.20 | Correct computation of evaluation metrics |
| `output_completeness` | 0.15 | All required output fields present |
| `citation_accuracy` | 0.12 | Correctness and relevance of citations |
| `reasoning_quality` | 0.10 | Quality of reasoning in explanations |
| `robustness` | 0.08 | Handles edge cases gracefully |
| `efficiency` | 0.05 | Computational efficiency |
| `reproducibility` | 0.05 | Results are reproducible |

**Three-layer formula:**
```
blended_score = static_weight × static_score + judge_weight × judge_score
raw_composite = Σ (dim_weight × blended_score)
final_score   = raw_composite × anti_pattern_penalty
```

**Anti-pattern penalties** apply multiplicative reductions when common failure modes are detected (empty output → 0.6×, hallucinated citation → 0.7×, parse failure → 0.5×).

### LLM Judge

The `LLMJudge` provides structured rubric-based scoring with 7 default quality rubrics (relevance, correctness, completeness, clarity, reasoning, citations, robustness). It builds a prompt containing the task, agent output, and anchored rubrics (5 levels per dimension, 0.0–1.0), then parses the LLM's JSON response into per-dimension scores with justifications.

### Verification Coverage

The `build_coverage_report()` function aggregates multiple `ScoreReport` objects into a coverage report showing which dimensions were evaluated, how many tasks had judge-layer scoring, anti-pattern breakdowns, and coverage gaps — mirroring PluginEval's coverage reporting concept.

### Evaluator Priority

Evaluators can be assigned a `priority` field in the evaluation config YAML (higher = runs first). This allows running expensive models first (e.g., BERTScore) before lightweight checks (e.g., Exact Match), optimizing total evaluation time:

```yaml
eval_metrics:
  exact_match:
    priority: 4    # Runs last (fastest)
  rouge:
    priority: 3
  bert_score:
    priority: 2    # Runs early (slowest, uses transformer model)
  citation:
    priority: 1    # Runs first
```

The `sort_evaluators_by_priority()` function in `src/eval.py` sorts evaluators by priority before execution.

### Structured Output (OpenAI)

When `structured_output: true` is set in a task config, the OpenAI agent generates a JSON Schema from the gold-answer file at runtime, stripping any keys containing `id` (case-insensitive). This is passed as `text.format` with `strict: true` to `client.responses.create()`, enforcing the schema during generation.

### Eval Loop (Two-Phase Pipeline)

The evaluation is split into **two independent phases**, allowing re-evaluation with different metrics without re-running the LLM:

**Phase 1 — Agent Output** (`src/llm_pred.py`):

1. Load agent config and task config from YAML
2. Load task instances from the dataset JSON
3. Launch parallel agent sessions (semaphore-based concurrency in `AgentRunner`)
4. Save raw agent outputs as a JSONL file

**Phase 2 — Evaluation** (`src/eval.py`):

1. Load agent outputs from the saved JSONL file
2. Create evaluator from task config's evaluation config
3. Score outputs with automated metrics (and optional LLM judge)
4. Aggregate into composite score reports with anti-pattern detection
5. Save aggregated results as JSON + per-task results as JSONL

---

## CLI Reference

### `python src/llm_pred.py` — Phase 1: LLM Inference

```
Usage: python src/llm_pred.py [options]

Arguments:
  -c, --config PATH          LLM config file path (default: configs/llm/qwen_openai.yaml)
  -o, --output PATH          Output JSONL path (default: results/qwen3.8-max_pred.jsonl)
  -n, --max-samples N        Max samples (overrides max_samples in config)
  --concurrency N            Concurrency (overrides concurrency in config)
  --dry-run                  Show config only, do not run inference
  --log-level LEVEL          Logging level: DEBUG/INFO/WARNING/ERROR/CRITICAL (default: INFO)
```

### `python src/eval.py` — Phase 2: Evaluation Engine

```
Usage: python src/eval.py [options]

Arguments:
  -i, --input PATH           Agent outputs JSONL file (default: self_test/dataset/paper_info_extract/test_v1.jsonl)
  -g, --gold PATH            Gold standard answer dataset file (JSON) (default: dataset/paper_info_extract/dataset_json/gold_answer_v1.json)
  -e, --eval-config PATH     Evaluation configuration YAML file (default: configs/evaluations/paper_info_extract.yaml)
  -o, --output PATH          Output results path (default: results/eval_results.json)
  --system-config PATH       System config path (default: configs/system/template.yaml)
  --log-level LEVEL          Logging level (default: INFO)
  --log-file PATH            Log file path
```

---

## Scripts

### Standalone Utility Scripts

```bash
# List supported OpenAI models
python utils/list_openai_support_models.py

# Convert paper data to dataset format
python utils/paper_data_to_dateset.py
```

### Report Generation

```bash
# Generate HTML/Markdown report from evaluation results
python src/utils/generate_report.py results/eval_results.json --format html
```

---

## Testing

```bash
# Run all tests
uv run pytest tests/

# Specific test file
uv run pytest tests/test_evaluator_base.py

# With coverage
uv run pytest tests/ --cov=src --cov-report=html

# Skip network-dependent tests
uv run pytest tests/ --ignore=tests/test_bert_score_eval.py
```

---

## Customization

### Adding a New LLM Provider

1. Create a provider class in `src/llm/models/` extending `BaseLLM`
2. Implement `chat(messages, tools)` → `LLMOutput` and `close()`
3. Add a YAML config in `configs/llm/`

### Adding a New Evaluator

1. Create an evaluator class in `src/evaluators/` extending `BaseEvaluator`
2. Register it in `create_evaluator()` factory (`src/evaluators/factory.py`)
3. Reference the scoring method in task config YAML

### Adding a New Task

1. Prepare a dataset JSON file (array of records)
2. Add task configuration to an existing LLM config file (`configs/llm/*.yaml`) under the `task` section
3. Create a prompt template in `prompts/` task-specific subdirectory
4. (Optional) Add a `structured_output` section and `gold_answer_path` for JSON-schema-constrained generation

---

## Dependencies

**Core**: `openai`, `anthropic`, `google-genai`, `mistralai`, `pydantic`, `pyyaml`, `python-dotenv`, `pandas`, `numpy`, `scipy`, `loguru`, `tqdm`, `httpx`, `aiohttp`, `bert-score`, `rouge-score`, `sentencepiece`, `tiktoken`, `pypdf2`, `python-docx`, `accelerate`, `protobuf`

**Dev**: `pytest`, `pytest-asyncio`, `pytest-cov`, `black`, `isort`, `ruff`, `mypy`, `pre-commit`

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- [Vercel Labs benchmark-agents / PluginEval](https://www.skills.sh/vercel-labs/vercel-plugin/benchmark-agents) — Composite weighted scoring, LLM judge evaluation, anti-pattern penalties, and coverage reporting methodology ([MIT License](https://github.com/vercel-labs/skills))
- [AgentBench](https://github.com/OpenGVLab/AgentBench) — Evaluation framework design inspiration
- [Zemax OpticStudio](https://www.zemax.com/) — Optical design software
