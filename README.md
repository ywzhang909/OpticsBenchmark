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
- **Multi-dimensional evaluation** — 7 metric modules: Exact Match, ROUGE, BLEU, BERTScore, Sentence Similarity, Edit Distance, Citation F1
- **Composite scoring** — PluginEval-inspired weighted scoring with LLM judge and anti-pattern penalties
- **Configuration-driven** — YAML-based configuration for LLMs and tasks; switching LLMs or tasks (across 8 task types) requires no code changes
- **Parallel execution** — Async concurrency with semaphore-based task control
- **Report generation** — Automatic HTML/Markdown reports with statistics and model comparison
- **Quick LLM selector** — Interactive CLI tool for comparing providers without writing code

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
MISTRAL_API_KEY=your-mistral-key
GROQ_API_KEY=your-groq-key
TOGETHER_API_KEY=your-together-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
EOF
```

### Download Datasets

```bash
# Dataset files are included in the repository under dataset/
# If data is hosted remotely, clone or download to the appropriate subdirectory
ls dataset/paper_info_extract/data_v1/  # Verify PDF papers are present
```

### Run an Evaluation (Two-Phase Pipeline)

**Phase 1** — Run agent to generate output dataset (via `optis` CLI or `src/main.py`):

```bash
# Using the installed CLI entry point
optis -a configs/llm/GPT_OpenAI.yaml -t paper_info_extract

# Or run directly
python src/main.py -a configs/llm/GPT_OpenAI.yaml -t paper_info_extract

# All tasks with 4 concurrent workers
python src/main.py -a configs/llm/claude_anthropic.yaml --all-tasks -c 4

# Specify output path
python src/main.py -a configs/llm/GPT_OpenAI.yaml -t paper_info_extract -o results/agent_outputs.jsonl

# Dry run to validate config without calling APIs
python src/main.py -a configs/llm/GPT_OpenAI.yaml -t paper_info_extract --dry-run
```

**Phase 2** — Evaluate agent outputs with scoring metrics:

```bash
# Evaluate agent outputs using eval config and gold answers
python src/eval.py -i results/agent_outputs.jsonl -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json -e configs/evaluations/paper_info_extract.yaml

# Specify output path for evaluation results
python src/eval.py -i results/agent_outputs.jsonl -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json -e configs/evaluations/paper_info_extract.yaml -o results/eval_results.json
```

---

## Project Structure

```
OpticsBenchmark/
├── configs/                        # Configuration center
│   ├── system/template.yaml       # Global system config template
│   ├── evaluations/               # Evaluation configs
│   ├── llm/                       # LLM provider configs (9 YAML files)
│   │   ├── GPT_OpenAI.yaml        # OpenAI GPT-4/4o
│   │   ├── claude_anthropic.yaml  # Anthropic Claude 3.5 Sonnet
│   │   ├── gemini_google.yaml     # Google Gemini 1.5 Pro
│   │   ├── kimi_openai.yaml       # Moonshot Kimi (K3/K2.x)
│   │   ├── glm_openai.yaml        # Zhipu GLM-4/GLM-Z1
│   │   ├── deepseek_openai.yaml   # DeepSeek V4 Pro
│   │   ├── qwen_openai.yaml       # Alibaba Qwen 3.5/3.7
│   │   ├── llama_openai.yaml      # Meta Llama 4 via Together AI
│   │   └── mistral_official.yaml  # Mistral via official SDK
│   └── tasks/                     # Task configs (3 task types + template)
├── src/                           # Core source package
│   ├── __init__.py
│   ├── main.py                    # Phase 1: Agent output generator (CLI: optis)
│   ├── eval.py                    # Phase 2: Evaluation engine
│   ├── llm_pred.py               # LLM prediction runner
│   ├── core/
│   │   ├── config.py             # Shared TaskConfig dataclass
│   │   ├── llm_judge.py          # LLM-as-judge with anchored rubrics
│   │   ├── llm_runner.py         # LLMPredRunner for prediction
│   │   └── runner.py             # Async parallel AgentRunner (259 lines)
│   ├── evaluators/               # Metric evaluators
│   │   ├── base.py               # BaseEvaluator ABC
│   │   ├── factory.py            # create_evaluator() config-driven factory
│   │   ├── helpers.py            # JSON parsing, sentence matching, dict normalization
│   │   ├── exact_match_evaluator.py
│   │   ├── rouge_evaluator.py
│   │   ├── bert_score_evaluator.py
│   │   ├── citation_evaluator.py
│   │   └── scorer/               # Thin wrappers → algorithm/* functions
│   ├── algorithm/                # Pure-math evaluation algorithms (12 modules)
│   │   ├── em_eval_utils.py
│   │   ├── rouge_eval_utils.py
│   │   ├── bleu_eval_utils.py
│   │   ├── meteor_eval_utils.py
│   │   ├── cider_eval_utils.py
│   │   ├── bertScore_eval_utils.py
│   │   ├── perplexity_eval_utils.py
│   │   ├── edit_distance_utils.py
│   │   ├── jaccard_similarity_utils.py
│   │   ├── hungarian_algorithm_utils.py
│   │   ├── sentence_similarity_utils.py
│   │   └── citation_eval_utils.py
│   ├── llm/                      # LLM abstraction layer (10 model classes, 8 providers)
│   │   ├── base.py               # BaseLLM ABC
│   │   ├── models/               # Model-specific LLM implementations
│   │   │   ├── ClaudeLLM.py      # Anthropic Claude (official SDK)
│   │   │   ├── DeepSeekLLM.py    # DeepSeek (OpenAI-compatible)
│   │   │   ├── GeminiLLM.py      # Google Gemini (Interactions API)
│   │   │   ├── GlmLLM.py         # Zhipu GLM (OpenAI SDK)
│   │   │   ├── GPTLLM.py         # OpenAI GPT (Chat Completions + Responses)
│   │   │   ├── KimiLLM.py        # Moonshot Kimi (OpenAI-compatible)
│   │   │   ├── LlamaLLM.py       # Meta Llama (Together AI + OpenAI SDK)
│   │   │   ├── MistralLLM.py     # Mistral (official mistralai SDK)
│   │   │   ├── OllamaLLM.py      # Ollama (local)
│   │   │   └── QwenLLM.py        # Alibaba Qwen (OpenAI-compatible)
│   │   └── providers/            # Provider-specific API clients
│   │       ├── AnthropicProvider.py
│   │       ├── BedrockProvider.py
│   │       ├── GoogleProvider.py
│   │       ├── MistralProvider.py
│   │       ├── OllamaProvider.py
│   │       ├── OpenAIProvider.py
│   │       └── TogetherAIProvider.py
│   ├── environments/
│   │   ├── base_env.py           # BaseEnvironment ABC + LocalEnvironment
│   │   └── zos_env.py            # Zemax ZOS-API integration (stub)
│   ├── module/
│   │   └── result.py             # EvaluationResult + AggregatedResults dataclasses
│   ├── utils/
│   │   ├── logger.py             # Loguru-based logging (console + file + rotation)
│   │   ├── parser.py             # YAML/JSONL/config parser with env-var expansion
│   │   └── generate_report.py    # HTML/Markdown report generator
│   └── tools/
│       └── quick_llm_selector.py # Interactive CLI tool for testing/comparing providers
├── dataset/                       # Evaluation datasets
│   ├── paper_info_extract/       # 15 PDF papers + JSON dataset/gold-answer files
│   ├── info_extraction/          # AO paper analysis files (274 texts)
│   ├── paper_review/             # Paper review dataset (empty, pending)
│   └── optics_question_answer/   # Q&A dataset (empty, pending)
├── prompts/                       # LLM prompt templates
│   ├── system/                   # System prompts (optical_agent, research_agent)
│   ├── templates/                # Task-specific templates
│   └── paper_info_extract/       # Custom prompt for paper info extraction task
├── utils/                         # Standalone utility scripts
│   ├── list_openai_support_models.py
│   └── paper_data_to_dateset.py
├── self_test/                     # Self-test datasets
├── tests/                         # Pytest test suite (20 test files)
├── docs/                          # Chinese technical documentation
│   ├── foundation/               # Optical basics, agent theory, evaluation methodology
│   ├── theory.md                 # Evaluation theory
│   ├── contribution.md           # Contribution guide
│   └── ...                       # Design docs
├── pyproject.toml                 # Single-source config (build, deps, tools)
├── requirements.txt
└── environment.yml
```

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
| Groq | `configs/llm/groq_openai.yaml` | `groq` |
| Ollama (local) | via `src/llm/models/OllamaLLM.py` | `httpx` |
| AWS Bedrock | via `src/llm/providers/BedrockProvider.py` | `boto3` |
| Together AI | via `src/llm/providers/TogetherAIProvider.py` | `openai` (compatible) |

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

These 12 independent evaluation utility modules are self-contained with consistent interfaces:

| Module | Metric | Description |
|--------|--------|-------------|
| `em_eval_utils.py` | Exact Match | Normalized text equality |
| `rouge_eval_utils.py` | ROUGE-L F1 | Recall-oriented n-gram overlap |
| `bleu_eval_utils.py` | BLEU | N-gram precision |
| `meteor_eval_utils.py` | METEOR | Synonym-aware matching |
| `cider_eval_utils.py` | CIDEr | TF-IDF weighted n-gram consensus |
| `bertScore_eval_utils.py` | BERTScore | Semantic similarity via BERT embeddings |
| `perplexity_eval_utils.py` | Perplexity | GPT-2 based fluency evaluation |
| `edit_distance_utils.py` | Levenshtein / WER | Character-level edit distance |
| `jaccard_similarity_utils.py` | Jaccard / Dice | Token set overlap coefficients |
| `sentence_similarity_utils.py` | Sentence Embedding | Semantic similarity + Hungarian matching |
| `hungarian_algorithm_utils.py` | Optimal Assignment | Minimum-cost matching |
| `citation_eval_utils.py` | Citation F1 | Citation verification via NLI |

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

### Structured Output (OpenAI)

When `structured_output: true` is set in a task config, the OpenAI agent generates a JSON Schema from the gold-answer file at runtime, stripping any keys containing `id` (case-insensitive). This is passed as `text.format` with `strict: true` to `client.responses.create()`, enforcing the schema during generation.

### Eval Loop (Two-Phase Pipeline)

The evaluation is split into **two independent phases**, allowing re-evaluation with different metrics without re-running the LLM:

**Phase 1 — Agent Output** (`src/main.py`, CLI entry point: `optis`):

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

### `optis` / `python src/main.py` — Phase 1: Agent Output Generator

```
Usage: python src/main.py -a <agent_config> -t <task_set> [options]

Arguments:
  -a, --agent-config PATH    Agent YAML config file (default: configs/llm/GPT_OpenAI.yaml)
  -t, --task-set NAME        Task set name or path to task config (default: configs/tasks/paper_info_extract.yaml)
  --all-tasks                Run all available task sets
  -o, --output PATH          Output JSONL path (default: results/agent_outputs.jsonl)
  -c, --concurrency N        Max concurrent agent sessions (default: 1)
  --timeout SECONDS          Per-task timeout (default: 300)
  --max-samples N            Limit samples per task
  --system-config PATH       System config path (default: configs/system/template.yaml)
  --dry-run                  Validate config without calling APIs
  --log-level LEVEL          Logging level (default: INFO)
  --log-file PATH            Log file path
  --version                  Show version and exit
```

### `python src/eval.py` — Phase 2: Evaluation Engine

```
Usage: python src/eval.py -i <agent_outputs> -g <gold> -e <eval_config> [options]

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

## Quick LLM Selector

Test and compare LLM models interactively without writing code:

```bash
# List available providers
python -m src.tools.quick_llm_selector --list

# Interactive mode
python -m src.tools.quick_llm_selector

# Test a single provider
python -m src.tools.quick_llm_selector --provider gpt-4 --prompt "Explain optical refraction"

# Compare multiple providers
python -m src.tools.quick_llm_selector --compare gpt-4 claude-3 gemini --prompt "What is a convex lens?"

# Output as JSON
python -m src.tools.quick_llm_selector --provider gpt-4 --prompt "Hello" --format json
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
2. Create a YAML task config in `configs/tasks/`
3. Create a prompt template in `prompts/templates/`
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
