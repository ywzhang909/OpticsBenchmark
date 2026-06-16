# OptiS Benchmark

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Open-Source Optical Design Agent Evaluation Framework**

</div>

---

## Overview

OptiS Benchmark is a modular, extensible evaluation framework for assessing LLM-based agents in **optical design** and **optical science paper understanding** tasks. It provides standardized benchmarks, pluggable agent backends (7 LLM providers), multi-dimensional evaluation metrics, and async parallel execution.

### Key Features

- **Optics-focused environments** — Zemax OpticStudio ZOS-API integration for ray tracing, lens design, tolerance analysis
- **Pluggable LLM backends** — 7 providers: OpenAI, Anthropic, Google Gemini, Groq, Ollama, AWS Bedrock, Together AI
- **Structured output support** — JSON schema generation from gold-answer files for OpenAI Responses API
- **Multi-dimensional evaluation** — 12 metric modules: Exact Match, ROUGE, BLEU, METEOR, CIDEr, BERTScore, Perplexity, Sentence Similarity, Hungarian Matching, Jaccard Similarity, Edit Distance, Citation F1
- **Composite scoring** — PluginEval-inspired weighted scoring with LLM judge and anti-pattern penalties
- **Configuration-driven** — YAML-based agent and task configs; no code changes to switch models or tasks
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

If you have an **NVIDIA GPU**, auto-detect and install the GPU-accelerated version:

```bash
# Recommended: auto-detect CUDA driver and install matching PyTorch
uv run python scripts/install_torch.py

# Or manually specify a CUDA version
uv pip install torch --torch-backend=cu130 --upgrade
```

> The `install_torch.py` script updates `pyproject.toml` so subsequent `uv sync` calls use the correct PyTorch index.
>
> Supported backends: `auto`, `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`, `cu130`

---

### Environment Setup

```bash
# Create .env file with your API keys (Linux/Mac/Git Bash)
cat > .env << EOF
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=sk-ant-your-key
GOOGLE_API_KEY=your-gemini-key
GROQ_API_KEY=your-groq-key
TOGETHER_API_KEY=your-together-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
EOF
```

### Download Datasets

```bash
bash scripts/download_data.sh
```

### Run an Evaluation (Two-Phase Pipeline)

**Phase 1** — Run agent to generate output dataset (via `optis` CLI or `src/main.py`):

```bash
# Using the installed CLI entry point
optis -a configs/agents/openai/gpt-4.yaml -t lens_design

# Or run directly
python src/main.py -a configs/agents/openai/gpt-4.yaml -t lens_design

# All tasks with 4 concurrent workers
python src/main.py -a configs/agents/anthropic/claude-3.yaml --all-tasks -c 4

# Specify output path
python src/main.py -a configs/agents/openai/gpt-4.yaml -t lens_design -o results/agent_outputs.jsonl

# Dry run to validate config without calling APIs
python src/main.py -a configs/agents/openai/gpt-4.yaml -t lens_design --dry-run
```

**Phase 2** — Evaluate agent outputs with scoring metrics:

```bash
# Evaluate agent outputs using task config
python src/eval.py -i results/agent_outputs.jsonl -t configs/tasks/lens_design.yaml

# Specify output path for evaluation results
python src/eval.py -i results/agent_outputs.jsonl -t configs/tasks/lens_design.yaml -o results/eval_results.json
```

---

## Project Structure

```
OpticsBenchmark/
├── configs/                        # Configuration center
│   ├── system.yaml                # Global system config
│   ├── agents/                    # Agent configs (7 providers + template)
│   └── tasks/                     # Task configs (7 task types + template)
├── src/                           # Core source package
│   ├── __init__.py
│   ├── main.py                    # Phase 1: Agent output generator (CLI: optis)
│   ├── eval.py                    # Phase 2: Evaluation engine
│   ├── core/
│   │   ├── agent.py              # BaseAgent ABC + 7 LLM providers + factory
│   │   ├── config.py             # Shared TaskConfig dataclass
│   │   ├── evaluator.py          # 6 evaluator types + ROGUEScorer + analyzers + report generator
│   │   ├── composite_scorer.py   # Multi-dimensional weighted scoring (PluginEval-style)
│   │   ├── llm_judge.py          # LLM-as-judge with anchored rubrics
│   │   └── runner.py             # Async parallel AgentRunner
│   ├── environments/
│   │   ├── base_env.py           # BaseEnvironment ABC + LocalEnvironment
│   │   └── zos_env.py            # Zemax ZOS-API integration (stub)
│   ├── utils/
│   │   ├── logger.py             # Loguru-based logging (console + file + rotation)
│   │   └── parser.py             # YAML/JSONL/config parser with env-var expansion
│   └── tools/
│       └── quick_llm_selector.py # Interactive CLI tool for testing/comparing providers
├── dataset/                       # Evaluation datasets (downloaded via download_data.sh)
├── prompts/                       # Prompt templates
│   ├── system/                   # System prompts (optical_agent, research_agent)
│   ├── templates/                # Task-specific templates (6 tasks)
│   └── paper_info_extract/       # Custom prompt for paper info extraction task
├── scripts/                       # Evaluation & utility scripts
│   ├── install_torch.py          # PyTorch CUDA auto-installation
│   ├── optics_paper_extract_eval.py # Paper extraction evaluation pipeline
│   ├── generate_report.py        # Standalone HTML/Markdown report generator
│   ├── download_data.sh          # Dataset download script
│   ├── run_eval.sh               # Evaluation automation wrapper
│   └── utils/                    # 12 standalone evaluation utility modules
├── tests/                         # Pytest test suite (21 files, 367 tests)
├── docs/                          # Chinese-language technical documentation
├── website/                       # Static leaderboard page (index.html)
├── pyproject.toml                 # Single-source config (build, deps, tools)
├── requirements.txt
├── environment.yml
└── uv.lock
```

---

## Supported LLM Providers

| Provider | Config | Client Library |
|----------|--------|----------------|
| OpenAI (GPT-4, GPT-4 Turbo) | `configs/agents/openai/gpt-4.yaml` | `openai` |
| Anthropic (Claude 3.5 Sonnet) | `configs/agents/anthropic/claude-3.yaml` | `anthropic` |
| Google Gemini (1.5 Pro) | `configs/agents/google/gemini.yaml` | `google-genai` |
| Groq (free inference, Llama 3.1 70B) | `configs/agents/groq/groq.yaml` | `groq` |
| Ollama (local models) | `configs/agents/ollama/ollama.yaml` | `httpx` |
| AWS Bedrock (Claude 3.5 Sonnet) | `configs/agents/bedrock/bedrock.yaml` | `boto3` |
| Together AI (Llama 3.3 70B Instruct) | `configs/agents/together/together.yaml` | `httpx` |

Per-provider features:
- **Anthropic**: Configurable `thinking_budget` for extended thinking
- **OpenAI**: `api_params` dict for arbitrary OpenAI Responses API parameters (e.g. `store`, `metadata`, `include`, `reasoning`)
- **Ollama**: Configurable `ollama_host` for remote Ollama instances

---

## Supported Tasks

| Task ID | Description | Difficulty | Samples | Evaluation Metrics |
|---------|-------------|------------|---------|-------------------|
| `lens_design` | Lens design and optimization | 3 | 50 | Metric-based (MTF, spot size) |
| `system_analysis` | Optical system performance analysis | 2 | 75 | Metric-based |
| `paper_review` | Academic paper review | 3 | 50 | ROUGE-L, citation accuracy |
| `paper_retrieval_eval` | Paper retrieval and citation | 2 | 100 | Precision, recall, F1 |
| `paper_info_extract` | 13-field structured extraction from optical science papers | 2 | 100 | Structured output + metric |
| `multi_doc_summary` | Multi-document summarization | 3 | 50 | ROUGE-1/2/L composite |
| `research_overview` | Research area overview generation | 4 | 30 | ROUGE-L, coverage |

The `paper_info_extract` task supports structured output (`structured_output: true`), generating a JSON Schema from the gold-answer file at runtime. It also uses `file_input: true` to read PDF files directly.

---

## Evaluation Types

### Core Evaluators (`src/core/evaluator.py`)

| Evaluator | Scoring Method | Description |
|-----------|----------------|-------------|
| MetricBasedEvaluator | `metric_based` | Numeric optical performance metrics |
| ExactMatchEvaluator | `exact_match` | Normalized string equality |
| PartialMatchEvaluator | `partial_match` | Jaccard similarity for strings, key matching for dicts |
| SummarizationEvaluator | `summarization` / `rouge` | ROUGE-1/2/L weighted composite via ROGUEScorer |
| CitationEvaluator | `citation` / `retrieval` | Precision, recall, F1 for citation accuracy |
| CompositeEvaluator | `composite` | Multi-dimensional weighted scoring with LLM judge and anti-pattern penalties |

Additional utility classes in `evaluator.py`:
- **ROGUEScorer** — ROUGE-1/2/L computation engine
- **ResultAnalyzer** — Per-task result analysis and aggregation
- **ErrorAnalyzer** — Error classification and pattern detection
- **EvaluationQA** — Q&A evaluation support
- **ReportGenerator** — HTML/Markdown report generation from evaluation results

### Standalone Metric Modules (`scripts/utils/`)

These 12 independent evaluation utilities are used by `scripts/optics_paper_extract_eval.py` and also tested individually:

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

OptiS Benchmark implements a **three-layer evaluation architecture** inspired by [Vercel Labs benchmark-agents / PluginEval](https://www.skills.sh/vercel-labs/vercel-plugin/benchmark-agents):

### Composite Weighted Scoring

The composite scoring engine (`CompositeScorer`) blends scores across 8 optical-design dimensions:

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
  -a, --agent-config PATH    Agent YAML config file (required)
  -t, --task-set NAME        Task set to run (required unless --all-tasks)
  --all-tasks                Run all available task sets
  -o, --output PATH          Output JSONL path (default: results/agent_outputs.jsonl)
  -c, --concurrency N        Max concurrent agent sessions (default: 1)
  --timeout SECONDS          Per-task timeout (default: 300)
  --max-samples N            Limit samples per task
  --system-config PATH       System config path (default: configs/system.yaml)
  --dry-run                  Validate config without calling APIs
  --log-level LEVEL          Logging level (default: INFO)
  --log-file PATH            Log file path
  --version                  Show version and exit
```

### `python src/eval.py` — Phase 2: Evaluation Engine

```
Usage: python src/eval.py -i <agent_outputs> -t <task_config> [options]

Arguments:
  -i, --input PATH           Agent outputs JSONL file (required)
  -t, --task-config PATH     Task YAML config file (required)
  -o, --output PATH          Output results path (default: results/eval_results.json)
  --system-config PATH       System config path (default: configs/system.yaml)
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

### Paper Extraction Evaluation

Evaluates paper information extraction predictions against gold-standard data using all 12 metric modules:

```bash
python scripts/optics_paper_extract_eval.py \
  --pred-file results/predictions.json \
  --gold-file dataset/processed/gold.json \
  --match --rouge --bertScore
```

### Report Generation

Two report generation options are available:

```bash
# Standalone script (from evaluation results JSON)
python scripts/generate_report.py results/eval_results.json --format html
python scripts/generate_report.py results/eval_results.json --format markdown
python scripts/generate_report.py results/eval_results.json --format both

# In-code generation (via ReportGenerator in evaluator.py)
python -c "from src.core.evaluator import ReportGenerator; ..."
```

### Automation

```bash
# Full evaluation pipeline wrapper
bash scripts/run_eval.sh
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Specific test file
pytest tests/test_evaluator_base.py

# With coverage
pytest tests/ --cov=src --cov-report=html

# Skip network-dependent tests
pytest tests/ --ignore=tests/test_bert_score_eval.py
```

---

## Customization

### Adding a New LLM Provider

1. Create a provider class in `src/core/agent.py` extending `BaseAgent`
2. Implement `chat(messages, tools)` → `AgentOutput` and `close()`
3. Register it in the `create_agent()` factory
4. Add a YAML config in `configs/agents/`

### Adding a New Evaluator

1. Create an evaluator class in `src/core/evaluator.py` extending `BaseEvaluator`
2. Register it in the `create_evaluator()` factory
3. Reference the scoring method in task config YAML

### Adding a New Task

1. Prepare a dataset JSON file (array of records)
2. Create a YAML task config in `configs/tasks/`
3. Create a prompt template in `prompts/templates/`
4. (Optional) Add a `structured_output` section and `gold_answer_path` for JSON-schema-constrained generation

---

## Dependencies

**Core**: `openai`, `anthropic`, `pydantic`, `pyyaml`, `pandas`, `numpy`, `scipy`, `loguru`, `tqdm`, `httpx`, `aiohttp`, `bert-score`, `rouge-score`, `sentencepiece`, `tiktoken`, `pypdf2`, `python-docx`

**Dev**: `pytest`, `pytest-asyncio`, `pytest-cov`, `black`, `isort`, `ruff`, `mypy`

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- [Vercel Labs benchmark-agents / PluginEval](https://www.skills.sh/vercel-labs/vercel-plugin/benchmark-agents) — Composite weighted scoring, LLM judge evaluation, anti-pattern penalties, and coverage reporting methodology ([MIT License](https://github.com/vercel-labs/skills))
- [AgentBench](https://github.com/OpenGVLab/AgentBench) — Evaluation framework design inspiration
- [Zemax OpticStudio](https://www.zemax.com/) — Optical design software
