# OptiS Benchmark

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Open-Source Optical Design Agent Evaluation Framework**

</div>

---

## Overview

OptiS Benchmark is a modular, extensible evaluation framework for assessing LLM-based agents in **optical design** tasks. It provides standardized benchmarks, pluggable agent backends, multi-dimensional evaluation metrics, and parallel execution.

### Key Features

- **Optics-focused environments** — Zemax OpticStudio ZOS-API integration for ray tracing, lens design, tolerance analysis
- **Pluggable LLM backends** — 7 providers: OpenAI, Anthropic, Google Gemini, Groq, Ollama, AWS Bedrock, Together AI
- **Multi-dimensional evaluation** — Exact match, ROUGE, BERTScore, citation accuracy, sentence similarity, Hungarian matching
- **Configuration-driven** — YAML-based agent and task configs; no code changes to switch models or tasks
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

**Phase 1** — Run agent to generate output dataset:

```bash
# Single task
python src/main.py -a configs/agents/gpt-4.yaml -t lens_design

# All tasks with 4 concurrent workers
python src/main.py -a configs/agents/claude-3.yaml --all-tasks -c 4

# Specify output path
python src/main.py -a configs/agents/gpt-4.yaml -t lens_design -o results/agent_outputs.jsonl
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
│   └── tasks/                     # Task configs (6 task types + template)
├── src/                           # Core source
│   ├── main.py                    # Phase 1: Agent output generator
│   ├── eval.py                    # Phase 2: Evaluation engine
│   ├── core/
│   │   ├── agent.py              # Base agent + 7 LLM providers
│   │   ├── composite_scorer.py   # Multi-dimensional weighted scoring
│   │   ├── evaluator.py          # Evaluators + scorers + analyzers
│   │   ├── llm_judge.py          # LLM judge for rubric-based scoring
│   │   └── runner.py             # Async parallel runner + AgentOutput
│   ├── environments/
│   │   ├── base_env.py           # Base + local environment
│   │   └── zos_env.py           # Zemax ZOS-API integration
│   ├── utils/
│   │   ├── logger.py             # Loguru-based logging
│   │   └── parser.py             # YAML/JSONL/config parser
│   └── tools/
│       └── quick_llm_selector.py # Interactive LLM testing tool
├── scripts/                       # Evaluation and utility scripts
│   ├── install_torch.py          # PyTorch CUDA auto-installation
│   ├── optics_paper_extract_eval.py
│   ├── generate_report.py
│   ├── download_data.sh
│   ├── run_eval.sh
│   └── utils/                    # Evaluation utility modules
├── dataset/                       # Evaluation datasets (via download_data.sh)
├── prompts/                       # Prompt templates
│   ├── system/                   # System prompts (optical_agent, research_agent)
│   └── templates/                # Task-specific templates
├── tests/                         # Pytest test suite
├── docs/                          # Documentation
├── website/                       # Leaderboard page
├── pyproject.toml
├── requirements.txt
├── environment.yml
└── uv.lock
```

---

## Supported LLM Providers

| Provider | Config | Client Library |
|----------|--------|----------------|
| OpenAI (GPT-4, GPT-4 Turbo) | `configs/agents/gpt-4.yaml` | `openai` |
| Anthropic (Claude 3.5 Sonnet) | `configs/agents/claude-3.yaml` | `anthropic` |
| Google Gemini | `configs/agents/gemini.yaml` | `google-genai` |
| Groq (free inference) | `configs/agents/groq.yaml` | `groq` |
| Ollama (local models) | `configs/agents/ollama.yaml` | `httpx` |
| AWS Bedrock | `configs/agents/bedrock.yaml` | `boto3` |
| Together AI | `configs/agents/together.yaml` | `httpx` |

## Supported Tasks

| Task ID | Description | Evaluation Metrics |
|---------|-------------|-------------------|
| `lens_design` | Lens design and optimization | Metric-based (MTF, spot size, etc.) |
| `system_analysis` | Optical system performance analysis | Metric-based |
| `paper_review` | Academic paper review | ROUGE-L, citation accuracy |
| `paper_retrieval_eval` | Paper retrieval and citation | Precision, recall, F1 |
| `multi_doc_summary` | Multi-document summarization | ROUGE-1/2/L composite |
| `research_overview` | Research area overview generation | ROUGE-L, coverage |

## Evaluation Types

| Evaluator | Scoring Method | Description |
|-----------|----------------|-------------|
| MetricBasedEvaluator | `metric_based` | Numeric optical performance metrics |
| ExactMatchEvaluator | `exact_match` | Normalized string equality |
| PartialMatchEvaluator | `partial_match` | Jaccard similarity for strings, key matching for dicts |
| SummarizationEvaluator | `summarization` / `rouge` | ROUGE-1/2/L weighted composite |
| CitationEvaluator | `citation` / `retrieval` | Precision, recall, F1 for citation accuracy |
| CompositeEvaluator | `composite` | Multi-dimensional weighted scoring with LLM judge and anti-pattern penalties |

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

The `LLMJudge` provides structured rubric-based scoring when an LLM callable is configured. It builds a prompt containing the task, agent output, and anchored rubrics (5 levels per dimension, 0.0–1.0), then parses the LLM's JSON response into per-dimension scores with justifications.

### Verification Coverage

The `build_coverage_report()` function aggregates multiple `ScoreReport` objects into a coverage report showing which dimensions were evaluated, how many tasks had judge-layer scoring, anti-pattern breakdowns, and coverage gaps — mirroring PluginEval's coverage reporting concept.

### Eval Loop (Two-Phase Pipeline)

The evaluation is split into **two independent phases**, allowing re-evaluation with different metrics without re-running the LLM:

**Phase 1 — Agent Output (`src/main.py`):**
1. Load agent config and task config from YAML
2. Load task instances from the dataset JSONL
3. Launch parallel agent sessions (semaphore-based concurrency in `EvaluationRunner`)
4. Save raw agent outputs as a JSONL file (`AgentOutput`)

**Phase 2 — Evaluation (`src/eval.py`):**
1. Load agent outputs from the saved JSONL file
2. Create evaluator from task config's evaluation config
3. Score outputs with automated metrics (and optional LLM judge)
4. Aggregate into composite score reports with anti-pattern detection
5. Save aggregated results as JSON + per-task results as JSONL

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
```

## Scripts

### Paper Extraction Evaluation

```bash
python scripts/optics_paper_extract_eval.py \
  --pred-file results/predictions.json \
  --gold-file dataset/processed/gold.json \
  --match --rouge --bertScore
```

### Report Generation

```bash
# Generate report from evaluation results (output of eval.py)
python scripts/generate_report.py results/eval_results.json --format html
python scripts/generate_report.py results/eval_results.json --format markdown
python scripts/generate_report.py results/eval_results.json --format both
```

## Testing

```bash
# Run all tests
pytest tests/

# Specific test file
pytest tests/test_evaluator_base.py

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Customization

### Adding a New LLM Provider

1. Create a provider class in `src/core/agent.py` extending `BaseAgent`
2. Register it in the `create_agent()` factory
3. Add a YAML config in `configs/agents/`

### Adding a New Evaluator

1. Create an evaluator class in `src/core/evaluator.py` extending `BaseEvaluator`
2. Register it in the `create_evaluator()` factory
3. Reference the scoring method in task configs

### Adding a New Task

1. Prepare a JSONL dataset
2. Create a YAML task config in `configs/tasks/`
3. Create a prompt template in `prompts/templates/`

## Dependencies

Core: `openai`, `anthropic`, `pydantic`, `pyyaml`, `pandas`, `numpy`, `scipy`, `loguru`, `tqdm`, `httpx`, `aiohttp`, `bert-score`, `rouge-score`

Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `black`, `isort`, `ruff`, `mypy`

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- [Vercel Labs benchmark-agents / PluginEval](https://www.skills.sh/vercel-labs/vercel-plugin/benchmark-agents) — Composite weighted scoring, LLM judge evaluation, anti-pattern penalties, and coverage reporting methodology ([MIT License](https://github.com/vercel-labs/skills))
- [AgentBench](https://github.com/OpenGVLab/AgentBench) — Evaluation framework design inspiration
- [Zemax OpticStudio](https://www.zemax.com/) — Optical design software
