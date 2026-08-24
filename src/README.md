# Source Code

**Path:** `src/` — Core Python package for Optis Benchmark.

Two-phase evaluation pipeline: **Phase 1** (`llm_pred.py`) generates agent outputs, **Phase 2** (`eval.py`) evaluates them. LLM abstraction layer (10 model classes, 8 providers), metric evaluators (4 types), pure-math algorithms (8 modules), execution sandboxes, and CLI utilities are organized into 8 subpackages.

---

## Directory Structure

```
src/
├── __init__.py               # Package root; re-exports core symbols; __version__ = "1.0.0"
├── llm_pred.py               # Phase 1 CLI: run agents to generate outputs
├── eval.py                   # Phase 2 CLI: evaluate agent outputs against gold answers
│
├── core/                     # Pipeline orchestration
│   ├── __init__.py
│   ├── config.py             # TaskConfig dataclass (YAML-loaded)
│   ├── llm_judge.py          # LLM-as-judge evaluator (PluginEval Layer 2)
│   ├── llm_runner.py         # LLMPredRunner for prediction
│   └── runner.py             # Async AgentRunner with semaphore concurrency (259 lines)
│
├── evaluators/               # Metric evaluators
│   ├── __init__.py
│   ├── base.py               # BaseEvaluator ABC
│   ├── factory.py            # create_evaluator() — config-driven factory
│   ├── helpers.py            # JSON parsing, sentence matching, dict normalization
│   ├── exact_match_evaluator.py  # Dict field-by-field exact match
│   ├── rouge_evaluator.py        # ROUGE-1/2/L with Hungarian sentence alignment
│   ├── bert_score_evaluator.py   # BERTScore P/R/F1 with Hungarian alignment
│   ├── citation_evaluator.py     # Citation precision/recall/F1 + composite score
│   └── scorer/                   # Thin wrappers → algorithm/* functions
│       ├── __init__.py
│       ├── exact_match_scorer.py
│       ├── rouge_scorer.py
│       ├── bert_score_scorer.py
│       ├── bleu_scorer.py
│       └── citation_scorer.py
│
├── algorithm/                # Pure-math evaluation algorithms (8 modules)
│   ├── __init__.py
│   ├── em_eval_utils.py          # Text normalization + exact match
│   ├── rouge_eval_utils.py       # ROUGE-1/2/L via rouge_score library
│   ├── bert_score_eval_utils.py   # BERTScore via bert-score library
│   ├── bleu_eval_utils.py        # BLEU with smoothing (pure Python)
│   ├── citation_eval_utils.py    # AutoAIS-based citation F1
│   ├── hungarian_algorithm_utils.py  # Optimal assignment via scipy
│   ├── sentence_similarity_utils.py  # Transformer embedder (BAAI/bge-m3)
│   └── model_registry.py            # Model registry for evaluation
│
├── llm/                      # LLM abstraction layer (10 models, 8 providers)
│   ├── __init__.py
│   ├── base.py               # BaseLLM ABC
│   ├── models/               # Model-specific LLM implementations
│   │   ├── claude_llm.py      # Anthropic Claude (official SDK)
│   │   ├── deepseek_llm.py    # DeepSeek (OpenAI-compatible)
│   │   ├── gemini_llm.py      # Google Gemini (Interactions API)
│   │   ├── glm_llm.py         # Zhipu GLM (OpenAI SDK)
│   │   ├── gpt_llm.py         # OpenAI GPT (Chat Completions + Responses)
│   │   ├── kimi_llm.py        # Moonshot Kimi (OpenAI-compatible)
│   │   ├── llama_llm.py       # Meta Llama (Together AI + OpenAI SDK)
│   │   ├── mistral_llm.py     # Mistral (official mistralai SDK)
│   │   ├── ollama_llm.py      # Ollama (local)
│   │   └── qwen_llm.py        # Alibaba Qwen (OpenAI-compatible)
│   └── providers/            # Provider-specific API clients
│       ├── anthropic_provider.py
│       ├── bedrock_provider.py
│       ├── google_provider.py
│       ├── mistral_provider.py
│       ├── ollama_provider.py
│       ├── openai_provider.py
│       └── together_ai_provider.py
│
├── environments/              # Execution sandboxes
│   ├── __init__.py
│   ├── base_env.py           # BaseEnvironment ABC + LocalEnvironment (subprocess)
│   └── zos_env.py            # ZOSAPIEnvironment — Zemax OpticStudio stub (PythonNET)
│
├── module/                    # Shared data structures
│   ├── __init__.py
│   └── result.py              # EvaluationResult + AggregatedResults dataclasses
│
└── utils/                     # Infrastructure utilities
    ├── __init__.py
    ├── logger.py              # loguru-based singleton logger
    ├── parser.py              # JSONL/YAML/Config/Optical-data parsers
    ├── generate_report.py     # HTML/Markdown report generator
    └── general.py             # Standalone utilities (_dict_to_response_format)
```

---

## Two-Phase Pipeline

```
┌─────────────────────────────────────────────────────┐
│                   Phase 1: Generate                  │
│                   src/llm_pred.py                    │
│                                                     │
│  LLM Config   ──► AgentRunner                       │
│  Task Config   ──► TaskConfig       ──►  run_agent()│
│  Dataset JSONL ──► load_tasks()     ──►  AgentOutput │
│                                                     │
│  Output: agent_outputs/{agent}_{task}.json          │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                   Phase 2: Evaluate                  │
│                   src/eval.py                        │
│                                                     │
│  Agent Outputs  ──► load_agent_outputs()            │
│  Gold Answers    ──► _load_gold_data()              │
│  Eval Config     ──► create_evaluator()             │
│                       ├── ExactMatchEvaluator       │
│                       ├── RougeEvaluator            │
│                       ├── BertScoreEvaluator        │
│                       └── CitationEvaluator         │
│                                                     │
│  Output: evaluation_results/{eval}_{task}.json      │
└─────────────────────────────────────────────────────┘
```

---

## Core Subpackages

### `core/` — Pipeline Orchestration

| File | Key Symbols | Description |
|------|-------------|-------------|
| `config.py` | `TaskConfig` | YAML-loaded task configuration dataclass |
| `runner.py` | `AgentRunner`, `RunnerConfig`, `TaskInstance` | Async orchestrator with semaphore-based concurrency (259 lines) |
| `llm_judge.py` | `LLMJudge`, `JudgePromptBuilder`, `Rubric`, `DEFAULT_RUBRICS` | LLM-as-judge evaluator with structured rubrics |
| `llm_runner.py` | `LLMPredRunner` | LLM prediction runner |

**Note**: `agent.py` has been removed. LLM providers are now in `src/llm/models/`.

---

### `environments/` — Execution Sandboxes

| File | Key Symbols | Description |
|------|-------------|-------------|
| `base_env.py` | `BaseEnvironment` (ABC), `LocalEnvironment`, `EnvironmentConfig`, `EnvironmentResponse`, `create_environment()` | Abstract sandbox + subprocess-based local execution |
| `zos_env.py` | `ZOSAPIEnvironment`, `ZOSConnectionConfig` | Zemax OpticStudio ZOS-API stub via PythonNET |

`LocalEnvironment` executes commands via `subprocess.run()`. `ZOSAPIEnvironment` dispatches `python:`, `zemax:`, and shell commands; all high-level methods (`analyze_mtf()`, `analyze_spot()`, `optimize()`) return placeholder data — real integration requires PythonNET + Zemax.

---

### `evaluators/` — Metric Evaluators

| Evaluator | Metric | Method |
|-----------|--------|--------|
| `ExactMatchEvaluator` | `exact_match_avg` | Normalized string equality per JSON field |
| `RougeEvaluator` | ROUGE-1/2/L (P/R/F1) | `rouge_score` library + Hungarian sentence alignment |
| `BertScoreEvaluator` | BERTScore (P/R/F1) | `bert-score` library + transformer embeddings |
| `CitationEvaluator` | Precision, Recall, F1, citation_accuracy, composite | Paper ID matching + Jaccard title similarity |

**Architecture**:
```
evaluator/*.py        scorer/*.py            algorithm/*.py
────────────────      ───────────────        ───────────────────
ExactMatchEval  ──►   ExactMatchScorer  ──►  compute_exact_match()
RougeEval       ──►   ROUGEScorer       ──►  compute_rouge()
BertScoreEval   ──►   BERTScoreScorer   ──►  compute_bert_score()
CitationEval    ──►   CitationScorer    ──►  compute_citation_f1()
                    BLEUScorer         ──►  compute_bleu()
```

**Hungarian matching pipeline** (for structured ROUGE/BERTScore):
```
predicted_sentences ──┐
                       ├── SentenceEmbedder (BAAI/bge-m3) ──► similarity_matrix
gold_sentences     ───┘                                            │
                                                          hungarian_match()
                                                             (scipy)
                                                                │
                                                    aligned pairs ──► scorer
```

---

### `algorithm/` — Pure Evaluation Algorithms

| Module | Algorithm | Dependency |
|--------|-----------|------------|
| `em_eval_utils` | Text normalization + binary exact match | none |
| `rouge_eval_utils` | ROUGE-1/2/L | `rouge_score`, `nltk` |
| `bert_score_eval_utils` | BERTScore P/R/F1 | `bert-score` |
| `bleu_eval_utils` | BLEU with Chen & Cherry smoothing | none (pure Python) |
| `citation_eval_utils` | AutoAIS-based citation F1 | none |
| `hungarian_algorithm_utils` | `hungarian_match()` | `scipy` |
| `sentence_similarity_utils` | `SentenceEmbedder`, `compute_similarity_matrix()` | `transformers`, `torch` |
| `model_registry` | Model registry for evaluation | none |

---

### `llm/` — LLM Abstraction Layer

Model-specific implementations, each wrapping an official or compatible SDK.

| Module | Key Symbols | Description |
|--------|-------------|-------------|
| `base.py` | `BaseLLM` (ABC), `Message`, `LLMOutput`, `LLMConfig` | Base LLM interface, data models |
| `models/claude_llm.py` | `ClaudeLLM` | Anthropic Claude via `anthropic` SDK |
| `models/deepseek_llm.py` | `DeepSeekLLM` | DeepSeek via OpenAI-compatible API |
| `models/gemini_llm.py` | `GeminiLLM` | Google Gemini via Interactions API |
| `models/glm_llm.py` | `GlmLLM` | Zhipu GLM via OpenAI SDK |
| `models/gpt_llm.py` | `GPTLLM` | OpenAI GPT (Chat Completions + Responses) |
| `models/kimi_llm.py` | `KimiLLM` | Moonshot Kimi via OpenAI-compatible API |
| `models/llama_llm.py` | `LlamaLLM` | Meta Llama via Together AI or OpenAI SDK |
| `models/mistral_llm.py` | `MistralLLM` | Mistral via official `mistralai` SDK |
| `models/ollama_llm.py` | `OllamaLLM` | Ollama local inference via httpx |
| `models/qwen_llm.py` | `QwenLLM` | Alibaba Qwen via OpenAI-compatible API |
| `providers/anthropic_provider.py` | `AnthropicProvider` | Wraps `anthropic.AsyncAnthropic` |
| `providers/bedrock_provider.py` | `BedrockProvider` | Wraps `boto3` Bedrock Runtime |
| `providers/google_provider.py` | `GoogleProvider` | Wraps `google.genai.Client` |
| `providers/mistral_provider.py` | `MistralProvider` | Wraps `mistralai.Mistral` |
| `providers/ollama_provider.py` | `OllamaProvider` | Wraps httpx for Ollama API |
| `providers/openai_provider.py` | `OpenAIProvider` | Wraps `openai.AsyncOpenAI` |
| `providers/together_ai_provider.py` | `TogetherAIProvider` | Wraps httpx for Together AI API |

---

### `module/` — Shared Data Structures

```python
@dataclass
class EvaluationResult:
    task_id: str
    metrics: dict           # {"rouge1_f1": 0.85, "exact_match": 1.0, ...}
    execution_time: float

@dataclass
class AggregatedResults:
    total_tasks: int
    metrics_summary: dict   # {"rouge1_f1": {"mean": 0.82, "std": 0.12, ...}}
    avg_execution_time: float
    per_task_results: list[EvaluationResult]
```

---

### `utils/` — Infrastructure

| Module | Key Functions |
|--------|--------------|
| `logger.py` | `setup_logger()`, `get_logger()` — loguru singleton with console + file + rotation |
| `parser.py` | `JSONLParser` (JSONL read/write), `YAMLParser` (YAML read/write), `ConfigParser` (`${ENV_VAR}` expansion), `ResultsParser` (load/format), `OpticalDataParser` (Zemax .zmx, MTF, spot data stub) |
| `generate_report.py` | `load_results()`, `generate_html_report()`, `generate_markdown_report()` |
| `general.py` | `_dict_to_response_format()` — JSON schema to OpenAI response_format conversion |

---

---

## CLI Entry Points

```bash
# Phase 1: Run agent on tasks
uv run python src/llm_pred.py \
  -a configs/llm/GPT_OpenAI.yaml \
  -t paper_info_extract

# Phase 2: Evaluate agent outputs
uv run python src/eval.py \
  -i results/agent_outputs.jsonl \
  -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \
  -e configs/evaluations/paper_info_extract.yaml

# Generate HTML report from results
uv run python src/utils/generate_report.py results/eval_results.json --format html
```

---

## Module Dependency Graph

```
llm_pred.py ──► core/runner ──► core/config
                                │
eval.py  ──► core/runner ──► evaluators/factory ──► evaluators/* ──► scorer/* ──► algorithm/*
               │                                              │
               └── utils/ ──► logger.py, parser.py            └── helpers ──► algorithm/
                                                                   (sentence embedding, hungarian)

llm_pred.py ──► llm/* ──► providers/*
environments/base_env.py  ◄── environments/zos_env.py
module/result.py          ◄── used by evaluators + eval.py
```

---

## Known Issues

- `ZOSAPIEnvironment` high-level methods return placeholder data; real integration requires PythonNET + Zemax OpticStudio.
- `utils/general.py` contains standalone utility functions.
- `src/core/runner.py` has TODO markers for migration to `src.llm` abstraction.
- `pyproject.toml` references `src.main:main` but the actual entry point is `src/llm_pred.py`.
