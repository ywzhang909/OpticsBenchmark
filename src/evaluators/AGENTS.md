# EVALUATORS MODULE

**Path:** `src/evaluators/` — 5 evaluator types + scorer subpackage.

## OVERVIEW
Config-driven metric evaluators extending `BaseEvaluator` ABC. Each implements `evaluate()` → `EvaluationResult` and `aggregate()` → `AggregatedResults`. Loaded by `create_evaluator()` factory from YAML `eval_metrics` config.

## FILES
| File | Lines | Role |
|------|-------|------|
| `base.py` | 75 | `BaseEvaluator` ABC |
| `factory.py` | 86 | `create_evaluator()`, `sort_evaluators_by_priority()`, GPU classification |
| `helpers.py` | 61 | JSON parse, dict normalize, `sentenceMatch()` (Hungarian + BAAI/bge-m3) |
| `exact_match_evaluator.py` | 75 | Dict field-by-field normalized string equality |
| `rouge_evaluator.py` | 110 | ROUGE-1/2/L P/R/F1 + Hungarian sentence alignment |
| `bert_score_evaluator.py` | 119 | BERTScore P/R/F1 via transformer embeddings |
| `citation_evaluator.py` | 62 | Citation precision/recall/F1 via NLI model |
| `qualitative_evaluator.py` | 240 | LLM-as-judge with configurable rubrics, offline + online modes |
| `scorer/` | 6 files | Thin wrappers → `algorithm/*` compute functions |

## EVALUATOR TYPES
| Type | Class | GPU | Config Key |
|------|-------|-----|------------|
| exact_match | `ExactMatchEvaluator` | CPU | `info_names` (JSON fields) |
| rouge | `RougeEvaluator` | Light | `info_names`, `metrics` (rouge1/2/L) |
| bert_score | `BertScoreEvaluator` | Intensive | `info_names`, `model_name`, `hungarian_match.model` |
| citation | `CitationEvaluator` | Intensive | — (uses default NLI model) |
| qualitative | `QualitativeEvaluator` | CPU | `dimensions` (custom rubrics), `judge_config` (LLM provider) |

## KEY PATTERNS
- **Factory + priority**: YAML config maps type → class via `EVALUATOR_MAP`. Priority sorting: lower = first (citation:1 → exact_match:4). GPU-heavy evaluators run first to maximize utilization.
- **Hungarian pipeline**: `sentenceMatch()` → `SentenceEmbedder.encode()` → similarity matrix → `hungarian_match()` → aligned pairs → scorer. Shared across rouge + bert_score evaluators.
- **Lifecycle**: `setup()` (load GPU models) → `evaluate()` (per task) → `teardown()` (release GPU memory). Wrapped in try/finally in `src/eval.py`.
- **Details field**: `EvaluationResult.details` stores non-numeric data (judge justifications, error messages) alongside float `metrics`.

## QUALITATIVE EVALUATOR
- Wraps `LLMJudge` from `src/core/llm_judge.py`
- Offline mode: builds judge prompts without LLM callable
- Online mode: creates agent via `create_agent()` from `judge_config`
- Rubrics: configurable via `dimensions` array, falls back to `DEFAULT_RUBRICS`
- Output: `metrics` = `{"qualitative_{dim}": score}`, `details` = `{"justifications": {...}}`
