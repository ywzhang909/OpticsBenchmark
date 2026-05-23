# CORE MODULE

**Path:** `src/core/` — Agent, Evaluator, Runner abstractions.

## OVERVIEW
Three-file module providing the entire evaluation pipeline: LLM agent interface (7 providers), multi-metric evaluators (6 types), and async parallel runner.

## FILES
| File | Lines | Role |
|------|-------|------|
| `agent.py` | 955 | `BaseAgent` + `OpenAIAgent`, `AnthropicAgent`, factory. 7 providers in one file. |
| `evaluator.py` | 1474 | `BaseEvaluator` + 6 evaluators (Metric, ExactMatch, PartialMatch, Summarization, Citation, ResultAnalyzer). Scoring + aggregation. |
| `runner.py` | 413 | `EvaluationRunner`, semaphore-based concurrency, JSONL result persistence, progress bar. |
| `__init__.py` | 60 | Re-exports all public symbols. |

## KEY PATTERNS
- **Factory functions**: `create_agent()`, `create_evaluator()` — config-driven instantiation.
- **Dataclass configs**: `AgentConfig`, `RunnerConfig`, `TaskConfig`, `TaskInstance` — all `@dataclass` with `from_yaml()` classmethods.
- **Async+Semaphore**: Concurrent evaluation via `asyncio.Semaphore(max_concurrency)`.
- **Composition**: Runner owns Agent + Evaluator → `setup()` → `run()` → `teardown()` lifecycle.
- **No mocking in tests**: Tests use concrete fixture data, no `unittest.mock`.

## CONVENTIONS (module-specific)
- Agent classes must implement `chat(messages)` returning `AgentResponse`.
- New providers: subclass `BaseAgent`, register in `create_agent()` factory, add YAML in `configs/agents/`.
- New evaluators: subclass `BaseEvaluator`, register in `create_evaluator()` factory, add `scoring_method` to task YAML.
- `evaluator.py` is too large (1474 lines). Consider splitting: evaluator types → separate files, analyzers → separate module.
