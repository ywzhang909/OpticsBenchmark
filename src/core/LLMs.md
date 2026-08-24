# CORE MODULE

**Path:** `src/core/` — Runner, Config, LLM Judge abstractions.

## OVERVIEW
Pipeline orchestration module: async parallel runner, task configuration, and LLM-as-judge evaluator.

## FILES
| File | Lines | Role |
|------|-------|------|
| `config.py` | 44 | `TaskConfig` dataclass (YAML-loaded) |
| `runner.py` | 259 | `AgentRunner`, semaphore-based concurrency, JSONL result persistence |
| `llm_judge.py` | — | `LLMJudge`, `JudgePromptBuilder`, `Rubric`, `DEFAULT_RUBRICS` — LLM-as-judge evaluator |
| `llm_runner.py` | — | `LLMPredRunner` — LLM prediction runner |
| `__init__.py` | — | Re-exports all public symbols |

Note: Evaluators have been refactored into `src/evaluators/` subpackage.
Note: `agent.py` has been removed. LLM providers are now in `src/llm/models/`.

## KEY PATTERNS
- **Dataclass configs**: `RunnerConfig`, `TaskConfig`, `TaskInstance` — all `@dataclass` with `from_yaml()` classmethods.
- **Async+Semaphore**: Concurrent evaluation via `asyncio.Semaphore(max_concurrency)`.
- **Composition**: Runner owns Agent → `setup()` → `run()` → `teardown()` lifecycle.
- **No mocking in tests**: Tests use concrete fixture data, no `unittest.mock`.

## TODO
- `runner.py` needs migration from deleted `agent.py` to `src.llm` abstraction.
