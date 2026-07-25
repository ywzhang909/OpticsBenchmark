# CORE MODULE

**Path:** `src/core/` — Agent, Runner, Config, LLM Judge abstractions.

## OVERVIEW
Pipeline orchestration module: LLM agent interface (7 providers), async parallel runner, task configuration, and LLM-as-judge evaluator.

## FILES
| File | Lines | Role |
|------|-------|------|
| `agent.py` | 1306 | `BaseAgent` + `OpenAIAgent`, `AnthropicAgent`, `GoogleAgent`, `GroqAgent`, `OllamaAgent`, `BedrockAgent`, `TogetherAIAgent`, factory. 7 providers in one file. |
| `config.py` | 44 | `TaskConfig` dataclass (YAML-loaded) |
| `runner.py` | 259 | `AgentRunner`, semaphore-based concurrency, JSONL result persistence |
| `llm_judge.py` | — | `LLMJudge`, `JudgePromptBuilder`, `Rubric`, `DEFAULT_RUBRICS` — LLM-as-judge evaluator |
| `llm_runner.py` | — | `LLMPredRunner` — LLM prediction runner |
| `__init__.py` | — | Re-exports all public symbols |

Note: Evaluators have been refactored into `src/evaluators/` subpackage.
`LLMJudge` is consumed by `src/evaluators/qualitative_evaluator.py`.

## KEY PATTERNS
- **Factory functions**: `create_agent()` — config-driven instantiation.
- **Dataclass configs**: `AgentConfig`, `RunnerConfig`, `TaskConfig`, `TaskInstance` — all `@dataclass` with `from_yaml()` classmethods.
- **Async+Semaphore**: Concurrent evaluation via `asyncio.Semaphore(max_concurrency)`.
- **Composition**: Runner owns Agent → `setup()` → `run()` → `teardown()` lifecycle.
- **No mocking in tests**: Tests use concrete fixture data, no `unittest.mock`.

## CONVENTIONS (module-specific)
- Agent classes must implement `chat(messages)` returning `AgentOutput`.
- New providers: subclass `BaseAgent`, register in `create_agent()` factory, add YAML in `configs/agents/`.
