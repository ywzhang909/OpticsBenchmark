# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-23
**Commit:** a02e906
**Branch:** main

## OVERVIEW
OptiS Benchmark — open-source evaluation framework for LLM-based agents in optical design tasks. Python 3.10+, uv-managed, async-parallel architecture.

## STRUCTURE
```
OpticsBenchmark/
├── configs/          # YAML configs: agents/, tasks/, system.yaml
├── docs/             # Chinese-language design docs
├── prompts/          # LLM prompt templates: system/, templates/
├── scripts/          # Standalone eval scripts + utils (EM, ROUGE, BERTScore)
├── src/              # Core package: main.py, core/, environments/, utils/, tools/
├── tests/            # pytest suite (10 files, async-first)
├── website/          # Leaderboard page
└── pyproject.toml    # Single-source config (build, lint, format, test, coverage)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CLI entry point | `src/main.py` | argparse → `optis` command |
| Agent implementations | `src/core/agent.py` | 955 lines, 7 LLM providers |
| Evaluators & scorers | `src/core/evaluator.py` | 1474 lines, 6 evaluator types |
| Async runner | `src/core/runner.py` | Semaphore-based concurrency |
| Zemax integration | `src/environments/zos_env.py` | ZOS-API stub (PythonNET) |
| Local env | `src/environments/base_env.py` | Shell execution sandbox |
| Config loading | `src/utils/parser.py` | JSONL/YAML/env-var expansion |
| Interactive LLM test | `src/tools/quick_llm_selector.py` | CLI provider comparison |
| Test fixtures | `tests/conftest.py` | 15 fixtures, async event loop |
| System config | `configs/system.yaml` | Parallel, sandbox, rate-limit, security |

## CONVENTIONS
- **Package manager**: `uv` (recommended), `pip` fallback. `uv sync` to install.
- **Line length**: 100 chars (Black, Ruff, isort agreement).
- **Imports**: stdlib → third-party → `src` (isort profile=black).
- **Linting**: Ruff rules E, W, F, I, B, C4, UP. No naming/pydocstyle checks.
- **Types**: Optional (mypy warn_return_any=true, disallow_untyped_defs=false).
- **Async**: All evaluator methods + runner are async. `asyncio_mode = auto` in pytest.
- **Test naming**: `tests/test_*.py`, `class Test*`, `def test_*`.
- **No mocking**: Tests use dataclass fixtures, not `unittest.mock`.
- **Configuration**: YAML, snake_case fields, `${ENV_VAR}` for secrets.

## KNOWN ISSUES
- No CI/CD pipeline (`.github/workflows/` absent despite tooling being configured).
- `uv.lock` in `.gitignore` — non-deterministic installs.
- `pytest.ini` duplicates `[tool.pytest.ini_options]` in pyproject.toml.
- Dependencies in 3 sources (pyproject.toml, requirements.txt, environment.yml) with drift.
- ZOS-API integration is stub-only (ZOSAPIEnvironment methods return placeholder data).
- `configs/system.yaml` references Docker but no Dockerfile exists.
- `scripts/utils/` eval utilities are outside installable package.

## COMMANDS
```bash
uv sync                          # Install dependencies
uv run pytest tests/             # Run tests (verbose, short traceback)
uv run pytest --cov=src          # With coverage
uv run ruff check .              # Lint
uv run mypy src/                 # Type check (lenient mode)
uv run python src/main.py -a configs/agents/gpt-4.yaml -t lens_design  # Run eval
```
