# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-23
**Commit:** 24f7c61
**Branch:** main

## OVERVIEW
Optis Benchmark — open-source evaluation framework for LLM-based agents in optical design tasks. Python 3.10+, uv-managed, async-parallel architecture.

## STRUCTURE
```
OpticsBenchmark/
├── configs/          # YAML configs: llm/, evaluations/, system/
├── docs/             # Chinese-language design docs
├── prompts/          # LLM prompt templates: system/, templates/, task-specific dirs
├── src/              # Core package: llm_pred.py, eval.py, core/, evaluators/, algorithm/, llm/, environments/, utils/
├── dataset/          # Evaluation datasets (paper_info_extract, info_extraction)
├── tests/            # pytest suite (20 test files, async-first)
├── utils/            # Standalone utility scripts
└── pyproject.toml    # Single-source config (build, lint, format, test, coverage)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CLI entry point | `src/llm_pred.py` | argparse → `optis` command |
| Evaluators & scorers | `src/evaluators/` | 4 evaluator types + 5 scorers |
| LLM abstraction | `src/llm/` | 10 model classes, 7 provider classes |
| Algorithm modules | `src/algorithm/` | 8 pure-math evaluation modules |
| Async runner | `src/core/runner.py` | Semaphore-based concurrency (259 lines) |
| Zemax integration | `src/environments/zos_env.py` | ZOS-API stub (PythonNET) |
| Local env | `src/environments/base_env.py` | Shell execution sandbox |
| Config loading | `src/utils/parser.py` | JSONL/YAML/env-var expansion |
| Test fixtures | `tests/conftest.py` | 15 fixtures, async event loop |
| System config | `configs/system/template.yaml` | Parallel, sandbox, rate-limit, security |

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
- `pytest.ini` duplicates `[tool.pytest.ini_options]` in pyproject.toml.
- Dependencies in 3 sources (pyproject.toml, requirements.txt, environment.yml) with drift.
- ZOS-API integration is stub-only (ZOSAPIEnvironment methods return placeholder data).
- `configs/system/template.yaml` is a minimal template with most settings commented out.
- `pyproject.toml` references `src.main:main` but the actual entry point is `src/llm_pred.py`.
- `src/core/runner.py` has TODO markers for migration to `src.llm` abstraction.

## COMMANDS
```bash
uv sync                          # Install dependencies
uv run pytest tests/             # Run tests (verbose, short traceback)
uv run pytest --cov=src          # With coverage
uv run ruff check .              # Lint
uv run mypy src/                 # Type check (lenient mode)
uv run python src/llm_pred.py -a configs/llm/GPT_OpenAI.yaml -t paper_info_extract  # Run eval
```
