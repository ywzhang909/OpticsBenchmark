# src/core — Pipeline Orchestration

## Purpose

Core runtime components connecting configuration, LLM prediction, and evaluation.

## Files

| File | Description |
|------|-------------|
| `config.py` | `ConfigParser` — YAML loading with `${ENV_VAR}` expansion |
| `llm_runner.py` | `LLMPredRunner` / `LLMRunnerConfig` — concurrent LLM prediction over a dataset |
| `runner.py` | Generic async runner with concurrency semaphore |
| `llm_judge.py` | LLM-as-judge scoring configuration and logic |

## Usage

```python
from src.core.config import ConfigParser
cfg = ConfigParser.load_config("configs/llm/qwen_openai.yaml")
```
