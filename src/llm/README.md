# src/llm — LLM Abstraction Layer

## Purpose

Provider/model abstraction for all LLM backends. `create_llm()` and `create_provider()` resolve YAML config keys to concrete classes via lazy entry-point imports.

## Files

| File | Description |
|------|-------------|
| `base.py` | `BaseLLM` abstract base class, `build_response_format()` helper |
| `__init__.py` | Factory registries (`_PROVIDER_MAP`, `_LLM_MAP`), `create_provider()`, `create_llm()`, `_lazy_import()` |

## Usage

```python
from src.llm import create_llm

llm = create_llm({"type": "gpt", "name": "gpt-4o"})
```

Model implementations live in `models/`, provider clients in `providers/`. Configs: `configs/llm/*.yaml`.
