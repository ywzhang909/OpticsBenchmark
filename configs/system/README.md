# configs/system — System Configs

## Purpose

Global runtime settings. Currently only the `logging` section is consumed (see `load_system_config()` in `src/eval.py`).

## Files

| File | Description |
|------|-------------|
| `template.yaml` | Logging configuration template |

## Usage

```bash
uv run python src/eval.py --system-config configs/system/template.yaml ...
```
