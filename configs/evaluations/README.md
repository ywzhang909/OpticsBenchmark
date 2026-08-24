# configs/evaluations — Evaluation Configs

## Purpose

YAML configs selecting which metric evaluators run on agent outputs (`eval_metrics` key). See parent [README](../README.md).

## Files

| File | Description |
|------|-------------|
| `template.yaml` | Annotated template with all evaluator options |
| `paper_info_extract.yaml` | Task config: exact_match + rouge + bert_score + citation |

## Usage

```bash
uv run python src/eval.py --eval-config configs/evaluations/paper_info_extract.yaml ...
```
