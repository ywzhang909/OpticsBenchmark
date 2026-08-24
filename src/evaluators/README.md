# src/evaluators — Metric Evaluators

## Purpose

Evaluation entry layer. The factory reads `eval_metrics` from evaluation YAML configs and instantiates the requested evaluators.

## Files

| File | Description |
|------|-------------|
| `factory.py` | `EVALUATOR_MAP`, `create_evaluator(config)` |
| `base.py` | `BaseEvaluator` abstract base class |
| `exact_match_evaluator.py` | Normalized string equality per JSON field |
| `rouge_evaluator.py` | ROUGE-1/2/L precision/recall/F1 |
| `bert_score_evaluator.py` | BERTScore via transformer embeddings |
| `citation_evaluator.py` | Citation precision/recall/F1 |
| `helpers.py` | Shared evaluation helpers |

## Usage

```python
from src.evaluators import create_evaluator
evaluators = create_evaluator({"eval_metrics": {"exact_match": {...}}})
```

Scoring primitives live in `scorer/`; configs in `configs/evaluations/*.yaml`.
