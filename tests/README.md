# tests — Test Suite

## Purpose

Unit and integration tests for the benchmark pipeline.

## Running

```bash
# Full suite (GPU/NLI-dependent files excluded)
uv run python -m pytest tests/ -q \
  --ignore=tests/test_bert_score_eval.py \
  --ignore=tests/test_sentence_similarity.py \
  --ignore=tests/test_model_registry.py \
  -k "not paper_retrieval"
```

## Layout

| Area | Files |
|------|-------|
| Fixtures & stubs | `conftest.py`, `stubs.py` (stand-ins for removed/heavy modules) |
| Evaluators | `test_evaluator_base.py`, `test_exact_match*`, `test_rouge_*`, `test_citation_*`, `test_bert_score_eval.py` |
| Algorithms | `test_hungarian_algorithm.py`, `test_bleu_eval_utils.py`, `test_cider_eval_utils.py`, `test_perplexity_eval_utils.py` |
| Core & LLM | `test_llm_judge.py`, `test_model_registry.py`, `test_composite_scorer.py` |
| Integration | `test_integration.py` |

Test files mirror `snake_case` naming: `test_<module_under_test>.py` (STANDARDS.md §7).
