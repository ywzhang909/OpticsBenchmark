# TESTS

**Path:** `tests/` — pytest suite (20 test files, flat structure).

## OVERVIEW
Async-first test suite. All evaluator tests are async, using centralized fixtures from `conftest.py`. No mocking library — pure dataclass fixtures.

## STRUCTURE
| File | Tests | Coverage |
|------|-------|----------|
| `conftest.py` | 15 fixtures | Shared evaluator configs + pre-built instances + mock results |
| `stubs.py` | — | Test stubs and helpers |
| `test_evaluator_base.py` | BaseEvaluator, ExactMatchEvaluator | Base evaluator tests |
| `test_citation_evaluator.py` | CitationEvaluator | Unit + integration |
| `test_rouge_eval_utils.py` | ROUGE evaluation utilities | Algorithm tests |
| `test_rouge_scorer.py` | ROUGEScorer | Scorer tests |
| `test_em_eval_utils.py` | normalize_text, compute_exact_match | String utils |
| `test_bleu_eval_utils.py` | BLEU evaluation | Algorithm tests |
| `test_cider_eval_utils.py` | CIDEr evaluation | Algorithm tests |
| `test_meteor_eval_utils.py` | METEOR evaluation | Algorithm tests |
| `test_perplexity_eval_utils.py` | Perplexity evaluation | Algorithm tests |
| `test_edit_distance_utils.py` | Edit distance, WER | Algorithm tests |
| `test_jaccard_similarity_utils.py` | Jaccard, Dice | Algorithm tests |
| `test_hungarian_algorithm.py` | Hungarian matching | Algorithm tests |
| `test_sentence_similarity.py` | Sentence similarity | Algorithm tests |
| `test_bert_score_eval.py` | BERTScore evaluator | Network-dependent |
| `test_citation_eval_utils.py` | Citation F1 | Algorithm tests |
| `test_composite_scorer.py` | Composite scorer | Scoring tests |
| `test_llm_judge.py` | LLM judge | Integration tests |
| `test_model_registry.py` | Model registry | Registry tests |
| `test_quick_llm_selector.py` | QuickLLMSelector | Unit + CLI + async |
| `test_integration.py` | End-to-end pipeline | Factory, lens design, summarization, retrieval, model comparison |

## CONVENTIONS
- **Async by default**: `@pytest.mark.asyncio` + `async def test_*()` for evaluators.
- **asyncio_mode = auto**: `pytest.ini` configures auto async discovery.
- **Event loop**: Custom `event_loop` fixture (overrides pytest-asyncio default, creates new loop per test).
- **No parametrize**: Variants handled via separate test methods or inline loops.
- **No mocking**: Tests use `MockTask` dataclass, `tmp_path` for filesystem, `pytest.approx()` for floats.

## GAPS
- Zero tests for `src/core/agent.py` (1306 lines, 7 providers — untested).
- Zero tests for `src/core/runner.py` (259 lines — untested).
- Zero tests for `src/environments/` (environments module untested).
- Zero tests for `src/utils/parser.py` (untested).
- Zero tests for `src/utils/logger.py` (untested).
