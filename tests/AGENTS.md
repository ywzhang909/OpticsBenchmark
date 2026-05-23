# TESTS

**Path:** `tests/` — pytest suite (10 files, flat structure).

## OVERVIEW
Async-first test suite. All evaluator tests are async, using centralized fixtures from `conftest.py`. No mocking library — pure dataclass fixtures.

## STRUCTURE
| File | Tests | Coverage |
|------|-------|----------|
| `conftest.py` | 15 fixtures | Shared evaluator configs + pre-built instances + mock results |
| `test_evaluator_base.py` | Metric, ExactMatch, PartialMatch evaluators | 3 eval types |
| `test_citation_evaluator.py` | CitationEvaluator | Unit + integration classes |
| `test_rouge_scorer.py` | ROGUEScorer + SummarizationEvaluator | Scorer + eval |
| `test_result_analyzer.py` | ResultAnalyzer, ErrorAnalyzer, CompositeScore, EvaluationQA | 4 utility classes |
| `test_report_generator.py` | ReportGenerator | HTML + Markdown output |
| `test_em_eval_utils.py` | normalize_text, record_doi_punctuation | String utils |
| `test_quick_llm_selector.py` | QuickLLMSelector | Unit + CLI + async |
| `test_integration.py` | End-to-end pipeline | Factory, lens design, summarization, retrieval, model comparison |

## CONVENTIONS
- **Async by default**: `@pytest.mark.asyncio` + `async def test_*()` for evaluators.
- **asyncio_mode = auto**: `pytest.ini` configures auto async discovery.
- **Event loop**: Custom `event_loop` fixture (overrides pytest-asyncio default, creates new loop per test).
- **No parametrize**: Variants handled via separate test methods or inline loops.
- **No mocking**: Tests use `MockTask` dataclass, `tmp_path` for filesystem, `pytest.approx()` for floats.
- **Coverage targets**: `src/core/` ≥ 80%, `src/environments/` ≥ 70%, `src/utils/` ≥ 80%.

## GAPS
- Zero tests for `src/core/agent.py` (955 lines, 7 providers — untested).
- Zero tests for `src/core/runner.py` (413 lines — untested).
- Zero tests for `src/environments/` (environments module untested).
- Zero tests for `src/utils/parser.py` (326 lines — untested).
- Zero tests for `src/utils/logger.py` (112 lines — untested).
- No `@pytest.mark.parametrize` usage despite heavy combinatorial logic.
