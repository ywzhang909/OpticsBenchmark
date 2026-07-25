# ALGORITHM MODULE

**Path:** `src/algorithm/` — 13 pure-math evaluation modules.

## OVERVIEW
Self-contained compute functions for 12 metric types. No business logic — just numerics. Called by evaluator scorers. Consistent interface: `compute_*(...) → dict[str, float]`.

## MODULES
| Module | Metrics | Deps | Pure Python? |
|--------|---------|------|-------------|
| `em_eval_utils.py` | Exact match (normalized) | none | ✅ |
| `rouge_eval_utils.py` | ROUGE-1/2/L | rouge_score, nltk | ❌ |
| `bleu_eval_utils.py` | BLEU (Chen & Cherry smoothing) | none | ✅ |
| `cider_eval_utils.py` | CIDEr (TF-IDF weighted) | none | ✅ |
| `meteor_eval_utils.py` | METEOR | nltk | ❌ |
| `edit_distance_utils.py` | Levenshtein, WER, norm edit sim | none | ✅ |
| `jaccard_similarity_utils.py` | Jaccard, Dice, keyword F1 | none | ✅ |
| `hungarian_algorithm_utils.py` | `hungarian_match()` | scipy | ❌ |
| `sentence_similarity_utils.py` | `SentenceEmbedder`, sim matrix | transformers, torch | ❌ |
| `perplexity_eval_utils.py` | Perplexity (GPT-2 LM) | transformers | ❌ |
| `bertScore_eval_utils.py` | BERTScore | bert-score | ❌ |
| `citation_eval_utils.py` | Citation F1 | none | ✅ |
| `model_registry.py` | `get_or_load()` / `unload()` GPU models | none | ✅ |

## KEY PATTERNS
- **model_registry** singleton: lazy-loads GPU models (SentenceEmbedder, BERTScorer, Citation NLI) keyed by name. `unload()` releases GPU memory.
- **Hungarian matching**: `hungarian_match(sim_matrix)` → optimal assignment via `scipy.optimize.linear_sum_assignment`. Used for sentence-level alignment in ROUGE/BERTScore.
- **SentenceEmbedder**: wraps `BAAI/bge-m3` transformer. `encode(sentences)` → `np.ndarray`. Cached in model_registry.
