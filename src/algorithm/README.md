# Evaluation Utility Modules

This directory contains evaluation metric implementations used by the Optis Benchmark evaluators (`src/evaluators/`) and the evaluation pipeline. Each module is self-contained with a consistent interface, making them suitable for both standalone use and integration into the larger evaluation pipeline.

---

## Table of Contents

| Module | Category | Dependencies | Weight |
|--------|----------|-------------|--------|
| [em_eval_utils.py](#em_eval_utilspy) | Exact Match | none | Lightweight |
| [rouge_eval_utils.py](#rouge_eval_utilspy) | N-gram Overlap | nltk, rouge-score | Lightweight |
| [bleu_eval_utils.py](#bleu_eval_utilspy) | N-gram Precision | none (pure Python) | Lightweight |
| [bert_score_eval_utils.py](#bert_score_eval_utilspy) | Semantic Similarity | torch, bert-score | Heavy |
| [citation_eval_utils.py](#citation_eval_utilspy) | Citation Structure | torch, transformers, nltk | Heavy |
| [hungarian_algorithm_utils.py](#hungarian_algorithm_utilspy) | Assignment Matching | numpy, scipy | Lightweight |
| [sentence_similarity_utils.py](#sentence_similarity_utilspy) | Semantic Embeddings | torch, transformers | Heavy |
| [model_registry.py](#model_registrypy) | Model Registry | none | Lightweight |
| [consistency.py](#consistencypy) | Judge Agreement | numpy, scipy | Lightweight |
| [bias.py](#biaspy) | Judge Self-Preference | numpy | Lightweight |

---

## em_eval_utils.py

**Exact Match — the simplest form of text comparison.**

### Principle
Normalizes two strings (lowercase, remove punctuation, collapse whitespace) and checks for identity. A match scores 1, otherwise 0.

### Functions
- `normalize_text(text)` — Lowercases, removes punctuation, collapses whitespace, strips.
- `record_doi_punctuation(text)` — Records positions of punctuation marks in a DOI string (used to later distinguish real punctuation from DOI formatting).
- `compute_exact_match(pred_answer, gold_answer)` — Returns 1 if normalized strings are identical, 0 otherwise.

### Applications
- Tasks where outputs are expected to be verbatim (e.g., answer extraction, entity recognition).
- Sanity check / baseline metric before applying more sophisticated methods.

### Pros & Cons
| Pros | Cons |
|------|------|
| Zero dependencies; fast | Binary (0 or 1) — no graded similarity |
| Deterministic and interpretable | Sensitive to word order and content |
| No model bias or randomness | Punctuation can mask actual matches |

---

## rouge_eval_utils.py

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation) — measures n-gram overlap with a focus on recall.**

### Principle
Computes ROUGE-L (longest common subsequence based) F1-score between a predicted answer and one or more reference answers. The `rouge-score` library tokenizes text using nltk, then calculates precision, recall, and F1 for the longest common subsequence at the sentence level.

### Functions
- `ensure_nltk_resources()` — Downloads nltk tokenizer data if missing.
- `compute_rouge(pred_answer, gold_answers)` — Returns the maximum ROUGE-L F1 across all references.

### Applications
- Text summarization evaluation (original use case).
- Any task where recall of key content matters (e.g., paper reviews, knowledge QA).

### Pros & Cons
| Pros | Cons |
|------|------|
| Well-established in NLP | Requires nltk data download |
| Multiple reference support | Only LCS-based (not ROUGE-1/2) |
| F1 provides balanced scoring | Loses character-level information |

---

## bleu_eval_utils.py

**BLEU (Bilingual Evaluation Understudy) — measures n-gram precision with a brevity penalty.**

### Principle
Computes n-gram precision (unigram through 4-gram) between a predicted text and one or more references. A brevity penalty discourages overly short outputs. When multiple references exist, each n-gram is clipped against the reference that gives the highest match count.

Implements smoothing (method 1 from Chen & Cherry 2014): when a precision would be zero, adds a small positive count to avoid zero BLEU scores on short or unusual outputs.

### Functions
- `compute_bleu(pred_answer, gold_answers, max_n=4, smooth=True)` — Returns dict with `bleu`, `precisions`, `brevity_penalty`, `pred_len`, `ref_len`.

### Applications
- Machine translation evaluation (original use case).
- Complements ROUGE: BLEU is precision-oriented, ROUGE is recall-oriented.
- Tasks where output fluency and concise content coverage matter.

### Pros & Cons
| Pros | Cons |
|------|------|
| Pure Python, no external deps | Less sensitive to word choice variance |
| Multi-reference with best-match | Brevity penalty can be harsh on short answers |
| Smoothing prevents degenerate zeros | Word-order sensitive beyond n-gram window |

### Relationship to ROUGE
| Aspect | BLEU | ROUGE |
|--------|------|-------|
| Orientation | Precision | Recall |
| N-gram handling | Counts matches / total pred n-grams | Counts matches / total ref n-grams |
| Scoring | Geometric mean of n-gram precisions × BP | LCS-based F1 |
| Complements | Finds what is *correct* in output | Finds what is *missing* from output |

---

## bert_score_eval_utils.py

### Principle
Encodes both prediction and reference with a pre-trained BERT model, then computes cosine similarity between token embeddings, greedily matching tokens to maximize similarity. The final score is an aggregate (F1) of precision and recall over token alignments.

### Functions
- `compute_bert_score(pred_answer, gold_answers, model_type="microsoft/deberta-xlarge-mnli", batch_size=64)` — Returns max F1-BERTScore across references.

### Applications
- Open-ended generation evaluation where paraphrasing is expected.
- Semantic similarity tasks where exact wording varies but meaning is preserved.
- Image captioning, dialogue, story generation evaluation.

### Pros & Cons
| Pros | Cons |
|------|------|
| Captures semantic similarity well | Requires GPU for practical speed |
| Handles synonyms and paraphrasing | Model-dependent scores (bias from training data) |
| Multi-reference support | Slow without GPU |
| Strong correlation with human judgment | Large model download (≈1.5 GB for deberta) |

---

## sentence_similarity_utils.py

**Sentence Embedding Similarity — encodes sentences into dense vectors and computes pairwise cosine similarity.**

### Principle
Uses a SentenceTransformer model (`all-MiniLM-L6-v2`) to encode text into fixed-size embeddings. A similarity matrix is computed via cosine similarity between prediction and reference embeddings, optionally optimized with Hungarian matching for multi-sentence alignment.

### Functions
- `_mean_pooling(token_embeddings, attention_mask)` — Mean pooling over BERT token embeddings.
- `SentenceEmbedder` — Class wrapping model loading and encoding.
- `compute_similarity_matrix(pred_embeddings, ref_embeddings)` — Pairwise cosine similarity.

### Applications
- Multi-sentence / multi-document comparison.
- Sentence-level alignment evaluation.
- Retrieval evaluation (embedding-based search quality).

### Pros & Cons
| Pros | Cons |
|------|------|
| Fast (optimized MiniLM model) | Model size ≈ 80 MB |
| Batch encoding for efficiency | Less nuanced than BERTScore for fine-grained tokens |
| Hungarian matching for alignment | Embeddings lose token-level precision |

---

## hungarian_algorithm_utils.py

**Hungarian Algorithm — optimal assignment for matching predictions to references.**

### Principle
Given a cost matrix (e.g., 1 - similarity), the Hungarian algorithm finds the minimum-cost one-to-one assignment between predicted and reference items. This is useful when order doesn't matter (e.g., sets of facts, multi-sentence outputs).

### Functions
- `hungarian_match(pred, ref, similarity_fn, threshold=0.5)` — Returns matched pairs, unmatched preds, and unmatched refs.

### Applications
- Multi-objective / multi-label evaluation.
- Factual consistency evaluation (matching claims across texts).
- Any structured output where alignment is needed before scoring.

### Pros & Cons
| Pros | Cons |
|------|------|
| Optimal matching guaranteed | O(n³) complexity for n items |
| Flexible similarity function injection | Requires numerical similarity matrix |
| Provides interpretable alignments | May over-match when scores are noisy |

---

## citation_eval_utils.py

**Citation Evaluation — evaluates the accuracy of citations in generated text.**

### Principle
Removes citation markers from text, extracts citation IDs, runs Natural Language Inference (NLI) to verify whether cited passages actually support the claims they are attached to, and computes precision/recall/F1 for citation usage.

### Functions
- `remove_citations(text)` — Strips citation markers like `[1]`, `[2,3]`.
- `extract_citations(text)` — Extracts citation IDs from text.
- `_run_nli_autoais(premises, hypotheses)` — Runs NLI to check if premise entails hypothesis.
- `compute_citation_f1(pred, gold)` — Citation precision, recall, F1.
- `get_max_memory()` — Detects available GPU memory for model placement.

### Applications
- Academic citation accuracy evaluation.
- Fact-verification in retrieval-augmented generation (RAG).
- Paper review and literature survey generation.

### Pros & Cons
| Pros | Cons |
|------|------|
| End-to-end citation verification | Requires large NLI model + GPU |
| Distinguishes citation existence from correctness | NLI model may have biases |
| Structured output with per-citation analysis | Slower than lexical methods |

---

## model_registry.py

**Model Registry — centralized registry for evaluation model configurations.**

### Principle
Provides a centralized mapping of model names to their configurations, enabling consistent model loading across evaluation modules.

### Applications
- Used internally by other algorithm modules for model selection.
- Ensures consistent model configuration across the evaluation pipeline.

---

## consistency.py

**Judge Agreement — how much two judges' rankings of the same samples agree.**

### Principle
Pairs two score lists by `id` (order-independent, intersection only) and computes Pearson's r plus the two-sided p-value over the paired samples in a single `scipy.stats.pearsonr` call. Degenerate inputs are screened *before* that call — fewer than 2 pairs or zero variance on either side would make SciPy raise or return `NaN`, so they are reported as undefined instead (`pearson_r` falls back to `0.0` with `defined=False` and an `undefined_reason`).

### Functions
- `pearson_correlation(scores_a, scores_b)` — Pearson r for two equal-length score sequences, `None` when undefined.
- `evaluate_consistency(list_a, list_b)` — Returns `pearson_r`, `pearson_p`, `defined`, `undefined_reason`, `num_pairs`, `num_only_a` / `num_only_b`, `unpaired_ids`, `mean_a` / `mean_b`.

### Applications
- Comparing two LLM judges or a judge against human scores.
- Regression-style comparison of two evaluation runs (`python -m src.utils.eval_consistency`).

### Input Records
Judge score records `{"id", "output_model", "context", "judge_model", "score"}` are paired by `id`, which is unique within a file; only `id` and `score` enter the computation, the remaining keys are metadata and are ignored here. `src/utils/eval_consistency.py` reads those metadata fields to warn when the two files share a judge, or when an id was scored as different `output_model` / `context` on each side.

### Pros & Cons
| Pros | Cons |
|------|------|
| Order-independent pairing by `id` | Only measures linear association |
| Reports pairing coverage and undefined reasons | Undefined for < 2 pairs or zero variance |

---

## bias.py

**Judge Self-Preference — measures how much a judge inflates scores for its own answers.**

### Principle
Computes `ErrorRate_SE = y_self / y_other - 1` per sample, where `y_self` is the judge's score for its own model's output and `y_other` is the mean score the same judge gave the other m-1 models for the same sample. Since all outputs of one judge normally share one scale (e.g. 0–3, 0–5), scores are used raw; the metric is a ratio, so it is scale-free.

### Functions
- `evaluate_judge_bias(records, judge_model)` — Per-sample and aggregate results for one judge: `mean_error_rate_se` / `median_error_rate_se`, plus `mean_y_self` / `mean_y_other` and the `per_sample` breakdown. Samples without the judge's own output, without other models, or with `y_other <= 0` are reported in `undefined_ids` and excluded from the mean; if no sample is valid, `defined` is `False` and `undefined_reason` explains why.

### Applications
- Auditing LLM-as-judge harnesses before trusting absolute scores.
- Measuring one judge's self-preference (`python -m src.utils.eval_bias`).

### Input Records
A single file holds one judge's scores over the whole grid: n samples × m model outputs, `m * n` records `{"id", "output_model", "context", "judge_model", "score"}`. `id` repeats once per model, so the pairing key is `(id, output_model)`; only `id`, `output_model` and `score` enter the computation. The judge must be one of the m scored models, otherwise its own output cannot be identified — `src/utils/eval_bias.py` resolves the judge name from the unique `judge_model` field (overridable with `--judge-model`, which warns on a mismatch) and warns when a sample does not cover all m models, since `y_other` would then average a different model set. A repeated `(id, output_model)` pair keeps its last occurrence.

### Pros & Cons
| Pros | Cons |
|------|------|
| Scale-free ratio, easy to interpret | Needs ≥ 2 scored models to form `y_other` |
| Raw scores preserve level shifts | Undefined when `y_other` is not positive |
| One judge per file keeps the grid self-describing | Judge name is not unique to a file: parallel judges need separate files |

---

## Summary: Choosing the Right Metric

| When to Use | Recommended Metric(s) |
|-------------|----------------------|
| Output must be exact | Exact Match |
| Summarization quality | ROUGE + BLEU (recall + precision) |
| Semantic similarity (paraphrasing) | BERTScore |
| Multi-sentence alignment | Sentence Similarity + Hungarian |
| Citation verification | Citation Evaluation |
| Judge agreement across raters | Consistency (Pearson r) |
| Judge leniency toward own answers | Bias (ErrorRate_SE) |
| Need results fast, no GPU | Exact Match, ROUGE, BLEU |
| Need results accurate, GPU available | BERTScore, Citation Evaluation |
