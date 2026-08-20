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

## Summary: Choosing the Right Metric

| When to Use | Recommended Metric(s) |
|-------------|----------------------|
| Output must be exact | Exact Match |
| Summarization quality | ROUGE + BLEU (recall + precision) |
| Semantic similarity (paraphrasing) | BERTScore |
| Multi-sentence alignment | Sentence Similarity + Hungarian |
| Citation verification | Citation Evaluation |
| Need results fast, no GPU | Exact Match, ROUGE, BLEU |
| Need results accurate, GPU available | BERTScore, Citation Evaluation |
