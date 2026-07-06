# cider_eval_utils.py
"""
CIDEr Evaluation Utilities Module

CIDEr (Consensus-based Image Description Evaluation) measures consensus
between a candidate text and reference texts using TF-IDF weighted n-gram
similarity (cosine of TF-IDF vectors).

While originally designed for image captioning, it can be applied to any
text generation task where multiple reference outputs capture the range
of acceptable responses.

Reference: Vedantam et al., "CIDEr: Consensus-based Image Description
Evaluation", CVPR 2015.

Pure Python implementation — no external ML dependencies.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _validate_input(pred_answer: str, gold_answer: list) -> dict | None:
    """Return zero-score dict if degenerate, else None."""
    if not gold_answer:
        return {"cider": 0.0, "cider_n": [0.0] * 4}
    if not pred_answer or not pred_answer.strip():
        return {"cider": 0.0, "cider_n": [0.0] * 4}
    return None


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization, lowercased."""
    return text.lower().split()


def _compute_tf(
    tokens: list[str],
    vocab: set[str],
) -> Counter:
    """Term frequency (raw count) for a document."""
    return Counter(t for t in tokens if t in vocab)


def _compute_idf(
    all_ref_tokens: list[list[str]],
    vocab: set[str],
) -> dict[str, float]:
    """Inverse document frequency for each term in the vocabulary."""
    n = len(all_ref_tokens)
    df: Counter = Counter()
    for tokens in all_ref_tokens:
        unique = set(t for t in tokens if t in vocab)
        df.update(unique)
    return {term: math.log(n / (1.0 + df[term])) for term in vocab}


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Generate n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_cider(
    pred_answer: str,
    gold_answer: list[str],
    max_n: int = 4,
    sigma: float = 6.0,
) -> dict[str, Any]:
    """Compute CIDEr score between a prediction and multiple references.

    Uses TF-IDF weighted n-gram cosine similarity averaged over n-gram
    orders 1 to ``max_n``, with a Gaussian decay weighting:

        CIDEr_n = (1 / |R|) * sum_r [ g(p) · g(r) / (||g(p)|| * ||g(r)||) ]

        CIDEr = sum_{n=1}^{max_n} w_n * CIDEr_n

    where ``w_n = exp(-(n - 1)^2 / (2 * sigma^2))``.

    Args:
        pred_answer: The model output text string.
        gold_answer: A list of reference text strings.
        max_n: Maximum n-gram order (default 4).
        sigma: Spread of the Gaussian weighting over n (default 6.0).

    Returns:
        dict with keys ``cider`` (aggregate), ``cider_n`` (per-n list).
    """
    early = _validate_input(pred_answer, gold_answer)
    if early is not None:
        return early

    pred_tokens = _tokenize(pred_answer)
    all_ref_tokens = [_tokenize(ref) for ref in gold_answer]

    # Build vocabulary from prediction + all references
    vocab: set[str] = set(pred_tokens)
    for tokens in all_ref_tokens:
        vocab.update(tokens)

    # Pre-compute IDF from references only
    idf = _compute_idf(all_ref_tokens, vocab)

    # Gaussian weights for each n-gram order
    weights = [math.exp(-((n - 1) ** 2) / (2 * sigma**2)) for n in range(1, max_n + 1)]
    weight_sum = sum(weights)

    cider_n = []

    for n in range(1, max_n + 1):
        pred_ngrams = _ngrams(pred_tokens, n)
        pred_tf = Counter(pred_ngrams)

        # Build TF-IDF vector for prediction
        pred_vec: dict[tuple[str, ...], float] = {}
        for ng, tf in pred_tf.items():
            # IDF: use average IDF of constituent unigrams
            ng_idf = sum(idf.get(w, 0.0) for w in ng) / n
            pred_vec[ng] = tf * ng_idf

        scores = []
        for ref_tokens in all_ref_tokens:
            ref_ngrams = _ngrams(ref_tokens, n)
            ref_tf = Counter(ref_ngrams)

            ref_vec: dict[tuple[str, ...], float] = {}
            for ng, tf in ref_tf.items():
                ng_idf = sum(idf.get(w, 0.0) for w in ng) / n
                ref_vec[ng] = tf * ng_idf

            # Cosine similarity
            all_keys = set(pred_vec) | set(ref_vec)
            dot = sum(pred_vec.get(k, 0.0) * ref_vec.get(k, 0.0) for k in all_keys)
            pred_norm = math.sqrt(sum(v**2 for v in pred_vec.values()))
            ref_norm = math.sqrt(sum(v**2 for v in ref_vec.values()))

            if pred_norm == 0 or ref_norm == 0:
                scores.append(0.0)
            else:
                scores.append(dot / (pred_norm * ref_norm))

        cider_n.append(sum(scores) / len(scores) if scores else 0.0)

    # Weighted sum over n-gram orders
    cider_val = sum(w * c for w, c in zip(weights, cider_n)) / weight_sum if weight_sum > 0 else 0.0

    return {
        "cider": round(cider_val, 4),
        "cider_n": [round(c, 4) for c in cider_n],
    }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="CIDEr Evaluation")
    parser.add_argument("--pred", type=str, required=True, help="Predicted text")
    parser.add_argument("--gold", type=str, required=True, nargs="+", help="Gold/reference text(s)")
    args = parser.parse_args()

    result = compute_cider(args.pred, args.gold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
