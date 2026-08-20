# bleu_eval_utils.py
"""
BLEU Score Evaluation Utilities Module

This module provides BLEU (Bilingual Evaluation Understudy) score computation
with smoothing, supporting multiple references. BLEU measures n-gram precision
between generated and reference texts with a brevity penalty to discourage
overly short outputs.

BLEU complements ROUGE (which is recall-oriented) by being precision-oriented.
Together they provide a balanced view of text generation quality.
"""

import argparse
import json
import math
from collections import Counter


def _count_ngrams(text: str, n: int) -> Counter:
    """Count n-grams in text.

    Args:
        text: Input string.
        n: N-gram size (1 = unigram, 2 = bigram, etc.).

    Returns:
        Counter mapping each n-gram tuple to its count.
    """
    words = text.lower().split()
    ngrams = Counter()
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i : i + n])
        ngrams[ngram] += 1
    return ngrams


def _clip_count(
    pred_ngrams: Counter, ref_ngrams: Counter
) -> int:
    """Count clipped n-gram matches between prediction and a single reference.

    For each n-gram, takes min(pred_count, ref_count) to avoid
    over-counting repeated n-grams.

    Args:
        pred_ngrams: N-gram counts from the predicted text.
        ref_ngrams: N-gram counts from the reference text.

    Returns:
        Total clipped count of matching n-grams.
    """
    total = 0
    for ngram, count in pred_ngrams.items():
        total += min(count, ref_ngrams.get(ngram, 0))
    return total


def compute_bleu(
    pred_answer: str,
    gold_answers: list[str],
    max_n: int = 4,
    smooth: bool = True,
) -> dict:
    """Compute BLEU score between a prediction and one or more references.

    When multiple references are provided, each n-gram in the prediction
    is clipped against the reference that gives the highest count for
    that n-gram (maximising matches across all references).

    Implements smoothing (method 1 from Chen & Cherry 2014): add 1 to
    the n-gram match count when the precision would otherwise be 0,
    preventing zero BLEU scores on short or unusual outputs.

    Args:
        pred_answer: The model output text string.
        gold_answers: One or more reference text strings.
        max_n: Maximum n-gram order (default: 4, standard BLEU).
        smooth: Whether to apply smoothing (default: True).

    Returns:
        Dict with keys:
            - 'bleu': Corpus-level BLEU score (0-1).
            - 'precisions': List of n-gram precisions for n=1..max_n.
            - 'brevity_penalty': The brevity penalty factor.
            - 'pred_len': Length of predicted text in words.
            - 'ref_len': Effective reference length (closest to pred_len).
    """
    pred_words = pred_answer.lower().split()
    pred_len = len(pred_words)

    # Find best-matching reference length (for brevity penalty)
    ref_lens = [len(r.lower().split()) for r in gold_answers]
    # Choose reference length closest to prediction length
    ref_len = min(ref_lens, key=lambda x: abs(x - pred_len))

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _count_ngrams(pred_answer, n)
        if not pred_ngrams:
            precisions.append(0.0)
            continue

        # Clip against each reference and take maximum
        max_matches = 0
        for ref in gold_answers:
            ref_ngrams = _count_ngrams(ref, n)
            matches = _clip_count(pred_ngrams, ref_ngrams)
            if matches > max_matches:
                max_matches = matches

        total_pred_ngrams = sum(pred_ngrams.values())

        if total_pred_ngrams == 0:
            precisions.append(0.0)
        else:
            prec = max_matches / total_pred_ngrams
            # Smoothing method 1: add 1 to numerator when precision=0
            if smooth and prec == 0:
                prec = 1 / (total_pred_ngrams * (2**n))
            precisions.append(prec)

    # Brevity penalty
    if pred_len < ref_len:
        brevity_penalty = math.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0
    else:
        brevity_penalty = 1.0

    # Geometric mean of precisions
    if all(p == 0 for p in precisions):
        bleu = 0.0
    else:
        log_avg = sum(math.log(p) for p in precisions if p > 0) / max_n
        bleu = brevity_penalty * math.exp(log_avg)

    return {
        "bleu": round(bleu, 4),
        "precisions": [round(p, 4) for p in precisions],
        "brevity_penalty": round(brevity_penalty, 4),
        "pred_len": pred_len,
        "ref_len": ref_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BLEU Score Evaluation")
    parser.add_argument("--pred", type=str, required=True, help="Predicted text")
    parser.add_argument("--gold", type=str, required=True, nargs="+", help="Gold/reference text(s)")
    parser.add_argument("--max-n", type=int, default=4, help="Maximum n-gram order (default: 4)")
    parser.add_argument("--no-smooth", action="store_false", dest="smooth", help="Disable smoothing")
    args = parser.parse_args()

    result = compute_bleu(args.pred, args.gold, max_n=args.max_n, smooth=args.smooth)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
