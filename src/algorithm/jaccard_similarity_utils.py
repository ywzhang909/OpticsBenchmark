# jaccard_similarity_utils.py
"""
Jaccard Similarity and Keyword Coverage Utilities Module

This module provides set-based similarity metrics between texts, using
word-level and character n-gram overlap. These metrics are simple,
deterministic, and require no external dependencies — making them ideal
for quick sanity checks and interpretable evaluation.

In addition to raw similarity, keyword coverage measures how well a
predicted output covers the important terms present in the reference.
"""

import argparse
import json
import re
from collections import Counter
from typing import List, Set


def _tokenize(text: str) -> Set[str]:
    """Tokenize text into a set of lowercase words, removing punctuation.

    Args:
        text: Input string.

    Returns:
        Set of lowercase word tokens.
    """
    return set(re.findall(r"\b\w+\b", text.lower()))


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity coefficient between two texts.

    Jaccard = |intersection| / |union| of the word sets.
    Ranges from 0.0 (no overlap) to 1.0 (identical sets).

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Jaccard similarity in [0.0, 1.0].
    """
    set1 = _tokenize(text1)
    set2 = _tokenize(text2)

    if not set1 and not set2:
        return 1.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def dice_coefficient(text1: str, text2: str) -> float:
    """Compute Dice (Sørensen-Dice) coefficient between two texts.

    Dice = 2 * |intersection| / (|set1| + |set2|).
    Ranges from 0.0 to 1.0. Generally gives slightly higher values
    than Jaccard for the same input.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Dice coefficient in [0.0, 1.0].
    """
    set1 = _tokenize(text1)
    set2 = _tokenize(text2)

    if not set1 and not set2:
        return 1.0

    intersection = len(set1 & set2)
    return 2 * intersection / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0.0


def char_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """Compute Jaccard similarity on character n-grams.

    Captures substring-level overlap that word-level Jaccard misses.
    Useful for languages without spaces or for detecting surface-form
    similarity.

    Args:
        text1: First text string.
        text2: Second text string.
        n: Character n-gram size (default: 3, i.e. trigrams).

    Returns:
        Character n-gram Jaccard similarity in [0.0, 1.0].
    """
    def _char_ngrams(text: str) -> Set[str]:
        text = text.lower()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    set1 = _char_ngrams(text1)
    set2 = _char_ngrams(text2)

    if not set1 and not set2:
        return 1.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def extract_keywords_by_tf(text: str, top_n: int = 10, stop_words: List[str] = None) -> List[str]:
    """Extract keywords from text using term frequency (TF).

    Simple unsupervised keyword extraction: counts word frequencies
    after removing stop words and punctuation, and returns the top-N
    most frequent terms.

    Args:
        text: Input text.
        top_n: Number of top keywords to return.
        stop_words: Optional list of stop words to filter. If None,
            uses a small built-in English stop word list.

    Returns:
        List of (keyword, frequency) tuples sorted by frequency descending.
    """
    if stop_words is None:
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "can", "could", "shall", "should", "may", "might",
            "it", "its", "this", "that", "these", "those", "i", "you", "he",
            "she", "we", "they", "not", "no", "so", "as", "if", "about",
            "into", "over", "after", "before", "between", "under", "more",
            "most", "some", "any", "each", "every", "all", "both", "few",
        }

    words = re.findall(r"\b\w+\b", text.lower())
    filtered = [w for w in words if w not in stop_words and len(w) > 2]

    counter = Counter(filtered)
    return [word for word, _ in counter.most_common(top_n)]


def keyword_precision_recall(
    pred_keywords: List[str],
    gold_keywords: List[str],
) -> dict:
    """Compute precision, recall, and F1 for keyword overlap.

    Treats predicted keywords as a set of retrieved items and gold
    keywords as the relevant set.

    Args:
        pred_keywords: List of keywords extracted from prediction.
        gold_keywords: List of keywords from reference/ground truth.

    Returns:
        Dict with 'precision', 'recall', 'f1' in [0.0, 1.0].
    """
    pred_set = set(pred_keywords)
    gold_set = set(gold_keywords)

    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def keyword_coverage(pred_text: str, gold_keywords: List[str]) -> dict:
    """Measure how many gold keywords appear in the predicted text.

    This is a simpler, recall-focused metric that checks for the
    presence of each gold keyword in the prediction (as substrings).

    Args:
        pred_text: The predicted/student text.
        gold_keywords: List of keywords/phrases to check for.

    Returns:
        Dict with:
            - 'coverage': Fraction of gold keywords found in [0.0, 1.0].
            - 'found': List of keywords that were found.
            - 'missing': List of keywords not found.
    """
    pred_lower = pred_text.lower()
    found = []
    missing = []

    for kw in gold_keywords:
        if kw.lower() in pred_lower:
            found.append(kw)
        else:
            missing.append(kw)

    total = len(gold_keywords)
    coverage = len(found) / total if total > 0 else 1.0

    return {
        "coverage": round(coverage, 4),
        "found": found,
        "missing": missing,
    }


def main():
    parser = argparse.ArgumentParser(description="Jaccard Similarity & Keyword Coverage")
    parser.add_argument("--s1", type=str, required=True, help="First text")
    parser.add_argument("--s2", type=str, required=True, help="Second text")
    parser.add_argument("--topn", type=int, default=10, help="Top-N keywords (default: 10)")
    parser.add_argument("--char-n", type=int, default=3, help="Character n-gram size (default: 3)")
    args = parser.parse_args()

    kw1 = extract_keywords_by_tf(args.s1, top_n=args.topn)
    kw2 = extract_keywords_by_tf(args.s2, top_n=args.topn)

    result = {
        "jaccard": round(jaccard_similarity(args.s1, args.s2), 4),
        "dice": round(dice_coefficient(args.s1, args.s2), 4),
        f"char_{args.char_n}_gram_jaccard": round(char_ngram_jaccard(args.s1, args.s2, n=args.char_n), 4),
        "keyword_p/r/f1_s1_vs_s2": keyword_precision_recall(kw1, kw2),
        "keyword_p/r/f1_s2_vs_s1": keyword_precision_recall(kw2, kw1),
        "keyword_coverage_s1_in_s2": keyword_coverage(args.s2, kw1),
        "keyword_coverage_s2_in_s1": keyword_coverage(args.s1, kw2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
