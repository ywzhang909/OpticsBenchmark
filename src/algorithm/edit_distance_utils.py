# edit_distance_utils.py
"""
Edit Distance Evaluation Utilities Module

This module provides edit distance based metrics for comparing texts
at both the character and word level. These metrics measure how many
insertions, deletions, or substitutions are needed to transform one
string into another.

Edit distance complements exact match by providing a graded similarity
score even when strings are not identical, capturing small typos,
spacing differences, and word reorderings.
"""


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute character-level Levenshtein (edit) distance.

    The minimum number of single-character edits (insertions, deletions,
    or substitutions) required to change *s1* into *s2*.

    Uses a space-optimised dynamic programming approach (O(min(m,n))
    memory) for efficiency.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The edit distance (0 = identical strings).
    """
    # Optimise: make s1 the shorter string
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    prev = list(range(m + 1))

    for j in range(1, n + 1):
        curr = [j] + [0] * m
        for i in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[i] = min(
                curr[i - 1] + 1,      # insertion
                prev[i] + 1,          # deletion
                prev[i - 1] + cost,   # substitution
            )
        prev = curr

    return prev[m]


def normalized_edit_similarity(s1: str, s2: str) -> float:
    """Compute normalised edit similarity between two strings.

    Maps edit distance to a [0, 1] similarity score where 1.0 means
    identical and 0.0 means completely different. Uses the formula:
        similarity = 1 - distance / max(len(s1), len(s2))

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Similarity score in [0.0, 1.0].
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(s1, s2) / max_len


def _levenshtein_on_sequences(a: list, b: list) -> int:
    """Compute Levenshtein distance on two sequences (e.g., word lists).

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        Edit distance between the sequences.
    """
    if len(a) > len(b):
        a, b = b, a

    m, n = len(a), len(b)
    prev = list(range(m + 1))

    for j in range(1, n + 1):
        curr = [j] + [0] * m
        for i in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(
                curr[i - 1] + 1,      # insertion
                prev[i] + 1,          # deletion
                prev[i - 1] + cost,   # substitution
            )
        prev = curr

    return prev[m]


def word_error_rate(pred: str, ref: str) -> float:
    """Compute Word Error Rate (WER) between predicted and reference text.

    WER is the Levenshtein distance at the word level divided by the
    number of words in the reference. Lower is better; 0.0 = perfect.

    Words are compared as-is (case-sensitive).

    Args:
        pred: Predicted/recognised text.
        ref: Reference/ground-truth text.

    Returns:
        WER in [0.0, inf). 0.0 means a perfect word-level match.
    """
    pred_words = pred.split()
    ref_words = ref.split()

    if not ref_words:
        return 0.0 if not pred_words else 1.0

    distance = _levenshtein_on_sequences(pred_words, ref_words)
    return distance / len(ref_words)


def word_edit_similarity(pred: str, ref: str) -> float:
    """Compute word-level edit similarity.

    Like WER but inverted to a [0, 1] similarity score.

    Args:
        pred: Predicted text.
        ref: Reference text.

    Returns:
        Similarity score in [0.0, 1.0].
    """
    pred_words = pred.split()
    ref_words = ref.split()

    max_len = max(len(pred_words), len(ref_words))
    if max_len == 0:
        return 1.0

    distance = _levenshtein_on_sequences(pred_words, ref_words)
    return 1.0 - distance / max_len
