# em_eval_utils.py
"""
Exact Match Evaluation Utilities Module

This module provides utility functions for exact match evaluation,
including text normalization for consistent comparison.
"""

import re
import string


def normalize_text(text: str) -> str:
    """Normalize text by collapsing spaces, stripping, removing punctuation, and lowercasing.

    Args:
        text: Input string to normalize.

    Returns:
        Normalized string with:
            - Multiple consecutive spaces collapsed to one
            - Leading/trailing whitespace removed
            - All punctuation characters removed
            - All letters converted to lowercase
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return text


def record_doi_punctuation(doi: str) -> dict[str, list[int]]:
    """Record indices of punctuation characters in a DOI string.

    Strips a trailing period (sentence-ending), then removes the DOI
    prefix up to and including the first '.' (i.e. '10.xxxx.'),
    and records 0-indexed positions of remaining punctuation.

    Args:
        doi: DOI string (e.g. '10.1109/ICIT.2016.7474909.').

    Returns:
        Dict mapping each punctuation character to a list of its
        0-indexed positions in the DOI body (after prefix removal).
    """
    if doi.endswith("."):
        doi = doi[:-1]

    first_dot = doi.find(".")
    if first_dot != -1:
        doi = doi[first_dot + 1 :]

    result: dict[str, list[int]] = {}
    for i, ch in enumerate(doi):
        if ch in string.punctuation:
            result.setdefault(ch, []).append(i)
    return result


def compute_exact_match(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_text(a_gold) == normalize_text(a_pred))
