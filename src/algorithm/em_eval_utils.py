# em_eval_utils.py
"""
Exact Match Evaluation Utilities Module

This module provides utility functions for exact match evaluation,
including text normalization for consistent comparison.
"""

import argparse
import json
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


def compute_exact_match(a_gold: str, a_pred: str) -> int:
    """Check whether two strings are equal up to normalization."""

    return int(normalize_text(a_gold) == normalize_text(a_pred))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact Match Evaluation")
    parser.add_argument("--gold", type=str, required=True, help="Gold/reference text")
    parser.add_argument("--pred", type=str, required=True, help="Predicted text")
    args = parser.parse_args()

    result = {
        "exact_match": compute_exact_match(args.gold, args.pred),
        "normalized_gold": normalize_text(args.gold),
        "normalized_pred": normalize_text(args.pred),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
