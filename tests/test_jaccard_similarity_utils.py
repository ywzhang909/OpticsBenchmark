"""
OptiS Benchmark - Jaccard Similarity Evaluation Utils Tests

Tests for set-based similarity functions from scripts/utils/jaccard_similarity_utils.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algorithm.jaccard_similarity_utils import (
    char_ngram_jaccard,
    dice_coefficient,
    extract_keywords_by_tf,
    jaccard_similarity,
    keyword_coverage,
    keyword_precision_recall,
)


class TestJaccardSimilarity:
    """Tests for jaccard_similarity function."""

    def test_identical(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert jaccard_similarity("hello world", "goodbye moon") == 0.0

    def test_partial_overlap(self):
        sim = jaccard_similarity("the cat sat", "the dog ran")
        # intersection: {the}, union: {the, cat, sat, dog, ran} => 1/5
        assert sim == pytest.approx(0.2, abs=0.01)

    def test_both_empty(self):
        assert jaccard_similarity("", "") == 1.0

    def test_one_empty(self):
        assert jaccard_similarity("hello", "") == 0.0

    def test_case_insensitivity(self):
        assert jaccard_similarity("Hello World", "hello world") == 1.0

    def test_punctuation(self):
        assert jaccard_similarity("hello, world!", "hello world") == 1.0


class TestDiceCoefficient:
    """Tests for dice_coefficient function."""

    def test_identical(self):
        assert dice_coefficient("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert dice_coefficient("hello world", "goodbye moon") == 0.0

    def test_partial_overlap(self):
        sim = dice_coefficient("the cat sat", "the dog ran")
        # intersection: {the}, |s1|+|s2| = 3+3 => 2*1/6 = 1/3
        assert sim == pytest.approx(1 / 3, abs=0.01)

    def test_both_empty(self):
        assert dice_coefficient("", "") == 1.0

    def test_dice_higher_than_jaccard(self):
        """Dice typically > Jaccard for same overlap."""
        d = dice_coefficient("the cat sat", "the dog ran")
        j = jaccard_similarity("the cat sat", "the dog ran")
        assert d > j


class TestCharNgramJaccard:
    """Tests for char_ngram_jaccard function."""

    def test_identical(self):
        assert char_ngram_jaccard("hello", "hello", n=2) == 1.0

    def test_no_overlap(self):
        sim = char_ngram_jaccard("abc", "xyz", n=2)
        assert sim == 0.0

    def test_partial(self):
        # "abc" bigrams: {ab, bc}, "abd" bigrams: {ab, bd} => intersect=1 union=3 => 1/3
        sim = char_ngram_jaccard("abc", "abd", n=2)
        assert sim == pytest.approx(1 / 3, abs=0.01)

    def test_both_empty(self):
        assert char_ngram_jaccard("", "") == 1.0


class TestExtractKeywordsByTf:
    """Tests for extract_keywords_by_tf function."""

    def test_basic_extraction(self):
        text = "optical design uses lens lens lens and mirror mirror"
        keywords = extract_keywords_by_tf(text, top_n=3)
        assert keywords[0] == "lens"

    def test_top_n_respected(self):
        text = "one two three four five"
        keywords = extract_keywords_by_tf(text, top_n=2)
        assert len(keywords) == 2

    def test_stop_words_removed(self):
        text = "the and of lens design"
        keywords = extract_keywords_by_tf(text, top_n=5)
        assert "lens" in keywords
        assert "the" not in keywords

    def test_empty_text(self):
        assert extract_keywords_by_tf("") == []


class TestKeywordPrecisionRecall:
    """Tests for keyword_precision_recall function."""

    def test_perfect_match(self):
        result = keyword_precision_recall(["lens", "design"], ["lens", "design"])
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_no_overlap(self):
        result = keyword_precision_recall(["lens"], ["mirror"])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_partial(self):
        result = keyword_precision_recall(
            ["lens", "design", "optics"],
            ["lens", "mirror"],
        )
        # precision: 1/3, recall: 1/2, f1: 2*(1/3)*(1/2)/(1/3+1/2)
        assert result["precision"] == pytest.approx(1 / 3, abs=0.01)
        assert result["recall"] == pytest.approx(0.5, abs=0.01)

    def test_both_empty(self):
        result = keyword_precision_recall([], [])
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_empty_prediction(self):
        result = keyword_precision_recall([], ["lens"])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


class TestKeywordCoverage:
    """Tests for keyword_coverage function."""

    def test_all_present(self):
        result = keyword_coverage(
            "lens design for optical systems",
            ["lens", "optical"],
        )
        assert result["coverage"] == 1.0
        assert len(result["missing"]) == 0

    def test_some_missing(self):
        result = keyword_coverage(
            "lens design",
            ["lens", "optical", "mirror"],
        )
        assert result["coverage"] == pytest.approx(1 / 3, abs=0.01)
        assert "optical" in result["missing"]
        assert "mirror" in result["missing"]

    def test_none_present(self):
        result = keyword_coverage("nothing here", ["lens", "mirror"])
        assert result["coverage"] == 0.0

    def test_empty_gold(self):
        result = keyword_coverage("some text", [])
        assert result["coverage"] == 1.0
