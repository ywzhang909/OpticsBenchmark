"""
Optis Benchmark - Edit Distance Evaluation Utils Tests

Tests for edit distance functions from scripts/utils/edit_distance_utils.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algorithm.edit_distance_utils import (
    levenshtein_distance,
    normalized_edit_similarity,
    word_error_rate,
    word_edit_similarity,
)


class TestLevenshteinDistance:
    """Tests for levenshtein_distance function."""

    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_one_insertion(self):
        assert levenshtein_distance("hello", "hello!") == 1

    def test_one_deletion(self):
        assert levenshtein_distance("hello!", "hello") == 1

    def test_one_substitution(self):
        assert levenshtein_distance("hello", "hallo") == 1

    def test_completely_different(self):
        assert levenshtein_distance("abc", "xyz") == 3

    def test_both_empty(self):
        assert levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert levenshtein_distance("hello", "") == 5
        assert levenshtein_distance("", "hello") == 5

    def test_longer_strings(self):
        d = levenshtein_distance("kitten", "sitting")
        assert d == 3  # k->s, e->i, +g

    def test_case_sensitive(self):
        assert levenshtein_distance("Hello", "hello") > 0

    def test_unicode(self):
        assert levenshtein_distance("café", "cafe") == 1


class TestNormalizedEditSimilarity:
    """Tests for normalized_edit_similarity function."""

    def test_identical(self):
        assert normalized_edit_similarity("hello", "hello") == 1.0

    def test_completely_different(self):
        sim = normalized_edit_similarity("abcde", "vwxyz")
        # All 5 chars differ -> dist=5, max_len=5 -> sim=0.0
        assert sim == 0.0

    def test_partial_match(self):
        sim = normalized_edit_similarity("hello", "hallo")
        assert 0.0 < sim < 1.0

    def test_both_empty(self):
        assert normalized_edit_similarity("", "") == 1.0

    def test_one_empty(self):
        sim = normalized_edit_similarity("hello", "")
        assert sim == 0.0

    def test_symmetric(self):
        sim1 = normalized_edit_similarity("abc", "ab")
        sim2 = normalized_edit_similarity("ab", "abc")
        assert sim1 == sim2


class TestWordErrorRate:
    """Tests for word_error_rate function."""

    def test_perfect_match(self):
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_completely_wrong(self):
        wer = word_error_rate("goodbye", "hello world")
        assert wer > 0

    def test_empty_prediction(self):
        assert word_error_rate("", "hello world") == 1.0

    def test_empty_reference(self):
        assert word_error_rate("hello", "") == 1.0  # not 0 because ref empty means non-match
        # Actually let's re-read: ref empty -> return 0 if pred also empty else 1
        assert word_error_rate("", "") == 0.0

    def test_half_correct(self):
        # "the cat mat" vs "the dog mat" -> 1 substitution out of 3 ref words
        wer = word_error_rate("the cat mat", "the dog mat")
        assert wer == pytest.approx(1 / 3, abs=0.01)


class TestWordEditSimilarity:
    """Tests for word_edit_similarity function."""

    def test_identical(self):
        assert word_edit_similarity("hello world", "hello world") == 1.0

    def test_different(self):
        sim = word_edit_similarity("hello world", "goodbye moon")
        assert sim < 1.0

    def test_both_empty(self):
        assert word_edit_similarity("", "") == 1.0
