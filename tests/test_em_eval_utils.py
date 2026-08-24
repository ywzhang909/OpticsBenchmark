"""
Optis Benchmark - Exact Match Evaluation Utils Tests

Tests for text normalization utilities in em_eval_utils.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algorithm.em_eval_utils import compute_exact_match, normalize_text, record_doi_punctuation

# =============================================================================
# Classes
# =============================================================================


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_normalize_collapses_multiple_spaces(self):
        result = normalize_text("hello    world")
        assert result == "hello world"

    def test_normalize_strips_leading_trailing_spaces(self):
        result = normalize_text("   hello world   ")
        assert result == "hello world"

    def test_normalize_removes_punctuation(self):
        result = normalize_text("hello, world!")
        assert result == "hello world"

    def test_normalize_lowercases(self):
        result = normalize_text("Hello World")
        assert result == "hello world"

    def test_normalize_combined(self):
        result = normalize_text("  Hello,   World!!  ")
        assert result == "hello world"

    def test_normalize_empty_string(self):
        result = normalize_text("")
        assert result == ""

    def test_normalize_only_spaces(self):
        result = normalize_text("     ")
        assert result == ""

    def test_normalize_only_punctuation(self):
        result = normalize_text("!!!,.;?")
        assert result == ""

    def test_normalize_mixed_punctuation_and_text(self):
        result = normalize_text("Note: This is a (test) example - let's see!")
        assert result == "note this is a test example  lets see"

    def test_normalize_numbers_and_punctuation(self):
        result = normalize_text("Item #1: price = $99.99?")
        assert result == "item 1 price  9999"

    def test_normalize_tabs_and_newlines(self):
        result = normalize_text("hello\tworld\nfoo  bar")
        assert result == "hello world foo bar"

    def test_normalize_already_normalized(self):
        result = normalize_text("hello world")
        assert result == "hello world"


class TestRecordDoiPunctuation:
    """Tests for record_doi_punctuation function."""

    def test_example_from_spec(self):
        result = record_doi_punctuation("10.1109/ICIT.2016.7474909.")
        assert result == {"/": [4], ".": [9, 14]}

    def test_no_trailing_period(self):
        result = record_doi_punctuation("10.1109/ICIT.2016.7474909")
        assert result == {"/": [4], ".": [9, 14]}

    def test_simple_doi(self):
        result = record_doi_punctuation("10.1000/abc123.")
        assert result == {"/": [4]}

    def test_doi_with_multiple_segments(self):
        result = record_doi_punctuation("10.1234/5678.9012.3456.")
        assert result == {"/": [4], ".": [9, 14]}

    def test_no_punctuation_after_prefix(self):
        result = record_doi_punctuation("10.1000/abc123")
        assert result == {"/": [4]}

    def test_only_trailing_period(self):
        result = record_doi_punctuation("10.1000/abc.")
        assert result == {"/": [4]}

    def test_empty_after_prefix_stripped(self):
        result = record_doi_punctuation("10.x.")
        assert result == {}

    def test_no_dot_prefix_no_punctuation(self):
        result = record_doi_punctuation("10/abcdef.")
        assert result == {"/": [2]}

    def test_mixed_punctuation(self):
        result = record_doi_punctuation("10.test/foo.bar-baz.")
        assert result == {"/": [4], ".": [8], "-": [12]}


class TestComputeExactMatch:
    """Tests for compute_exact_match function."""

    def test_exact_match_identical(self):
        result = compute_exact_match("hello world", "hello world")
        assert result == 1

    def test_exact_match_normalized_equal(self):
        result = compute_exact_match("  Hello,   World!!  ", "hello world")
        assert result == 1

    def test_exact_match_different(self):
        result = compute_exact_match("hello world", "goodbye world")
        assert result == 0

    def test_exact_match_case_difference(self):
        result = compute_exact_match("Hello World", "hello world")
        assert result == 1

    def test_exact_match_punctuation_difference(self):
        result = compute_exact_match("hello, world!", "hello world")
        assert result == 1

    def test_exact_match_empty_strings(self):
        result = compute_exact_match("", "")
        assert result == 1

    def test_exact_match_one_empty(self):
        result = compute_exact_match("", "hello")
        assert result == 0

    def test_exact_match_numbers(self):
        result = compute_exact_match("Item #1 costs $99.99", "Item 1 costs 9999")
        assert result == 1

    def test_exact_match_whitespace_differences(self):
        result = compute_exact_match("hello\tworld\nfoo  bar", "hello world foo bar")
        assert result == 1

    def test_exact_match_returns_int(self):
        result = compute_exact_match("a", "b")
        assert isinstance(result, int)
        assert result == 0

    def test_exact_match_unicode(self):
        result = compute_exact_match("café", "café")
        assert result == 1
