"""
OptiS Benchmark - ROUGE Evaluation Utils Tests

Tests for compute_rouge and ensure_nltk_resources from scripts/utils.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def rouge_module():
    """Import rouge_eval_utils once per module (downloads nltk data)."""
    from algorithm.rouge_eval_utils import compute_rouge, ensure_nltk_resources

    ensure_nltk_resources()
    return compute_rouge


class TestComputeRouge:
    """Tests for compute_rouge function."""

    def test_perfect_match(self, rouge_module):
        """Test with identical prediction and reference."""
        compute_rouge = rouge_module
        score = compute_rouge("The cat is on the mat.", ["The cat is on the mat."])
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self, rouge_module):
        """Test with partially matching text."""
        compute_rouge = rouge_module
        score = compute_rouge(
            "The cat is on the mat.",
            ["The cat sat on the mat."],
        )
        assert 0.5 < score < 1.0

    def test_no_overlap(self, rouge_module):
        """Test with completely different text."""
        compute_rouge = rouge_module
        score = compute_rouge(
            "Completely unrelated topic about cooking.",
            ["Deep learning for optical design optimization."],
        )
        assert score < 0.3

    def test_empty_prediction(self, rouge_module):
        """Test with empty prediction string."""
        compute_rouge = rouge_module
        score = compute_rouge("", ["Some reference text."])
        assert score == 0.0

    def test_multiple_references_best(self, rouge_module):
        """Test that best score among multiple references is chosen."""
        compute_rouge = rouge_module
        score = compute_rouge(
            "The cat is on the mat.",
            [
                "Completely different.",
                "The cat is on the mat.",
                "Something else entirely.",
            ],
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_case_insensitivity(self, rouge_module):
        """Test that ROUGE scoring is case-insensitive."""
        compute_rouge = rouge_module
        score = compute_rouge(
            "THE CAT IS ON THE MAT.",
            ["the cat is on the mat."],
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_single_word(self, rouge_module):
        """Test with single-word strings."""
        compute_rouge = rouge_module
        score = compute_rouge("optics", ["optics"])
        assert score == pytest.approx(1.0, abs=0.01)

    def test_different_lengths(self, rouge_module):
        """Test with significantly different length texts."""
        compute_rouge = rouge_module
        long_text = "The quick brown fox jumps over the lazy dog near the river."
        short_text = "fox jumps dog"
        score = compute_rouge(long_text, [short_text])
        assert 0 < score < 1.0

    def test_example_from_module(self, rouge_module):
        """Test the example given in the module docstring."""
        compute_rouge = rouge_module
        score = compute_rouge(
            "The cat is on the mat.",
            ["The cat is on the mat.", "The cat sat on the mat."],
        )
        assert score == pytest.approx(1.0, abs=0.01)
