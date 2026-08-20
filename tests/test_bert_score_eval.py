"""
Optis Benchmark - BERTScore Evaluation Utils Tests

Tests for compute_bert_score from scripts/utils.
Requires bert_score, torch, and transformers packages (installed via uv sync).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from algorithm.bert_score_eval_utils import compute_bert_score

    BERTSCORE_AVAILABLE = True
except (ImportError, OSError):
    BERTSCORE_AVAILABLE = False


@pytest.mark.skipif(not BERTSCORE_AVAILABLE, reason="bert_score_eval_utils not available")
class TestComputeBertScore:
    """Tests for compute_bert_score function."""

    def test_perfect_match(self):
        """Test with identical prediction and reference."""
        result = compute_bert_score(
            "The cat is on the mat.",
            ["The cat is on the mat."],
        )
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert result["f1"] == pytest.approx(1.0, abs=0.05)

    def test_multiple_references_best_f1(self):
        """Test that best F1 among multiple references is returned."""
        result = compute_bert_score(
            "The cat is on the mat.",
            [
                "The cat is on the mat.",
                "Something completely different.",
            ],
        )
        assert result["f1"] == pytest.approx(1.0, abs=0.05)

    def test_partial_match(self):
        """Test with partially matching text."""
        result = compute_bert_score(
            "The cat sat on the mat.",
            ["The dog sat on the rug."],
        )
        assert 0 < result["f1"] < 1.0

    def test_no_overlap(self):
        """Test with semantically different text."""
        result = compute_bert_score(
            "Quantum physics and black holes.",
            ["I like to bake chocolate chip cookies."],
        )
        assert result["f1"] < 0.7

    def test_empty_prediction(self):
        """Test with empty prediction."""
        result = compute_bert_score("", ["Some reference text."])
        assert result["f1"] == 0.0

    def test_empty_gold_references(self):
        """Test with empty reference list."""
        result = compute_bert_score("Some text.", [])
        assert result["f1"] == 0.0

    def test_all_scores_in_range(self):
        """Test that all scores are within [0, 1]."""
        result = compute_bert_score(
            "Machine learning is transforming optics.",
            ["Deep learning is used in optical design."],
        )
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1

    def test_return_keys_match(self):
        """Test that returned dict has expected keys."""
        result = compute_bert_score(
            "Test sentence.",
            ["Test sentence."],
        )
        assert set(result.keys()) == {"precision", "recall", "f1"}
