"""
Optis Benchmark - BLEU Evaluation Utils Tests

Tests for compute_bleu from scripts/utils/bleu_eval_utils.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algorithm.bleu_eval_utils import compute_bleu


class TestComputeBleu:
    """Tests for compute_bleu function."""

    def test_perfect_match(self):
        """Identical prediction and single reference -> BLEU ~1.0."""
        result = compute_bleu("the cat is on the mat", ["the cat is on the mat"])
        assert result["bleu"] == pytest.approx(1.0, abs=0.05)
        assert result["brevity_penalty"] == 1.0

    def test_no_overlap(self):
        """Completely different words -> BLEU near 0 (smoothing gives small boost)."""
        result = compute_bleu(
            "completely unrelated",
            ["deep learning for optical design"],
        )
        assert result["bleu"] < 0.2

    def test_partial_overlap(self):
        """Some overlapping words -> intermediate BLEU."""
        result = compute_bleu(
            "the cat sat on the mat",
            ["the cat is on the mat"],
        )
        assert 0.15 < result["bleu"] < 0.9

    def test_empty_prediction(self):
        """Empty prediction -> BLEU 0."""
        result = compute_bleu("", ["some reference text"])
        assert result["bleu"] == 0.0

    def test_multiple_references_best(self):
        """Should pick best reference match."""
        result = compute_bleu(
            "the cat is on the mat",
            [
                "completely different",
                "the cat is on the mat",
                "something else entirely",
            ],
        )
        assert result["bleu"] == pytest.approx(1.0, abs=0.05)

    def test_shorter_prediction_brevity_penalty(self):
        """Short prediction should incur brevity penalty."""
        result = compute_bleu("cat mat", ["the cat is on the mat"])
        assert result["brevity_penalty"] < 1.0
        assert result["bleu"] < 1.0

    def test_longer_prediction_no_penalty(self):
        """Longer prediction should not incur brevity penalty."""
        result = compute_bleu(
            "the cat is on the mat and the dog is in the yard",
            ["the cat is on the mat"],
        )
        assert result["brevity_penalty"] == 1.0

    def test_precisions_structure(self):
        """Should return precision values for each n-gram level."""
        result = compute_bleu("the cat is on the mat", ["the cat is on the mat"])
        assert len(result["precisions"]) == 4  # unigram to 4-gram
        assert all(0.0 <= p <= 1.0 for p in result["precisions"])

    def test_case_insensitivity(self):
        """Should be case-insensitive."""
        result1 = compute_bleu("THE CAT IS ON THE MAT", ["the cat is on the mat"])
        result2 = compute_bleu("the cat is on the mat", ["the cat is on the mat"])
        assert result1["bleu"] == pytest.approx(result2["bleu"], abs=0.01)

    def test_pred_len_ref_len(self):
        """Should output correct length statistics."""
        result = compute_bleu("cat mat", ["the cat is on the mat"])
        assert result["pred_len"] == 2
        assert result["ref_len"] == 6

    def test_smoothing_prevents_zero(self):
        """Smoothing should prevent zero BLEU on very short outputs."""
        # Short prediction with no overlapping 4-grams
        result = compute_bleu("hello world", ["goodbye universe"])
        assert result["bleu"] >= 0.0  # smoothing prevents -inf
        # With smoothing, precisions should be > 0 where n-gram mismatch exists
        assert all(p >= 0 for p in result["precisions"])

    def test_single_word(self):
        """Single word both sides should produce BLEU near 1."""
        result = compute_bleu("optics", ["optics"])
        assert result["bleu"] == pytest.approx(1.0, abs=0.01)

    def test_no_smoothing(self):
        """Without smoothing, zero matches can yield zero BLEU."""
        result = compute_bleu(
            "completely unrelated",
            ["deep learning optical design"],
            smooth=False,
        )
        # Unsmoothed, should still be near zero
        assert result["bleu"] < 0.05
