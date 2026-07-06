"""
OptiS Benchmark - METEOR Evaluation Utils Tests

Tests for compute_meteor from scripts/utils/meteor_eval_utils.
Requires nltk with WordNet data.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import nltk

    try:
        nltk.data.find("wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    NLTK_WORDNET_AVAILABLE = True
except ImportError:
    NLTK_WORDNET_AVAILABLE = False

try:
    from algorithm.meteor_eval_utils import compute_meteor

    METEOR_AVAILABLE = True
except (ImportError, OSError):
    METEOR_AVAILABLE = False


@pytest.mark.skipif(
    not METEOR_AVAILABLE or not NLTK_WORDNET_AVAILABLE,
    reason="meteor_eval_utils or WordNet not available",
)
class TestComputeMeteor:
    """Tests for compute_meteor function."""

    def test_return_keys(self):
        """Test returned dict has expected keys."""
        result = compute_meteor("the cat is on the mat", ["the cat is on the mat"])
        assert "meteor" in result

    def test_perfect_match(self):
        """Test identical strings give high score."""
        result = compute_meteor("the cat is on the mat", ["the cat is on the mat"])
        assert result["meteor"] == pytest.approx(1.0, abs=0.05)

    def test_no_overlap(self):
        """Test completely different text gives low score."""
        result = compute_meteor(
            "quantum physics black holes",
            ["I like to bake chocolate chip cookies"],
        )
        assert result["meteor"] < 0.5

    def test_empty_prediction(self):
        """Test empty prediction returns zero."""
        result = compute_meteor("", ["some reference text"])
        assert result["meteor"] == 0.0

    def test_empty_references(self):
        """Test empty references returns zero."""
        result = compute_meteor("some text", [])
        assert result["meteor"] == 0.0

    def test_multiple_references(self):
        """Test best score among multiple references is used."""
        result = compute_meteor(
            "the cat is on the mat",
            [
                "the cat is on the mat",
                "something completely different",
            ],
        )
        assert result["meteor"] == pytest.approx(1.0, abs=0.05)

    def test_partial_overlap(self):
        """Test partial overlap score is in (0, 1)."""
        result = compute_meteor("the cat sat", ["the dog ran"])
        assert 0 < result["meteor"] < 1.0
