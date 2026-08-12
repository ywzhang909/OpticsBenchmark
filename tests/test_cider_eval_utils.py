"""
Optis Benchmark - CIDEr Evaluation Utils Tests

Tests for compute_cider from scripts/utils/cider_eval_utils.
Pure Python, no external ML dependencies.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from algorithm.cider_eval_utils import compute_cider

    CIDER_AVAILABLE = True
except (ImportError, OSError):
    CIDER_AVAILABLE = False


@pytest.mark.skipif(not CIDER_AVAILABLE, reason="cider_eval_utils not available")
class TestComputeCider:
    """Tests for compute_cider function."""

    def test_return_keys(self):
        """Test returned dict has expected keys."""
        result = compute_cider("the cat is on the mat", ["the cat is on the mat"])
        assert "cider" in result
        assert "cider_n" in result
        assert len(result["cider_n"]) == 4

    def test_perfect_match_single_ref(self):
        """Test identical prediction and reference."""
        result = compute_cider("the cat is on the mat", ["the cat is on the mat"])
        assert result["cider_n"][0] > 0.9  # unigram match
        assert result["cider"] > 0.5

    def test_no_overlap(self):
        """Test completely different text."""
        result = compute_cider(
            "quantum physics black holes",
            ["I like to bake chocolate chip cookies"],
        )
        assert result["cider"] < 0.5

    def test_empty_prediction(self):
        """Test empty prediction returns zero."""
        result = compute_cider("", ["some reference text"])
        assert result["cider"] == 0.0

    def test_empty_references(self):
        """Test empty references returns zero."""
        result = compute_cider("some text", [])
        assert result["cider"] == 0.0

    def test_multi_reference_boost(self):
        """Test more references typically give higher consensus."""
        result = compute_cider(
            "the cat",
            [
                "the cat is on the mat",
                "a cat sits on a mat",
                "the furry cat lounges",
            ],
        )
        assert result["cider"] > 0

    def test_cider_n_length(self):
        """Test cider_n length matches max_n."""
        result = compute_cider("test", ["test"], max_n=2)
        assert len(result["cider_n"]) == 2

    def test_partial_overlap(self):
        """Test partial overlap."""
        r1 = compute_cider("the cat sat on the mat", ["the dog sat on the rug"])
        r2 = compute_cider("the cat sat on the mat", ["quantum physics"])
        assert r1["cider"] > r2["cider"]

    def test_cider_symmetric(self):
        """Test CIDEr is approximately symmetric."""
        r1 = compute_cider("cat on mat", ["dog on rug"])
        r2 = compute_cider("dog on rug", ["cat on mat"])
        assert r1["cider"] == pytest.approx(r2["cider"], abs=0.05)
