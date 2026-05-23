"""
OptiS Benchmark - Perplexity Evaluation Utils Tests

Tests for compute_perplexity from scripts/utils/perplexity_eval_utils.
Requires torch and transformers.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import torch
    import transformers

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from utils.perplexity_eval_utils import compute_perplexity

    PERPLEXITY_AVAILABLE = True
except (ImportError, OSError):
    PERPLEXITY_AVAILABLE = False


@pytest.mark.skipif(
    not PERPLEXITY_AVAILABLE or not TORCH_AVAILABLE,
    reason="perplexity_eval_utils or torch not available",
)
class TestComputePerplexity:
    """Tests for compute_perplexity function."""

    def test_return_keys(self):
        """Test returned dict has expected keys."""
        result = compute_perplexity("Hello world.")
        assert "perplexity" in result
        assert "avg_log_likelihood" in result
        assert "num_tokens" in result
        assert "model_name" in result

    def test_perplexity_positive(self):
        """Test perplexity is positive for valid text."""
        result = compute_perplexity("The cat sat on the mat.")
        assert result["perplexity"] > 0
        assert result["num_tokens"] > 0

    def test_empty_text(self):
        """Test empty text returns inf."""
        result = compute_perplexity("")
        assert result["perplexity"] == float("inf")
        assert result["num_tokens"] == 0

    def test_blank_text(self):
        """Test whitespace-only text returns inf."""
        result = compute_perplexity("   ")
        assert result["perplexity"] == float("inf")
        assert result["num_tokens"] == 0

    def test_model_name_in_output(self):
        """Test model_name is returned."""
        result = compute_perplexity("Hello.", model_name="gpt2")
        assert result["model_name"] == "gpt2"

    def test_short_vs_long(self):
        """Test longer text does not crash (sliding window)."""
        text = "The quick brown fox jumps over the lazy dog. " * 50
        result = compute_perplexity(text, max_length=128, stride=64)
        assert result["perplexity"] > 0
        assert result["num_tokens"] > 100

    def test_same_text_same_model(self):
        """Test deterministic: same text gives approximately same score."""
        r1 = compute_perplexity("Machine learning is interesting.")
        r2 = compute_perplexity("Machine learning is interesting.")
        assert r1["perplexity"] == pytest.approx(r2["perplexity"], abs=1e-4)

    def test_different_texts_different_scores(self):
        """Test that different texts do not have identical scores."""
        r1 = compute_perplexity("The cat sat on the mat.")
        r2 = compute_perplexity("zxcvbnm qwertyuiop asdfghjkl")
        # Highly unlikely both perplexities are exactly equal
        assert r1["perplexity"] != r2["perplexity"]
