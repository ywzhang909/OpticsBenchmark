"""
Optis Benchmark - Sentence Similarity Utils Tests

Tests for _mean_pooling, SentenceEmbedder, and compute_similarity_matrix.
Heavy model-dependent tests are skipped when torch/transformers are missing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import sentencepiece  # needed for most HF tokenizers

    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

try:
    from algorithm.sentence_similarity_utils import SentenceEmbedder, _mean_pooling, compute_similarity_matrix

    SENTENCE_SIM_AVAILABLE = True
except (ImportError, OSError):
    SENTENCE_SIM_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestMeanPooling:
    """Tests for _mean_pooling function."""

    def test_mean_pooling_basic(self):
        """Test mean pooling with uniform attention."""
        emb = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]], dtype=torch.float32
        )
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
        result = _mean_pooling(emb, mask)
        assert result.shape == (1, 2)
        expected = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_mean_pooling_full_attention(self):
        """Test mean pooling with full attention (no padding)."""
        emb = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32
        )
        mask = torch.tensor([[1, 1]], dtype=torch.long)
        result = _mean_pooling(emb, mask)
        assert result.shape == (1, 3)
        expected = torch.tensor([[2.5, 3.5, 4.5]], dtype=torch.float32)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_mean_pooling_single_token(self):
        """Test mean pooling with a single token."""
        emb = torch.tensor([[[5.0, 10.0]]], dtype=torch.float32)
        mask = torch.tensor([[1]], dtype=torch.long)
        result = _mean_pooling(emb, mask)
        expected = torch.tensor([[5.0, 10.0]], dtype=torch.float32)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_mean_pooling_batch(self):
        """Test mean pooling with batch dimension."""
        emb = torch.tensor(
            [
                [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                [[3.0, 3.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
        result = _mean_pooling(emb, mask)
        assert result.shape == (2, 2)
        expected = torch.tensor([[1.5, 1.5], [3.0, 3.0]], dtype=torch.float32)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_mean_pooling_small_mask(self):
        """Test mean pooling with mask clamping edge case."""
        emb = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
        mask = torch.tensor([[0]], dtype=torch.long)
        result = _mean_pooling(emb, mask)
        # mask.sum() = 0, clamped to 1e-9
        assert result.shape == (1, 2)
        assert not torch.isnan(result).any()

    def test_mean_pooling_float_mask(self):
        """Test mean pooling with float attention mask."""
        emb = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32
        )
        mask = torch.tensor([[0.5, 1.0, 0.0]], dtype=torch.float32)
        result = _mean_pooling(emb, mask)
        assert result.shape == (1, 2)
        # weighted: (0.5*[1,2] + 1.0*[3,4]) / 1.5 = [2.333, 3.333]
        expected = torch.tensor([[7.0 / 3, 10.0 / 3]], dtype=torch.float32)
        assert torch.allclose(result, expected, atol=1e-5)


@pytest.mark.skipif(
    not SENTENCE_SIM_AVAILABLE or not TORCH_AVAILABLE or not SENTENCEPIECE_AVAILABLE,
    reason="torch, transformers, or sentencepiece not available",
)
class TestComputeSimilarityMatrix:
    """Tests for compute_similarity_matrix function.

    Uses a tiny model to avoid long download times.
    """

    def test_similarity_matrix_shape(self):
        """Test output shape with 2 predictions and 3 gold sentences."""
        pred = ["hello world", "foo bar"]
        gold = ["hello world", "baz qux", "test sentence"]
        sim = compute_similarity_matrix(pred, gold, model_name="prajjwal1/bert-tiny")
        assert sim.shape == (2, 3)
        assert sim.dtype == np.float32

    def test_self_similarity_diagonal_max(self):
        """Test that diagonal elements are highest for identical sentences."""
        sentences = ["The cat sat on the mat.", "The dog ran in the park."]
        sim = compute_similarity_matrix(
            sentences, sentences, model_name="prajjwal1/bert-tiny"
        )
        assert sim[0, 0] >= sim[0, 1]
        assert sim[1, 1] >= sim[1, 0]

    def test_similarity_values_in_range(self):
        """Test that similarity values are in [-1, 1]."""
        pred = ["first sentence here", "second one there"]
        gold = ["reference one", "reference two"]
        sim = compute_similarity_matrix(pred, gold, model_name="prajjwal1/bert-tiny")
        assert np.all(sim >= -1.0) and np.all(sim <= 1.0)

    def test_single_prediction(self):
        """Test with a single prediction sentence."""
        sim = compute_similarity_matrix(
            ["hello"], ["hello", "world"], model_name="prajjwal1/bert-tiny"
        )
        assert sim.shape == (1, 2)

    def test_single_gold(self):
        """Test with a single gold sentence."""
        sim = compute_similarity_matrix(
            ["hello", "world"], ["test"], model_name="prajjwal1/bert-tiny"
        )
        assert sim.shape == (2, 1)
