"""
OptiS Benchmark - ROUGE Scorer Tests

Tests for ROGUEScorer and SummarizationEvaluator.
"""

import pytest
import asyncio
from src.core.evaluator import ROGUEScorer, SummarizationEvaluator


class TestROGUEScorer:
    """Tests for ROUGE scorer implementation."""

    def test_rouge_1_perfect_match(self):
        """Test ROUGE-1 with perfect match."""
        text = "the quick brown fox"

        result = ROGUEScorer.rouge_n(text, text, n=1)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f_score"] == 1.0

    def test_rouge_1_partial_match(self):
        """Test ROUGE-1 with partial match."""
        pred = "the quick fox"
        ref = "the quick brown fox"

        result = ROGUEScorer.rouge_n(pred, ref, n=1)

        # 3 common unigrams, 4 reference unigrams
        assert result["recall"] == pytest.approx(3 / 4)
        # 3 common unigrams, 3 predicted unigrams
        assert result["precision"] == 1.0

    def test_rouge_1_no_overlap(self):
        """Test ROUGE-1 with no overlap."""
        pred = "hello world"
        ref = "goodbye earth"

        result = ROGUEScorer.rouge_n(pred, ref, n=1)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f_score"] == 0.0

    def test_rouge_2_basic(self):
        """Test ROUGE-2 (bigrams)."""
        pred = "the quick brown"
        ref = "the quick red"

        result = ROGUEScorer.rouge_n(pred, ref, n=2)

        # Common bigrams: "the quick"
        assert result["precision"] == pytest.approx(1 / 2)
        assert result["recall"] == pytest.approx(1 / 2)

    def test_rouge_2_perfect(self):
        """Test ROUGE-2 with perfect match."""
        text = "the quick brown"

        result = ROGUEScorer.rouge_n(text, text, n=2)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f_score"] == 1.0

    def test_rouge_2_empty_output(self):
        """Test ROUGE-2 with empty predicted output."""
        pred = ""
        ref = "the quick brown fox"

        result = ROGUEScorer.rouge_n(pred, ref, n=2)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f_score"] == 0.0

    def test_rouge_l_basic(self):
        """Test ROUGE-L (LCS)."""
        pred = "the quick brown fox"
        ref = "the quick brown fox"

        result = ROGUEScorer.rouge_l(pred, ref)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f_score"] == 1.0

    def test_rouge_l_partial(self):
        """Test ROUGE-L with partial match."""
        pred = "the quick fox"
        ref = "the quick brown fox jumps"

        result = ROGUEScorer.rouge_l(pred, ref)

        # LCS = "the quick fox" = 4 words
        # Precision = 4/4 = 1.0
        # Recall = 4/6 = 0.667
        assert result["precision"] == 1.0
        assert result["recall"] == pytest.approx(4 / 6)

    def test_rouge_l_no_overlap(self):
        """Test ROUGE-L with no overlap."""
        pred = "hello world"
        ref = "goodbye earth"

        result = ROGUEScorer.rouge_l(pred, ref)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f_score"] == 0.0

    def test_rouge_l_empty(self):
        """Test ROUGE-L with empty strings."""
        result = ROGUEScorer.rouge_l("", "")

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_calculate_all_metrics(self):
        """Test calculating all ROUGE metrics at once."""
        pred = "the quick brown fox jumps"
        ref = "the quick red fox jumps over"

        metrics = ROGUEScorer.calculate_all(pred, ref)

        # Check all expected keys exist
        assert "rouge_1_precision" in metrics
        assert "rouge_1_recall" in metrics
        assert "rouge_1_f_score" in metrics
        assert "rouge_2_precision" in metrics
        assert "rouge_2_recall" in metrics
        assert "rouge_2_f_score" in metrics
        assert "rouge_l_precision" in metrics
        assert "rouge_l_recall" in metrics
        assert "rouge_l_f_score" in metrics

        # All values should be in [0, 1]
        for key, value in metrics.items():
            assert 0 <= value <= 1, f"{key} = {value} out of range"

    def test_f_score_calculation(self):
        """Test F-score formula: 2 * P * R / (P + R)."""
        pred = "a b c d"
        ref = "a b e f"

        # Common: a, b = 2
        # Pred unique: c, d = 2
        # Ref unique: e, f = 2
        precision = 2 / 4  # 2 common / 4 predicted
        recall = 2 / 4  # 2 common / 4 reference
        expected_f = 2 * precision * recall / (precision + recall)

        result = ROGUEScorer.rouge_n(pred, ref, n=1)

        assert result["f_score"] == pytest.approx(expected_f)


class TestSummarizationEvaluator:
    """Tests for SummarizationEvaluator."""

    @pytest.mark.asyncio
    async def test_summarization_perfect_match(
        self,
        summarization_evaluator: SummarizationEvaluator,
    ):
        """Test summarization with perfect match."""
        summary = "This is a test summary about machine learning."

        result = await summarization_evaluator.evaluate(
            task_id="sum_001",
            predicted_output=summary,
            expected_output=summary,
        )

        assert result.success is True
        assert result.score == 1.0
        assert "rouge_1" in result.metrics
        assert "rouge_2" in result.metrics
        assert "rouge_l" in result.metrics
        assert "content_coverage" in result.metrics

    @pytest.mark.asyncio
    async def test_summarization_partial_match(
        self,
        summarization_evaluator: SummarizationEvaluator,
    ):
        """Test summarization with partial match."""
        predicted = "Deep learning has revolutionized computer vision."
        reference = "Deep learning techniques have transformed computer vision research."

        result = await summarization_evaluator.evaluate(
            task_id="sum_002",
            predicted_output=predicted,
            expected_output=reference,
        )

        assert 0 < result.score < 1
        assert "rouge_1" in result.metrics
        assert "composite_score" in result.metrics

    @pytest.mark.asyncio
    async def test_summarization_no_overlap(
        self,
        summarization_evaluator: SummarizationEvaluator,
    ):
        """Test summarization with no overlap."""
        predicted = "Completely different topic about cooking pasta."
        reference = "Deep learning for optical design optimization."

        result = await summarization_evaluator.evaluate(
            task_id="sum_003",
            predicted_output=predicted,
            expected_output=reference,
        )

        assert result.score < 0.5  # Should have low score

    @pytest.mark.asyncio
    async def test_summarization_with_json(
        self,
        summarization_evaluator: SummarizationEvaluator,
    ):
        """Test summarization with JSON input."""
        import json

        predicted = json.dumps({"summary": "Machine learning is great.", "length": 3})
        reference = json.dumps({"summary": "Machine learning is great.", "length": 3})

        result = await summarization_evaluator.evaluate(
            task_id="sum_004",
            predicted_output=predicted,
            expected_output=reference,
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_summarization_weighted_score(
        self,
        summarization_evaluator: SummarizationEvaluator,
    ):
        """Test that weighted scoring is applied correctly."""
        predicted = "deep learning neural networks"
        reference = "neural networks deep learning artificial intelligence"

        result = await summarization_evaluator.evaluate(
            task_id="sum_005",
            predicted_output=predicted,
            expected_output=reference,
        )

        # Calculate expected weighted score
        rouge_metrics = ROGUEScorer.calculate_all(predicted, reference)
        expected = (
            summarization_evaluator.weight_rouge_1 * rouge_metrics["rouge_1_f_score"]
            + summarization_evaluator.weight_rouge_2 * rouge_metrics["rouge_2_f_score"]
            + summarization_evaluator.weight_rouge_l * rouge_metrics["rouge_l_f_score"]
        )

        assert result.metrics["composite_score"] == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_summarization_empty_output(
        self,
        summarization_evaluator: SummarizationEvaluator,
    ):
        """Test summarization with empty predicted output."""
        predicted = ""
        reference = "This is a reference summary."

        result = await summarization_evaluator.evaluate(
            task_id="sum_006",
            predicted_output=predicted,
            expected_output=reference,
        )

        # Should handle gracefully with low score
        assert result.score >= 0
        assert result.error is None  # Should not raise exception

    @pytest.mark.asyncio
    async def test_summarization_case_insensitivity(
        self,
        summarization_evaluator: SummarizationEvaluator,
    ):
        """Test that ROUGE is case-insensitive."""
        predicted = "THE QUICK BROWN FOX"
        reference = "the quick brown fox"

        result = await summarization_evaluator.evaluate(
            task_id="sum_007",
            predicted_output=predicted,
            expected_output=reference,
        )

        assert result.score == 1.0  # Should be perfect match after lowercasing
