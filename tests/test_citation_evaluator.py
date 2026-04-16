"""
OptiS Benchmark - Citation Evaluator Tests

Tests for CitationEvaluator and retrieval-related functionality.
"""

import pytest
import json
from src.core.evaluator import CitationEvaluator, EvaluationResult


class TestCitationEvaluator:
    """Tests for CitationEvaluator."""

    @pytest.mark.asyncio
    async def test_citation_perfect_match(
        self,
        citation_evaluator: CitationEvaluator,
    ):
        """Test citation with perfect match."""
        predicted = {
            "papers": [
                {"doi": "10.1234/test.2023.001", "title": "Paper A"},
                {"doi": "10.1234/test.2023.002", "title": "Paper B"},
            ]
        }
        expected = {
            "papers": [
                {"doi": "10.1234/test.2023.001", "title": "Paper A"},
                {"doi": "10.1234/test.2023.002", "title": "Paper B"},
            ]
        }

        result = await citation_evaluator.evaluate(
            task_id="cite_001",
            predicted_output=predicted,
            expected_output=expected,
        )

        assert result.success is True
        assert result.metrics["precision"] == 1.0
        assert result.metrics["recall"] == 1.0
        assert result.metrics["f1"] == 1.0
        assert result.metrics["citation_accuracy"] == 1.0

    @pytest.mark.asyncio
    async def test_citation_partial_match(
        self,
        citation_evaluator: CitationEvaluator,
    ):
        """Test citation with partial match."""
        predicted = {
            "papers": [
                {"doi": "10.1234/test.2023.001", "title": "Paper A"},
                {"doi": "10.1234/test.2023.003", "title": "Paper C"},
            ]
        }
        expected = {
            "papers": [
                {"doi": "10.1234/test.2023.001", "title": "Paper A"},
                {"doi": "10.1234/test.2023.002", "title": "Paper B"},
            ]
        }

        result = await citation_evaluator.evaluate(
            task_id="cite_002",
            predicted_output=predicted,
            expected_output=expected,
        )

        # 1 correct out of 2 predicted
        assert result.metrics["precision"] == pytest.approx(1 / 2)
        # 1 correct out of 2 expected
        assert result.metrics["recall"] == pytest.approx(1 / 2)
        assert "composite_score" in result.metrics

    @pytest.mark.asyncio
    async def test_citation_no_match(
        self,
        citation_evaluator: CitationEvaluator,
    ):
        """Test citation with no matching papers."""
        predicted = {
            "papers": [
                {"doi": "10.1234/test.2023.999", "title": "Completely Different Paper"},
            ]
        }
        expected = {
            "papers": [
                {"doi": "10.1234/test.2023.001", "title": "Original Paper"},
            ]
        }

        result = await citation_evaluator.evaluate(
            task_id="cite_003",
            predicted_output=predicted,
            expected_output=expected,
        )

        assert result.metrics["precision"] == 0.0
        assert result.metrics["recall"] == 0.0
        assert result.metrics["f1"] == 0.0

    @pytest.mark.asyncio
    async def test_citation_empty_outputs(
        self,
        citation_evaluator: CitationEvaluator,
    ):
        """Test citation with empty paper lists."""
        predicted = {"papers": []}
        expected = {"papers": []}

        result = await citation_evaluator.evaluate(
            task_id="cite_004",
            predicted_output=predicted,
            expected_output=expected,
        )

        # Both empty should be perfect match
        assert result.metrics["precision"] == 1.0
        assert result.metrics["recall"] == 1.0

    @pytest.mark.asyncio
    async def test_citation_json_string_input(
        self,
        citation_evaluator: CitationEvaluator,
    ):
        """Test citation with JSON string input."""
        predicted = json.dumps({"papers": [{"doi": "10.1234/test.2023.001"}]})
        expected = json.dumps({"papers": [{"doi": "10.1234/test.2023.001"}]})

        result = await citation_evaluator.evaluate(
            task_id="cite_005",
            predicted_output=predicted,
            expected_output=expected,
        )

        assert result.metrics["precision"] == 1.0

    @pytest.mark.asyncio
    async def test_citation_list_format(
        self,
        citation_evaluator: CitationEvaluator,
    ):
        """Test citation with list format (no dict wrapper)."""
        predicted = [
            {"doi": "10.1234/test.2023.001"},
            {"doi": "10.1234/test.2023.002"},
        ]
        expected = [
            {"doi": "10.1234/test.2023.001"},
            {"doi": "10.1234/test.2023.003"},
        ]

        result = await citation_evaluator.evaluate(
            task_id="cite_006",
            predicted_output=predicted,
            expected_output=expected,
        )

        # 1 match out of 2 predicted
        assert result.metrics["precision"] == pytest.approx(1 / 2)
        # 1 match out of 2 expected
        assert result.metrics["recall"] == pytest.approx(1 / 2)

    def test_extract_papers_with_doi(self):
        """Test paper extraction with DOI."""
        evaluator = CitationEvaluator({})

        data = [
            {"doi": "10.1234/test.001", "title": "Paper 1"},
            {"doi": "10.1234/test.002", "title": "Paper 2"},
        ]

        papers = evaluator._extract_papers(data)

        assert len(papers) == 2
        assert "10.1234/test.001" in papers
        assert "10.1234/test.002" in papers

    def test_extract_papers_with_title(self):
        """Test paper extraction with title (fallback)."""
        evaluator = CitationEvaluator({})

        data = [
            {"title": "Deep Learning for Optics"},  # No DOI
            {"title": "Neural Networks in Photography"},
        ]

        papers = evaluator._extract_papers(data)

        assert len(papers) == 2
        assert "deep learning for optics" in papers
        assert "neural networks in photography" in papers

    def test_extract_papers_string_list(self):
        """Test paper extraction from string list."""
        evaluator = CitationEvaluator({})

        data = ["paper_1", "paper_2", "paper_3"]

        papers = evaluator._extract_papers(data)

        assert len(papers) == 3
        assert "paper_1" in papers

    def test_extract_papers_nested(self):
        """Test paper extraction from nested structure."""
        evaluator = CitationEvaluator({})

        data = {
            "results": [
                {"doi": "10.1234/test.001"},
                {"doi": "10.1234/test.002"},
            ]
        }

        papers = evaluator._extract_papers(data)

        assert len(papers) == 2

    def test_calculate_retrieval_metrics(self):
        """Test retrieval metrics calculation."""
        evaluator = CitationEvaluator({})

        pred = {"paper1", "paper2", "paper3"}
        ref = {"paper1", "paper4", "paper5"}

        metrics = evaluator._calculate_retrieval_metrics(pred, ref)

        # TP = 1 (paper1)
        assert metrics["num_correct"] == 1
        assert metrics["precision"] == pytest.approx(1 / 3)
        assert metrics["recall"] == pytest.approx(1 / 3)
        # F1 = 2 * (1/3 * 1/3) / (1/3 + 1/3) = 2/6 = 1/3
        assert metrics["f1"] == pytest.approx(1 / 3)

    def test_calculate_retrieval_metrics_empty(self):
        """Test retrieval metrics with empty inputs."""
        evaluator = CitationEvaluator({})

        # Both empty
        metrics = evaluator._calculate_retrieval_metrics(set(), set())
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

        # Pred empty
        metrics = evaluator._calculate_retrieval_metrics(set(), {"a", "b"})
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

        # Ref empty
        metrics = evaluator._calculate_retrieval_metrics({"a", "b"}, set())
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_title_similarity(self):
        """Test title similarity calculation."""
        title1 = "Deep Learning for Optical Design"
        title2 = "Deep Learning for Optical Engineering"

        similarity = CitationEvaluator._title_similarity(title1, title2)

        # Common words: deep, learning, for, optical (4)
        # All words union: deep, learning, for, optical, design, engineering (6)
        expected = 4 / 6
        assert similarity == pytest.approx(expected, rel=0.01)

    def test_title_similarity_identical(self):
        """Test title similarity with identical titles."""
        title = "Machine Learning in Optics"

        similarity = CitationEvaluator._title_similarity(title, title)

        assert similarity == 1.0

    def test_title_similarity_no_overlap(self):
        """Test title similarity with no overlap."""
        title1 = "cat dog bird"
        title2 = "car bike train"

        similarity = CitationEvaluator._title_similarity(title1, title2)

        assert similarity == 0.0

    def test_calculate_citation_accuracy(self):
        """Test citation accuracy calculation."""
        evaluator = CitationEvaluator({})

        pred = {"paper1", "paper2"}
        ref = {"paper1", "paper3"}

        accuracy = evaluator._calculate_citation_accuracy(pred, ref)

        # 1 exact match, 1 partial (paper2 vs paper3 with similarity > 0.8)
        # paper2 vs paper3: no common words, similarity = 0
        # So only 1 exact match / 2 expected
        assert accuracy == pytest.approx(1 / 2)

    def test_calculate_citation_accuracy_partial_match(self):
        """Test citation accuracy with partial title matches."""
        evaluator = CitationEvaluator({})

        # paper1 exact, paper2 partial match with paper3
        pred = {"paper1", "deep learning optics"}
        ref = {"paper1", "deep learning systems"}

        accuracy = evaluator._calculate_citation_accuracy(pred, ref)

        # paper1 exact match
        # "deep learning optics" vs "deep learning systems": 3 common / 4 total = 0.75
        # partial match with 0.5 weight = 0.375
        # Total = (1 + 0.375) / 2 = 0.6875
        assert 0.5 < accuracy < 0.8


class TestCitationEvaluatorIntegration:
    """Integration tests for CitationEvaluator with sample data."""

    @pytest.mark.asyncio
    async def test_realistic_retrieval_scenario(
        self,
        citation_evaluator: CitationEvaluator,
        sample_paper_retrieval: tuple[dict, dict],
    ):
        """Test a realistic paper retrieval scenario."""
        predicted, expected = sample_paper_retrieval

        result = await citation_evaluator.evaluate(
            task_id="realistic_001",
            predicted_output=predicted,
            expected_output=expected,
        )

        # Should have 1 exact match (paper 001)
        assert result.metrics["num_correct"] == 1
        assert result.metrics["precision"] == pytest.approx(1 / 3)
        assert result.metrics["recall"] == pytest.approx(1 / 3)

        # Details should include counts
        assert result.details["num_predicted"] == 3
        assert result.details["num_expected"] == 3
