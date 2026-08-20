"""
Optis Benchmark - Citation Evaluator Tests

Tests for CitationEvaluator and retrieval-related functionality.
"""

import json

import pytest

from src.evaluators import CitationEvaluator


class TestCitationEvaluator:
    """Tests for CitationEvaluator helper methods."""

    def test_extract_papers_with_doi(self):
        """Test paper extraction with DOI."""
        data = [
            {"doi": "10.1234/test.001", "title": "Paper 1"},
            {"doi": "10.1234/test.002", "title": "Paper 2"},
        ]

        papers = CitationEvaluator._extract_papers(data)

        assert len(papers) == 2
        assert "10.1234/test.001" in papers
        assert "10.1234/test.002" in papers

    def test_extract_papers_with_title(self):
        """Test paper extraction with title (fallback)."""
        data = [
            {"title": "Deep Learning for Optics"},
            {"title": "Neural Networks in Photography"},
        ]

        papers = CitationEvaluator._extract_papers(data)

        assert len(papers) == 2
        assert "deep learning for optics" in papers
        assert "neural networks in photography" in papers

    def test_extract_papers_string_list(self):
        """Test paper extraction from string list."""
        data = ["paper_1", "paper_2", "paper_3"]

        papers = CitationEvaluator._extract_papers(data)

        assert len(papers) == 3
        assert "paper_1" in papers

    def test_extract_papers_nested(self):
        """Test paper extraction from nested dict with 'papers' key."""
        data = {
            "papers": [
                {"doi": "10.1234/test.001"},
                {"doi": "10.1234/test.002"},
            ]
        }

        papers = CitationEvaluator._extract_papers(data)

        assert len(papers) == 2

    def test_calculate_retrieval_metrics(self):
        """Test retrieval metrics calculation."""
        pred = {"paper1", "paper2", "paper3"}
        ref = {"paper1", "paper4", "paper5"}

        metrics = CitationEvaluator._calculate_retrieval_metrics(pred, ref)

        assert metrics["num_correct"] == 1
        assert metrics["precision"] == pytest.approx(1 / 3)
        assert metrics["recall"] == pytest.approx(1 / 3)
        assert metrics["f1"] == pytest.approx(1 / 3)

    def test_calculate_retrieval_metrics_empty(self):
        """Test retrieval metrics with empty inputs."""
        metrics = CitationEvaluator._calculate_retrieval_metrics(set(), set())
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

        metrics = CitationEvaluator._calculate_retrieval_metrics(set(), {"a", "b"})
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

        metrics = CitationEvaluator._calculate_retrieval_metrics({"a", "b"}, set())
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_title_similarity(self):
        """Test title similarity calculation."""
        title1 = "Deep Learning for Optical Design"
        title2 = "Deep Learning for Optical Engineering"

        similarity = CitationEvaluator._title_similarity(title1, title2)

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
        pred = {"paper1", "paper2"}
        ref = {"paper1", "paper3"}

        accuracy = CitationEvaluator._calculate_citation_accuracy(pred, ref)

        assert accuracy == pytest.approx(1 / 2)

    def test_calculate_citation_accuracy_partial_match(self):
        """Test citation accuracy with partial title matches."""
        pred = {"paper1", "deep learning optics"}
        ref = {"paper1", "deep learning systems"}

        accuracy = CitationEvaluator._calculate_citation_accuracy(pred, ref)

        assert accuracy == pytest.approx(0.5)


class TestCitationEvaluatorIntegration:
    """Integration tests for CitationEvaluator helper methods with sample data."""

    def test_realistic_retrieval_scenario(
        self,
        sample_paper_retrieval: tuple[dict, dict],
    ):
        """Test a realistic paper retrieval scenario using helper methods."""
        predicted, expected = sample_paper_retrieval

        pred_papers = CitationEvaluator._extract_papers(predicted)
        ref_papers = CitationEvaluator._extract_papers(expected)

        metrics = CitationEvaluator._calculate_retrieval_metrics(pred_papers, ref_papers)

        assert metrics["num_correct"] == 1
        assert metrics["precision"] == pytest.approx(1 / 3)
        assert metrics["recall"] == pytest.approx(1 / 3)
        assert len(pred_papers) == 3
        assert len(ref_papers) == 3
