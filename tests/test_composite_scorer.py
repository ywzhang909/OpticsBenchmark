"""
Tests for the composite weighted scoring engine (``composite_scorer.py``).

Inspired by Vercel Labs benchmark-agents / PluginEval evaluation methodology.
"""

from __future__ import annotations

import pytest

from src.core.composite_scorer import (
    AntiPattern,
    CompositeScoreConfig,
    CompositeScorer,
    CoverageReport,
    DimensionCoverage,
    DimensionScore,
    ScoreReport,
    ScoringDimension,
    VerificationCatch,
    build_coverage_report,
)


class TestCompositeScoreConfig:
    """Configuration construction and serialisation."""

    def test_default_optical_has_8_dimensions(self):
        config = CompositeScoreConfig.default_optical()
        assert len(config.dimensions) == 8

    def test_default_optical_weights_sum_to_one(self):
        config = CompositeScoreConfig.default_optical()
        total = sum(d.weight for d in config.dimensions)
        assert abs(total - 1.0) < 1e-6

    def test_default_optical_has_4_anti_patterns(self):
        config = CompositeScoreConfig.default_optical()
        assert len(config.anti_patterns) == 4

    def test_round_trip_to_dict(self):
        config = CompositeScoreConfig.default_optical()
        d = config.to_dict()
        restored = CompositeScoreConfig.from_dict(d)
        assert len(restored.dimensions) == 8
        assert restored.llm_judge_weight == 0.3
        assert restored.static_weight == 0.7
        for orig, new in zip(config.dimensions, restored.dimensions):
            assert orig.name == new.name
            assert orig.weight == new.weight

    def test_from_dict_empty(self):
        config = CompositeScoreConfig.from_dict({})
        assert config.dimensions == []
        assert config.anti_patterns == []
        assert config.llm_judge_weight == 0.3

    def test_from_dict_with_data(self):
        data = {
            "dimensions": [
                {"name": "test_dim", "weight": 0.5, "description": "Test", "rubric": "A rubric"},
            ],
            "anti_patterns": [
                {"name": "bad", "description": "Bad thing", "penalty": 0.5},
            ],
            "llm_judge_weight": 0.4,
            "static_weight": 0.6,
        }
        config = CompositeScoreConfig.from_dict(data)
        assert len(config.dimensions) == 1
        assert config.dimensions[0].name == "test_dim"
        assert config.anti_patterns[0].penalty == 0.5


class TestCompositeScorer:
    """Core scoring logic."""

    def test_score_all_perfect(self):
        scorer = CompositeScorer()
        dims = {d.name: 1.0 for d in scorer.config.dimensions}
        report = scorer.score(static_scores=dims)
        assert abs(report.final_composite - 1.0) < 1e-6
        assert report.grade == "S"
        assert report.anti_pattern_penalty == 1.0

    def test_score_all_zero(self):
        scorer = CompositeScorer()
        dims = {d.name: 0.0 for d in scorer.config.dimensions}
        report = scorer.score(static_scores=dims)
        assert report.final_composite == 0.0
        assert report.grade == "F"

    def test_score_with_anti_pattern_penalty(self):
        scorer = CompositeScorer()
        dims = {d.name: 1.0 for d in scorer.config.dimensions}
        report = scorer.score(
            static_scores=dims,
            anti_patterns_triggered=["empty_output"],
        )
        # 1.0 * 0.6 = 0.6
        assert abs(report.final_composite - 0.6) < 1e-6
        assert report.anti_patterns_triggered == ["empty_output"]

    def test_score_with_judge_layer(self):
        scorer = CompositeScorer()
        dims = {d.name: 1.0 for d in scorer.config.dimensions}
        # Judge says everything is 0.0
        judge = {d.name: 0.0 for d in scorer.config.dimensions}
        report = scorer.score(static_scores=dims, judge_scores=judge)
        # blended = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert abs(report.final_composite - 0.7) < 1e-6

    def test_score_partial_static(self):
        """Only a subset of dimensions provided — missing ones default to 0."""
        scorer = CompositeScorer()
        report = scorer.score(static_scores={"optical_accuracy": 1.0})
        assert report.final_composite > 0.0
        assert report.final_composite < 1.0

    def test_anti_pattern_worst_dominates(self):
        scorer = CompositeScorer()
        dims = {d.name: 1.0 for d in scorer.config.dimensions}
        report = scorer.score(
            static_scores=dims,
            anti_patterns_triggered=["empty_output", "parse_failure"],
        )
        # worst penalty: parse_failure = 0.5
        assert abs(report.anti_pattern_penalty - 0.5) < 1e-6
        assert abs(report.final_composite - 0.5) < 1e-6

    def test_unknown_anti_pattern_ignored(self):
        scorer = CompositeScorer()
        dims = {d.name: 1.0 for d in scorer.config.dimensions}
        report = scorer.score(
            static_scores=dims,
            anti_patterns_triggered=["nonexistent_pattern"],
        )
        assert report.anti_pattern_penalty == 1.0

    def test_score_from_results_averaging(self):
        scorer = CompositeScorer()
        results = [
            {"optical_accuracy": 0.8, "metric_correctness": 0.9},
            {"optical_accuracy": 0.6, "metric_correctness": 0.7},
        ]
        report = scorer.score_from_results(results)
        # optical_accuracy: avg(0.8, 0.6) = 0.7
        # metric_correctness: avg(0.9, 0.7) = 0.8
        assert report.final_composite < 1.0

    def test_empty_results_list(self):
        scorer = CompositeScorer()
        report = scorer.score_from_results([])
        assert report.final_composite == 0.0

    def test_custom_config(self):
        config = CompositeScoreConfig(
            dimensions=[
                ScoringDimension(name="d1", weight=0.6, description="", rubric=""),
                ScoringDimension(name="d2", weight=0.4, description="", rubric=""),
            ],
            anti_patterns=[
                AntiPattern(name="fail", description="", penalty=0.5),
            ],
        )
        scorer = CompositeScorer(config)
        report = scorer.score(static_scores={"d1": 1.0, "d2": 0.0})
        # 1.0*0.6 + 0.0*0.4 = 0.6
        assert abs(report.final_composite - 0.6) < 1e-6

    def test_grade_thresholds(self):
        scorer = CompositeScorer()
        cases = [
            (0.96, "S"),
            (0.90, "A"),
            (0.80, "B"),
            (0.65, "C"),
            (0.50, "D"),
            (0.30, "F"),
            (0.00, "F"),
        ]
        for score, expected_grade in cases:
            # Fake a report by constructing manually
            dims = {d.name: 1.0 for d in scorer.config.dimensions}
            report = scorer.score(static_scores=dims, anti_patterns_triggered=[])
            # Override raw to test grade
            report = ScoreReport(
                dimension_scores=report.dimension_scores,
                final_composite=score,
                grade=scorer._to_grade(score),
            )
            assert report.grade == expected_grade, f"score {score} → {report.grade} ≠ {expected_grade}"

    def test_report_structure(self):
        scorer = CompositeScorer()
        dims = {d.name: 0.85 for d in scorer.config.dimensions}
        report = scorer.score(
            static_scores=dims,
            anti_patterns_triggered=["empty_output"],
        )
        assert len(report.dimension_scores) == 8
        assert isinstance(report.dimension_scores[0], DimensionScore)
        assert report.raw_composite > 0.0
        assert isinstance(report.details, dict)
        assert "config" in report.details
        assert "num_dimensions" in report.details


class TestCoverageReport:
    """Verification coverage report (PluginEval coverage concept)."""

    def test_empty_reports(self):
        report = build_coverage_report([])
        assert report.total_tasks == 0
        assert report.dimensions_covered == []

    def test_single_report_all_perfect(self):
        scorer = CompositeScorer()
        dims = {d.name: 1.0 for d in scorer.config.dimensions}
        report = scorer.score(static_scores=dims)
        cover = build_coverage_report([report])
        assert cover.total_tasks == 1
        assert all(d.tasks_evaluated == 1 for d in cover.dimensions_covered)
        assert all(d.mean_blended == 1.0 for d in cover.dimensions_covered)

    def test_single_report_with_judge(self):
        scorer = CompositeScorer()
        dims = {d.name: 1.0 for d in scorer.config.dimensions}
        judge = {d.name: 0.5 for d in scorer.config.dimensions}
        report = scorer.score(static_scores=dims, judge_scores=judge)
        cover = build_coverage_report([report])
        # blended = 0.7*1.0 + 0.3*0.5 = 0.85
        assert all(abs(d.mean_blended - 0.85) < 1e-6 for d in cover.dimensions_covered)
        assert all(d.tasks_with_judge == 1 for d in cover.dimensions_covered)

    def test_anti_pattern_breakdown(self):
        scorer = CompositeScorer()
        dims = {d.name: 1.0 for d in scorer.config.dimensions}
        r1 = scorer.score(static_scores=dims, anti_patterns_triggered=["empty_output"])
        r2 = scorer.score(static_scores=dims, anti_patterns_triggered=["parse_failure"])
        r3 = scorer.score(static_scores=dims, anti_patterns_triggered=["empty_output"])
        cover = build_coverage_report([r1, r2, r3])
        assert cover.total_anti_patterns_caught == 3
        assert cover.anti_pattern_breakdown["empty_output"] == 2
        assert cover.anti_pattern_breakdown["parse_failure"] == 1

    def test_judge_layer_counting(self):
        scorer = CompositeScorer()
        dims = {d.name: 1.0 for d in scorer.config.dimensions}
        r1 = scorer.score(static_scores=dims)  # no judge
        r2 = scorer.score(static_scores=dims, judge_scores={d.name: 1.0 for d in scorer.config.dimensions})
        cover = build_coverage_report([r1, r2])
        # Only r2 has judge scores
        assert cover.tasks_with_judge_layer == 8  # one report × 8 dims

    def test_coverage_report_to_dict_roundtrip(self):
        scorer = CompositeScorer()
        dims = {d.name: 1.0 for d in scorer.config.dimensions}
        report = scorer.score(static_scores=dims, anti_patterns_triggered=["empty_output"])
        cover = build_coverage_report([report])
        d = cover.to_dict()
        assert d["total_tasks"] == 1
        assert len(d["dimensions_covered"]) == 8
        assert d["total_anti_patterns_caught"] == 1
        assert "empty_output" in d["anti_pattern_breakdown"]

    def test_verification_catch_structure(self):
        catch = VerificationCatch(
            task_id="task_001",
            dimension="optical_accuracy",
            catch_type="anti_pattern",
            severity="error",
            message="Empty output detected",
            score_impact=0.4,
        )
        assert catch.task_id == "task_001"
        assert catch.catch_type == "anti_pattern"
        assert catch.severity == "error"

    def test_missing_dimension_gap_detected(self):
        """If a dimension is never in any static_scores, it still appears
        in the coverage report (with 0 weight_contribution)."""
        scorer = CompositeScorer()
        # Only provide some dimensions
        report = scorer.score(static_scores={"optical_accuracy": 1.0})
        cover = build_coverage_report([report])
        # All config dimensions are always present in dimension_scores
        assert len(cover.dimensions_covered) == 8
