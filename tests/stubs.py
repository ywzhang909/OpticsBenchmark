"""
Stubs for symbols that no longer exist in the refactored codebase
(removed from src.core.evaluator and src.core.composite_scorer).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# Classes
# =============================================================================


class MetricBasedEvaluator:
    """Stub — replaced by src.evaluators.* evaluators."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def _extract_metrics_config(self) -> tuple[list[str], list[dict[str, Any]]]:
        """Collect metric names and success criteria from config."""
        names: list[str] = []
        for m in self.config.get("metrics", []):
            if isinstance(m, dict):
                name = m.get("name", "")
            elif isinstance(m, str):
                name = m
            else:
                continue
            if name and name not in names:
                names.append(name)

        criteria = [c for c in self.config.get("success_criteria", []) if isinstance(c, dict)]
        for c in criteria:
            name = c.get("metric", "")
            if name and name not in names:
                names.append(name)
        return names, criteria

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Evaluate numeric metrics against configured success criteria.

        Missing metrics default to ``0.0``; invalid JSON input is treated
        as an empty prediction instead of raising.
        """
        from src.module import EvaluationResult

        predicted = predicted_output
        if isinstance(predicted, str):
            try:
                predicted = json.loads(predicted)
            except json.JSONDecodeError:
                predicted = {}
        if not isinstance(predicted, dict):
            predicted = {}

        names, criteria = self._extract_metrics_config()
        metrics: dict[str, float] = {name: float(predicted.get(name, 0.0)) for name in names}

        all_met = True
        for c in criteria:
            value = metrics.get(c.get("metric", ""), 0.0)
            if not self._compare(value, c.get("operator", ">="), float(c.get("value", 0.0))):
                all_met = False
                break
        metrics["all_criteria_met"] = 1.0 if all_met else 0.0

        return EvaluationResult(task_id=task_id, metrics=metrics)

    async def aggregate(self, results: list) -> Any:
        """Average metrics across results."""
        from src.module import AggregatedResults

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        exec_times: list[float] = []
        for r in results:
            exec_times.append(float(getattr(r, "execution_time", 0.0)))
            for k, v in getattr(r, "metrics", {}).items():
                totals[k] = totals.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1

        return AggregatedResults(
            total_tasks=len(results),
            metrics_summary={k: totals[k] / counts[k] for k in totals},
            avg_execution_time=sum(exec_times) / len(exec_times) if exec_times else 0.0,
            per_task_results=list(results),
        )

    @staticmethod
    def _compare(value: float, operator: str, threshold: float) -> bool:
        if operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        return False


class SummarizationEvaluator:
    """Stub — replaced by RougeEvaluator in src.evaluators.rouge_evaluator."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.weight_rouge_1 = config.get("weight_rouge_1", 0.2)
        self.weight_rouge_2 = config.get("weight_rouge_2", 0.3)
        self.weight_rouge_l = config.get("weight_rouge_l", 0.5)

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        from src.evaluators.scorer import ROGUEScorer
        from src.module import EvaluationResult

        predicted = str(predicted_output) if predicted_output else ""
        reference = str(expected_output) if expected_output else ""

        rouge_metrics = ROGUEScorer.calculate_all(predicted, reference)

        composite = (
            self.weight_rouge_1 * rouge_metrics.get("rouge_1_f_score", 0.0)
            + self.weight_rouge_2 * rouge_metrics.get("rouge_2_f_score", 0.0)
            + self.weight_rouge_l * rouge_metrics.get("rouge_l_f_score", 0.0)
        )

        metrics = {
            "rouge_1": rouge_metrics.get("rouge_1_f_score", 0.0),
            "rouge_2": rouge_metrics.get("rouge_2_f_score", 0.0),
            "rouge_l": rouge_metrics.get("rouge_l_f_score", 0.0),
            "content_coverage": rouge_metrics.get("rouge_l_recall", 0.0),
            "composite_score": composite,
        }
        return EvaluationResult(task_id=task_id, metrics=metrics)


class CompositeScore:
    """Stub — was in old src.core.evaluator."""

    @staticmethod
    def calculate(results: list) -> dict[str, float]:
        values = [float(v) for r in results for v in getattr(r, "metrics", {}).values()]
        mean = sum(values) / len(values) if values else 0.0
        return {"composite_score": mean}


class ResultAnalyzer:
    """Stub — was in old src.core.evaluator."""

    @staticmethod
    def _mean_metric_value(results: list) -> float:
        values = [float(v) for r in results for v in getattr(r, "metrics", {}).values()]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def compute_statistics(results: list) -> dict[str, Any]:
        return {
            "num_tasks": len(results),
            "mean_score": ResultAnalyzer._mean_metric_value(results),
        }

    @staticmethod
    def compare_models(
        a: list, b: list, a_name: str, b_name: str
    ) -> Any:
        mean_a = ResultAnalyzer._mean_metric_value(a)
        mean_b = ResultAnalyzer._mean_metric_value(b)

        @dataclass
        class _Comparison:
            mean_a: float = 0.0
            mean_b: float = 0.0
            winner: str = ""

        return _Comparison(mean_a=mean_a, mean_b=mean_b, winner="A" if mean_a >= mean_b else "B")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ScoringDimension:
    name: str = ""
    weight: float = 0.0
    description: str = ""
    rubric: str = ""


@dataclass
class AntiPattern:
    name: str = ""
    description: str = ""
    penalty: float = 0.0


@dataclass
class DimensionScore:
    name: str = ""
    static_score: float = 0.0
    judge_score: float | None = None
    blended_score: float = 0.0
    weight: float = 0.0


@dataclass
class ScoreReport:
    dimension_scores: list[DimensionScore] = field(default_factory=list)
    raw_composite: float = 0.0
    final_composite: float = 0.0
    grade: str = "F"
    anti_patterns_triggered: list[str] = field(default_factory=list)
    anti_pattern_penalty: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionCoverage:
    name: str = ""
    tasks_evaluated: int = 0
    tasks_with_judge: int = 0
    mean_blended: float = 0.0
    weight_contribution: float = 0.0


@dataclass
class CoverageReport:
    total_tasks: int = 0
    dimensions_covered: list[DimensionCoverage] = field(default_factory=list)
    total_anti_patterns_caught: int = 0
    anti_pattern_breakdown: dict[str, int] = field(default_factory=dict)
    tasks_with_judge_layer: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "dimensions_covered": [
                {
                    "name": d.name,
                    "tasks_evaluated": d.tasks_evaluated,
                    "tasks_with_judge": d.tasks_with_judge,
                    "mean_blended": d.mean_blended,
                    "weight_contribution": d.weight_contribution,
                }
                for d in self.dimensions_covered
            ],
            "total_anti_patterns_caught": self.total_anti_patterns_caught,
            "anti_pattern_breakdown": self.anti_pattern_breakdown,
        }


@dataclass
class VerificationCatch:
    task_id: str = ""
    dimension: str = ""
    catch_type: str = ""
    severity: str = ""
    message: str = ""
    score_impact: float = 0.0


# =============================================================================
# Constants
# =============================================================================

_DEFAULT_DIMENSIONS = [
    ScoringDimension(name="optical_accuracy", weight=0.15, description="", rubric=""),
    ScoringDimension(name="metric_correctness", weight=0.15, description="", rubric=""),
    ScoringDimension(name="output_completeness", weight=0.15, description="", rubric=""),
    ScoringDimension(name="citation_accuracy", weight=0.15, description="", rubric=""),
    ScoringDimension(name="reasoning_quality", weight=0.10, description="", rubric=""),
    ScoringDimension(name="robustness", weight=0.10, description="", rubric=""),
    ScoringDimension(name="efficiency", weight=0.10, description="", rubric=""),
    ScoringDimension(name="reproducibility", weight=0.10, description="", rubric=""),
]

_DEFAULT_ANTI_PATTERNS = [
    AntiPattern(name="empty_output", description="", penalty=0.6),
    AntiPattern(name="parse_failure", description="", penalty=0.5),
    AntiPattern(name="hallucination", description="", penalty=0.3),
    AntiPattern(name="refusal", description="", penalty=0.4),
]

_GRADE_THRESHOLDS = [
    ("S", 0.95),
    ("A", 0.85),
    ("B", 0.75),
    ("C", 0.60),
    ("D", 0.40),
    ("F", 0.0),
]


def _to_grade(score: float) -> str:
    for grade, threshold in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


class CompositeScoreConfig:
    def __init__(
        self,
        dimensions: list[ScoringDimension] | None = None,
        anti_patterns: list[AntiPattern] | None = None,
        llm_judge_weight: float = 0.3,
        static_weight: float = 0.7,
    ):
        self.dimensions = dimensions or []
        self.anti_patterns = anti_patterns or []
        self.llm_judge_weight = llm_judge_weight
        self.static_weight = static_weight

    @classmethod
    def default_optical(cls) -> CompositeScoreConfig:
        return cls(
            dimensions=list(_DEFAULT_DIMENSIONS),
            anti_patterns=list(_DEFAULT_ANTI_PATTERNS),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": [
                {
                    "name": d.name,
                    "weight": d.weight,
                    "description": d.description,
                    "rubric": d.rubric,
                }
                for d in self.dimensions
            ],
            "anti_patterns": [
                {"name": ap.name, "description": ap.description, "penalty": ap.penalty}
                for ap in self.anti_patterns
            ],
            "llm_judge_weight": self.llm_judge_weight,
            "static_weight": self.static_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompositeScoreConfig:
        dims = [
            ScoringDimension(
                name=d.get("name", ""),
                weight=d.get("weight", 0.0),
                description=d.get("description", ""),
                rubric=d.get("rubric", ""),
            )
            for d in data.get("dimensions", [])
        ]
        aps = [
            AntiPattern(
                name=ap.get("name", ""),
                description=ap.get("description", ""),
                penalty=ap.get("penalty", 0.0),
            )
            for ap in data.get("anti_patterns", [])
        ]
        return cls(
            dimensions=dims,
            anti_patterns=aps,
            llm_judge_weight=data.get("llm_judge_weight", 0.3),
            static_weight=data.get("static_weight", 0.7),
        )


class CompositeScorer:
    def __init__(self, config: CompositeScoreConfig | None = None):
        self.config = config or CompositeScoreConfig.default_optical()

    def score(
        self,
        static_scores: dict[str, float] | None = None,
        judge_scores: dict[str, float] | None = None,
        anti_patterns_triggered: list[str] | None = None,
    ) -> ScoreReport:
        static_scores = static_scores or {}
        judge_scores = judge_scores or {}
        anti_patterns_triggered = anti_patterns_triggered or []

        dim_scores: list[DimensionScore] = []
        raw_total = 0.0
        for dim in self.config.dimensions:
            static = static_scores.get(dim.name, 0.0)
            judge = judge_scores.get(dim.name, None)
            if judge is not None:
                blended = self.config.static_weight * static + self.config.llm_judge_weight * judge
            else:
                blended = static
            ds = DimensionScore(
                name=dim.name,
                static_score=static,
                judge_score=judge,
                blended_score=blended,
                weight=dim.weight,
            )
            dim_scores.append(ds)
            raw_total += blended * dim.weight

        penalty = 1.0
        triggered = []
        for ap_name in anti_patterns_triggered:
            for ap in self.config.anti_patterns:
                if ap.name == ap_name:
                    if ap.penalty < penalty:
                        penalty = ap.penalty
                    triggered.append(ap_name)
                    break

        final = raw_total * penalty

        return ScoreReport(
            dimension_scores=dim_scores,
            raw_composite=raw_total,
            final_composite=final,
            grade=_to_grade(final),
            anti_patterns_triggered=triggered,
            anti_pattern_penalty=penalty,
            details={"config": self.config.to_dict(), "num_dimensions": len(dim_scores)},
        )

    def score_from_results(self, results: list[dict[str, float]]) -> ScoreReport:
        if not results:
            return ScoreReport()
        avg_scores: dict[str, float] = {}
        for result in results:
            for key, value in result.items():
                if key not in avg_scores:
                    avg_scores[key] = []
                avg_scores[key].append(value)
        static = {k: sum(v) / len(v) for k, v in avg_scores.items()}
        return self.score(static_scores=static)

    def _to_grade(self, score: float) -> str:
        return _to_grade(score)


def build_coverage_report(reports: list[ScoreReport]) -> CoverageReport:
    if not reports:
        return CoverageReport()

    dim_map: dict[str, DimensionCoverage] = {}
    total_ap = 0
    ap_breakdown: dict[str, int] = {}
    layer_count = 0

    for report in reports:
        for ds in report.dimension_scores:
            if ds.name not in dim_map:
                dim_map[ds.name] = DimensionCoverage(name=ds.name)
            dim_map[ds.name].tasks_evaluated += 1
            dim_map[ds.name].mean_blended += ds.blended_score
            if ds.judge_score is not None:
                dim_map[ds.name].tasks_with_judge += 1
        for ap in report.anti_patterns_triggered:
            total_ap += 1
            ap_breakdown[ap] = ap_breakdown.get(ap, 0) + 1
        if any(ds.judge_score is not None for ds in report.dimension_scores):
            layer_count += sum(1 for ds in report.dimension_scores if ds.judge_score is not None)

    dims = sorted(dim_map.values(), key=lambda d: d.name)
    for d in dims:
        if d.tasks_evaluated > 0:
            d.mean_blended /= d.tasks_evaluated

    return CoverageReport(
        total_tasks=len(reports),
        dimensions_covered=dims,
        total_anti_patterns_caught=total_ap,
        anti_pattern_breakdown=ap_breakdown,
        tasks_with_judge_layer=layer_count,
    )
