"""
Stubs for symbols that no longer exist in the refactored codebase
(removed from src.core.evaluator and src.core.composite_scorer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Stubs for symbols formerly in src.core.evaluator
# =============================================================================

class MetricBasedEvaluator:
    """Stub — replaced by src.evaluators.* evaluators."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        from src.module import EvaluationResult
        return EvaluationResult(task_id=task_id, metrics={})

    async def aggregate(self, results: list) -> Any:
        from src.module import AggregatedResults
        return AggregatedResults(
            total_tasks=len(results),
            metrics_summary={},
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


class PartialMatchEvaluator:
    """Stub — was in old src.core.evaluator."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.threshold = config.get("threshold", 0.8)

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        from src.module import EvaluationResult
        return EvaluationResult(task_id=task_id, metrics={})

    def _string_similarity(self, s1: str, s2: str) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())
        if not tokens1 and not tokens2:
            return 1.0
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def _dict_similarity(self, d1: dict, d2: dict) -> float:
        if not d1 and not d2:
            return 1.0
        if not d1 or not d2:
            return 0.0
        all_keys = set(d1) | set(d2)
        matches = sum(1 for k in all_keys if d1.get(k) == d2.get(k))
        return matches / len(all_keys)


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
        from src.module import EvaluationResult
        return EvaluationResult(task_id=task_id, metrics={})


class CompositeScore:
    """Stub — was in old src.core.evaluator."""

    @staticmethod
    def calculate(results: list) -> dict[str, float]:
        return {"composite_score": 0.0}


class ResultAnalyzer:
    """Stub — was in old src.core.evaluator."""

    @staticmethod
    def compute_statistics(results: list) -> dict[str, Any]:
        return {
            "num_tasks": len(results),
            "mean_score": 0.0,
        }

    @staticmethod
    def compare_models(
        a: list, b: list, a_name: str, b_name: str
    ) -> Any:
        @dataclass
        class _Comparison:
            mean_a: float = 0.0
            mean_b: float = 0.0
            winner: str = ""
        return _Comparison()


class ErrorAnalyzer:
    """Stub — was in old src.core.evaluator."""
    pass


class EvaluationQA:
    """Stub — was in old src.core.evaluator."""
    pass


class ReportGenerator:
    """Stub — was in old src.core.evaluator or separate module."""
    pass


# =============================================================================
# Stubs for symbols formerly in src.core.composite_scorer
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
                {"name": d.name, "weight": d.weight, "description": d.description, "rubric": d.rubric}
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
