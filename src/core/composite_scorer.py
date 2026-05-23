"""
OptiS Benchmark — Composite Weighted Scoring Engine

Inspired by the Vercel Labs benchmark-agents / PluginEval evaluation
methodology which uses multi-dimensional scoring with configurable
dimension weights, layer blending, and anti-pattern penalties.

Reference: https://www.skills.sh/vercel-labs/vercel-plugin/benchmark-agents

Typical usage::

    config = CompositeScoreConfig.from_dict({...})
    scorer = CompositeScorer(config)
    report = scorer.score(metric_results, llm_judge_scores)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class ScoringDimension:
    """A single evaluation dimension with rubric.

    Inspired by PluginEval's 10 scoring dimensions, each dimension
    represents one quality axis that contributes to the final score.
    """

    name: str
    weight: float
    description: str = ""
    rubric: str = ""


@dataclass
class AntiPattern:
    """An anti-pattern that triggers a score penalty.

    Analogous to PluginEval's anti-pattern flags which detect common
    failure modes and apply multiplicative penalties.
    """

    name: str
    description: str
    penalty: float  # 0.0–1.0 multiplier (e.g. 0.8 = 20 % off)


@dataclass
class CompositeScoreConfig:
    """Configuration for the composite scorer.

    Mirrors PluginEval's three-layer evaluation architecture:
      1. Static analysis (automated metric scoring)
      2. LLM Judge (rubric-based quality assessment)
      3. Aggregate blending with anti-pattern penalties
    """

    dimensions: list[ScoringDimension] = field(default_factory=list)
    anti_patterns: list[AntiPattern] = field(default_factory=list)
    llm_judge_weight: float = 0.3  # blend weight for LLM judge scores
    static_weight: float = 0.7  # blend weight for static analysis scores

    @classmethod
    def default_optical(cls) -> CompositeScoreConfig:
        """Return a default configuration tailored to optical design evaluation."""
        return cls(
            dimensions=[
                ScoringDimension(
                    name="optical_accuracy",
                    weight=0.25,
                    description="Accuracy of optical design outputs (MTF, spot size, etc.)",
                    rubric="Higher is better. 1.0 = design meets all optical criteria.",
                ),
                ScoringDimension(
                    name="metric_correctness",
                    weight=0.20,
                    description="Correct computation of evaluation metrics",
                    rubric="1.0 = all metrics computed correctly with proper normalization.",
                ),
                ScoringDimension(
                    name="output_completeness",
                    weight=0.15,
                    description="All required fields and task elements present in output",
                    rubric="1.0 = every requested field is present and non-empty.",
                ),
                ScoringDimension(
                    name="citation_accuracy",
                    weight=0.12,
                    description="Correctness and relevance of citations and references",
                    rubric="1.0 = all citations are real and correctly attributed.",
                ),
                ScoringDimension(
                    name="reasoning_quality",
                    weight=0.10,
                    description="Quality of reasoning in explanations and justifications",
                    rubric="1.0 = clear, logical, well-structured reasoning.",
                ),
                ScoringDimension(
                    name="robustness",
                    weight=0.08,
                    description="Handles edge cases gracefully without crashing",
                    rubric="1.0 = performs correctly on all tested edge cases.",
                ),
                ScoringDimension(
                    name="efficiency",
                    weight=0.05,
                    description="Computational efficiency of the solution",
                    rubric="1.0 = completes within reasonable time/memory bounds.",
                ),
                ScoringDimension(
                    name="reproducibility",
                    weight=0.05,
                    description="Results are reproducible with the same inputs",
                    rubric="1.0 = identical inputs produce identical outputs.",
                ),
            ],
            anti_patterns=[
                AntiPattern(
                    name="empty_output",
                    description="Agent returned empty or null output",
                    penalty=0.6,  # -40 %
                ),
                AntiPattern(
                    name="hallucinated_citation",
                    description="Citation to a non-existent or irrelevant paper",
                    penalty=0.7,  # -30 %
                ),
                AntiPattern(
                    name="incorrect_calculation",
                    description="Fundamentally incorrect numerical calculation",
                    penalty=0.75,  # -25 %
                ),
                AntiPattern(
                    name="parse_failure",
                    description="Output could not be parsed or decoded",
                    penalty=0.5,  # -50 %
                ),
            ],
            llm_judge_weight=0.3,
            static_weight=0.7,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
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
                {"name": a.name, "description": a.description, "penalty": a.penalty}
                for a in self.anti_patterns
            ],
            "llm_judge_weight": self.llm_judge_weight,
            "static_weight": self.static_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompositeScoreConfig:
        """Deserialize from a dictionary (e.g. loaded from YAML)."""
        dims = [
            ScoringDimension(
                name=d["name"],
                weight=d.get("weight", 0.1),
                description=d.get("description", ""),
                rubric=d.get("rubric", ""),
            )
            for d in data.get("dimensions", [])
        ]
        patterns = [
            AntiPattern(
                name=a["name"],
                description=a.get("description", ""),
                penalty=a.get("penalty", 1.0),
            )
            for a in data.get("anti_patterns", [])
        ]
        return cls(
            dimensions=dims,
            anti_patterns=patterns,
            llm_judge_weight=data.get("llm_judge_weight", 0.3),
            static_weight=data.get("static_weight", 0.7),
        )


@dataclass
class DimensionScore:
    """Score for a single dimension."""

    name: str
    static_score: float = 0.0
    judge_score: float | None = None
    blended_score: float = 0.0
    weight: float = 0.0
    weight_contribution: float = 0.0


@dataclass
class ScoreReport:
    """Full composite score report, analogous to PluginEval's structured JSON output.

    Contains per-dimension breakdown, anti-pattern flags, and the final
    weighted composite score.
    """

    dimension_scores: list[DimensionScore] = field(default_factory=list)
    anti_patterns_triggered: list[str] = field(default_factory=list)
    anti_pattern_penalty: float = 1.0
    raw_composite: float = 0.0
    final_composite: float = 0.0
    grade: str = "F"
    details: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Verification Coverage (PluginEval coverage report concept)
# =============================================================================


@dataclass
class DimensionCoverage:
    """Coverage info for a single dimension across a batch of tasks.

    Analogous to PluginEval's verification tracking — records how many
    tasks were evaluated on this dimension, what the min/max/mean scores
    were, and whether any anti-patterns were caught.
    """

    name: str
    tasks_evaluated: int = 0
    tasks_with_judge: int = 0
    mean_blended: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    anti_patterns_caught: list[str] = field(default_factory=list)


@dataclass
class CoverageReport:
    """Aggregated verification coverage report for a full eval run.

    Mirrors PluginEval's structured coverage output: which dimensions
    were verified, which anti-patterns were caught, how many tasks
    had judge-layer scoring, and any coverage gaps.
    """

    total_tasks: int = 0
    dimensions_covered: list[DimensionCoverage] = field(default_factory=list)
    total_anti_patterns_caught: int = 0
    anti_pattern_breakdown: dict[str, int] = field(default_factory=dict)
    tasks_with_judge_layer: int = 0
    gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "total_tasks": self.total_tasks,
            "dimensions_covered": [
                {
                    "name": d.name,
                    "tasks_evaluated": d.tasks_evaluated,
                    "tasks_with_judge": d.tasks_with_judge,
                    "mean_blended": d.mean_blended,
                    "min_score": d.min_score,
                    "max_score": d.max_score,
                    "anti_patterns_caught": d.anti_patterns_caught,
                }
                for d in self.dimensions_covered
            ],
            "total_anti_patterns_caught": self.total_anti_patterns_caught,
            "anti_pattern_breakdown": self.anti_pattern_breakdown,
            "tasks_with_judge_layer": self.tasks_with_judge_layer,
            "gaps": self.gaps,
        }


@dataclass
class VerificationCatch:
    """A single verification catch — an anti-pattern or quality issue detected.

    Maps to PluginEval's "PostToolUse validation catches": automated
    checks that flag issues in agent output for human review.
    """

    task_id: str
    dimension: str
    catch_type: str  # "anti_pattern", "low_score", "judge_unavailable"
    severity: str  # "error", "warning", "info"
    message: str
    score_impact: float = 0.0  # how much it affected the final score


def build_coverage_report(
    reports: list[ScoreReport],
    task_ids: list[str] | None = None,
) -> CoverageReport:
    """Aggregate multiple ``ScoreReport`` objects into a coverage report.

    Args:
        reports: List of ``ScoreReport`` objects from ``CompositeScorer.score()``.
        task_ids: Optional list of task IDs corresponding to the reports.

    Returns:
        A ``CoverageReport`` with per-dimension coverage, anti-pattern
        breakdowns, and any coverage gaps.
    """
    if not reports:
        return CoverageReport()

    total = len(reports)
    task_ids = task_ids or [f"task_{i}" for i in range(total)]

    # Collect per-dimension data
    dim_data: dict[str, dict[str, Any]] = {}
    ap_breakdown: dict[str, int] = {}
    tasks_with_judge = 0

    for report, tid in zip(reports, task_ids):
        if report.anti_patterns_triggered:
            for ap in report.anti_patterns_triggered:
                ap_breakdown[ap] = ap_breakdown.get(ap, 0) + 1

        for ds in report.dimension_scores:
            if ds.name not in dim_data:
                dim_data[ds.name] = {
                    "tasks_evaluated": 0,
                    "tasks_with_judge": 0,
                    "scores": [],
                    "anti_patterns_caught": set(),
                }
            dim_data[ds.name]["tasks_evaluated"] += 1
            if ds.judge_score is not None:
                dim_data[ds.name]["tasks_with_judge"] += 1
                tasks_with_judge += 1
            dim_data[ds.name]["scores"].append(ds.blended_score)
            if report.anti_patterns_triggered:
                dim_data[ds.name]["anti_patterns_caught"].update(
                    report.anti_patterns_triggered
                )

    dim_coverages = []
    gaps = []

    for name, data in dim_data.items():
        scores = data["scores"]
        mean_sc = sum(scores) / len(scores) if scores else 0.0
        ap_caught = sorted(data["anti_patterns_caught"])
        cov = DimensionCoverage(
            name=name,
            tasks_evaluated=data["tasks_evaluated"],
            tasks_with_judge=data["tasks_with_judge"],
            mean_blended=mean_sc,
            min_score=min(scores) if scores else 0.0,
            max_score=max(scores) if scores else 0.0,
            anti_patterns_caught=ap_caught,
        )
        dim_coverages.append(cov)

        # Detect gaps
        if data["tasks_evaluated"] < total:
            gaps.append(
                f"Dimension '{name}' only evaluated on {data['tasks_evaluated']}/{total} tasks"
            )
        if data["tasks_with_judge"] == 0:
            gaps.append(
                f"Dimension '{name}' has no LLM judge coverage on any task"
            )

    total_ap_caught = sum(ap_breakdown.values())

    return CoverageReport(
        total_tasks=total,
        dimensions_covered=dim_coverages,
        total_anti_patterns_caught=total_ap_caught,
        anti_pattern_breakdown=ap_breakdown,
        tasks_with_judge_layer=tasks_with_judge,
        gaps=gaps,
    )


# =============================================================================
# Scorer
# =============================================================================


class CompositeScorer:
    """Weighted composite scorer with multi-layer blending.

    Implements a scoring model inspired by PluginEval's three-layer
    evaluation architecture:

    1. **Static layer**: Scores derived from automated metric computation
       (exact match, ROUGE, BERTScore, BLEU, etc.)
    2. **Judge layer**: Scores from an LLM-based judge using structured
       rubrics (optional, configurable weight)
    3. **Anti-pattern penalty**: Multiplicative penalty if common failure
       modes are detected

    The final formula is::

        final = raw_composite × penalty
        raw_composite = Σ (dim_weight × blended_dim_score)
        blended_dim_score = static_weight × static + judge_weight × judge
    """

    def __init__(self, config: CompositeScoreConfig | None = None):
        self.config = config or CompositeScoreConfig.default_optical()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        static_scores: dict[str, float] | None = None,
        judge_scores: dict[str, float] | None = None,
        anti_patterns_triggered: list[str] | None = None,
    ) -> ScoreReport:
        """Compute the full composite score report.

        Args:
            static_scores: Per-dimension scores from automated metrics.
            judge_scores: Per-dimension scores from an LLM judge.
            anti_patterns_triggered: List of anti-pattern names detected.

        Returns:
            A ``ScoreReport`` with per-dimension breakdown and final score.
        """
        static_scores = static_scores or {}
        judge_scores = judge_scores or {}
        anti_patterns_triggered = anti_patterns_triggered or []

        # --- Compute per-dimension blended scores --------------------
        dim_scores: list[DimensionScore] = []
        raw_total = 0.0

        for dim in self.config.dimensions:
            static = static_scores.get(dim.name, 0.0)
            judge = judge_scores.get(dim.name)

            if judge is not None:
                blended = (
                    self.config.static_weight * static
                    + self.config.llm_judge_weight * judge
                )
            else:
                blended = static

            weighted = blended * dim.weight
            raw_total += weighted

            dim_scores.append(
                DimensionScore(
                    name=dim.name,
                    static_score=static,
                    judge_score=judge,
                    blended_score=blended,
                    weight=dim.weight,
                    weight_contribution=weighted,
                )
            )

        # --- Anti-pattern penalty ------------------------------------
        penalty = self._compute_penalty(anti_patterns_triggered)

        return ScoreReport(
            dimension_scores=dim_scores,
            anti_patterns_triggered=anti_patterns_triggered,
            anti_pattern_penalty=penalty,
            raw_composite=raw_total,
            final_composite=raw_total * penalty,
            grade=self._to_grade(raw_total * penalty),
            details={
                "config": self.config.to_dict(),
                "num_dimensions": len(self.config.dimensions),
                "static_layer_used": bool(static_scores),
                "judge_layer_used": bool(judge_scores),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_penalty(self, triggered: list[str]) -> float:
        """Compute multiplicative penalty from triggered anti-patterns.

        The penalty is the **minimum** of all matching anti-pattern
        penalties (i.e. the worst violation dominates). If none are
        triggered, penalty = 1.0 (no reduction).
        """
        if not triggered:
            return 1.0
        penalties = [
            ap.penalty
            for ap in self.config.anti_patterns
            if ap.name in triggered
        ]
        return min(penalties) if penalties else 1.0

    @staticmethod
    def _to_grade(score: float) -> str:
        """Convert a numeric score to a letter grade (PluginEval-style)."""
        if score >= 0.95:
            return "S"
        if score >= 0.85:
            return "A"
        if score >= 0.75:
            return "B"
        if score >= 0.60:
            return "C"
        if score >= 0.40:
            return "D"
        return "F"

    # ------------------------------------------------------------------
    # Convenience: score from EvaluationResult lists
    # ------------------------------------------------------------------

    def score_from_results(
        self,
        results_flat: list[dict[str, float]] | None = None,
        **kwargs: Any,
    ) -> ScoreReport:
        """Alternative entry point that accepts flat dict metric lists.

        Each dict in ``results_flat`` should map dimension names to
        scores, and they are averaged to produce the static layer input.
        """
        static: dict[str, float] = {}
        if results_flat:
            keys: set[str] = set()
            for r in results_flat:
                keys.update(r.keys())
            for k in keys:
                vals = [r.get(k, 0.0) for r in results_flat]
                static[k] = sum(vals) / len(vals)

        return self.score(static_scores=static, **kwargs)
