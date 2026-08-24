"""
Optis Benchmark — LLM Judge Evaluator

Inspired by Vercel Labs PluginEval's "Layer 2 — LLM Judge" which uses
a language model to score output quality against anchored rubrics.

Reference: https://www.skills.sh/vercel-labs/vercel-plugin/benchmark-agents

The judge evaluates agent outputs across multiple quality dimensions
using structured rubrics and returns a JSON object with per-dimension
scores and justifications.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RubricCriterion:
    """A single criterion within a rubric."""

    score: float  # 0.0–1.0
    label: str  # e.g. "Excellent", "Good", "Fair", "Poor"
    description: str


@dataclass
class Rubric:
    """An anchored rubric for scoring a single dimension.

    Each dimension has 4–5 anchored levels that describe what performance
    looks like at each score point.
    """

    dimension: str
    criteria: list[RubricCriterion] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Format the rubric as a prompt string for the LLM."""
        lines = [f"## {self.dimension}"]
        for c in self.criteria:
            lines.append(f"- {c.score:.1f} ({c.label}): {c.description}")
        return "\n".join(lines)


@dataclass
class JudgeResult:
    """Structured output from the LLM judge."""

    dimension_scores: dict[str, float] = field(default_factory=dict)
    justifications: dict[str, str] = field(default_factory=dict)
    raw_response: str = ""
    model_used: str = ""
    error: str | None = None


# =============================================================================
# Default rubrics for optical design evaluation
# =============================================================================

DEFAULT_RUBRICS: list[Rubric] = [
    Rubric(
        dimension="optical_accuracy",
        criteria=[
            RubricCriterion(1.0, "Excellent", "All optical metrics meet or exceed targets"),
            RubricCriterion(0.75, "Good", "Most optical metrics meet targets, minor deviations"),
            RubricCriterion(
                0.5, "Fair", "Some metrics meet targets, significant deviations in others"
            ),
            RubricCriterion(0.25, "Poor", "Few metrics meet targets, major errors"),
            RubricCriterion(0.0, "Fail", "No metrics meet targets or output is empty"),
        ],
    ),
    Rubric(
        dimension="metric_correctness",
        criteria=[
            RubricCriterion(1.0, "Excellent", "All metrics correctly computed and normalized"),
            RubricCriterion(0.75, "Good", "Minor calculation issues, mostly correct"),
            RubricCriterion(0.5, "Fair", "Several calculation errors present"),
            RubricCriterion(0.25, "Poor", "Majority of calculations are wrong"),
            RubricCriterion(0.0, "Fail", "No correct calculations or output missing"),
        ],
    ),
    Rubric(
        dimension="output_completeness",
        criteria=[
            RubricCriterion(
                1.0, "Complete", "All requested fields present with substantive content"
            ),
            RubricCriterion(0.75, "Mostly", "Most fields present, some minor omissions"),
            RubricCriterion(0.5, "Partial", "Several fields missing or placeholder content"),
            RubricCriterion(0.25, "Incomplete", "Most fields missing or trivial content"),
            RubricCriterion(0.0, "Empty", "Output is empty or unusable"),
        ],
    ),
    Rubric(
        dimension="citation_accuracy",
        criteria=[
            RubricCriterion(
                1.0, "Accurate", "All citations are real, relevant, and correctly attributed"
            ),
            RubricCriterion(0.75, "Mostly", "Minor citation errors, mostly correct"),
            RubricCriterion(0.5, "Mixed", "Some real citations mixed with questionable ones"),
            RubricCriterion(0.25, "Poor", "Most citations are incorrect or hallucinated"),
            RubricCriterion(0.0, "Hallucinated", "Citations appear fabricated or non-existent"),
        ],
    ),
    Rubric(
        dimension="reasoning_quality",
        criteria=[
            RubricCriterion(
                1.0, "Excellent", "Clear, logical, well-structured reasoning with evidence"
            ),
            RubricCriterion(0.75, "Good", "Mostly clear reasoning, minor gaps"),
            RubricCriterion(0.5, "Fair", "Some reasoning present but with logical gaps"),
            RubricCriterion(0.25, "Poor", "Minimal reasoning, largely unsupported claims"),
            RubricCriterion(0.0, "None", "No reasoning or explanation provided"),
        ],
    ),
    Rubric(
        dimension="robustness",
        criteria=[
            RubricCriterion(
                1.0, "Robust", "Handles all edge cases and boundary conditions correctly"
            ),
            RubricCriterion(0.75, "Good", "Handles common edge cases, some gaps"),
            RubricCriterion(0.5, "Fair", "Basic cases work, edge cases fail"),
            RubricCriterion(0.25, "Fragile", "Fails on most non-standard inputs"),
            RubricCriterion(0.0, "Brittle", "Fails on standard inputs"),
        ],
    ),
]


# =============================================================================
# Judge Prompt Builder
# =============================================================================


class JudgePromptBuilder:
    """Builds structured prompts for LLM-based evaluation judges."""

    @staticmethod
    def build(
        task_description: str,
        predicted_output: str,
        expected_output: str | None = None,
        rubrics: list[Rubric] | None = None,
    ) -> str:
        """Build a judge prompt asking the LLM to evaluate output quality.

        Args:
            task_description: Description of the task the agent was asked to perform.
            predicted_output: The agent's output to evaluate.
            expected_output: Optional ground-truth / expected output for comparison.
            rubrics: Scoring rubrics for each dimension (defaults to ``DEFAULT_RUBRICS``).

        Returns:
            A prompt string ready to send to an LLM.
        """
        rubrics = rubrics or DEFAULT_RUBRICS

        dim_names = [r.dimension for r in rubrics]

        sections = [
            "# Evaluation Judge",
            "",
            "You are an expert evaluator. Score the agent's "
            "output below on the specified dimensions.",
            "",
            "## Task",
            task_description,
            "",
            "## Agent Output",
            predicted_output,
            "",
        ]

        if expected_output:
            sections.extend(["## Expected Output (Ground Truth)", expected_output, ""])

        sections.append("## Scoring Rubrics")
        sections.append("")

        for rubric in rubrics:
            sections.append(rubric.to_prompt_block())
            sections.append("")

        sections.extend(
            [
                "## Instructions",
                "- Score each dimension on a scale of 0.0 to 1.0",
                f"- Dimensions to score: {', '.join(dim_names)}",
                "- Provide a brief justification for each score (1–2 sentences)",
                "- Return ONLY valid JSON with this exact structure:",
                "  {",
                '    "dimension_scores": {',
                '      "optical_accuracy": 0.0,',
                '      ...',
                "    },",
                '    "justifications": {',
                '      "optical_accuracy": "...",',
                "      ...",
                "    }",
                "  }",
                "- Do NOT include markdown fences, code blocks, or any text outside the JSON.",
            ]
        )

        return "\n".join(sections)

    @staticmethod
    def parse_response(raw: str) -> JudgeResult:
        """Parse the LLM's response into a ``JudgeResult``.

        Handles both plain JSON and JSON-within-markdown-fences.
        """
        text = raw.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove opening fence (```json, ```, etc.)
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return JudgeResult(error=f"Failed to parse judge response: {e}")

        dim_scores = data.get("dimension_scores", {})
        justifications = data.get("justifications", {})

        # Normalise scores to [0.0, 1.0]
        for k in list(dim_scores):
            v = dim_scores[k]
            if isinstance(v, (int, float)):
                dim_scores[k] = max(0.0, min(1.0, float(v)))
            else:
                dim_scores[k] = 0.0

        return JudgeResult(
            dimension_scores=dim_scores,
            justifications=justifications,
            raw_response=raw,
        )


# =============================================================================
# LLM Judge
# =============================================================================


class LLMJudge:
    """An LLM-based evaluation judge.

    Uses a language model to score agent outputs across multiple
    quality dimensions using structured rubrics. Mirrors PluginEval's
    Layer 2 evaluation approach.

    The judge can work in two modes:

    * **Live mode**: Calls an LLM via the existing ``create_agent()``
      factory (requires API keys).
    * **Mock mode**: Uses a provided callable for testing without API keys.

    Usage::

        from src.core.llm_judge import LLMJudge, DEFAULT_RUBRICS

        judge = LLMJudge()
        result = await judge.evaluate(
            task_description="Design a lens with ...",
            predicted_output=agent_output,
            expected_output=gold_standard,
        )
        print(result.dimension_scores)
    """

    def __init__(
        self,
        rubrics: list[Rubric] | None = None,
        llm_callable: Any = None,
    ):
        """Initialise the judge.

        Args:
            rubrics: Scoring rubrics. Defaults to ``DEFAULT_RUBRICS``.
            llm_callable: An async callable ``(prompt: str) -> str`` used to
                query the LLM. If ``None``, the judge stores the prompt but
                cannot execute live scoring (useful for offline or testing).
        """
        self.rubrics = rubrics or DEFAULT_RUBRICS
        self._llm = llm_callable

    async def evaluate(
        self,
        task_description: str,
        predicted_output: str,
        expected_output: str | None = None,
    ) -> JudgeResult:
        """Evaluate agent output against rubrics using an LLM judge.

        Returns a ``JudgeResult`` with per-dimension scores and justifications.
        If no ``llm_callable`` was provided, the result will contain the built
        prompt in ``error`` and empty scores.
        """
        prompt = JudgePromptBuilder.build(
            task_description=task_description,
            predicted_output=predicted_output,
            expected_output=expected_output,
            rubrics=self.rubrics,
        )

        if self._llm is None:
            return JudgeResult(
                error=(
                    "No LLM callable configured. Use llm_callable parameter "
                    "or set up via create_judge_from_config()."
                ),
            )

        try:
            raw = await self._llm(prompt)
        except Exception as e:
            logger.exception("LLM judge call failed")
            return JudgeResult(error=f"LLM call failed: {e}")

        result = JudgePromptBuilder.parse_response(raw)
        result.model_used = getattr(self._llm, "__name__", "unknown")
        return result


# =============================================================================
# Factory helpers
# =============================================================================


def create_judge_from_config(
    config: dict[str, Any],
    llm_callable: Any = None,
) -> LLMJudge:
    """Create an ``LLMJudge`` from a configuration dictionary.

    The config may optionally specify custom rubric criteria to override
    the defaults::

        {
            "dimensions": [
                {
                    "name": "optical_accuracy",
                    "criteria": [
                        {"score": 1.0, "label": "Perfect", "description": "..."},
                        ...
                    ]
                }
            ]
        }
    """
    rubrics = list(DEFAULT_RUBRICS)
    custom_dims = config.get("dimensions")

    if custom_dims:
        for cd in custom_dims:
            name = cd.get("name", "")
            criteria_list = cd.get("criteria", [])
            criteria = [
                RubricCriterion(
                    score=c.get("score", 0.0),
                    label=c.get("label", ""),
                    description=c.get("description", ""),
                )
                for c in criteria_list
            ]
            if criteria:
                # Replace matching dimension or append
                found = False
                for i, r in enumerate(rubrics):
                    if r.dimension == name:
                        rubrics[i] = Rubric(dimension=name, criteria=criteria)
                        found = True
                        break
                if not found:
                    rubrics.append(Rubric(dimension=name, criteria=criteria))

    return LLMJudge(rubrics=rubrics, llm_callable=llm_callable)
