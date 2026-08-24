"""
Tests for the LLM Judge evaluator (``llm_judge.py``).

Inspired by Vercel Labs benchmark-agents / PluginEval "Layer 2 — LLM Judge"
evaluation approach.
"""

from __future__ import annotations

import pytest

from src.core.llm_judge import (
    DEFAULT_RUBRICS,
    JudgePromptBuilder,
    JudgeResult,
    LLMJudge,
    Rubric,
    RubricCriterion,
    create_judge_from_config,
)

# =============================================================================
# Classes
# =============================================================================


class TestRubric:
    """Rubric construction and formatting."""

    def test_to_prompt_block(self):
        rubric = Rubric(
            dimension="test_dim",
            criteria=[
                RubricCriterion(1.0, "Great", "Perfect score"),
                RubricCriterion(0.0, "Bad", "Terrible score"),
            ],
        )
        block = rubric.to_prompt_block()
        assert "## test_dim" in block
        assert "1.0 (Great)" in block
        assert "0.0 (Bad)" in block
        assert "Perfect score" in block
        assert "Terrible score" in block


class TestJudgePromptBuilder:
    """Prompt construction and response parsing."""

    def test_build_contains_task_and_output(self):
        prompt = JudgePromptBuilder.build(
            task_description="Design a lens",
            predicted_output="focal length: 50mm",
        )
        assert "Design a lens" in prompt
        assert "focal length: 50mm" in prompt

    def test_build_with_expected_output(self):
        prompt = JudgePromptBuilder.build(
            task_description="Design a lens",
            predicted_output="focal length: 50mm",
            expected_output="focal length: 50mm, aperture: f/1.8",
        )
        assert "Expected Output (Ground Truth)" in prompt

    def test_build_contains_rubrics(self):
        prompt = JudgePromptBuilder.build(
            task_description="Test",
            predicted_output="output",
        )
        for rubric in DEFAULT_RUBRICS:
            assert rubric.dimension in prompt

    def test_build_contains_json_instructions(self):
        prompt = JudgePromptBuilder.build(
            task_description="Test",
            predicted_output="output",
        )
        assert "dimension_scores" in prompt
        assert "justifications" in prompt
        assert "Return ONLY valid JSON" in prompt

    def test_parse_response_plain_json(self):
        raw = (
            '{"dimension_scores": {"optical_accuracy": 0.8}, '
            '"justifications": {"optical_accuracy": "Good"}}'
        )
        result = JudgePromptBuilder.parse_response(raw)
        assert abs(result.dimension_scores["optical_accuracy"] - 0.8) < 1e-6
        assert result.justifications["optical_accuracy"] == "Good"

    def test_parse_response_with_fences(self):
        raw = """```json
{"dimension_scores": {"optical_accuracy": 0.9}, "justifications": {"optical_accuracy": "Great"}}
```"""
        result = JudgePromptBuilder.parse_response(raw)
        assert abs(result.dimension_scores["optical_accuracy"] - 0.9) < 1e-6

    def test_parse_response_with_code_block(self):
        raw = """```
{"dimension_scores": {"metric_correctness": 0.75}, "justifications": {"metric_correctness": "OK"}}
```"""
        result = JudgePromptBuilder.parse_response(raw)
        assert abs(result.dimension_scores["metric_correctness"] - 0.75) < 1e-6

    def test_parse_response_invalid_json(self):
        raw = "not json at all"
        result = JudgePromptBuilder.parse_response(raw)
        assert result.error is not None
        assert "Failed to parse" in result.error

    def test_parse_response_clamps_scores(self):
        raw = '{"dimension_scores": {"optical_accuracy": 2.5, "bad": -1.0}}'
        result = JudgePromptBuilder.parse_response(raw)
        assert result.dimension_scores["optical_accuracy"] == 1.0
        assert result.dimension_scores["bad"] == 0.0

    def test_parse_response_empty(self):
        raw = ""
        result = JudgePromptBuilder.parse_response(raw)
        assert result.error is not None


class TestLLMJudge:
    """LLM Judge logic."""

    @pytest.mark.asyncio
    async def test_evaluate_no_llm_callable(self):
        judge = LLMJudge()
        result = await judge.evaluate(
            task_description="Test",
            predicted_output="output",
        )
        # Should return an error result
        assert isinstance(result, JudgeResult)
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_evaluate_with_mock_llm(self):
        async def mock_llm(prompt: str) -> str:
            return (
                '{"dimension_scores": {"optical_accuracy": 0.9}, '
                '"justifications": {"optical_accuracy": "Good job"}}'
            )

        judge = LLMJudge(llm_callable=mock_llm)
        result = await judge.evaluate(
            task_description="Design a lens",
            predicted_output="MTF: 0.85",
        )
        assert result.error is None
        assert abs(result.dimension_scores["optical_accuracy"] - 0.9) < 1e-6
        assert result.justifications["optical_accuracy"] == "Good job"

    @pytest.mark.asyncio
    async def test_evaluate_with_expected_output(self):
        async def mock_llm(prompt: str) -> str:
            return '{"dimension_scores": {"optical_accuracy": 0.7}, "justifications": {}}'

        judge = LLMJudge(llm_callable=mock_llm)
        result = await judge.evaluate(
            task_description="Test",
            predicted_output="output",
            expected_output="expected",
        )
        assert result.error is None

    @pytest.mark.asyncio
    async def test_evaluate_llm_raises(self):
        async def broken_llm(prompt: str) -> str:
            raise RuntimeError("API failure")

        judge = LLMJudge(llm_callable=broken_llm)
        result = await judge.evaluate(
            task_description="Test",
            predicted_output="output",
        )
        assert result.error is not None
        assert "API failure" in result.error

    @pytest.mark.asyncio
    async def test_custom_rubrics(self):
        custom = [
            Rubric(
                dimension="custom_dim",
                criteria=[
                    RubricCriterion(1.0, "Great", "Perfect"),
                    RubricCriterion(0.0, "Bad", "Terrible"),
                ],
            )
        ]

        async def mock_llm(prompt: str) -> str:
            assert "custom_dim" in prompt
            return '{"dimension_scores": {"custom_dim": 1.0}, "justifications": {}}'

        judge = LLMJudge(rubrics=custom, llm_callable=mock_llm)
        result = await judge.evaluate(
            task_description="Test",
            predicted_output="output",
        )
        assert abs(result.dimension_scores["custom_dim"] - 1.0) < 1e-6


class TestCreateJudgeFromConfig:
    """Factory helper tests."""

    def test_default_config(self):
        judge = create_judge_from_config({})
        assert len(judge.rubrics) == len(DEFAULT_RUBRICS)

    def test_custom_dimension_overrides(self):
        config = {
            "dimensions": [
                {
                    "name": "optical_accuracy",
                    "criteria": [
                        {"score": 1.0, "label": "Perfect", "description": "Custom"},
                    ],
                }
            ]
        }
        judge = create_judge_from_config(config)
        target = [r for r in judge.rubrics if r.dimension == "optical_accuracy"][0]
        assert target.criteria[0].description == "Custom"
        assert len(target.criteria) == 1

    def test_new_dimension_appended(self):
        config = {
            "dimensions": [
                {
                    "name": "new_dim",
                    "criteria": [
                        {"score": 0.5, "label": "OK", "description": "New"},
                    ],
                }
            ]
        }
        judge = create_judge_from_config(config)
        names = [r.dimension for r in judge.rubrics]
        assert "new_dim" in names
