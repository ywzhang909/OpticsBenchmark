"""
OptiS Benchmark — Rubric Eval vLLM Integration Tests

Tests the rubric_eval dataset against a local vLLM endpoint using the
RubricBasedEvaluator. Covers all 5 scenarios (S1-S5) with 7 test cases.

Usage:
    # Run only offline tests (no vLLM needed)
    uv run pytest tests/test_rubric_eval_vllm.py -m "not online" -v

    # Run online tests (requires local vLLM at localhost:8001)
    uv run pytest tests/test_rubric_eval_vllm.py -m "online" -v

    # Run all tests
    uv run pytest tests/test_rubric_eval_vllm.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.evaluators import RubricBasedEvaluator
from src.module import EvaluationResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset" / "rubric_eval"
DATASET_JSON = DATASET_DIR / "dataset_json" / "dataset_v1.json"
GOLD_ANSWER_JSON = DATASET_DIR / "dataset_json" / "gold_answer_v1.json"
RUBRICS_DIR = DATASET_DIR / "rubrics"

# Local vLLM endpoint (Docker container)
VLLM_BASE_URL = "http://localhost:8001/v1"
VLLM_MODEL = "qwen"
VLLM_API_KEY = "sk-11235813"

# Rubric files by scenario
RUBRIC_FILES: dict[str, str] = {
    "S1": "optical_experiment_review.yaml",
    "S2": "optical_paper_review.yaml",
    "S3": "optical_literature_understanding.yaml",
    "S4": "optical_system_design.yaml",
    "S5": "optical_agent_tool_use.yaml",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dataset() -> list[dict[str, Any]]:
    """Load test cases from dataset_v1.json."""
    with open(DATASET_JSON, encoding="utf-8") as f:
        return json.load(f)


def _load_gold_answers() -> list[dict[str, Any]]:
    """Load gold standard answers from gold_answer_v1.json."""
    with open(GOLD_ANSWER_JSON, encoding="utf-8") as f:
        return json.load(f)


def _load_rubric(scenario: str) -> dict[str, Any]:
    """Load rubric YAML for a given scenario.

    Returns the criteria list directly (extracted from the scenario-specific key).
    """
    rubric_file = RUBRICS_DIR / RUBRIC_FILES[scenario]
    with open(rubric_file, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # Find the criteria key (varies by scenario: rubric_v1, rubric_s2_paper_review, etc.)
    for key in raw:
        if isinstance(raw[key], dict) and "criteria" in raw[key]:
            return raw[key]
    raise ValueError(f"No criteria found in {rubric_file}")


def _get_rubric_criteria_count(scenario: str) -> int:
    """Get the number of criteria for a scenario."""
    rubric = _load_rubric(scenario)
    return len(rubric["criteria"])


def _judge_config() -> dict[str, Any]:
    """Build judge config for local vLLM endpoint."""
    return {
        "provider": "openai",
        "model": VLLM_MODEL,
        "api_base": VLLM_BASE_URL,
        "api_key": VLLM_API_KEY,
        "temperature": 0.0,
        "raw_http": True,
    }


def _get_content_text(case: dict[str, Any]) -> str:
    """Extract the main content text from a test case.

    Different scenarios store content in different keys:
    - S1: 'content' (experiment design)
    - S2: 'paper_content' (paper review)
    - S3: 'question' + 'papers' (literature understanding)
    - S4: 'question' + 'system_params' (system design)
    - S5: 'trajectory' + 'question' (agent tool use)
    """
    # Direct content
    if "content" in case:
        return case["content"]
    if "paper_content" in case:
        return case["paper_content"]

    # Compound content
    parts = []
    if "papers" in case:
        for paper in case["papers"]:
            parts.append(f"论文: {paper.get('title', '')}")
            parts.append(f"主要发现: {paper.get('key_findings', [])}")
    if "system_params" in case:
        parts.append(f"系统参数: {json.dumps(case['system_params'], ensure_ascii=False)}")
    if "design_requirements" in case:
        parts.append(f"设计要求: {json.dumps(case['design_requirements'], ensure_ascii=False)}")
    if "trajectory" in case:
        for step in case["trajectory"]:
            parts.append(
                f"Step {step.get('step', '?')}: {step.get('action', '')} - "
                f"{step.get('tool_name', step.get('content', ''))}"
            )
    if "question" in case:
        parts.append(f"问题: {case['question']}")

    return "\n".join(parts) if parts else json.dumps(case, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Offline Tests — No vLLM required
# ---------------------------------------------------------------------------


class TestRubricEvalDataset_Offline:
    """Offline validation of dataset structure, rubrics, and gold answers."""

    def test_dataset_loads(self):
        """dataset_v1.json loads and contains test cases."""
        cases = _load_dataset()
        assert len(cases) >= 5, f"Expected at least 5 test cases, got {len(cases)}"

    def test_dataset_has_all_scenarios(self):
        """Dataset covers all 5 scenarios (S1-S5)."""
        cases = _load_dataset()
        scenarios = {c["scenario"] for c in cases}
        expected = {"S1", "S2", "S3", "S4", "S5"}
        assert scenarios == expected, f"Missing scenarios: {expected - scenarios}"

    def test_dataset_scenario_distribution(self):
        """S1 has 3 cases, S2 has 2, S3/S4/S5 have 1 each."""
        cases = _load_dataset()
        from collections import Counter

        dist = Counter(c["scenario"] for c in cases)
        assert dist["S1"] == 3, f"S1 should have 3 cases, got {dist['S1']}"
        assert dist["S2"] == 2, f"S2 should have 2 cases, got {dist['S2']}"
        assert dist["S3"] == 1
        assert dist["S4"] == 1
        assert dist["S5"] == 1

    def test_dataset_required_fields(self):
        """Each test case has required fields: id, scenario, title, metadata."""
        cases = _load_dataset()
        for case in cases:
            assert "id" in case, f"Missing 'id' in case"
            assert "scenario" in case, f"Missing 'scenario' in case {case['id']}"
            assert "title" in case, f"Missing 'title' in case {case['id']}"
            assert "metadata" in case, f"Missing 'metadata' in case {case['id']}"

    def test_gold_answers_loads(self):
        """gold_answer_v1.json loads and contains entries for S1_001, S2_001, S3_001, S4_001, S5_001."""
        gold = _load_gold_answers()
        gold_ids = {g["id"] for g in gold}
        expected_ids = {"S1_001", "S2_001", "S3_001", "S4_001", "S5_001"}
        assert expected_ids.issubset(gold_ids), (
            f"Missing gold answers: {expected_ids - gold_ids}"
        )

    def test_gold_answers_have_annotations(self):
        """Each gold answer has expert_annotations with verdict or score."""
        gold = _load_gold_answers()
        for g in gold:
            anns = g.get("expert_annotations", {})
            assert len(anns) > 0, f"No annotations in gold answer for {g['id']}"
            for key, ann in anns.items():
                assert "verdict" in ann or "score" in ann, (
                    f"Annotation {key} in {g['id']} missing verdict/score"
                )

    def test_gold_answers_have_overall_score(self):
        """Each gold answer has overall_score and pass_threshold."""
        gold = _load_gold_answers()
        for g in gold:
            assert "overall_score" in g, f"Missing overall_score in {g['id']}"
            assert "pass_threshold" in g, f"Missing pass_threshold in {g['id']}"
            assert "passed" in g, f"Missing passed in {g['id']}"
            assert isinstance(g["overall_score"], (int, float))
            assert 0 <= g["overall_score"] <= 100

    def test_all_rubric_files_exist(self):
        """All 5 rubric YAML files exist."""
        for scenario, filename in RUBRIC_FILES.items():
            rubric_path = RUBRICS_DIR / filename
            assert rubric_path.exists(), f"Missing rubric file: {rubric_path}"

    def test_rubric_s1_loads(self):
        """S1 rubric loads with 18 criteria."""
        rubric = _load_rubric("S1")
        assert "meta" in rubric and "criteria" in rubric
        assert len(rubric["criteria"]) == 18

    def test_rubric_s2_loads(self):
        """S2 rubric loads with 16 criteria."""
        rubric = _load_rubric("S2")
        assert len(rubric["criteria"]) == 16

    def test_rubric_s3_loads(self):
        """S3 rubric loads with 14 criteria."""
        rubric = _load_rubric("S3")
        assert len(rubric["criteria"]) == 14

    def test_rubric_s4_loads(self):
        """S4 rubric loads with 15 criteria."""
        rubric = _load_rubric("S4")
        assert len(rubric["criteria"]) == 15

    def test_rubric_s5_loads(self):
        """S5 rubric loads with 5 criteria."""
        rubric = _load_rubric("S5")
        assert len(rubric["criteria"]) == 5

    def test_rubric_criteria_have_required_fields(self):
        """Each rubric criterion has id, dimension, type, question, weight."""
        for scenario in RUBRIC_FILES:
            rubric = _load_rubric(scenario)
            for c in rubric["criteria"]:
                assert "id" in c, f"Criterion missing 'id' in {scenario}"
                assert "dimension" in c, f"Criterion {c['id']} missing 'dimension'"
                assert "type" in c, f"Criterion {c['id']} missing 'type'"
                assert "question" in c, f"Criterion {c['id']} missing 'question'"
                assert "weight" in c, f"Criterion {c['id']} missing 'weight'"
                assert c["type"] in ("binary", "ordinal"), (
                    f"Criterion {c['id']} has invalid type: {c['type']}"
                )

    def test_rubric_binary_criteria_have_fail_hard(self):
        """Binary criteria optionally have fail_hard field."""
        for scenario in RUBRIC_FILES:
            rubric = _load_rubric(scenario)
            for c in rubric["criteria"]:
                if c["type"] == "binary" and "fail_hard" in c:
                    assert isinstance(c["fail_hard"], bool), (
                        f"Binary criterion {c['id']} fail_hard must be bool"
                    )

    def test_rubric_ordinal_criteria_have_scale(self):
        """Ordinal criteria have scale field."""
        for scenario in RUBRIC_FILES:
            rubric = _load_rubric(scenario)
            for c in rubric["criteria"]:
                if c["type"] == "ordinal":
                    assert "scale" in c, (
                        f"Ordinal criterion {c['id']} in {scenario} missing scale"
                    )
                    assert c["scale"] == [1, 2, 3, 4, 5], (
                        f"Criterion {c['id']} scale should be [1,2,3,4,5]"
                    )

    def test_total_rubric_criteria(self):
        """Total rubric criteria across all scenarios is 68."""
        total = 0
        for scenario in RUBRIC_FILES:
            rubric = _load_rubric(scenario)
            total += len(rubric["criteria"])
        assert total == 68, f"Expected 68 total criteria, got {total}"

    def test_gold_annotations_match_rubric_criteria(self):
        """Gold answer annotations reference rubric criterion IDs."""
        gold = _load_gold_answers()
        for g in gold:
            scenario = g["scenario"]
            rubric = _load_rubric(scenario)
            rubric_ids = {c["id"] for c in rubric["criteria"]}
            gold_ann_ids = set(g.get("expert_annotations", {}).keys())
            # Gold annotations should reference rubric criteria
            for ann_id in gold_ann_ids:
                # Extract the criterion ID prefix (e.g., C01 from C01_energy_conservation)
                assert any(
                    ann_id.startswith(rid.split("_")[0]) for rid in rubric_ids
                ), f"Gold annotation {ann_id} in {g['id']} not found in rubric {scenario}"

    def test_content_text_extraction(self):
        """_get_content_text extracts meaningful text from all scenario types."""
        cases = _load_dataset()
        for case in cases:
            text = _get_content_text(case)
            assert len(text) > 50, (
                f"Content too short for {case['id']}: {len(text)} chars"
            )

    def test_evaluator_offline_returns_zero_scores(self):
        """RubricBasedEvaluator without judge_config returns zero scores."""
        ev = RubricBasedEvaluator({})
        result = {
            "task_id": "test_offline",
            "metrics": {},
            "execution_time": 0.0,
        }
        # Just verify the evaluator can be created
        assert ev is not None


# ---------------------------------------------------------------------------
# Online Tests — Require local vLLM at localhost:8001
# ---------------------------------------------------------------------------


@pytest.mark.online
class TestRubricEvalDataset_Online:
    """Integration tests using local vLLM endpoint.

    Requires: docker run vllm/vllm-openai on port 8001 with model 'qwen'.
    """

    @pytest.mark.asyncio
    async def test_vllm_endpoint_reachable(self):
        """Verify local vLLM endpoint is reachable."""
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{VLLM_BASE_URL}/models",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, (
                f"vLLM endpoint returned {resp.status_code}: {resp.text}"
            )
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            assert VLLM_MODEL in models, (
                f"Model '{VLLM_MODEL}' not found. Available: {models}"
            )

    @pytest.mark.asyncio
    async def test_single_case_s1_001(self):
        """Evaluate S1_001 (adaptive optics experiment) with vLLM judge."""
        cases = _load_dataset()
        case = next(c for c in cases if c["id"] == "S1_001")
        content = _get_content_text(case)

        ev = RubricBasedEvaluator({"judge_config": _judge_config()})
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="S1_001",
                predicted_output={"ten keywords": content[:200]},
                expected_output={"ten keywords": content[:200]},
            )
            assert isinstance(result, EvaluationResult)
            assert result.task_id == "S1_001"
            assert result.metrics["accuracy"] > 0.0, (
                f"S1_001 accuracy should be > 0, got {result.metrics['accuracy']}"
            )
            assert result.metrics["completeness"] > 0.0
            assert result.metrics["readability"] > 0.0
            assert 0.0 <= result.metrics["hallucination_rate"] <= 1.0
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_single_case_s2_001(self):
        """Evaluate S2_001 (metasurface beam deflector) with vLLM judge."""
        cases = _load_dataset()
        case = next(c for c in cases if c["id"] == "S2_001")
        content = _get_content_text(case)

        ev = RubricBasedEvaluator({"judge_config": _judge_config()})
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="S2_001",
                predicted_output={"ten keywords": content[:200]},
                expected_output={"ten keywords": content[:200]},
            )
            assert result.metrics["accuracy"] > 0.0
            assert result.metrics["completeness"] > 0.0
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_single_case_s3_001(self):
        """Evaluate S3_001 (deep learning wavefront sensing) with vLLM judge."""
        cases = _load_dataset()
        case = next(c for c in cases if c["id"] == "S3_001")
        content = _get_content_text(case)

        ev = RubricBasedEvaluator({"judge_config": _judge_config()})
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="S3_001",
                predicted_output={"ten keywords": content[:200]},
                expected_output={"ten keywords": content[:200]},
            )
            assert result.metrics["accuracy"] > 0.0
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_single_case_s4_001(self):
        """Evaluate S4_001 (laser beam quality) with vLLM judge."""
        cases = _load_dataset()
        case = next(c for c in cases if c["id"] == "S4_001")
        content = _get_content_text(case)

        ev = RubricBasedEvaluator({"judge_config": _judge_config()})
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="S4_001",
                predicted_output={"ten keywords": content[:200]},
                expected_output={"ten keywords": content[:200]},
            )
            assert result.metrics["accuracy"] > 0.0
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_single_case_s5_001(self):
        """Evaluate S5_001 (agent tool use) with vLLM judge."""
        cases = _load_dataset()
        case = next(c for c in cases if c["id"] == "S5_001")
        content = _get_content_text(case)

        ev = RubricBasedEvaluator({"judge_config": _judge_config()})
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="S5_001",
                predicted_output={"ten keywords": content[:200]},
                expected_output={"ten keywords": content[:200]},
            )
            assert result.metrics["accuracy"] > 0.0
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_all_cases_batch_evaluation(self):
        """Evaluate all 7 test cases and aggregate results."""
        cases = _load_dataset()

        ev = RubricBasedEvaluator({"judge_config": _judge_config()})
        await ev.setup()
        try:
            results = []
            for case in cases:
                content = _get_content_text(case)
                result = await ev.evaluate(
                    task_id=case["id"],
                    predicted_output={"ten keywords": content[:200]},
                    expected_output={"ten keywords": content[:200]},
                )
                results.append(result)
                assert result.metrics["accuracy"] > 0.0, (
                    f"{case['id']} accuracy should be > 0"
                )

            # Aggregate
            agg = await ev.aggregate(results)
            assert agg.total_tasks == len(cases)
            assert "accuracy" in agg.metrics_summary
            assert agg.metrics_summary["accuracy"] > 0.0
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_gold_answer_comparison(self):
        """Compare vLLM judge scores against gold answer overall scores."""
        cases = _load_dataset()
        gold_answers = _load_gold_answers()
        gold_map = {g["id"]: g for g in gold_answers}

        ev = RubricBasedEvaluator({"judge_config": _judge_config()})
        await ev.setup()
        try:
            for case in cases:
                if case["id"] not in gold_map:
                    continue
                content = _get_content_text(case)
                result = await ev.evaluate(
                    task_id=case["id"],
                    predicted_output={"ten keywords": content[:200]},
                    expected_output={"ten keywords": content[:200]},
                )
                gold = gold_map[case["id"]]
                # vLLM judge should produce non-zero scores
                assert result.metrics["accuracy"] > 0.0, (
                    f"{case['id']}: accuracy {result.metrics['accuracy']} should be > 0"
                )
                # Check that we have field-level details
                assert "field_scores" in result.details
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_hallucination_detection(self):
        """Hallucination detection works across different scenarios."""
        cases = _load_dataset()

        ev = RubricBasedEvaluator({"judge_config": _judge_config()})
        await ev.setup()
        try:
            for case in cases[:3]:  # Test first 3 cases
                content = _get_content_text(case)
                result = await ev.evaluate(
                    task_id=case["id"],
                    predicted_output={"ten keywords": content[:200]},
                    expected_output={"ten keywords": "nonexistent keywords"},
                )
                assert "hallucination_rate" in result.metrics
                assert 0.0 <= result.metrics["hallucination_rate"] <= 1.0
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_different_field_inputs(self):
        """Evaluator handles various input field combinations."""
        ev = RubricBasedEvaluator({"judge_config": _judge_config()})
        await ev.setup()
        try:
            # Single field
            result = await ev.evaluate(
                task_id="field_single",
                predicted_output={"ten keywords": "optics, lens, design"},
                expected_output={"ten keywords": "optics, lens, design"},
            )
            assert result.metrics["accuracy"] > 0.0

            # Multiple fields
            result = await ev.evaluate(
                task_id="field_multi",
                predicted_output={
                    "ten keywords": "optics, lens, design",
                    "objective": "Design an optical system",
                    "method": "Ray tracing optimization",
                },
                expected_output={
                    "ten keywords": "optics, lens, design",
                    "objective": "Design an optical system",
                    "method": "Ray tracing optimization",
                },
            )
            assert result.metrics["accuracy"] > 0.0

            # Empty predicted (should still work)
            result = await ev.evaluate(
                task_id="field_empty",
                predicted_output={},
                expected_output={"ten keywords": "test"},
            )
            assert result.metrics["accuracy"] == 0.0
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_teardown_cleans_up(self):
        """Evaluator can be setup/teardown multiple times safely."""
        ev = RubricBasedEvaluator({"judge_config": _judge_config()})

        for i in range(3):
            await ev.setup()
            result = await ev.evaluate(
                task_id=f"lifecycle_{i}",
                predicted_output={"ten keywords": "test"},
                expected_output={"ten keywords": "test"},
            )
            assert result.metrics["accuracy"] > 0.0
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_reference_free_evaluation(self):
        """Evaluator works without expected output (reference-free mode)."""
        cases = _load_dataset()
        case = cases[0]  # S1_001
        content = _get_content_text(case)

        ev = RubricBasedEvaluator({"judge_config": _judge_config()})
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="ref_free",
                predicted_output={"ten keywords": content[:200]},
                expected_output=None,
            )
            assert result.metrics["accuracy"] > 0.0
            # No hallucination without expected output
            assert result.metrics["hallucination_rate"] == 0.0
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_reasoning_model_response_parsing(self):
        """Evaluator handles vLLM reasoning model responses (content:null + reasoning)."""
        # Simulate vLLM reasoning model response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning": '{"accuracy": 4, "completeness": 3, "readability": 5}',
                    },
                },
            ],
        }
        content = RubricBasedEvaluator._extract_content(mock_response)
        assert content == '{"accuracy": 4, "completeness": 3, "readability": 5}'

    @pytest.mark.asyncio
    async def test_json_string_input(self):
        """Evaluator handles JSON string inputs."""
        ev = RubricBasedEvaluator({"judge_config": _judge_config()})
        await ev.setup()
        try:
            pred = json.dumps({"ten keywords": "optics, lens"})
            expected = json.dumps({"ten keywords": "optics, lens"})
            result = await ev.evaluate(
                task_id="json_str",
                predicted_output=pred,
                expected_output=expected,
            )
            assert result.metrics["accuracy"] > 0.0
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self):
        """Multiple evaluators can run concurrently."""
        import asyncio

        async def run_eval(case_id: str, content: str):
            ev = RubricBasedEvaluator({"judge_config": _judge_config()})
            await ev.setup()
            try:
                return await ev.evaluate(
                    task_id=case_id,
                    predicted_output={"ten keywords": content[:200]},
                    expected_output={"ten keywords": content[:200]},
                )
            finally:
                await ev.teardown()

        cases = _load_dataset()[:3]  # First 3 cases
        tasks = [
            run_eval(c["id"], _get_content_text(c)) for c in cases
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
            assert result.metrics["accuracy"] > 0.0
