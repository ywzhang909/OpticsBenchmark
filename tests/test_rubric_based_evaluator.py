"""
OptiS Benchmark — RubricBasedEvaluator Tests

Tests for the per-field rubric-based evaluator with hallucination detection.

* Offline tests  — pure unit tests, no external dependencies.
* Online tests   — integration test against a live vLLM endpoint
  (``--vllm-judge`` CLI option required).
"""

from __future__ import annotations

import json

import pytest

from src.evaluators import RubricBasedEvaluator
from src.module import EvaluationResult


# ===========================================================================
# Offline (unit) tests — no LLM callable needed
# ===========================================================================


class TestRubricBasedEvaluator_Offline:
    """Offline-mode tests — no ``judge_config`` provided."""

    @pytest.mark.asyncio
    async def test_offline_returns_zero_scores(self):
        """Without judge_config, all dimension scores are 0.0."""
        ev = RubricBasedEvaluator({})
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="offline_1",
                predicted_output={"ten keywords": "lens, optics, MTF"},
                expected_output={"ten keywords": "lens, optical design, MTF"},
            )
            assert isinstance(result, EvaluationResult)
            assert result.task_id == "offline_1"
            assert result.metrics["accuracy"] == 0.0
            assert result.metrics["completeness"] == 0.0
            assert result.metrics["readability"] == 0.0
            assert "hallucination_rate" in result.metrics
            assert "field_scores" in result.details
            assert "hallucination" in result.details
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_hallucination_detects_mismatch(self):
        """Hallucination detection flags items in predicted but not expected."""
        ev = RubricBasedEvaluator({})
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="hallu_1",
                predicted_output={
                    "ten keywords": "lens, optics, MTF, aberration",
                    "objective": "Design a lens",
                },
                expected_output={
                    "ten keywords": "lens, optical design, MTF",
                    "objective": "Design a lens",
                },
            )
            hallu = result.details.get("hallucination", {})
            assert hallu["total_checked_items"] >= 2
            hallu_fields = {h["field"] for h in hallu.get("hallucinated_items", [])}
            assert "keywords" in hallu_fields
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_no_hallucination_on_exact_match(self):
        """No hallucinations when predicted equals expected."""
        ev = RubricBasedEvaluator({})
        await ev.setup()
        try:
            pred = {"ten keywords": "lens, optics, MTF"}
            result = await ev.evaluate(
                task_id="no_hallu",
                predicted_output=pred,
                expected_output=pred,
            )
            hallu = result.details.get("hallucination", {})
            assert hallu.get("hallucinated_items", []) == []
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_field_skipping_when_missing(self):
        """Fields absent from predicted output are skipped without crash."""
        ev = RubricBasedEvaluator({})
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="partial",
                predicted_output={"ten keywords": "lens, optics"},
                expected_output={"ten keywords": "lens, optics", "objective": "test"},
            )
            # All 5 field entries present in field_scores
            assert len(result.details["field_scores"]) == 5
            # Skipped fields have 0.0
            assert (
                result.details["field_scores"]["innovation_points"]["accuracy"] == 0.0
            )
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_json_string_input_parsed(self):
        """Inputs as JSON strings are parsed automatically."""
        ev = RubricBasedEvaluator({})
        await ev.setup()
        try:
            pred = json.dumps({"ten keywords": "lens, optics"})
            expected = json.dumps({"ten keywords": "lens, optics"})
            result = await ev.evaluate(
                task_id="json_str",
                predicted_output=pred,
                expected_output=expected,
            )
            assert "accuracy" in result.metrics
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_empty_predicted_dict(self):
        """Completely empty predicted dict does not crash."""
        ev = RubricBasedEvaluator({})
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="empty",
                predicted_output={},
                expected_output={"ten keywords": "test"},
            )
            assert result.metrics["accuracy"] == 0.0
            assert result.metrics["hallucination_rate"] == 0.0
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_custom_field_map(self):
        """Custom field_map routes JSON keys correctly."""
        ev = RubricBasedEvaluator({
            "field_map": {"keywords": "my_keys"},
        })
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="custom_map",
                predicted_output={"my_keys": "lens, optics"},
                expected_output={"my_keys": "lens, optics"},
            )
            # Only keywords has data → avg accuracy = 0 because all other
            # fields get 0.0 and are excluded, then avg(0.0) = 0.0
            assert "accuracy" in result.metrics
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_aggregate_two_results(self):
        """Aggregation averages metrics and preserves per-task results."""
        ev = RubricBasedEvaluator({})
        await ev.setup()
        try:
            r1 = await ev.evaluate(
                task_id="agg_1",
                predicted_output={"ten keywords": "a, b, c"},
                expected_output={"ten keywords": "a, b, c"},
            )
            r2 = await ev.evaluate(
                task_id="agg_2",
                predicted_output={"ten keywords": "x, y, z"},
                expected_output={"ten keywords": "x, y"},
            )
            agg = await ev.aggregate([r1, r2])
            assert agg.total_tasks == 2
            assert "accuracy" in agg.metrics_summary
            assert "hallucination_rate" in agg.metrics_summary
            assert len(agg.per_task_results) == 2
        finally:
            await ev.teardown()

    # ---- _split_items unit tests ---------------------------------------

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("lens, optics, MTF", ["lens", "optics", "MTF"]),
            ("lens; optics; MTF", ["lens", "optics", "MTF"]),
            ("lens\noptics\nMTF", ["lens", "optics", "MTF"]),
            ("- lens\n- optics\n- MTF", ["lens", "optics", "MTF"]),
            ("1. lens\n2. optics\n3. MTF", ["lens", "optics", "MTF"]),
            ('["lens", "optics", "MTF"]', ["lens", "optics", "MTF"]),
            ("", []),
        ],
    )
    def test_split_items(self, text: str, expected: list[str]):
        """Item splitting handles all common formats."""
        assert RubricBasedEvaluator._split_items(text) == expected

    # ---- _ensure_dict unit tests ---------------------------------------

    def test_ensure_dict_from_dict(self):
        """Dict input returned as-is."""
        d = {"a": 1}
        assert RubricBasedEvaluator._ensure_dict(d) is d

    def test_ensure_dict_from_json_string(self):
        """JSON string parsed."""
        assert RubricBasedEvaluator._ensure_dict('{"a": 1}') == {"a": 1}

    def test_ensure_dict_invalid_string(self):
        """Invalid JSON string returns empty dict."""
        assert RubricBasedEvaluator._ensure_dict("not json") == {}

    def test_ensure_dict_none(self):
        """None returns empty dict."""
        assert RubricBasedEvaluator._ensure_dict(None) == {}

    # ---- _avg_field_score unit tests -----------------------------------

    @pytest.mark.parametrize(
        "scores, dim, expected",
        [
            ({"a": {"acc": 4.0, "comp": 3.0}, "b": {"acc": 2.0, "comp": 5.0}}, "acc", 3.0),
            ({"a": {"acc": 4.0}, "b": {"acc": 0.0}}, "acc", 4.0),  # excludes 0.0
            ({"a": {"acc": 0.0}}, "acc", 0.0),  # all zero
            ({}, "acc", 0.0),  # empty
        ],
    )
    def test_avg_field_score(self, scores, dim, expected):
        """Averaging correctly excludes skipped (0.0) fields."""
        assert RubricBasedEvaluator._avg_field_score(scores, dim) == expected

    # ---- _parse_review_response unit tests -------------------------------

    @pytest.mark.parametrize(
        ("raw", "expected_acc", "expected_just"),
        [
            # Invalid JSON → fallback zeros + "Parse error"
            ("not json", 0.0, "Parse error"),
            ("{broken", 0.0, "Parse error"),
            ("", 0.0, "Parse error"),
            # Markdown fences stripped
            (
                '```json\n{"accuracy": 4.0, "completeness": 3.0, '
                '"readability": 5.0}\n```',
                4.0,
                "",
            ),
            # Score >5 clamped to 5
            (
                '{"accuracy": 6.0, "completeness": 5.0, "readability": 5.0}',
                5.0,
                "",
            ),
            # Score <1 clamped to 1
            (
                '{"accuracy": 0.5, "completeness": 1.0, "readability": 1.0}',
                1.0,
                "",
            ),
            # Non-numeric score → 0.0 (_clamp catches TypeError)
            (
                '{"accuracy": "bad", "completeness": 3.0, "readability": 4.0}',
                0.0,
                "",
            ),
            # Missing field → default 0
            (
                '{"accuracy": 4.0}',
                4.0,
                "",
            ),
        ],
    )
    def test_parse_review_response(
        self, raw: str, expected_acc: float, expected_just: str,
    ):
        """Response parsing handles invalid JSON, fences, clamping, defaults."""
        result = RubricBasedEvaluator._parse_review_response(raw)
        assert result["accuracy"] == expected_acc, (
            f"accuracy: expected {expected_acc}, got {result['accuracy']}"
        )
        # Justification check: if Parse error expected, verify it
        if expected_just:
            assert expected_just in result.get("accuracy_justification", "")

    # ---- _rubric_block unit test -----------------------------------------

    def test_rubric_block_contains_all_dimensions(self):
        """Static rubric block includes Accuracy, Completeness, Readability."""
        block = RubricBasedEvaluator._rubric_block()
        assert "## Accuracy (1-5)" in block
        assert "## Completeness (1-5)" in block
        assert "## Readability (1-5)" in block
        # Each dimension has 5 anchor levels
        for score in range(1, 6):
            assert f"  {score}:" in block, (
                f"Missing anchor level '{score}:' in rubric block"
            )

    # ---- _build_field_prompt unit tests ----------------------------------

    def test_build_field_prompt_with_expected(self):
        """Prompt includes <response> when expected_value is provided."""
        ev = RubricBasedEvaluator({})
        prompt = ev._build_field_prompt(
            field_name="keywords",
            predicted_value="lens, optics",
            expected_value="lens, optics, MTF",
        )
        assert "<answer>" in prompt
        assert "lens, optics" in prompt
        assert "<response>" in prompt
        assert "MTF" in prompt
        assert '"accuracy": <1-5>' in prompt
        assert "Return ONLY valid JSON" in prompt

    def test_build_field_prompt_without_expected(self):
        """Prompt omits <response> when expected_value is None."""
        ev = RubricBasedEvaluator({})
        prompt = ev._build_field_prompt(
            field_name="keywords",
            predicted_value="lens, optics",
            expected_value=None,
        )
        assert "<answer>" in prompt
        assert "<response>" not in prompt, (
            "<response> should be absent in reference-free mode"
        )

    # ---- hallucination edge cases ----------------------------------------

    def test_hallucination_empty_expected_dict(self):
        """When expected={}, hallucination count is 0 (no items to match)."""
        ev = RubricBasedEvaluator({})
        hallu_count, total_items, details = ev._detect_hallucinations(
            predicted={"ten keywords": "lens, optics, MTF"},
            expected={},
        )
        assert hallu_count == 0, (
            f"Expected 0 hallucinations with empty expected, got {hallu_count}"
        )
        assert total_items > 0, (
            "Should still count items from predicted"
        )
        assert details["hallucinated_items"] == [], (
            "No items should be marked hallucinated"
        )

    def test_hallucination_both_empty(self):
        """No hallucination when both predicted and expected are empty."""
        ev = RubricBasedEvaluator({})
        hallu_count, total_items, details = ev._detect_hallucinations(
            predicted={},
            expected={},
        )
        assert hallu_count == 0
        assert total_items == 0
        assert details["hallucinated_items"] == []


# ===========================================================================
# Online (integration) test — against a real vLLM endpoint
# ===========================================================================

VLLM_BASE_URL = "https://impecunious909.asia/vllm/v1"
VLLM_MODEL = "qwen"
VLLM_API_KEY = "sk-11235813"


@pytest.mark.online
class TestRubricBasedEvaluator_Online:
    """Integration tests using a live vLLM/OpenAI-compatible endpoint.

    These tests make real LLM calls to the endpoint specified via the
    module-level constants ``VLLM_BASE_URL`` / ``VLLM_MODEL`` / ``VLLM_API_KEY``.

    The endpoint is OpenAI-compatible (vLLM serves an OpenAI-like API),
    so ``judge_config.provider`` is set to ``"openai"``.
    """

    @staticmethod
    def _judge_config() -> dict:
        return {
            "provider": "openai",
            "model": VLLM_MODEL,
            "api_base": VLLM_BASE_URL,
            "api_key": VLLM_API_KEY,
            "temperature": 0.0,
            "raw_http": True,  # bypass OpenAI client lib (blocked by WAF)
        }

    @pytest.mark.asyncio
    async def test_online_full_evaluation(self):
        """Full evaluation with real LLM judge via vLLM endpoint.

        Covers all 5 fields with a complete predicted/expected pair.
        """
        ev = RubricBasedEvaluator({"judge_config": self._judge_config()})
        await ev.setup()
        try:
            predicted = {
                "ten keywords": "diffractive optics, meta-lens, "
                "wavefront shaping, computational imaging, "
                "point spread function",
                "objective": "Design a meta-lens for wide-field "
                "imaging in the visible spectrum",
                "novelty": "We propose a new inverse-design "
                "algorithm for meta-lens optimization",
                "method": "Finite-difference time-domain (FDTD) "
                "simulations with adjoint optimization",
                "performance metrics": "Focusing efficiency: 85%, "
                "Strehl ratio: 0.92, FOV: 60 degrees",
            }
            expected = {
                "ten keywords": "meta-lens, diffractive optics, "
                "wavefront engineering, computational imaging, "
                "point spread function",
                "objective": "Design and optimize a meta-lens for "
                "wide-field imaging in visible spectrum",
                "novelty": "Novel inverse-design approach for "
                "meta-lens optimization",
                "method": "FDTD simulations with adjoint-based "
                "topology optimization",
                "performance metrics": "Focusing efficiency: 85%, "
                "Strehl ratio: 0.95, FOV: 60 degrees",
            }

            result = await ev.evaluate(
                task_id="online_full",
                predicted_output=predicted,
                expected_output=expected,
            )

            # Should have meaningful scores (not 0.0) from the LLM
            assert result.metrics["accuracy"] > 0.0, (
                f"Expected accuracy > 0, got {result.metrics['accuracy']}"
            )
            assert result.metrics["completeness"] > 0.0
            assert result.metrics["readability"] > 0.0
            assert 0.0 <= result.metrics["hallucination_rate"] <= 1.0

            # Per-field details should be populated
            for field in RubricBasedEvaluator.FIELDS:
                fs = result.details["field_scores"].get(field, {})
                assert fs.get("accuracy", 0) >= 1.0, (
                    f"Field '{field}' accuracy < 1.0: {fs}"
                )

            # Justifications should exist
            for field in RubricBasedEvaluator.FIELDS:
                fj = result.details["field_justifications"].get(field, {})
                assert fj.get("accuracy", ""), (
                    f"Field '{field}' missing accuracy justification"
                )

        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_online_partial_output(self):
        """Online mode handles partial predicted output gracefully."""
        ev = RubricBasedEvaluator({"judge_config": self._judge_config()})
        await ev.setup()
        try:
            # Only provide a subset of fields — others should be skipped
            result = await ev.evaluate(
                task_id="online_partial",
                predicted_output={
                    "ten keywords": "lens, meta-lens, imaging",
                },
                expected_output={
                    "ten keywords": "meta-lens, lens, imaging",
                    "objective": "Design a lens",
                },
            )

            # The keywords field should be scored
            ks = result.details["field_scores"].get("keywords", {})
            assert ks.get("accuracy", 0) >= 1.0
            assert result.metrics["accuracy"] > 0.0

        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_online_no_expected_output(self):
        """Online evaluation works without expected output (reference-free)."""
        ev = RubricBasedEvaluator({"judge_config": self._judge_config()})
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="online_no_ref",
                predicted_output={
                    "ten keywords": "meta-lens, diffractive optics, "
                    "wavefront shaping",
                },
                expected_output=None,
            )

            # Should still produce scores (no expected = reference-free eval)
            ks = result.details["field_scores"].get("keywords", {})
            assert ks.get("accuracy", 0) >= 1.0, f"keywords accuracy: {ks}"
            assert result.metrics["accuracy"] > 0.0
            # With no expected, hallucination is not computed
            assert result.metrics["hallucination_rate"] == 0.0

        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_offline_after_teardown(self):
        """Teardown cleans up LLM callable — subsequent calls stay safe."""
        ev = RubricBasedEvaluator({"judge_config": self._judge_config()})
        await ev.setup()
        await ev.teardown()
        # Re-setup for fresh state
        await ev.setup()
        try:
            result = await ev.evaluate(
                task_id="lifecycle",
                predicted_output={"ten keywords": "test"},
                expected_output={"ten keywords": "test"},
            )
            assert "accuracy" in result.metrics
        finally:
            await ev.teardown()

    @pytest.mark.asyncio
    async def test_online_aggregate(self):
        """Aggregation works after online evaluations."""
        ev = RubricBasedEvaluator({"judge_config": self._judge_config()})
        await ev.setup()
        try:
            r1 = await ev.evaluate(
                task_id="on_agg_1",
                predicted_output={"ten keywords": "a, b, c"},
                expected_output={"ten keywords": "a, b, c"},
            )
            r2 = await ev.evaluate(
                task_id="on_agg_2",
                predicted_output={"ten keywords": "x, y, z"},
                expected_output={"ten keywords": "x, y"},
            )
            agg = await ev.aggregate([r1, r2])
            assert agg.total_tasks == 2
            assert agg.metrics_summary["accuracy"] > 0.0
            assert agg.metrics_summary["hallucination_rate"] >= 0.0
        finally:
            await ev.teardown()
