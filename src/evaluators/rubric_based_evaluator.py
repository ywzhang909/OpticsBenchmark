"""
Rubric-Based Evaluator with per-field scoring and hallucination detection.

Evaluates agent outputs across five semantic fields using three quality
rubrics (Accuracy, Completeness, Readability) rated 1–5, plus
hallucination detection.

Uses XML-tagged prompts (``<answer>``, ``<rule>``, ``<response>``) for
structured LLM judge interaction.

Config example::

    eval_metrics:
      rubric_based:
        priority: 5
        judge_config:
          provider: "openai"
          model: "gpt-4"
          temperature: 0.0
        field_map:
          keywords: "ten keywords"
          research_objectives: "objective"
          innovation_points: "novelty"
          research_methods: "method"
          performance_metrics: "performance metrics"

Metrics output::

    {
      "accuracy": <float 1-5>,
      "completeness": <float 1-5>,
      "readability": <float 1-5>,
      "hallucination_rate": <float 0-1>,
    }
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from src.module import AggregatedResults, EvaluationResult
from src.utils import logger
from src.utils.prompt_manager import PromptManager

from .base import BaseEvaluator

# ---------------------------------------------------------------------------
# Default field mapping (paper_info_extract JSON keys → evaluated fields)
# ---------------------------------------------------------------------------
DEFAULT_FIELD_MAP: dict[str, str] = {
    "keywords": "ten keywords",
    "research_objectives": "objective",
    "innovation_points": "novelty",
    "research_methods": "method",
    "performance_metrics": "performance metrics",
}

# ---------------------------------------------------------------------------
# Anchor-based rubric criteria (1-5 Likert)
# ---------------------------------------------------------------------------
ACCURACY_RUBRIC: dict[int, str] = {
    5: "All extracted information is factually correct and fully consistent "
    "with the source material. No errors.",
    4: "Minor inaccuracies present (e.g. slightly imprecise wording). No "
    "factual errors that change meaning.",
    3: "Some inaccuracies present. Multiple minor errors or one significant "
    "error.",
    2: "Significant inaccuracies. Key facts are wrong or misrepresented.",
    1: "Completely incorrect or fabricated information that does not reflect "
    "the source.",
}

COMPLETENESS_RUBRIC: dict[int, str] = {
    5: "All required information items are present and comprehensive. No "
    "omissions.",
    4: "Most information present, with only minor omissions that do not "
    "affect understanding.",
    3: "Some information present, but noticeable gaps exist in coverage.",
    2: "Significant missing information. Only partial content is provided.",
    1: "Almost entirely missing. Most required information is absent.",
}

READABILITY_RUBRIC: dict[int, str] = {
    5: "Excellent clarity, organization, and formatting. Easy to read and "
    "understand.",
    4: "Good clarity with minor issues in wording or structure. Still clear "
    "overall.",
    3: "Adequate readability. Some parts are unclear or poorly organized.",
    2: "Poor readability. Difficult to follow or extract meaning.",
    1: "Unreadable or incomprehensible. Does not convey useful information.",
}


# ===========================================================================
# Evaluator
# ===========================================================================


class RubricBasedEvaluator(BaseEvaluator):
    """Per-field rubric-based evaluator with hallucination detection.

    Evaluates each of five semantic fields (keywords, research objectives,
    innovation points, research methods, quantitative performance metrics)
    on three quality dimensions (Accuracy, Completeness, Readability) using
    a 1–5 Likert scale via LLM judge.

    Supports **online mode** (LLM callable configured via ``judge_config``)
    and **offline / prompt-only mode** (no LLM callable — returns zero
    scores with an explanation in ``details``).
    """

    FIELDS: list[str] = [
        "keywords",
        "research_objectives",
        "innovation_points",
        "research_methods",
        "performance_metrics",
    ]

    DISPLAY_NAMES: dict[str, str] = {
        "keywords": "Keywords",
        "research_objectives": "Research Objectives",
        "innovation_points": "Innovation Points",
        "research_methods": "Research Methods",
        "performance_metrics": "Quantitative Performance Metrics",
    }

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._llm_callable: Any = None
        self._field_map: dict[str, str] = dict(
            config.get("field_map", DEFAULT_FIELD_MAP)
        )

    # ---- lifecycle -------------------------------------------------------

    async def setup(self) -> None:
        """Set up the LLM callable for judge-based evaluation."""
        self._llm_callable = await self._create_llm_callable()

    async def teardown(self) -> None:
        """Clean up the LLM callable if it has a close method."""
        if self._llm_callable is not None:
            try:
                await self._llm_callable.close()
            except Exception:
                pass
            self._llm_callable = None

    # ---- public API ------------------------------------------------------

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
    ) -> EvaluationResult:
        """Evaluate a single prediction using rubric-based LLM judging."""
        start_time = time.time()

        try:
            predicted_dict = self._ensure_dict(predicted_output)
            expected_dict = (
                self._ensure_dict(expected_output)
                if expected_output is not None
                else {}
            )

            field_scores: dict[str, dict[str, float]] = {}
            field_justifications: dict[str, dict[str, str]] = {}

            for field in self.FIELDS:
                json_key = self._field_map.get(field, field)
                pred_val = predicted_dict.get(json_key, "")
                exp_val = expected_dict.get(json_key, "")

                if not pred_val:
                    logger.info(
                        "RubricBasedEvaluator task {}: field '{}' has no "
                        "predicted value — skipping",
                        task_id,
                        field,
                    )
                    field_scores[field] = {
                        "accuracy": 0.0,
                        "completeness": 0.0,
                        "readability": 0.0,
                    }
                    field_justifications[field] = {
                        "accuracy": "No predicted output",
                        "completeness": "No predicted output",
                        "readability": "No predicted output",
                    }
                    continue

                field_result = await self._evaluate_field(
                    field_name=field,
                    predicted_value=str(pred_val),
                    expected_value=str(exp_val) if exp_val else None,
                )
                field_scores[field] = {
                    "accuracy": field_result.get("accuracy", 0.0),
                    "completeness": field_result.get("completeness", 0.0),
                    "readability": field_result.get("readability", 0.0),
                }
                field_justifications[field] = {
                    "accuracy": field_result.get("accuracy_justification", ""),
                    "completeness": field_result.get(
                        "completeness_justification", ""
                    ),
                    "readability": field_result.get(
                        "readability_justification", ""
                    ),
                }

            # Per-dimension averages across fields
            avg_accuracy = self._avg_field_score(field_scores, "accuracy")
            avg_completeness = self._avg_field_score(field_scores, "completeness")
            avg_readability = self._avg_field_score(field_scores, "readability")

            # Hallucination detection
            hallu_count, total_items, hallu_details = self._detect_hallucinations(
                predicted_dict, expected_dict
            )
            hallucination_rate = hallu_count / max(total_items, 1)

            metrics: dict[str, float] = {
                "accuracy": round(avg_accuracy, 4),
                "completeness": round(avg_completeness, 4),
                "readability": round(avg_readability, 4),
                "hallucination_rate": round(hallucination_rate, 4),
            }

            details: dict[str, Any] = {
                "field_scores": {
                    f: field_scores[f]
                    for f in self.FIELDS
                    if f in field_scores
                },
                "field_justifications": {
                    f: field_justifications[f]
                    for f in self.FIELDS
                    if f in field_justifications
                },
                "hallucination": hallu_details,
            }

            return EvaluationResult(
                task_id=task_id,
                metrics=metrics,
                execution_time=time.time() - start_time,
                details=details,
            )

        except Exception as e:
            logger.error(
                "Error in RubricBasedEvaluator for task {}: {}", task_id, e
            )
            return EvaluationResult(
                task_id=task_id,
                execution_time=time.time() - start_time,
                details={"evaluation_error": str(e)},
            )

    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        """Aggregate rubric-based evaluation results."""
        total = len(results)
        return AggregatedResults(
            total_tasks=total,
            metrics_summary=self._avg_metrics(results),
            avg_execution_time=(
                sum(r.execution_time for r in results) / total
                if total > 0
                else 0.0
            ),
            per_task_results=results,
        )

    # ---- LLM callable creation ------------------------------------------

    async def _create_llm_callable(self) -> Any:
        """Create an LLM callable from ``judge_config``.

        Two modes:

        * **Agent mode** (default) — uses ``create_agent()`` from
          ``src.core.agent``. Suitable for standard OpenAI / Anthropic / etc.
          endpoints.

        * **Raw HTTP mode** (``judge_config.raw_http: true``) — uses ``httpx``
          directly, bypassing the OpenAI client library. Useful when the
          endpoint's WAF (e.g. Cloudflare) blocks the official OpenAI Python
          library's User-Agent.

        Returns ``None`` if no ``judge_config`` is provided (offline mode).
        """
        judge_cfg = self.config.get("judge_config")
        if not judge_cfg:
            logger.info(
                "RubricBasedEvaluator: No judge_config — "
                "operating in offline mode"
            )
            return None
        logger.debug("judge_config present: provider={}, model={}, raw_http={}",
                     judge_cfg.get("provider"), judge_cfg.get("model"),
                     judge_cfg.get("raw_http", False))

        # -- raw HTTP mode (bypasses the OpenAI client library) -------------
        if judge_cfg.get("raw_http", False):
            return await self._create_raw_http_callable(
                judge_cfg, self._expand_env_var,
            )

        # -- agent mode (default) ------------------------------------------
        try:
            from src.core.agent import (
                AgentConfig,
                AgentProvider,
                Message,
                create_agent,
            )
            from src.core.config import TaskConfig

            provider_str = judge_cfg.get("provider", "openai")
            model = judge_cfg.get("model", "gpt-4")
            temperature = judge_cfg.get("temperature", 0.0)

            agent_config = AgentConfig(
                name="rubric-judge",
                provider=AgentProvider(provider_str),
                model_name=model,
                api_base=self._expand_env_var(judge_cfg.get("api_base", "")),
                api_key=self._expand_env_var(judge_cfg.get("api_key", "")),
                setup_config={
                    "temperature": temperature,
                    "max_completion_tokens": 2048,
                },
            )
            task_config = TaskConfig(
                task={"id": "rubric-eval"},
                dataset_config={},
                prompt_config={},
            )

            agent = create_agent(agent_config, task_config)
            self._llm_callable = agent

            async def llm_callable(prompt: str) -> str:
                msg = Message(role="user", content=prompt)
                result = await agent.chat(messages=[msg])
                return result.response if result.response else str(result)

            return llm_callable

        except Exception as e:
            logger.warning(
                "RubricBasedEvaluator: Failed to create LLM callable ({}) — "
                "falling back to offline mode",
                e,
            )
            return None

    async def _create_raw_http_callable(
        self,
        judge_cfg: dict[str, Any],
        expand_val: Any,
    ) -> Any:
        """Create an LLM callable using raw ``httpx``.

        Useful when the target API endpoint blocks the official OpenAI
        Python library's User-Agent (e.g. Cloudflare WAF).
        """
        import httpx

        api_base = expand_val(judge_cfg.get("api_base", ""))
        api_key = expand_val(judge_cfg.get("api_key", ""))
        model = judge_cfg.get("model", "gpt-4")
        temperature = judge_cfg.get("temperature", 0.0)
        max_tokens = judge_cfg.get("max_tokens", 4096)

        # Ensure trailing slash for proper URL joining
        base = api_base.rstrip("/") + "/"
        endpoint = base + "chat/completions"

        logger.debug("Raw HTTP callable — endpoint={}, model={}, max_tokens={}",
                     endpoint, model, max_tokens)

        timeout = httpx.Timeout(120.0)

        async def llm_callable(prompt: str) -> str:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            prompt_preview = prompt[:120].replace("\n", " ")
            logger.debug("LLM request — prompt preview: {}...", prompt_preview)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            content = self._extract_content(data)
            logger.debug("LLM response — length={}, preview: {}...",
                         len(content), content[:120].replace("\n", " "))
            return content

        return llm_callable

    # ---- per-field evaluation -------------------------------------------

    async def _evaluate_field(
        self,
        field_name: str,
        predicted_value: str,
        expected_value: str | None,
    ) -> dict[str, Any]:
        """Evaluate a single field using the LLM judge.

        Returns a dict with keys ``accuracy``, ``completeness``,
        ``readability`` and their ``*_justification`` counterparts.
        """
        if self._llm_callable is None:
            return {
                "accuracy": 0.0,
                "completeness": 0.0,
                "readability": 0.0,
                "accuracy_justification": "No LLM callable (offline mode)",
                "completeness_justification": "No LLM callable (offline mode)",
                "readability_justification": "No LLM callable (offline mode)",
            }

        prompt = self._build_field_prompt(
            field_name=field_name,
            predicted_value=predicted_value,
            expected_value=expected_value,
        )

        try:
            raw = await self._llm_callable(prompt)
            return self._parse_review_response(raw)
        except Exception as e:
            logger.error(
                "Field evaluation failed for '{}': {}", field_name, e
            )
            return {
                "accuracy": 0.0,
                "completeness": 0.0,
                "readability": 0.0,
                "accuracy_justification": f"LLM error: {e}",
                "completeness_justification": f"LLM error: {e}",
                "readability_justification": f"LLM error: {e}",
            }

    # ---- prompt building ------------------------------------------------

    PROMPT_TEMPLATE = "evaluators/rubric_judge/field_prompt.jinja2"
    """Path to the Jinja2 prompt template, relative to ``prompts/``."""

    def _build_field_prompt(
        self,
        field_name: str,
        predicted_value: str,
        expected_value: str | None,
    ) -> str:
        """Build the XML-tagged judge prompt using the Jinja2 template.

        Template: ``prompts/evaluators/rubric_judge/field_prompt.jinja2``

        The three rubric dictionaries (``ACCURACY_RUBRIC``,
        ``COMPLETENESS_RUBRIC``, ``READABILITY_RUBRIC``) are passed as
        template variables so the rubric text is rendered server-side.
        """
        display = self.DISPLAY_NAMES.get(field_name, field_name)
        pm = PromptManager.get_instance()
        return pm.render(
            self.PROMPT_TEMPLATE,
            display=display,
            predicted_value=predicted_value,
            expected_value=expected_value,
            accuracy_rubric=ACCURACY_RUBRIC,
            completeness_rubric=COMPLETENESS_RUBRIC,
            readability_rubric=READABILITY_RUBRIC,
        )

    @staticmethod
    def _rubric_block() -> str:
        """Build the ``<rule>`` block text (delegates to the Jinja2 template)."""
        pm = PromptManager.get_instance()
        return pm.render(
            "evaluators/rubric_judge/field_prompt.jinja2",
            display="",
            predicted_value="",
            expected_value=None,
            accuracy_rubric=ACCURACY_RUBRIC,
            completeness_rubric=COMPLETENESS_RUBRIC,
            readability_rubric=READABILITY_RUBRIC,
        )

    # ---- response parsing -----------------------------------------------

    @staticmethod
    def _parse_review_response(raw: str) -> dict[str, Any]:
        """Parse the LLM response into a field-score dict.

        Handles both plain JSON and JSON-within-markdown-fences.
        """
        text = raw.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            logger.debug("_parse_review_response OK — keys: {}",
                         list(data.keys()))
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse judge response as JSON: {} ...",
                text[:200],
            )
            logger.debug("Raw text that failed JSON parse: {!r:.300}", text)
            return {
                "accuracy": 0.0,
                "completeness": 0.0,
                "readability": 0.0,
                "accuracy_justification": "Parse error",
                "completeness_justification": "Parse error",
                "readability_justification": "Parse error",
            }

        def _clamp(val: Any) -> float:
            try:
                v = float(val)
                return max(1.0, min(5.0, v))
            except (TypeError, ValueError):
                return 0.0

        return {
            "accuracy": _clamp(data.get("accuracy", 0)),
            "completeness": _clamp(data.get("completeness", 0)),
            "readability": _clamp(data.get("readability", 0)),
            "accuracy_justification": data.get("accuracy_justification", ""),
            "completeness_justification": data.get(
                "completeness_justification", ""
            ),
            "readability_justification": data.get(
                "readability_justification", ""
            ),
        }

    # ---- hallucination detection ----------------------------------------

    def _detect_hallucinations(
        self,
        predicted: dict[str, Any],
        expected: dict[str, Any],
    ) -> tuple[int, int, dict[str, Any]]:
        """Detect hallucinated content by field-level item comparison.

        Returns ``(hallucinated_count, total_checked_items, details_dict)``.
        """
        hallu_details: dict[str, Any] = {
            "hallucinated_items": [],
            "total_checked_items": 0,
        }
        hallu_count = 0
        total_items = 0

        for field, json_key in self._field_map.items():
            pred_val = predicted.get(json_key)
            exp_val = expected.get(json_key)

            if not pred_val:
                continue

            pred_str = str(pred_val).strip()
            exp_str = str(exp_val).strip() if exp_val else ""

            if not pred_str:
                continue

            pred_items = self._split_items(pred_str)
            exp_items_lower = (
                {item.lower().strip() for item in self._split_items(exp_str)}
                if exp_str
                else set()
            )

            for item in pred_items:
                item_clean = item.strip()
                if not item_clean:
                    continue
                total_items += 1
                if exp_str and item_clean.lower() not in exp_items_lower:
                    hallu_count += 1
                    hallu_details["hallucinated_items"].append({
                        "field": field,
                        "item": item_clean[:200],
                    })

        hallu_details["total_checked_items"] = total_items
        return hallu_count, total_items, hallu_details

    @staticmethod
    def _split_items(text: str) -> list[str]:
        """Split a text value into individual items.

        Handles JSON arrays, comma-separated, semicolon-separated,
        newline-separated, numbered lists, and bullet lists.
        """
        text_stripped = text.strip()
        # Try JSON array
        if text_stripped.startswith("[") and text_stripped.endswith("]"):
            try:
                parsed = json.loads(text_stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item]
            except (json.JSONDecodeError, TypeError):
                pass

        lines = text.split("\n")
        items: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Bullet / numbered-list prefixes
            if (
                line.startswith("- ")
                or line.startswith("* ")
                or line.startswith("• ")
            ):
                line = line[2:]
            if (
                len(line) > 2
                and line[0].isdigit()
                and line[1] in (".", ")", ":")
            ):
                line = line[2:].strip()
            if (
                len(line) > 3
                and line[0].isdigit()
                and line[1].isdigit()
                and line[2] in (".", ")", ":")
            ):
                line = line[3:].strip()
            # Split by semicolons or commas within this line
            if ";" in line:
                items.extend(
                    part.strip() for part in line.split(";") if part.strip()
                )
            elif ", " in line:
                items.extend(
                    part.strip() for part in line.split(",") if part.strip()
                )
            else:
                items.append(line)
        return items

    # ---- utility --------------------------------------------------------

    @staticmethod
    def _ensure_dict(value: Any) -> dict[str, Any]:
        """Convert a value to a dict if it is a JSON string."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _avg_field_score(
        field_scores: dict[str, dict[str, float]],
        dimension: str,
    ) -> float:
        """Average a specific dimension across all fields.

        Only includes fields that scored > 0.0 to avoid counting skipped
        fields.
        """
        values = [
            scores.get(dimension, 0.0) for scores in field_scores.values()
        ]
        values = [v for v in values if v > 0.0]
        if not values:
            return 0.0
        return sum(values) / len(values)

    # ---- extractable helpers for testability -----------------------------

    @staticmethod
    def _expand_env_var(value: str) -> str:
        """Expand ``${ENV_VAR}`` patterns using the process environment.

        Returns the raw value unchanged if it is not an env-var reference
        or if the variable is not set.
        """
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Extract the content string from an OpenAI-compatible response dict.

        Handles standard ``choices[0].message.content`` as well as the vLLM
        quirk where ``content`` is ``null`` and the actual response is placed
        in ``choices[0].message.reasoning``.
        """
        try:
            choice = data["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            if not content and msg.get("reasoning"):
                content = msg["reasoning"]
                logger.debug("content is null, using reasoning fallback "
                             "(len={})", len(content))
            logger.debug("_extract_content: content_len={}, has_reasoning={}",
                         len(content), bool(msg.get("reasoning")))
            return content
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("Failed to extract content from response: %s", exc)
            return ""
