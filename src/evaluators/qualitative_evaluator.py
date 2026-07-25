"""
Qualitative Evaluator using LLM-as-judge with structured rubrics.

Evaluates agent outputs across multiple quality dimensions using
a language model judge with anchored rubrics. Produces both
numeric dimension scores and textual justifications.

Inspired by Vercel Labs PluginEval's "Layer 2 — LLM Judge" approach.
"""

from __future__ import annotations

import os
import time
from typing import Any

from src.core.llm_judge import DEFAULT_RUBRICS, LLMJudge, Rubric, RubricCriterion
from src.module import AggregatedResults, EvaluationResult
from src.utils import logger

from .base import BaseEvaluator


class QualitativeEvaluator(BaseEvaluator):
    """
    Evaluator that uses an LLM judge to score outputs across qualitative
    dimensions with anchored rubrics.

    Supports three modes:
      1. **Live mode**: An LLM agent is configured via ``judge_config`` and
         called per-task to produce dimension scores + justifications.
      2. **Score-only mode**: The judge prompt is sent to the LLM but only
         the numeric dimension scores are stored in ``metrics``.
      3. **Offline / prompt-only mode**: No LLM callable is provided — the
         evaluator builds the judge prompt and stores it in ``details``
         for later manual or batch evaluation.

    Config example::

        eval_metrics:
          qualitative:
            priority: 0
            task_description: "Extract structured information from optical papers"
            judge_config:
              provider: "openai"
              model: "gpt-4"
              temperature: 0.0
            dimensions:
              - name: "optical_accuracy"
                criteria:
                  - {score: 1.0, label: "Excellent", description: "All metrics meet targets"}
                  - {score: 0.5, label: "Fair", description: "Some metrics met"}
                  - {score: 0.0, label: "Fail", description: "No metrics met"}
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._judge: LLMJudge | None = None
        self._llm_callable: Any = None

    async def setup(self) -> None:
        """Set up the LLM judge with configured rubrics."""
        rubrics = self._build_rubrics()
        llm_callable = await self._create_llm_callable()
        self._judge = LLMJudge(rubrics=rubrics, llm_callable=llm_callable)

    async def teardown(self) -> None:
        """Clean up the LLM callable if it has a close method."""
        if self._llm_callable is not None:
            try:
                await self._llm_callable.close()
            except Exception:
                pass
        self._judge = None
        self._llm_callable = None

    def _build_rubrics(self) -> list[Rubric]:
        """Build rubrics from config, falling back to ``DEFAULT_RUBRICS``.

        Config format::

            dimensions:
              - name: "optical_accuracy"
                criteria:
                  - {score: 1.0, label: "Excellent", description: "..."}
                  - {score: 0.5, label: "Fair", description: "..."}
        """
        custom_dims = self.config.get("dimensions")
        if not custom_dims:
            return list(DEFAULT_RUBRICS)

        rubrics: list[Rubric] = []
        for cd in custom_dims:
            name = cd.get("name", "")
            raw_criteria = cd.get("criteria", [])
            criteria = [
                RubricCriterion(
                    score=c.get("score", 0.0),
                    label=c.get("label", ""),
                    description=c.get("description", ""),
                )
                for c in raw_criteria
            ]
            if criteria:
                rubrics.append(Rubric(dimension=name, criteria=criteria))
        return rubrics if rubrics else list(DEFAULT_RUBRICS)

    async def _create_llm_callable(self) -> Any:
        """Create an LLM callable from judge_config, if configured.

        Returns ``None`` if no judge_config is provided, placing the
        evaluator in offline / prompt-only mode.
        """
        judge_cfg = self.config.get("judge_config")
        if not judge_cfg:
            logger.info("QualitativeEvaluator: No judge_config — operating in offline mode")
            return None

        try:
            from src.core.agent import AgentConfig, AgentProvider, Message, create_agent
            from src.core.config import TaskConfig

            provider_str = judge_cfg.get("provider", "openai")
            model = judge_cfg.get("model", "gpt-4")
            temperature = judge_cfg.get("temperature", 0.0)

            def _expand_val(v: str) -> str:
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    return os.environ.get(v[2:-1], "")
                return v

            raw_key = judge_cfg.get("api_key", "")
            api_key = _expand_val(raw_key)

            agent_config = AgentConfig(
                name="qualitative-judge",
                provider=AgentProvider(provider_str),
                model_name=model,
                api_base=_expand_val(judge_cfg.get("api_base", "")),
                api_key=api_key,
                setup_config={"temperature": temperature, "max_completion_tokens": 2048},
            )
            task_config = TaskConfig(
                task={"id": "qualitative-eval"},
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
                f"QualitativeEvaluator: Failed to create LLM callable ({e}) — "
                f"falling back to offline mode"
            )
            return None

    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
    ) -> EvaluationResult:
        """Evaluate a single prediction using LLM judge."""
        start_time = time.time()

        try:
            predicted_str = str(predicted_output)
            expected_str = str(expected_output) if expected_output is not None else None

            task_desc = self.config.get(
                "task_description",
                "Evaluate the quality of the optical design agent's output.",
            )

            if self._judge is None:
                return EvaluationResult(
                    task_id=task_id,
                    execution_time=time.time() - start_time,
                    details={"error": "Judge not initialized — call setup() first"},
                )

            result = await self._judge.evaluate(
                task_description=task_desc,
                predicted_output=predicted_str,
                expected_output=expected_str,
            )

            if result.error:
                logger.warning(
                    f"QualitativeEvaluator task {task_id}: judge error — {result.error}"
                )
                return EvaluationResult(
                    task_id=task_id,
                    execution_time=time.time() - start_time,
                    details={"judge_error": result.error},
                )

            metrics: dict[str, float] = {}
            justifications: dict[str, str] = {}

            for dim, score in result.dimension_scores.items():
                metrics[f"qualitative_{dim}"] = score
                justifications[dim] = result.justifications.get(dim, "")

            # Store the full judge response for transparency
            details: dict[str, Any] = {
                "justifications": justifications,
                "judge_model": result.model_used,
            }

            return EvaluationResult(
                task_id=task_id,
                metrics=metrics,
                execution_time=time.time() - start_time,
                details=details,
            )

        except Exception as e:
            logger.error(f"Error in QualitativeEvaluator for task {task_id}: {e}")
            return EvaluationResult(
                task_id=task_id,
                execution_time=time.time() - start_time,
                details={"evaluation_error": str(e)},
            )

    async def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> AggregatedResults:
        """Aggregate qualitative evaluation results."""
        total = len(results)
        return AggregatedResults(
            total_tasks=total,
            metrics_summary=self._avg_metrics(results),
            avg_execution_time=(
                sum(r.execution_time for r in results) / total if total > 0 else 0.0
            ),
            per_task_results=results,
        )
