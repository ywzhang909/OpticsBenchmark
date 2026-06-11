"""
OptiS Benchmark - Runner Module

This module defines the main evaluation runner that coordinates
agents, environments, and evaluators.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .agent import AgentConfig, BaseAgent, create_agent
from .evaluator import (
    AggregatedResults,
    BaseEvaluator,
    EvaluationResult,
    create_evaluator,
)


@dataclass
class TaskInstance:
    """A single task instance to be evaluated."""

    task_id: str
    instruction: str
    expected_output: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Raw output from running an agent on a single task."""

    task_id: str
    instruction: str
    response: str
    expected_output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "response": self.response,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
            "cost": self.cost,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentOutput:
        return cls(**data)


@dataclass
class TaskConfig:
    """Configuration for a task set."""

    task_id: str
    name: str
    dataset_path: str
    evaluation_config: dict[str, Any]
    max_samples: int | None = None
    shuffle: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskConfig:
        """Load task configuration from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        task_data = data.get("task", {})
        dataset_data = data.get("dataset", {})
        eval_data = data.get("evaluation", {})

        return cls(
            task_id=task_data.get("id", "unknown"),
            name=task_data.get("name", "Unknown Task"),
            dataset_path=dataset_data.get("path", ""),
            evaluation_config=eval_data,
            max_samples=dataset_data.get("num_samples"),
            shuffle=dataset_data.get("shuffle", False),
        )


@dataclass
class RunnerConfig:
    """Configuration for the evaluation runner."""

    agent_config: AgentConfig
    task_config: TaskConfig
    output_path: str
    max_concurrency: int = 1
    timeout: int = 300
    save_intermediate: bool = True
    verbose: bool = True

    @classmethod
    def from_files(
        cls,
        agent_config_path: str | Path,
        task_config_path: str | Path,
        output_path: str = "results/output.jsonl",
        **kwargs,
    ) -> RunnerConfig:
        """Create runner config from files."""
        agent_config = AgentConfig.from_yaml(agent_config_path)
        task_config = TaskConfig.from_yaml(task_config_path)

        return cls(
            agent_config=agent_config,
            task_config=task_config,
            output_path=output_path,
            **kwargs,
        )


class EvaluationRunner:
    """
    Main evaluation runner that coordinates the evaluation process.

    The runner:
    1. Loads task instances from the dataset
    2. Creates agent and evaluator instances
    3. Runs evaluations in parallel (with configurable concurrency)
    4. Aggregates and saves results
    """

    def __init__(self, config: RunnerConfig):
        """Initialize the runner with configuration."""
        self.config = config
        self.agent: BaseAgent | None = None
        self.evaluator: BaseEvaluator | None = None
        self.results: list[EvaluationResult] = []
        self._semaphore: asyncio.Semaphore | None = None

    async def setup(self) -> None:
        """Set up agent"""
        self.agent = create_agent(self.config.agent_config)
        self.evaluator = create_evaluator(self.config.task_config.evaluation_config)

        # Add system prompt
        if self.config.agent_config.system_prompt:
            self.agent.add_system_message(self.config.agent_config.system_prompt)

        # Set up concurrency limiter
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

    async def teardown(self) -> None:
        """Clean up resources."""
        if self.agent:
            await self.agent.close()

    def load_tasks(self) -> list[TaskInstance]:
        """Load task instances from dataset file."""
        dataset_path = Path(self.config.task_config.dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        tasks = []
        dataset_format = self.config.task_config.evaluation_config.get("dataset", {}).get(
            "format", {}
        )

        input_field = dataset_format.get("input_field", "instruction")
        output_field = dataset_format.get("output_field", "expected_output")
        metadata_fields = dataset_format.get("metadata_fields", [])

        with open(dataset_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    metadata = {
                        "task_id": data.get("task_id", f"task_{line_num}"),
                    }
                    for field in metadata_fields:
                        if field in data:
                            metadata[field] = data[field]

                    tasks.append(
                        TaskInstance(
                            task_id=metadata["task_id"],
                            instruction=data.get(input_field, ""),
                            expected_output=data.get(output_field, ""),
                            metadata=metadata,
                        )
                    )
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue

        # Limit number of samples
        if self.config.task_config.max_samples:
            tasks = tasks[: self.config.task_config.max_samples]

        # Shuffle if requested
        if self.config.task_config.shuffle:
            import random

            random.shuffle(tasks)

        return tasks

    async def run_agent(self) -> list[AgentOutput]:
        """Phase 1: Run agent on all tasks without evaluation.

        Returns:
            List of AgentOutput with raw agent responses.
        """
        from loguru import logger

        await self.setup()

        tasks = self.load_tasks()
        logger.info(f"Loaded {len(tasks)} task instances")

        if not tasks:
            raise ValueError("No tasks loaded")

        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)

        sem = asyncio.Semaphore(self.config.max_concurrency)

        async def run_one(task: TaskInstance) -> AgentOutput:
            async with sem:
                return await self._execute_agent(task)

        coros = [run_one(t) for t in tasks]
        outputs = await asyncio.gather(*coros)

        await self.teardown()
        return outputs

    async def _execute_agent(self, task: TaskInstance) -> AgentOutput:
        """Run agent on a single task without evaluation."""
        from loguru import logger

        start = time.time()
        try:
            user_message = task.instruction
            if task.metadata:
                meta_str = "\n\nTask Metadata:\n"
                for k, v in task.metadata.items():
                    if k != "task_id":
                        meta_str += f"- {k}: {v}\n"
                user_message += meta_str

            self.agent.reset()
            self.agent.add_user_message(user_message)
            messages = self.agent.conversation_history.copy()
            response = await self.agent.chat(messages)

            return AgentOutput(
                task_id=task.task_id,
                instruction=task.instruction,
                response=response.content,
                expected_output=task.expected_output,
                metadata=task.metadata,
                cost=response.cost,
                execution_time=time.time() - start,
            )
        except asyncio.TimeoutError:
            return AgentOutput(
                task_id=task.task_id,
                instruction=task.instruction,
                response="",
                expected_output=task.expected_output,
                metadata=task.metadata,
                cost=0.0,
                execution_time=time.time() - start,
            )
        except Exception as e:
            logger.error(f"Error running agent on task {task.task_id}: {e}")
            return AgentOutput(
                task_id=task.task_id,
                instruction=task.instruction,
                response="",
                expected_output=task.expected_output,
                metadata=task.metadata,
                cost=0.0,
                execution_time=time.time() - start,
            )

    async def evaluate_outputs(self, agent_outputs: list[AgentOutput]) -> AggregatedResults:
        """Phase 2: Evaluate agent outputs using the evaluator.

        Args:
            agent_outputs: Raw agent outputs from run_agent()

        Returns:
            AggregatedResults with evaluation metrics
        """
        from loguru import logger

        self.evaluator = create_evaluator(self.config.task_config.evaluation_config)
        logger.info(f"Evaluating {len(agent_outputs)} agent outputs")

        results = []
        for ao in agent_outputs:
            result = await self.evaluator.evaluate(
                task_id=ao.task_id,
                predicted_output=ao.response,
                expected_output=ao.expected_output,
                metadata=ao.metadata,
            )
            result.cost = ao.cost
            results.append(result)

        return await self.evaluator.aggregate(results)

    async def run(self) -> AggregatedResults:
        """
        Run the full evaluation (Phase 1 + Phase 2).

        Returns:
            AggregatedResults with summary statistics
        """
        outputs = await self.run_agent()
        aggregated = await self.evaluate_outputs(outputs)
        self.results = aggregated.per_task_results
        self._save_final_results(aggregated)
        return aggregated

    def _save_final_results(self, aggregated: AggregatedResults) -> None:
        """Save aggregated results."""
        from loguru import logger

        output_path = Path(self.config.output_path)

        # Save as JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "task_id": self.config.task_config.task_id,
                    "agent_name": self.config.agent_config.name,
                    "model": self.config.agent_config.model_name,
                    "total_tasks": aggregated.total_tasks,
                    "successful_tasks": aggregated.successful_tasks,
                    "success_rate": aggregated.success_rate,
                    "avg_score": aggregated.avg_score,
                    "avg_execution_time": aggregated.avg_execution_time,
                    "total_cost": aggregated.total_cost,
                    "metrics_summary": aggregated.metrics_summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("Results saved to:")
        logger.info(f"  - Individual results: {output_path}")
        logger.info(f"  - Aggregated results: {json_path}")

    @staticmethod
    def save_agent_outputs(outputs: list[AgentOutput], path: str | Path) -> None:
        """Save agent outputs to a JSONL file.

        Args:
            outputs: List of agent outputs to save
            path: Output file path (JSONL format)
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for o in outputs:
                f.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def load_agent_outputs(path: str | Path) -> list[AgentOutput]:
        """Load agent outputs from a JSONL file.

        Args:
            path: Input file path (JSONL format)

        Returns:
            List of AgentOutput loaded from file
        """
        outputs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    outputs.append(AgentOutput.from_dict(json.loads(line)))
        return outputs

    def get_statistics(self) -> dict[str, Any]:
        """Get run statistics."""
        if not self.agent:
            return {}

        return {
            "agent_stats": self.agent.get_statistics(),
            "num_results": len(self.results),
        }
