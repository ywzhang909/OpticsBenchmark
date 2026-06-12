"""
OptiS Benchmark - Runner Module

This module defines the main runner that coordinates
agents across task instances.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .agent import AgentConfig, BaseAgent, create_agent


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
    dataset_config: dict[str, Any]
    prompt_config: dict[str, Any]
    max_samples: int | None = None
    shuffle: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskConfig:
        """Load task configuration from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        task_data = data.get("task", {})
        dataset_data = data.get("dataset", {})
        prompt_data = data.get("prompt", {})

        return cls(
            task_id=task_data.get("id", "unknown"),
            name=task_data.get("name", "Unknown Task"),
            dataset_config=dataset_data,
            prompt_config=prompt_data,
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


class AgentRunner:
    """
    Main runner that coordinates agent execution across task instances.

    The runner:
    1. Loads task instances from the dataset
    2. Creates agent instance
    3. Runs agent on tasks in parallel (with configurable concurrency)
    4. Saves agent outputs
    """

    def __init__(self, config: RunnerConfig):
        self.config = config
        self.agent: BaseAgent | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def setup(self) -> None:
        """Set up agent"""
        self.agent = create_agent(self.config.agent_config)

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
        dataset_path = Path(self.config.task_config.dataset_config.get("path", ""))

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        tasks = []
        task_id_prefix = self.config.task_config.task_id

        with open(dataset_path, encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Expected JSON array in dataset, got {type(records).__name__}")

        for i, record in enumerate(records):
            task_id = f"{task_id_prefix}_{i + 1:03d}"
            title = record.get("title", "")
            location = record.get("location", "")

            tasks.append(
                TaskInstance(
                    task_id=task_id,
                    instruction=title,
                    expected_output="",
                    metadata={
                        "task_id": task_id,
                        "title": title,
                        "location": location,
                    },
                )
            )

        if self.config.task_config.max_samples:
            tasks = tasks[: self.config.task_config.max_samples]

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
        if not self.agent:
            return {}

        return {
            "agent_stats": self.agent.get_statistics(),
        }
