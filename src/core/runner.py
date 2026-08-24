"""
Optis Benchmark - Runner Module

This module defines the main runner that coordinates
agents across task instances.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# TODO: 重构为使用 src.llm 抽象层替代已删除的 agent.py
# from src.core.agent import AgentConfig, AgentOutput, BaseAgent, create_agent
from src.core.config import TaskConfig
from src.utils import logger


@dataclass
class TaskInstance:
    """A single task instance to be evaluated."""

    task_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    """Configuration for the evaluation runner."""

    agent_config: Any
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
        task_config = TaskConfig.from_yaml(task_config_path)

        return cls(
            agent_config={},
            task_config=task_config,
            output_path=output_path,
            **kwargs,
        )


class AgentRunner:
    """Main runner that coordinates agent execution across task instances.

    The runner:
    1. Loads task instances from the dataset
    2. Creates agent instance
    3. Runs agent on tasks in parallel (with configurable concurrency)
    4. Saves agent outputs
    """

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.agent: Any = None
        self._semaphore: asyncio.Semaphore | None = None

    async def setup(self) -> None:
        """Set up agent (requires src.core.agent migration)."""
        logger.warning(
            "AgentRunner.setup() requires src.core.agent migration — agent is None"
        )

    async def teardown(self) -> None:
        """Clean up resources."""
        pass

    def load_tasks(self) -> list[TaskInstance]:
        """Load task instances from dataset file."""
        dataset_path = Path(self.config.task_config.dataset_config.get("path", ""))

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        tasks = []

        with open(dataset_path, encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Expected JSON array in dataset, got {type(records).__name__}")

        # Build prompt from prompt_config: task_file
        prompt = ""
        prompt_cfg = self.config.task_config.prompt_config
        task_file = prompt_cfg.get("task_file", "")
        if task_file:
            prompt_path = Path(task_file)
            if prompt_path.exists():
                prompt = prompt_path.read_text(encoding="utf-8")
                # 删除前两行（注释行 + 空行）
                lines = prompt.split("\n")
                if len(lines) > 2:
                    prompt = "\n".join(lines[2:])
                else:
                    logger.warning(
                        f"Prompt file '{task_file}' has only {len(lines)} lines, "
                        f"expected at least 3. Using raw content."
                    )
            else:
                logger.warning(f"Prompt file not found: {task_file}")
        else:
            logger.warning("No 'task_file' specified in prompt_config")

        for i, record in enumerate(records):
            task_id = str(i + 1)
            title = record.get("title", "")
            location = record.get("location", "")

            tasks.append(
                TaskInstance(
                    task_id=task_id,
                    prompt=prompt,
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
            random.shuffle(tasks)

        return tasks

    async def run_agent(self) -> list[Any]:
        """Phase 1: Run agent on all tasks without evaluation.

        Returns:
            List of agent outputs with raw responses.
        """
        await self.setup()

        tasks = self.load_tasks()
        logger.info(f"Loaded {len(tasks)} task instances")

        if not tasks:
            raise ValueError("No tasks loaded")

        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)

        sem = asyncio.Semaphore(self.config.max_concurrency)

        async def run_one(task: TaskInstance) -> Any:
            async with sem:
                return await self._execute_agent(task)

        coros = [run_one(t) for t in tasks]
        outputs = await asyncio.gather(*coros)

        await self.teardown()
        return outputs

    async def _execute_agent(self, task: TaskInstance) -> Any:
        """Run agent on a single task without evaluation."""
        logger.warning(
            f"[{task.task_id}] AgentRunner._execute_agent() requires "
            f"src.core.agent migration — returning empty result"
        )
        return {}

    @staticmethod
    def save_agent_outputs(outputs: list[Any], path: str | Path) -> None:
        """Save agent outputs to a JSONL file.

        Args:
            outputs: List of agent outputs to save
            path: Output file path (JSONL format)
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for o in outputs:
                if hasattr(o, "to_dict"):
                    f.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")

    @staticmethod
    def load_agent_outputs(path: str | Path) -> list[Any]:
        """Load agent outputs from a JSONL file.

        Args:
            path: Input file path (JSONL format)

        Returns:
            List of agent outputs loaded from file
        """
        outputs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    outputs.append(json.loads(line))
        return outputs
