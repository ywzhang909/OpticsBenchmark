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

from src.core.agent import AgentConfig, AgentOutput, BaseAgent, create_agent
from src.core.config import TaskConfig


@dataclass
class TaskInstance:
    """A single task instance to be evaluated."""

    task_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

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
        self.agent = create_agent(self.config.agent_config, self.config.task_config)

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

        with open(dataset_path, encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Expected JSON array in dataset, got {type(records).__name__}")

        # Build prompt from prompt_config: try task_file first, then system_file + template_file
        prompt = ""
        prompt_cfg = self.config.task_config.prompt_config
        task_file = prompt_cfg.get("task_file", "")
        if task_file:
            prompt_path = Path(task_file)
            if prompt_path.exists():
                prompt = prompt_path.read_text(encoding="utf-8")

        if not prompt:
            system_file = prompt_cfg.get("system_file", "")
            template_file = prompt_cfg.get("template_file", "")
            parts = []
            if system_file:
                sp = Path(system_file)
                if sp.exists():
                    parts.append(sp.read_text(encoding="utf-8"))
            if template_file:
                tp = Path(template_file)
                if tp.exists():
                    parts.append(tp.read_text(encoding="utf-8"))
            prompt = "\n\n".join(parts)

        for i, record in enumerate(records):
            task_id = i + 1
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
            self.agent.reset()
            if self.config.task_config.task.get("file_input", False):
                self.agent.set_file(task.metadata.get("location"))
            self.agent.add_user_message(task.prompt)
            messages = self.agent.conversation_history.copy()
            result = await self.agent.chat(messages)

            elapsed = time.time() - start
            logger.info(f"[{task.task_id}] cost: ${result.cost:.4f}, time: {elapsed:.2f}s")

            return AgentOutput(
                task_id=task.task_id,
                response=result.response,
                cost=result.cost,
                latency=elapsed,
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            logger.info(f"[{task.task_id}] cost: $0.0000, time: {elapsed:.2f}s (timeout)")
            return AgentOutput(
                task_id=task.task_id,
                response="",
                cost=0.0,
                latency=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Error running agent on task {task.task_id}: {e}")
            logger.info(f"[{task.task_id}] cost: $0.0000, time: {elapsed:.2f}s (error)")
            return AgentOutput(
                task_id=task.task_id,
                response="",
                cost=0.0,
                latency=elapsed,
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
                line = line.strip()
                if line:
                    outputs.append(AgentOutput.from_dict(json.loads(line)))
        return outputs
