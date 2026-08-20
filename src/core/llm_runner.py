"""
Optis Benchmark - LLM Prediction Runner Module

LLM inference runner aligned with AgentRunner.
Uses Provider + LLM architecture to execute inference tasks.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.llm import create_llm, create_provider
from src.utils import logger

# ---------------------------------------------------------------------------
# 环境变量展开
# ---------------------------------------------------------------------------


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables ${VAR_NAME}."""
    if isinstance(data, str):
        if data.startswith("${") and data.endswith("}"):
            return os.environ.get(data[2:-1], "")
        return data
    elif isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    return data


# ---------------------------------------------------------------------------
# LLM 输出数据类（对齐 AgentOutput）
# ---------------------------------------------------------------------------


@dataclass
class LLMOutput:
    """Inference output for a single task."""

    task_id: str = ""
    response: str = ""
    cost: float = 0.0
    latency: float = 0.0
    model: str = ""
    provider: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.task_id,
            "data": self.response,
            "cost": self.cost,
            "latency": self.latency,
            "model": self.model,
            "provider": self.provider,
        }
        if self.error:
            result["error"] = self.error
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMOutput:
        """Create from dictionary."""
        return cls(
            task_id=data.get("id", ""),
            response=data.get("data", ""),
            cost=data.get("cost", 0.0),
            latency=data.get("latency", 0.0),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            error=data.get("error"),
        )


# ---------------------------------------------------------------------------
# LLM Runner 配置（对齐 RunnerConfig）
# ---------------------------------------------------------------------------


@dataclass
class LLMRunnerConfig:
    """LLM inference configuration."""

    provider_config: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)
    setup_config: dict[str, Any] = field(default_factory=dict)
    task_config: dict[str, Any] = field(default_factory=dict)
    execution_config: dict[str, Any] = field(default_factory=dict)
    output_path: str = "results/llm_outputs.jsonl"
    max_concurrency: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMRunnerConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        data = _expand_env_vars(data)

        llm_config = data.get("llm", {})
        task_config = data.get("task", {})
        execution_config = data.get("execution", {})

        return cls(
            provider_config=llm_config.get("provider", {}),
            model_config=llm_config.get("model", {}),
            setup_config=llm_config.get("setup", {}),
            task_config=task_config,
            execution_config=execution_config,
            output_path=execution_config.get("output_path", "results/llm_outputs.jsonl"),
            max_concurrency=execution_config.get("concurrency", 1),
        )


# ---------------------------------------------------------------------------
# LLM Prediction Runner（对齐 AgentRunner）
# ---------------------------------------------------------------------------


class LLMPredRunner:
    """LLM inference runner.

    Execution flow:
    1. setup() - Create Provider and LLM
    2. load_tasks() - Load dataset
    3. run() - Execute inference concurrently
    4. teardown() - Clean up resources
    """

    def __init__(self, config: LLMRunnerConfig):
        self.config = config
        self.provider: Any | None = None
        self.llm: Any | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def setup(self) -> None:
        """Create Provider and LLM instances."""
        self.provider = create_provider(self.config.provider_config)
        self.llm = create_llm(self.config.model_config)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

        logger.info(f"Provider: {type(self.provider).__name__}")
        logger.info(f"LLM: {self.llm.model_name}")

    async def teardown(self) -> None:
        """Clean up resources."""
        if self.provider:
            try:
                await self.provider.close()
            except Exception:
                pass

    def load_tasks(self) -> list[dict[str, Any]]:
        """Load dataset."""
        dataset_path = self.config.task_config.get("dataset_path", "")
        if not dataset_path:
            raise ValueError("dataset_path not specified")

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Expected JSON array, got {type(records).__name__}")

        # Load prompt
        prompt = ""
        prompt_file = self.config.task_config.get("prompt_file", "")
        if prompt_file:
            prompt = self._load_prompt(prompt_file)

        # Load gold_answer, build title -> id mapping (case-insensitive)
        gold_answer_path = self.config.task_config.get("gold_answer_path", "")
        title_to_id: dict[str, int] = {}
        if gold_answer_path:
            ga_path = Path(gold_answer_path)
            if not ga_path.exists():
                raise FileNotFoundError(f"Gold answer file not found: {gold_answer_path}")
            with open(ga_path, encoding="utf-8") as f:
                gold_records = json.load(f)
            for gr in gold_records:
                title = gr.get("data", {}).get("title", "")
                if title:
                    title_to_id[title.lower()] = gr.get("id", -1)

        # Limit sample count
        max_samples = self.config.task_config.get("max_samples")
        if max_samples is not None:
            records = records[:max_samples]

        # Shuffle order
        if self.config.task_config.get("shuffle", False):
            random.shuffle(records)

        tasks = []
        for i, record in enumerate(records):
            title = record.get("title", "")
            if title_to_id:
                task_id = title_to_id.get(title.lower())
                if task_id is None:
                    logger.warning(f"Cannot match title '{title}', skipping record")
                    continue
            else:
                task_id = i + 1
            tasks.append({
                "task_id": task_id,
                "prompt": prompt,
                "record": record,
            })

        return tasks

    def _load_prompt(self, prompt_file: str) -> str:
        """Load prompt file."""
        path = Path(prompt_file)
        if not path.exists():
            logger.warning(f"Prompt file not found: {prompt_file}")
            return ""

        content = path.read_text(encoding="utf-8")

        # Remove first two lines (comment line + blank line)
        lines = content.split("\n")
        if len(lines) > 2:
            content = "\n".join(lines[2:])
        else:
            logger.warning(
                f"Prompt file '{prompt_file}' has only {len(lines)} lines, "
                f"expected at least 3. Using raw content."
            )

        return content

    async def run(self) -> list[LLMOutput]:
        """Execute inference."""
        await self.setup()

        tasks = self.load_tasks()
        logger.info(f"Loaded {len(tasks)} task instances")

        if not tasks:
            raise ValueError("No tasks loaded")

        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)

        async def run_one(task: dict[str, Any]) -> LLMOutput:
            async with self._semaphore:
                return await self._execute_task(task)

        coros = [run_one(t) for t in tasks]
        outputs = await asyncio.gather(*coros)

        await self.teardown()
        return outputs

    async def _execute_task(self, task: dict[str, Any]) -> LLMOutput:
        """Execute a single task."""
        task_id = task["task_id"]
        start_time = time.time()

        try:
            # Build messages
            messages: list[dict[str, str]] = []

            record = task["record"]
            if isinstance(record, dict):
                for key, value in record.items():
                    messages.append({key: value})
            elif isinstance(record, str):
                messages.append({"record": record})

            prompt = task.get("prompt", "")
            if prompt:
                messages.append({"prompt": prompt})

            # Call LLM
            setup = self.config.setup_config
            gold_answer_path = self.config.task_config.get("gold_answer_path")
            result = await self.llm.chat(
                messages=messages,
                provider=self.provider,
                setup=setup,
                gold_answer_path=gold_answer_path,
            )

            latency = time.time() - start_time

            logger.info(
                f"[{task_id}] cost: ${result.get('cost', 0.0):.4f}, "
                f"time: {latency:.2f}s"
            )

            return LLMOutput(
                task_id=str(task_id),
                response=result.get("content", ""),
                cost=result.get("cost", 0.0),
                latency=latency,
                model=self.llm.model_name,
                provider=self.config.provider_config.get("type", ""),
            )

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"[{task_id}] Inference failed: {e}")
            logger.info(f"[{task_id}] cost: $0.0000, time: {latency:.2f}s (error)")

            return LLMOutput(
                task_id=str(task_id),
                response="",
                cost=0.0,
                latency=latency,
                model=self.llm.model_name if self.llm else "",
                provider=self.config.provider_config.get("type", ""),
                error=str(e),
            )

    @staticmethod
    def save_outputs(outputs: list[LLMOutput], path: str | Path) -> None:
        """Save outputs to JSONL file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for o in outputs:
                f.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Results saved: {path}")

    @staticmethod
    def load_outputs(path: str | Path) -> list[LLMOutput]:
        """Load outputs from JSONL file."""
        outputs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    outputs.append(LLMOutput.from_dict(json.loads(line)))
        return outputs
