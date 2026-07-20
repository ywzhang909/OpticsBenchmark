"""
OptiS Benchmark - LLM Prediction Runner Module

与 AgentRunner 对齐的 LLM 推理 Runner。
使用 Provider + LLM 架构执行推理任务。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils import logger

# ---------------------------------------------------------------------------
# 环境变量展开
# ---------------------------------------------------------------------------


def _expand_env_vars(data: Any) -> Any:
    """递归展开环境变量 ${VAR_NAME}。"""
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
    """单个任务的推理输出。"""

    task_id: str = ""
    response: str = ""
    cost: float = 0.0
    latency: float = 0.0
    model: str = ""
    provider: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
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
        """从字典创建。"""
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
    """LLM 推理配置。"""

    provider_config: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)
    setup_config: dict[str, Any] = field(default_factory=dict)
    task_config: dict[str, Any] = field(default_factory=dict)
    execution_config: dict[str, Any] = field(default_factory=dict)
    output_path: str = "results/llm_outputs.jsonl"
    max_concurrency: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMRunnerConfig:
        """从 YAML 文件加载配置。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

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
    """
    LLM 推理 Runner。

    执行流程：
    1. setup() - 创建 Provider 和 LLM
    2. load_tasks() - 加载数据集
    3. run() - 并发执行推理
    4. teardown() - 清理资源
    """

    def __init__(self, config: LLMRunnerConfig):
        self.config = config
        self.provider: Any | None = None
        self.llm: Any | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def setup(self) -> None:
        """创建 Provider 和 LLM 实例。"""
        from src.llm import create_llm, create_provider

        self.provider = create_provider(self.config.provider_config)
        self.llm = create_llm(self.config.model_config)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

        logger.info(f"Provider: {type(self.provider).__name__}")
        logger.info(f"LLM: {self.llm.model_name}")

    async def teardown(self) -> None:
        """清理资源。"""
        if self.provider:
            try:
                await self.provider.close()
            except Exception:
                pass

    def load_tasks(self) -> list[dict[str, Any]]:
        """加载数据集。"""
        dataset_path = self.config.task_config.get("dataset_path", "")
        if not dataset_path:
            raise ValueError("未指定 dataset_path")

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")

        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"数据集应为 JSON 数组，实际为 {type(records).__name__}")

        # 加载 prompt
        prompt = ""
        prompt_file = self.config.task_config.get("prompt_file", "")
        if prompt_file:
            prompt = self._load_prompt(prompt_file)

        # 限制样本数
        max_samples = self.config.task_config.get("max_samples")
        if max_samples is not None:
            records = records[:max_samples]

        # 打乱顺序
        if self.config.task_config.get("shuffle", False):
            import random
            random.shuffle(records)

        tasks = []
        for i, record in enumerate(records):
            tasks.append({
                "task_id": i + 1,
                "prompt": prompt,
                "record": record,
            })

        return tasks

    def _load_prompt(self, prompt_file: str) -> str:
        """加载 prompt 文件。"""
        path = Path(prompt_file)
        if not path.exists():
            logger.warning(f"Prompt 文件不存在: {prompt_file}")
            return ""

        content = path.read_text(encoding="utf-8")

        # 删除前两行（注释行 + 空行）
        lines = content.split("\n")
        if len(lines) > 2:
            content = "\n".join(lines[2:])
        else:
            logger.warning(
                f"Prompt 文件 '{prompt_file}' 仅有 {len(lines)} 行，"
                f"预期至少 3 行，使用原始内容。"
            )

        return content

    async def run(self) -> list[LLMOutput]:
        """执行推理。"""
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
        """执行单个任务。"""
        task_id = task["task_id"]
        start_time = time.time()

        try:
            # 构建消息
            messages: list[dict[str, str]] = []
            prompt = task.get("prompt", "")
            if prompt:
                messages.append({"role": "system", "content": prompt})

            # 用户消息
            record = task["record"]
            user_content = record.get("instruction", record.get("content", ""))
            messages.append({"role": "user", "content": user_content})

            # 调用 LLM
            setup = self.config.setup_config
            result = await self.llm.chat(
                messages=messages,
                provider=self.provider,
                setup=setup,
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
            logger.error(f"[{task_id}] 推理失败: {e}")
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
        """保存输出到 JSONL 文件。"""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for o in outputs:
                f.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"结果已保存: {path}")

    @staticmethod
    def load_outputs(path: str | Path) -> list[LLMOutput]:
        """从 JSONL 文件加载输出。"""
        outputs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    outputs.append(LLMOutput.from_dict(json.loads(line)))
        return outputs
