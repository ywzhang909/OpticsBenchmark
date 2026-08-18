"""
LLM Base - 抽象基类
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.utils import logger


def build_response_format(gold_answer_path: str | None) -> dict[str, Any] | None:
    """从 gold_answer_path 构建 response_format schema。

    读取 gold answer JSON 文件，推断第一条数据的类型并生成 JSON schema。

    Args:
        gold_answer_path: gold answer JSON 文件路径。

    Returns:
        OpenAI response_format 字典，失败时返回 None。
    """
    if not gold_answer_path:
        return None
    gold_path_obj = Path(gold_answer_path)
    if not gold_path_obj.exists():
        return None
    with open(gold_path_obj, encoding="utf-8") as f:
        gold_data = json.load(f)
    if not isinstance(gold_data, list) or not gold_data:
        return None
    first = gold_data[0]
    payload = first.get("data", first)
    if not isinstance(payload, dict):
        return None

    def _infer_type(value: Any) -> dict[str, Any]:
        if value is None:
            return {"type": "string"}
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if isinstance(value, str):
            return {"type": "string"}
        if isinstance(value, list):
            items = {"type": "string"}
            for item in value:
                if isinstance(item, (dict, list)):
                    items = _infer_type(item)
                    break
                t = _infer_type(item)
                if t != items:
                    items = {"type": "string"}
            return {"type": "array", "items": items}
        if isinstance(value, dict):
            props = {}
            required = []
            for k, v in value.items():
                props[k] = _infer_type(v)
                required.append(k)
            obj: dict[str, Any] = {
                "type": "object",
                "properties": props,
                "required": required,
            }
            obj["additionalProperties"] = False
            return obj
        return {"type": "string"}

    schema = _infer_type(payload)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "response_schema",
            "schema": schema,
            "strict": True,
        },
    }


class BaseLLM(ABC):
    """LLM 模型抽象基类，定义统一接口。"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        发送聊天请求。

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            provider: Provider 实例
            **kwargs: 额外参数 (temperature, max_tokens 等)

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}
        """
        pass

    @staticmethod
    def _log_usage(
        usage: dict[str, Any],
        cost: float,
        latency: float,
    ) -> None:
        """打印 Token 消耗信息。"""
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens") or input_tokens + output_tokens

        logger.info(
            f"\nToken Usage:\n"
            f"  Input:    {input_tokens:>8,} tokens\n"
            f"  Output:   {output_tokens:>8,} tokens\n"
            f"  Total:    {total_tokens:>8,} tokens\n"
            f"  Cost:     ${cost:.6f}\n"
            f"  Latency:  {latency:.2f}s"
        )

    @abstractmethod
    async def close(self, provider: Any) -> None:
        """关闭 Provider 连接。"""
        pass
