"""
OllamaLLM - 本地 Ollama 模型调用类

支持通过 OllamaProvider 进行本地 Ollama HTTP API 调用。
"""

from __future__ import annotations

import time
from typing import Any

from src.llm.base import BaseLLM
from src.llm.providers.ollama_provider import OllamaProvider

# =============================================================================
# Classes
# =============================================================================


class OllamaLLM(BaseLLM):
    """本地 Ollama 模型，支持 OllamaProvider。"""

    def __init__(self, model_name: str = "llama3.1"):
        super().__init__(model_name)

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """发送聊天请求。

        根据 provider 类型分发到对应实现，仅支持 OllamaProvider。

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            provider: Provider 实例（须为 OllamaProvider）
            **kwargs: 额外参数:
                - setup: API 调用参数字典（temperature、max_tokens 等）
                - 其他透传给底层 API 的参数

        Returns:
            {"content": str, "usage": dict, "cost": float, "latency": float}

        Raises:
            ValueError: provider 类型不受支持时
        """
        if isinstance(provider, OllamaProvider):
            return await self._chat_ollama(messages, provider, **kwargs)
        raise ValueError(
            f"OllamaLLM 不支持 provider: {type(provider).__name__}，"
            f"仅支持 OllamaProvider"
        )

    async def _chat_ollama(
        self,
        messages: list[dict[str, str]],
        provider: OllamaProvider,
        **kwargs: Any,
    ) -> dict[str, Any]:
        start_time = time.time()
        setup = kwargs.get("setup", {})

        request_body = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": setup.get("temperature", 0.0),
                "num_predict": setup.get("max_completion_tokens", 4096),
            },
        }

        # Ollama 本地模型不支持 response_format 结构化输出
        # if setup.get("response_format", False):
        #     rf = build_response_format(kwargs.get("gold_answer_path"))
        #     if rf:
        #         request_body["format"] = rf["json_schema"]["schema"]

        try:
            response = await provider.client.post("/api/chat", json=request_body)
            response.raise_for_status()
            data = response.json()
            latency = time.time() - start_time

            content = data.get("message", {}).get("content", "")

            # Ollama 本地模型无 API 费用
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
            cost = 0.0
            self._log_usage(usage, cost, latency)

            return {
                "content": content,
                "usage": usage,
                "cost": cost,
                "latency": latency,
            }
        except Exception as e:
            return {
                "content": "",
                "usage": {},
                "cost": 0.0,
                "latency": time.time() - start_time,
                "error": str(e),
            }

    async def close(self, provider: Any) -> None:
        """关闭 Provider 连接。"""
        if isinstance(provider, OllamaProvider):
            await provider.close()
