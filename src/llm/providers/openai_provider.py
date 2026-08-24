"""
OpenAI Provider - 封装 OpenAI AsyncOpenAI SDK

支持所有 OpenAI 兼容 API（OpenAI, DeepSeek, Qwen, Llama, Mistral 等）。
"""

from __future__ import annotations

from openai import AsyncOpenAI

# =============================================================================
# Classes
# =============================================================================


class OpenAIProvider:
    """封装 OpenAI AsyncOpenAI SDK，提供异步调用支持。"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        """
        初始化 OpenAI Provider。

        Args:
            api_key: API 密钥
            base_url: API 端点地址
        """
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def client(self):
        """返回 AsyncOpenAI 客户端实例。"""
        return self._client

    async def close(self) -> None:
        """关闭 API 客户端连接。"""
        await self._client.close()
