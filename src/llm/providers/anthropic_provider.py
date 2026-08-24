"""
Anthropic Provider - 封装 Anthropic AsyncAnthropic SDK

支持 Anthropic Claude 系列模型的异步调用。
"""

from __future__ import annotations

from anthropic import AsyncAnthropic


class AnthropicProvider:
    """封装 Anthropic AsyncAnthropic SDK，提供异步调用支持。"""

    def __init__(self, api_key: str):
        """
        初始化 Anthropic Provider。

        Args:
            api_key: Anthropic API 密钥
        """
        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def client(self):
        """返回 AsyncAnthropic 客户端实例。"""
        return self._client

    async def close(self) -> None:
        """关闭 API 客户端连接。"""
        await self._client.close()
