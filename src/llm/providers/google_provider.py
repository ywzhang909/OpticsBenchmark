"""
Google Provider - 封装 Google GenAI Client

支持 Google Gemini 系列模型的异步调用（通过 client.aio）。
"""

from __future__ import annotations

from typing import Any

from google import genai

# =============================================================================
# Classes
# =============================================================================


class GoogleProvider:
    """封装 Google GenAI Client，通过 client.aio 实现异步调用。"""

    def __init__(
        self,
        api_key: str,
        vertexai: bool = False,
        project_id: str | None = None,
        location: str | None = None,
    ):
        """
        初始化 Google Provider。

        Args:
            api_key: Google API 密钥
            vertexai: 是否使用 Vertex AI
            project_id: Google Cloud 项目 ID（Vertex AI 时需要）
            location: Google Cloud 区域（Vertex AI 时需要）
        """
        self._client = genai.Client(
            api_key=api_key,
            vertexai=vertexai,
            project_id=project_id,
            location=location,
        )

    @property
    def client(self):
        """返回 genai.Client 实例。"""
        return self._client

    @property
    def aio(self) -> Any:
        """返回异步接口 client.aio，用于 models.generate_content() 等调用。"""
        return self._client.aio

    async def close(self) -> None:
        """关闭客户端（genai.Client 无需显式关闭）。"""
        pass
