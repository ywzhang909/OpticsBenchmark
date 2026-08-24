"""
Bedrock Provider - 封装 AWS Bedrock boto3 客户端

通过 asyncio.run_in_executor 将同步 boto3 调用包装为异步。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any


class BedrockProvider:
    """封装 AWS Bedrock boto3 客户端，通过线程池实现异步调用。"""

    def __init__(
        self,
        region: str = "us-east-1",
        aws_key: str | None = None,
        aws_secret: str | None = None,
    ):
        """
        初始化 Bedrock Provider。

        Args:
            region: AWS 区域
            aws_key: AWS Access Key ID
            aws_secret: AWS Secret Access Key
        Raises:
            ImportError: boto3 未安装时抛出
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 未安装。请运行: pip install boto3 或 uv sync --extra providers"
            )
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=aws_key or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

    @property
    def client(self):
        """返回 boto3 客户端实例。"""
        return self._client

    async def invoke_model_async(
        self,
        model_id: str,
        body: dict[str, Any],
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        """
        异步调用 Bedrock invoke_model。

        Args:
            model_id: 模型 ID
            body: 请求体（会自动序列化为 JSON）
            content_type: 内容类型

        Returns:
            响应体字典
        """
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.invoke_model(
                modelId=model_id,
                contentType=content_type,
                accept="application/json",
                body=json.dumps(body),
            ),
        )
        response_body = json.loads(response["body"].read())
        return response_body

    async def close(self) -> None:
        """关闭客户端（boto3 无需显式关闭）。"""
        pass
