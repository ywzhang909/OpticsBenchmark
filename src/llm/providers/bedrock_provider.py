"""
Bedrock Provider - Wrapper around the AWS Bedrock boto3 client

Wraps the synchronous boto3 calls as async via asyncio.run_in_executor.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

# =============================================================================
# Classes
# =============================================================================


class BedrockProvider:
    """Wraps the AWS Bedrock boto3 client, implementing async calls via a thread pool."""

    def __init__(
        self,
        region: str = "us-east-1",
        aws_key: str | None = None,
        aws_secret: str | None = None,
    ):
        """
        Initialize the Bedrock Provider.

        Args:
            region: AWS region
            aws_key: AWS Access Key ID
            aws_secret: AWS Secret Access Key
        Raises:
            ImportError: Raised when boto3 is not installed
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is not installed. Please run: pip install boto3 or uv sync --extra providers"
            )
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=aws_key or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

    @property
    def client(self):
        """Return the boto3 client instance."""
        return self._client

    async def invoke_model_async(
        self,
        model_id: str,
        body: dict[str, Any],
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        """
        Call Bedrock invoke_model asynchronously.

        Args:
            model_id: Model ID
            body: Request body (automatically serialized to JSON)
            content_type: Content type

        Returns:
            The response body dict
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
        """Close the client (boto3 requires no explicit close)."""
        pass
