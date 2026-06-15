"""
OptiS Benchmark - Core Agent Module

This module defines the base Agent class and agent implementations
for different LLM providers (OpenAI, Anthropic, Google Gemini, Groq, Ollama, etc.).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from src.core.config import TaskConfig


class AgentProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"  # Gemini
    GROQ = "groq"
    OLLAMA = "ollama"  # Local
    BEDROCK = "bedrock"  # AWS
    TOGETHER = "together"  # Together AI
    LOCAL = "local"


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    provider: AgentProvider
    model_name: str
    api_base: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    system_prompt: str = ""
    tools_enabled: list[str] = field(default_factory=list)
    timeout: int = 300
    max_retries: int = 3

    # Provider-specific settings
    thinking_budget: int | None = None  # Anthropic specific
    ollama_host: str = "http://localhost:11434"  # Ollama specific

    # Generic extra params passed directly to the API call
    api_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        """Load configuration from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Expand environment variables
        data = cls._expand_env_vars(data)

        model_cfg = data.get("model", {})
        agent_cfg = data.get("agent", {})
        tools_cfg = data.get("tools", {})
        exec_cfg = data.get("execution", {})
        provider_cfg = data.get("provider_settings", {})

        # Load system prompt if specified
        system_prompt = ""
        system_file = data.get("system_prompt_file")
        if system_file:
            system_path = Path(system_file)
            if not system_path.is_absolute():
                system_path = Path("prompts/system") / system_file
            if system_path.exists():
                system_prompt = system_path.read_text(encoding="utf-8")

        return cls(
            name=agent_cfg.get("name", "unnamed-agent"),
            provider=AgentProvider(model_cfg.get("provider", "openai")),
            model_name=model_cfg.get("name", "gpt-4"),
            api_base=model_cfg.get("api_base", ""),
            api_key=model_cfg.get("api_key", ""),
            temperature=model_cfg.get("temperature", 0.0),
            max_tokens=model_cfg.get("max_tokens", 4096),
            top_p=model_cfg.get("top_p", 1.0),
            frequency_penalty=model_cfg.get("frequency_penalty", 0.0),
            presence_penalty=model_cfg.get("presence_penalty", 0.0),
            system_prompt=system_prompt,
            tools_enabled=tools_cfg.get("enabled", []),
            timeout=exec_cfg.get("timeout", 300),
            max_retries=exec_cfg.get("max_retries", 3),
            thinking_budget=provider_cfg.get("thinking_budget"),
            ollama_host=provider_cfg.get("ollama_host", "http://localhost:11434"),
            api_params=model_cfg.get("api_params", {}),
        )

    @staticmethod
    def _expand_env_vars(data: Any) -> Any:
        """Recursively expand environment variables in data."""
        if isinstance(data, str):
            if data.startswith("${") and data.endswith("}"):
                env_var = data[2:-1]
                return os.environ.get(env_var, "")
            return data
        elif isinstance(data, dict):
            return {k: AgentConfig._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [AgentConfig._expand_env_vars(item) for item in data]
        return data


def _dict_to_response_format(
    name: str, d: dict[str, Any], strict: bool = True
) -> dict[str, Any]:
    """Generate a Chat Completions response_format from a Python dict, inferring types."""

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
            if strict:
                obj["additionalProperties"] = False
            return obj
        return {"type": "string"}

    schema = _infer_type(d)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": strict,
        },
    }


@dataclass
class Message:
    """A message in a conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None


@dataclass
class ToolCall:
    """A tool call made by the agent."""

    id: str
    name: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None


@dataclass
class AgentOutput:
    """Output from running an agent on a single task."""

    task_id: str = ""
    response: Any = ""
    finish_reason: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    cost: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    latency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "response": self.response,
            "finish_reason": self.finish_reason,
            "cost": self.cost,
            "usage": self.usage,
            "latency": self.latency,
        }

    def to_dataset_format(self) -> dict[str, Any]:
        """Convert to a format suitable for datasets."""
        return {
            "id": self.task_id,
            "data": self.response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentOutput:
        return cls(**data)

class BaseAgent(ABC):
    """
    Base class for all agents.

    This abstract class defines the interface that all agent
    implementations must follow.
    """

    def __init__(self, agent_config: AgentConfig, task_config: TaskConfig):
        """Initialize the agent with configuration."""
        self.agent_config = agent_config
        self.task_config = task_config
        self.conversation_history: list[Message] = []
        self.total_cost = 0.0
        self.total_tokens = 0
        self.file_path: str | None = None

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AgentOutput:
        """
        Send a chat request to the agent.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            AgentOutput containing the model's response
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass

    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []

    def set_file(self, path: str | None) -> None:
        """Set file path for file upload."""
        self.file_path = path

    def add_developer_message(self, content: str) -> None:
        """Add a developer message to the conversation."""
        self.conversation_history.append(Message(role="developer", content=content))

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.conversation_history.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.conversation_history.append(Message(role="assistant", content=content))

    def get_statistics(self) -> dict[str, Any]:
        """Get agent usage statistics."""
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "conversation_length": len(self.conversation_history),
        }


class OpenAIAgent(BaseAgent):
    """Agent implementation for OpenAI models."""

    def __init__(self, agent_config: AgentConfig, task_config: TaskConfig):
        super().__init__(agent_config, task_config)
        from openai import AsyncOpenAI

        api_base = self.agent_config.api_base or "https://api.openai.com/v1"
        self.client = AsyncOpenAI(
            api_key=self.agent_config.api_key,
            base_url=api_base,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AgentOutput:
        """Send chat request to OpenAI Chat Completions API."""
        import time

        start_time = time.time()

        # Build messages for Chat Completions
        chat_messages = []
        for msg in messages:
            if msg.role == "developer":
                chat_messages.append({"role": "system", "content": msg.content})
            elif msg.role == "assistant":
                if msg.content:
                    chat_messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == "user" and self.file_path:
                content = msg.content
                file_path_obj = Path(self.file_path)
                if file_path_obj.exists():
                    file_content = file_path_obj.read_text(encoding="utf-8")
                    content = f"File content ({self.file_path}):\n```\n{file_content}\n```\n\nUser query: {msg.content}"
                chat_messages.append({"role": "user", "content": content})
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        # Core request params
        kwargs = {
            "model": self.agent_config.model_name,
            "messages": chat_messages,
            "temperature": self.agent_config.temperature,
            "max_tokens": self.agent_config.max_tokens,
            "top_p": self.agent_config.top_p,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # api_params overrides any of the above or adds extras
        # (stop, seed, presence_penalty, frequency_penalty, etc.)
        kwargs.update(self.agent_config.api_params)

        # Structured output via JSON schema from gold answer file
        first_answer = None
        if self.task_config.task.get("structured_output", False) and self.task_config.task.get("gold_answer_path"):
            gold_path = Path(self.task_config.task.get("gold_answer_path"))
            if gold_path.exists():
                with open(gold_path, encoding="utf-8") as f:
                    gold_data = json.load(f)
                if isinstance(gold_data, list) and gold_data:
                    first = gold_data[0]
                    first_answer = first
                    payload = first.get("data", first)
                    if isinstance(payload, dict):
                        kwargs["response_format"] = _dict_to_response_format(
                            f"{self.task_config.task.get('id', 'Unknown')}_schema", payload, strict=True
                        )

        try:
            response = await self.client.chat.completions.create(**kwargs)

            latency = time.time() - start_time

            # Parse response
            choice = response.choices[0]
            content = choice.message.content or ""

            calls: list[ToolCall] = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=json.loads(tc.function.arguments),
                        )
                    )

            finish_reason = choice.finish_reason or "stop"
            usage = response.usage.model_dump() if response.usage else {}
            cost = self._calculate_cost(usage)
            self.total_cost += cost
            self.total_tokens += usage.get("total_tokens", 0)

            return AgentOutput(
                task_id=first_answer.get("id", "") if first_answer else "",
                response=content,
                finish_reason=finish_reason,
                tool_calls=calls,
                cost=cost,
                usage=usage,
                latency=latency,
            )
        except Exception:
            return AgentOutput(
                response="",
                finish_reason="error",
                cost=0.0,
                latency=time.time() - start_time,
            )

    def _calculate_cost(self, usage: dict[str, int]) -> float:
        """Calculate API cost based on token usage."""
        # GPT-4 Turbo pricing
        input_cost_per_1k = 0.01
        output_cost_per_1k = 0.03

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self) -> None:
        """Close the API client."""
        await self.client.close()


class AnthropicAgent(BaseAgent):
    """Agent implementation for Anthropic models."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        from anthropic import AsyncAnthropic

        api_base = self.config.api_base or "https://api.anthropic.com"
        self.client = AsyncAnthropic(
            api_key=self.config.api_key,
            base_url=api_base,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AgentOutput:
        """Send chat request to Anthropic API."""
        import time

        start_time = time.time()

        # Convert messages to API format
        api_messages = []
        system_content = ""

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                api_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        # Prepare request kwargs
        kwargs = {
            "model": self.config.model_name,
            "messages": api_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if system_content:
            kwargs["system"] = system_content

        # Anthropic extended thinking
        if self.config.thinking_budget:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.config.thinking_budget,
            }

        if tools:
            kwargs["tools"] = tools

        try:
            response = await self.client.messages.create(**kwargs)

            latency = time.time() - start_time

            # Parse response
            content = ""
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )

            # Calculate cost
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            cost = self._calculate_cost(usage)
            self.total_cost += cost
            self.total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            return AgentOutput(
                response=content,
                tool_calls=tool_calls,
                finish_reason="stop",
                usage=usage,
                cost=cost,
                latency=latency,
            )
        except Exception:
            return AgentOutput(
                response="",
                tool_calls=[],
                finish_reason="error",
                cost=0.0,
                latency=time.time() - start_time,
            )

    def _calculate_cost(self, usage: dict[str, int]) -> float:
        """Calculate API cost based on token usage."""
        # Claude 3.5 Sonnet pricing
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self) -> None:
        """Close the API client."""
        await self.client.close()


class GeminiAgent(BaseAgent):
    """Agent implementation for Google Gemini models."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        import google.genai as genai

        genai.configure(api_key=self.config.api_key)
        self.client = genai.GenerativeModel(self.config.model_name)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AgentOutput:
        """Send chat request to Google Gemini API."""
        import time

        start_time = time.time()

        # Convert messages to Gemini format
        contents = []
        system_instruction = ""

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg.content}]})

        try:
            generation_config = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }

            response = await self.client.generate_content_async(
                contents=contents,
                generation_config=generation_config,
                system_instruction=system_instruction if system_instruction else None,
            )

            latency = time.time() - start_time

            content = response.text

            # Estimate usage (Gemini doesn't always return token counts)
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count
                if hasattr(response.usage_metadata, "prompt_token_count")
                else 0,
                "completion_tokens": response.usage_metadata.candidates_token_count
                if hasattr(response.usage_metadata, "candidates_token_count")
                else len(content) // 4,
            }

            cost = self._calculate_cost(usage)
            self.total_cost += cost
            self.total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

            return AgentOutput(
                response=content,
                tool_calls=[],
                finish_reason="stop",
                usage=usage,
                cost=cost,
                latency=latency,
            )
        except Exception:
            return AgentOutput(
                response="",
                tool_calls=[],
                finish_reason="error",
                cost=0.0,
                latency=time.time() - start_time,
            )

    def _calculate_cost(self, usage: dict[str, int]) -> float:
        """Calculate API cost based on token usage."""
        # Gemini 1.5 Pro pricing (approximate)
        input_cost_per_1k = 0.00125
        output_cost_per_1k = 0.005

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self) -> None:
        """Close the API client (no-op for Gemini)."""
        pass


class GroqAgent(BaseAgent):
    """Agent implementation for Groq models."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        from groq import AsyncGroq

        self.client = AsyncGroq(api_key=self.config.api_key)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AgentOutput:
        """Send chat request to Groq API."""
        import time

        start_time = time.time()

        # Convert messages to API format
        api_messages = []
        for msg in messages:
            if msg.role == "system":
                continue  # Groq handles system differently
            api_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        # Add system message as first user message if exists
        for msg in messages:
            if msg.role == "system":
                api_messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": msg.content,
                    },
                )
                break

        kwargs = {
            "model": self.config.model_name,
            "messages": api_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        try:
            response = await self.client.chat.completions.create(**kwargs)

            latency = time.time() - start_time
            choice = response.choices[0]

            content = choice.message.content or ""
            usage = response.usage.model_dump() if response.usage else {}
            cost = self._calculate_cost(usage)

            self.total_cost += cost
            self.total_tokens += usage.get("total_tokens", 0)

            return AgentOutput(
                response=content,
                tool_calls=[],
                finish_reason=choice.finish_reason or "stop",
                usage=usage,
                cost=cost,
                latency=latency,
            )
        except Exception:
            return AgentOutput(
                response="",
                tool_calls=[],
                finish_reason="error",
                cost=0.0,
                latency=time.time() - start_time,
            )

    def _calculate_cost(self, usage: dict[str, int]) -> float:
        """Calculate API cost based on token usage."""
        # Groq pricing (varies by model)
        input_cost_per_1k = 0.0006  # llama-3.1-70b
        output_cost_per_1k = 0.0006

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self) -> None:
        """Close the API client."""
        await self.client.close()


class OllamaAgent(BaseAgent):
    """Agent implementation for Ollama (local models)."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        import httpx

        self.host = self.config.ollama_host
        self.client = httpx.AsyncClient(base_url=self.host, timeout=120.0)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AgentOutput:
        """Send chat request to Ollama API."""
        import time

        start_time = time.time()

        # Convert messages to Ollama format
        api_messages = []
        for msg in messages:
            api_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        kwargs = {
            "model": self.config.model_name,
            "messages": api_messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        try:
            response = await self.client.post("/api/chat", json=kwargs)
            response.raise_for_status()
            data = response.json()

            latency = time.time() - start_time

            content = data.get("message", {}).get("content", "")

            # Ollama doesn't provide token counts in response
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

            # Local models have no API cost
            cost = 0.0
            self.total_cost += cost

            return AgentOutput(
                response=content,
                tool_calls=[],
                finish_reason="stop",
                usage=usage,
                cost=cost,
                latency=latency,
            )
        except Exception:
            return AgentOutput(
                response="",
                tool_calls=[],
                finish_reason="error",
                cost=0.0,
                latency=time.time() - start_time,
            )

    def _calculate_cost(self, usage: dict[str, int]) -> float:
        """Calculate API cost - Ollama is free."""
        return 0.0

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


class BedrockAgent(BaseAgent):
    """Agent implementation for AWS Bedrock models."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        import boto3

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=config.api_base or "us-east-1",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AgentOutput:
        """Send chat request to AWS Bedrock API."""
        import time

        start_time = time.time()

        # Convert messages to Bedrock format (varies by model)
        # Using Claude on Bedrock format
        api_messages = []
        for msg in messages:
            if msg.role == "system":
                api_messages.append(
                    {
                        "role": "user",
                        "content": f"<system>{msg.content}</system>",
                    }
                )
            else:
                api_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_tokens,
            "messages": api_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

        try:
            response = await self.client.invoke_model_async(
                modelId=self.config.model_name,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            latency = time.time() - start_time
            response_body = json.loads(response.get("body").read())

            content = response_body.get("content", [{}])[0].get("text", "")

            usage = response_body.get("usage", {})
            cost = self._calculate_cost(usage)

            self.total_cost += cost

            return AgentOutput(
                response=content,
                tool_calls=[],
                finish_reason="stop",
                usage=usage,
                cost=cost,
                latency=latency,
            )
        except Exception:
            return AgentOutput(
                response="",
                tool_calls=[],
                finish_reason="error",
                cost=0.0,
                latency=time.time() - start_time,
            )

    def _calculate_cost(self, usage: dict[str, int]) -> float:
        """Calculate API cost based on token usage."""
        # Bedrock pricing varies by model, using Claude 3 as default
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self) -> None:
        """Close the API client (no-op for boto3)."""
        pass


class TogetherAIAgent(BaseAgent):
    """Agent implementation for Together AI models."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        import httpx

        api_base = self.config.api_base or "https://api.together.xyz"
        self.client = httpx.AsyncClient(
            base_url=api_base,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AgentOutput:
        """Send chat request to Together AI API."""
        import time

        start_time = time.time()

        # Convert messages to API format
        api_messages = []
        for msg in messages:
            api_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        body = {
            "model": self.config.model_name,
            "messages": api_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        try:
            response = await self.client.post("/v1/chat/completions", json=body)
            response.raise_for_status()
            data = response.json()

            latency = time.time() - start_time

            choice = data["choices"][0]
            content = choice.get("message", {}).get("content", "")

            usage = data.get("usage", {})
            cost = self._calculate_cost(usage)

            self.total_cost += cost
            self.total_tokens += usage.get("total_tokens", 0)

            return AgentOutput(
                response=content,
                tool_calls=[],
                finish_reason=choice.get("finish_reason", "stop"),
                usage=usage,
                cost=cost,
                latency=latency,
            )
        except Exception:
            return AgentOutput(
                response="",
                tool_calls=[],
                finish_reason="error",
                cost=0.0,
                latency=time.time() - start_time,
            )

    def _calculate_cost(self, usage: dict[str, int]) -> float:
        """Calculate API cost based on token usage."""
        # Together AI pricing varies by model
        input_cost_per_1k = 0.001
        output_cost_per_1k = 0.001

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return (input_tokens / 1000) * input_cost_per_1k + (
            output_tokens / 1000
        ) * output_cost_per_1k

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


def create_agent(agent_config: AgentConfig | str | Path,
                 task_config: TaskConfig | str | Path) -> BaseAgent:
    """
    Factory function to create an agent instance.

    Args:
        agent_config: AgentConfig object or path to config file
        task_config: TaskConfig object or path to config file

    Returns:
        Configured agent instance
    """
    if isinstance(agent_config, (str, Path)):
        agent_config = AgentConfig.from_yaml(agent_config)

    if isinstance(task_config, (str, Path)):
        task_config = TaskConfig.from_yaml(task_config)

    provider_map = {
        AgentProvider.OPENAI: OpenAIAgent,
        AgentProvider.ANTHROPIC: AnthropicAgent,
        AgentProvider.GOOGLE: GeminiAgent,
        AgentProvider.GROQ: GroqAgent,
        AgentProvider.OLLAMA: OllamaAgent,
        AgentProvider.BEDROCK: BedrockAgent,
        AgentProvider.TOGETHER: TogetherAIAgent,
        AgentProvider.LOCAL: OllamaAgent,  # Local defaults to Ollama
    }

    agent_class = provider_map.get(agent_config.provider)
    if agent_class is None:
        raise ValueError(f"Unsupported provider: {agent_config.provider}")

    return agent_class(agent_config, task_config)
