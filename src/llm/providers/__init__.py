"""
LLM Providers - Async wrappers around vendor SDKs

Each Provider wraps the corresponding vendor SDK client and provides async call
support. Imports are deferred, so a missing optional SDK does not raise at
module import time.
"""

from __future__ import annotations

from typing import Any

_PROVIDER_CLASSES: dict[str, str] = {
    "OpenAIProvider": "src.llm.providers.openai_provider",
    "AnthropicProvider": "src.llm.providers.anthropic_provider",
    "GoogleProvider": "src.llm.providers.google_provider",
    "OllamaProvider": "src.llm.providers.ollama_provider",
    "BedrockProvider": "src.llm.providers.bedrock_provider",
    "TogetherAIProvider": "src.llm.providers.together_ai_provider",
}

__all__ = list(_PROVIDER_CLASSES.keys())


def __getattr__(name: str) -> Any:
    """Deferred import: the real import happens only on access to src.llm.providers.XXX."""
    if name in _PROVIDER_CLASSES:
        import importlib

        mod = importlib.import_module(_PROVIDER_CLASSES[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
