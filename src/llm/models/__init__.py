"""
LLM Models - Per-model invocation logic

Each LLM model class wraps the invocation logic for that model and can be
called through different providers.
Lazy imports are used, so missing optional dependencies do not fail at
module import time.
"""

from __future__ import annotations

from typing import Any

_MODEL_CLASSES: dict[str, str] = {
    "QwenLLM": "src.llm.models.qwen_llm",
    "DeepSeekLLM": "src.llm.models.deepseek_llm",
    "LlamaLLM": "src.llm.models.llama_llm",
    "MistralLLM": "src.llm.models.mistral_llm",
    "GeminiLLM": "src.llm.models.gemini_llm",
    "ClaudeLLM": "src.llm.models.claude_llm",
    "OllamaLLM": "src.llm.models.ollama_llm",
    "GlmLLM": "src.llm.models.glm_llm",
    "GPTLLM": "src.llm.models.gpt_llm",
    "KimiLLM": "src.llm.models.kimi_llm",
}

__all__ = list(_MODEL_CLASSES.keys())


def __getattr__(name: str) -> Any:
    """Lazy import: only imported when src.llm.models.XXX is accessed."""
    if name in _MODEL_CLASSES:
        import importlib

        mod = importlib.import_module(_MODEL_CLASSES[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
