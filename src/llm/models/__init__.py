"""
LLM Models - 各模型调用逻辑

每个 LLM 模型类封装该模型的调用逻辑，支持通过不同 Provider 进行调用。
使用延迟导入，缺失可选依赖时不会在模块导入阶段报错。
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
    """延迟导入：访问 src.llm.models.XXX 时才真正导入。"""
    if name in _MODEL_CLASSES:
        import importlib

        mod = importlib.import_module(_MODEL_CLASSES[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
