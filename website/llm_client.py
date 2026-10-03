"""
OptiS 文献处理工作台 — LLM 客户端封装

基于 ``src/llm`` 抽象层（``create_provider`` / ``create_llm``）构建的同步
聊天客户端，供 Streamlit 应用使用。

- :class:`LLMConfig` — 用户可配置的模型参数（provider、模型、密钥、端点等）
- :class:`ChatResult` — 一次聊天的结构化返回（内容、用量、成本、延迟、错误）
- :class:`LLMClient` — 同步包装器，内部通过 ``asyncio.run`` 调用异步 ``chat``
- 配置读写 — 兼容 ``configs/llm/*.yaml`` 的加载与保存
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.llm import create_llm, create_provider

# ---------------------------------------------------------------------------
# 内置预设
# ---------------------------------------------------------------------------

#: 友好的预设名称 → 默认配置字段
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "OpenAI（官方）": {
        "provider_type": "openai",
        "model_type": "qwen",
        "model_name": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
    },
    "Qwen（阿里云百炼）": {
        "provider_type": "openai",
        "model_type": "qwen",
        "model_name": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "DeepSeek": {
        "provider_type": "openai",
        "model_type": "deepseek",
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
    },
    "GLM（智谱 AI）": {
        "provider_type": "openai",
        "model_type": "glm",
        "model_name": "glm-4-plus",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
    "Anthropic Claude": {
        "provider_type": "anthropic",
        "model_type": "claude",
        "model_name": "claude-3-5-sonnet-20241022",
        "base_url": "",
    },
    "Google Gemini": {
        "provider_type": "google",
        "model_type": "gemini",
        "model_name": "gemini-1.5-pro",
        "base_url": "",
    },
    "Groq": {
        "provider_type": "groq",
        "model_type": "groq",
        "model_name": "llama-3.3-70b-versatile",
        "base_url": "",
    },
    "Ollama（本地）": {
        "provider_type": "ollama",
        "model_type": "ollama",
        "model_name": "llama3.1",
        "base_url": "http://localhost:11434",
    },
    "Together AI": {
        "provider_type": "together",
        "model_type": "llama",
        "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "base_url": "https://api.together.xyz",
    },
    "自定义（OpenAI 兼容）": {
        "provider_type": "openai",
        "model_type": "qwen",
        "model_name": "",
        "base_url": "https://api.openai.com/v1",
    },
}

#: 可选 provider 类型（用于自定义配置下拉框）
PROVIDER_TYPES: list[str] = [
    "openai",
    "anthropic",
    "google",
    "groq",
    "ollama",
    "together",
    "bedrock",
]

#: 可选模型类型（与 provider 的兼容性由 src/llm 各模型类校验）
MODEL_TYPES: list[str] = [
    "qwen",
    "deepseek",
    "llama",
    "mistral",
    "gemini",
    "claude",
    "groq",
    "ollama",
    "glm",
]


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """LLM 调用配置。

    字段与 ``configs/llm/*.yaml`` 中的 ``llm.provider`` / ``llm.model`` /
    ``llm.setup`` 对应，并补充 UI 常用参数。
    """

    provider_type: str = "openai"
    model_type: str = "qwen"
    model_name: str = "gpt-4o"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    max_tokens: int = 4096

    def to_provider_config(self) -> dict[str, Any]:
        """构建 ``create_provider`` 所需的配置字典。

        各 Provider 构造参数不同（已按源码核对）：
        - openai / together: ``api_key`` + ``base_url``
        - anthropic / google / groq: 仅 ``api_key``
        - ollama: ``host``
        - bedrock: ``region`` / ``aws_key`` / ``aws_secret``（boto3，需自行安装）
        """
        cfg: dict[str, Any] = {"type": self.provider_type}
        if self.provider_type == "ollama":
            cfg["host"] = self.base_url or "http://localhost:11434"
        elif self.provider_type == "bedrock":
            cfg["region"] = self.base_url or "us-east-1"
            if self.api_key:
                cfg["aws_key"], cfg["aws_secret"] = (
                    self.api_key.split(":", 1) if ":" in self.api_key else (self.api_key, "")
                )
        elif self.provider_type in ("anthropic", "google", "groq"):
            cfg["api_key"] = self.api_key
        else:  # openai / together / 其他 OpenAI 兼容
            cfg["api_key"] = self.api_key
            if self.base_url:
                cfg["base_url"] = self.base_url
        return cfg

    def to_llm_config(self) -> dict[str, str]:
        """构建 ``create_llm`` 所需的模型配置字典。"""
        return {"type": self.model_type, "name": self.model_name}

    def to_setup(self) -> dict[str, Any]:
        """构建传给 ``chat(..., setup=...)`` 的生成参数。"""
        return {
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }

    def to_judge_config(self) -> dict[str, Any]:
        """构建 ``RubricBasedEvaluator`` 的 ``judge_config``。"""
        return {
            "provider": self.provider_type,
            "model": self.model_name or self.model_type,
            "api_base": self.base_url,
            "api_key": self.api_key,
            "temperature": self.temperature,
        }

    def signature(self) -> str:
        """唯一标识，用于 session_state 缓存 key。"""
        return "|".join(
            [
                self.provider_type,
                self.model_type,
                self.model_name,
                self.api_key,
                self.base_url,
                f"{self.temperature:g}",
                str(self.max_tokens),
            ]
        )

    def validate(self) -> list[str]:
        """返回配置问题列表（空列表 = 配置可用）。"""
        problems: list[str] = []
        if not self.model_name.strip():
            problems.append("模型名称不能为空")
        if self.provider_type == "bedrock":
            if not self.api_key:
                problems.append("Bedrock 需要 AWS Access Key:Secret（boto3 需自行安装）")
        elif self.provider_type not in ("ollama",):
            if not self.api_key.strip():
                problems.append("API Key 不能为空（本地 Ollama 可留空）")
        return problems


@dataclass
class ChatResult:
    """一次 LLM 调用的结构化结果。

    ``error`` 非空表示调用失败（``src/llm`` 的 ``chat`` 失败时不抛异常，
    而是在返回字典中携带 ``error`` 字段；此处统一收敛到本字段）。
    """

    content: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    latency: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """是否成功。"""
        return self.error is None


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


class LLMClient:
    """基于 ``src.llm`` 工厂的同步聊天客户端。

    用法::

        client = LLMClient(LLMConfig(api_key="sk-...", model_name="gpt-4o"))
        result = client.chat([{"role": "user", "content": "你好"}])
        if result.ok:
            print(result.content)
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._provider: Any = None
        self._llm: Any = None
        self._initialized = False

    # -- 初始化 -----------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """惰性创建 provider 与 LLM 实例（仅一次）。"""
        if self._initialized:
            return
        self._provider = create_provider(self.config.to_provider_config())
        self._llm = create_llm(self.config.to_llm_config())
        self._initialized = True

    # -- 同步调用 ----------------------------------------------------------

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResult:
        """同步发送聊天请求。

        Args:
            messages: 消息列表 ``[{"role": "user", "content": "..."}]``。
            **kwargs: 透传给底层 ``chat`` 的额外参数。

        Returns:
            :class:`ChatResult`；失败时 ``error`` 字段携带错误信息，
            不会抛出异常（构造 Provider/LLM 失败除外，见 :meth:`validate`）。
        """
        problems = self.config.validate()
        if problems:
            return ChatResult(error="；".join(problems))
        try:
            self._ensure_initialized()
        except Exception as exc:  # 构造失败（如 boto3 缺失、参数错误）
            return ChatResult(error=f"初始化模型失败: {exc}")
        try:
            raw = asyncio.run(
                self._llm.chat(messages, self._provider, setup=self.config.to_setup(), **kwargs)
            )
        except ValueError as exc:  # 模型与 Provider 不兼容（如 QwenLLM + GoogleProvider）
            return ChatResult(error=f"模型与 Provider 不兼容: {exc}")
        except Exception as exc:
            return ChatResult(error=f"调用失败: {exc}")

        if not raw.get("content") and raw.get("error"):
            return ChatResult(error=str(raw["error"]))
        return ChatResult(
            content=raw.get("content", "") or "",
            usage=raw.get("usage") or {},
            cost=float(raw.get("cost") or 0.0),
            latency=float(raw.get("latency") or 0.0),
            error=raw.get("error"),
        )

    async def chat_async(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResult:
        """异步版本（供测试或异步上下文使用）。"""
        problems = self.config.validate()
        if problems:
            return ChatResult(error="；".join(problems))
        try:
            self._ensure_initialized()
        except Exception as exc:
            return ChatResult(error=f"初始化模型失败: {exc}")
        try:
            raw = await self._llm.chat(
                messages, self._provider, setup=self.config.to_setup(), **kwargs
            )
        except ValueError as exc:
            return ChatResult(error=f"模型与 Provider 不兼容: {exc}")
        except Exception as exc:
            return ChatResult(error=f"调用失败: {exc}")
        return ChatResult(
            content=raw.get("content", "") or "",
            usage=raw.get("usage") or {},
            cost=float(raw.get("cost") or 0.0),
            latency=float(raw.get("latency") or 0.0),
            error=raw.get("error"),
        )

    def close(self) -> None:
        """释放底层连接（幂等）。"""
        if not self._initialized:
            return
        try:
            asyncio.run(self._llm.close(self._provider))
        except Exception:
            pass
        self._initialized = False


# ---------------------------------------------------------------------------
# 配置读写（兼容 configs/llm/*.yaml）
# ---------------------------------------------------------------------------


def load_llm_config_from_yaml(path: str | Path) -> LLMConfig:
    """从 YAML 加载 :class:`LLMConfig`。

    兼容两种结构：带 ``llm:`` 包装的 ``configs/llm/*.yaml`` 格式，以及
    平铺的 ``{provider: {...}, model: {...}, setup: {...}}`` 格式。
    ``${ENV_VAR}`` 会被展开。
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    llm = data.get("llm", data)
    provider = llm.get("provider", {}) or {}
    model = llm.get("model", {}) or {}
    setup = llm.get("setup", {}) or {}

    provider_type = provider.get("type", "openai")
    cfg = LLMConfig(
        provider_type=provider_type,
        model_type=model.get("type", "qwen"),
        model_name=model.get("name", ""),
        api_key=_expand_env(provider.get("api_key", "")),
        base_url=_expand_env(provider.get("base_url", "") or provider.get("host", "")),
        temperature=float(setup.get("temperature", 0.0)),
        max_tokens=int(setup.get("max_completion_tokens", setup.get("max_tokens", 4096))),
    )
    if provider_type == "ollama" and not cfg.base_url:
        cfg.base_url = _expand_env(provider.get("host", "http://localhost:11434"))
    return cfg


def save_llm_config_to_yaml(path: str | Path, config: LLMConfig) -> None:
    """将 :class:`LLMConfig` 保存为 ``configs/llm/*.yaml`` 风格 YAML。"""
    provider: dict[str, Any] = {"type": config.provider_type}
    if config.provider_type == "ollama":
        provider["host"] = config.base_url or "http://localhost:11434"
    elif config.provider_type in ("anthropic", "google", "groq"):
        provider["api_key"] = config.api_key
    else:
        provider["api_key"] = config.api_key
        if config.base_url:
            provider["base_url"] = config.base_url
    data = {
        "llm": {
            "provider": provider,
            "model": {"type": config.model_type, "name": config.model_name},
            "setup": {
                "temperature": config.temperature,
                "max_completion_tokens": config.max_tokens,
            },
        }
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def discover_preset_configs(configs_dir: str | Path = "configs/llm") -> dict[str, LLMConfig]:
    """扫描 ``configs/llm/*.yaml``，构建 ``{标签: LLMConfig}``。

    Args:
        configs_dir: 配置目录（默认项目根下 ``configs/llm``）。

    Returns:
        按文件名排序的映射；目录不存在或无 YAML 时返回空字典。
    """
    base = Path(configs_dir)
    result: dict[str, LLMConfig] = {}
    if not base.is_dir():
        return result
    for path in sorted(base.glob("*.yaml")):
        try:
            cfg = load_llm_config_from_yaml(path)
        except Exception:
            continue
        label = f"{path.stem}（{cfg.provider_type} / {cfg.model_name or cfg.model_type}）"
        result[label] = cfg
    return result


def _expand_env(value: str) -> str:
    """展开 ``${ENV_VAR}`` 环境变量引用。"""
    return os.path.expandvars(value)


__all__ = [
    "LLMConfig",
    "ChatResult",
    "LLMClient",
    "PROVIDER_PRESETS",
    "PROVIDER_TYPES",
    "MODEL_TYPES",
    "load_llm_config_from_yaml",
    "save_llm_config_to_yaml",
    "discover_preset_configs",
]
