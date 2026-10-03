"""离线测试 — LLM 配置与客户端封装（不发起任何网络请求）。

使用假 LLM 客户端 / 配置对象，验证：
- LLMConfig 各 Provider 的配置字典构建与校验
- YAML 配置的加载 / 保存 / 目录发现（兼容 configs/llm/*.yaml）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from website.llm_client import (
    LLMClient,
    LLMConfig,
    discover_preset_configs,
    load_llm_config_from_yaml,
    save_llm_config_to_yaml,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------


class TestLLMConfig:
    def test_openai_provider_config(self):
        cfg = LLMConfig(api_key="sk-abc", base_url="https://example.com/v1")
        assert cfg.to_provider_config() == {
            "type": "openai",
            "api_key": "sk-abc",
            "base_url": "https://example.com/v1",
        }

    def test_anthropic_provider_config_drops_base_url(self):
        cfg = LLMConfig(
            provider_type="anthropic",
            model_type="claude",
            model_name="claude-3-5-sonnet",
            api_key="sk-ant",
            base_url="https://ignored.example",
        )
        assert cfg.to_provider_config() == {"type": "anthropic", "api_key": "sk-ant"}

    def test_ollama_provider_config_uses_host(self):
        cfg = LLMConfig(provider_type="ollama", base_url="http://localhost:11434")
        assert cfg.to_provider_config() == {
            "type": "ollama",
            "host": "http://localhost:11434",
        }

    def test_ollama_provider_config_default_host(self):
        cfg = LLMConfig(provider_type="ollama", base_url="")
        assert cfg.to_provider_config()["host"] == "http://localhost:11434"

    def test_to_setup_and_judge_config(self):
        cfg = LLMConfig(
            provider_type="openai",
            model_type="qwen",
            model_name="qwen-plus",
            api_key="k",
            base_url="https://b/v1",
            temperature=0.3,
            max_tokens=2048,
        )
        assert cfg.to_setup() == {
            "temperature": 0.3,
            "max_completion_tokens": 2048,
        }
        assert cfg.to_judge_config() == {
            "provider": "openai",
            "model": "qwen-plus",
            "api_base": "https://b/v1",
            "api_key": "k",
            "temperature": 0.3,
        }

    def test_validate_missing_api_key(self):
        cfg = LLMConfig(api_key="", base_url="https://x/v1")
        assert any("API Key 不能为空" in p for p in cfg.validate())

    def test_validate_missing_model_name(self):
        cfg = LLMConfig(api_key="k", model_name="")
        assert "模型名称不能为空" in cfg.validate()

    def test_validate_ollama_needs_no_key(self):
        cfg = LLMConfig(provider_type="ollama", api_key="", model_name="llama3.1")
        assert cfg.validate() == []

    def test_signature_unique_per_config(self):
        a = LLMConfig(api_key="k1")
        b = LLMConfig(api_key="k2")
        assert a.signature() != b.signature()


# ---------------------------------------------------------------------------
# YAML 配置读写
# ---------------------------------------------------------------------------


class TestConfigIO:
    def test_load_llm_config_from_yaml_wrapped(self, tmp_path):
        cfg_file = tmp_path / "qwen.yaml"
        cfg_file.write_text(
            "llm:\n"
            "  provider:\n"
            "    type: openai\n"
            "    api_key: sk-123\n"
            "    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1\n"
            "  model:\n"
            "    type: qwen\n"
            "    name: qwen-plus\n"
            "  setup:\n"
            "    temperature: 0.1\n"
            "    max_completion_tokens: 2048\n",
            encoding="utf-8",
        )
        cfg = load_llm_config_from_yaml(cfg_file)
        assert cfg.provider_type == "openai"
        assert cfg.model_type == "qwen"
        assert cfg.model_name == "qwen-plus"
        assert cfg.api_key == "sk-123"
        assert cfg.max_tokens == 2048
        assert cfg.temperature == pytest.approx(0.1)

    def test_save_then_load_roundtrip(self, tmp_path):
        out = tmp_path / "custom.yaml"
        cfg = LLMConfig(
            provider_type="openai",
            model_type="glm",
            model_name="glm-4-plus",
            api_key="sk-glm",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            temperature=0.5,
            max_tokens=1024,
        )
        save_llm_config_to_yaml(out, cfg)
        loaded = load_llm_config_from_yaml(out)
        assert loaded.provider_type == cfg.provider_type
        assert loaded.model_type == cfg.model_type
        assert loaded.model_name == cfg.model_name
        assert loaded.api_key == cfg.api_key
        assert loaded.base_url == cfg.base_url
        assert loaded.temperature == pytest.approx(0.5)
        assert loaded.max_tokens == 1024

    def test_discover_preset_configs(self, tmp_path):
        (tmp_path / "a.yaml").write_text(
            "llm:\n  provider:\n    type: openai\n    api_key: k\n",
            encoding="utf-8",
        )
        (tmp_path / "b.yaml").write_text(
            "llm:\n  provider:\n    type: ollama\n    host: http://localhost:11434\n"
            "  model:\n    type: ollama\n    name: llama3.1\n",
            encoding="utf-8",
        )
        presets = discover_preset_configs(tmp_path)
        assert len(presets) == 2
        assert "ollama" in presets["b（ollama / llama3.1）"].provider_type

    def test_discover_preset_configs_missing_dir(self, tmp_path):
        assert discover_preset_configs(tmp_path / "nope") == {}

    def test_discover_preset_configs_real_repo(self):
        presets = discover_preset_configs(_PROJECT_ROOT / "configs" / "llm")
        assert len(presets) >= 1


# ---------------------------------------------------------------------------
# LLMClient（不真正初始化：validate 短路）
# ---------------------------------------------------------------------------


class TestLLMClient:
    def test_chat_rejects_invalid_config_without_network(self):
        client = LLMClient(LLMConfig(api_key="", model_name=""))
        result = client.chat([{"role": "user", "content": "hi"}])
        assert not result.ok
        assert "API Key 不能为空" in result.error
        assert "模型名称不能为空" in result.error
