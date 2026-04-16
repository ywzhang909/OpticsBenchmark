"""
OptiS Benchmark - Quick LLM Selector Tests

Tests for the QuickLLMSelector tool.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from src.tools.quick_llm_selector import (
    QuickLLMSelector,
    ProviderInfo,
    OutputFormat,
)


class TestProviderInfo:
    """Tests for ProviderInfo dataclass."""

    def test_display_name(self):
        """Test display name formatting."""
        provider = ProviderInfo(
            name="GPT-4 (OpenAI)",
            config_path=Path("configs/agents/gpt-4.yaml"),
            provider_type="openai",
            model_name="gpt-4-turbo",
            has_api_key=True,
            tools_enabled=["file_read", "bash_execute"],
        )

        assert provider.display_name == "GPT-4 (OpenAI)"
        assert provider.provider_type == "openai"
        assert provider.has_api_key is True
        assert len(provider.tools_enabled) == 2


class TestQuickLLMSelector:
    """Tests for QuickLLMSelector."""

    def test_init_default_config_dir(self):
        """Test initialization with default config directory."""
        selector = QuickLLMSelector()

        assert selector.config_dir == Path("configs/agents")
        assert selector.providers == {}

    def test_init_custom_config_dir(self):
        """Test initialization with custom config directory."""
        custom_dir = Path("custom/configs")
        selector = QuickLLMSelector(config_dir=custom_dir)

        assert selector.config_dir == custom_dir

    def test_discover_providers_nonexistent_dir(self):
        """Test discovery with non-existent directory."""
        selector = QuickLLMSelector(config_dir="nonexistent/path")
        providers = selector.discover_providers()

        assert providers == {}
        assert selector.providers == {}

    def test_discover_providers_loads_configs(self, tmp_path):
        """Test discovery loads valid YAML configs."""
        # Create mock config directory
        config_dir = tmp_path / "agents"
        config_dir.mkdir()

        # Create a mock config file
        config_file = config_dir / "test-provider.yaml"
        config_file.write_text("""
model:
  provider: openai
  name: gpt-4
  api_key: sk-test123
agent:
  name: Test Agent
tools:
  enabled:
    - file_read
    - bash_execute
""")

        selector = QuickLLMSelector(config_dir=config_dir)
        providers = selector.discover_providers()

        assert "test-provider" in providers
        assert providers["test-provider"].provider_type == "openai"
        assert providers["test-provider"].model_name == "gpt-4"
        assert providers["test-provider"].has_api_key is True

    def test_discover_providers_skips_template(self, tmp_path):
        """Test that template.yaml is skipped."""
        config_dir = tmp_path / "agents"
        config_dir.mkdir()

        # Create template file
        template = config_dir / "template.yaml"
        template.write_text("model:\n  provider: openai\n  name: template")

        selector = QuickLLMSelector(config_dir=config_dir)
        selector.discover_providers()

        assert "template" not in selector.providers

    def test_discover_providers_handles_invalid_yaml(self, tmp_path):
        """Test that invalid YAML files are handled gracefully."""
        config_dir = tmp_path / "agents"
        config_dir.mkdir()

        # Create invalid config
        invalid = config_dir / "invalid.yaml"
        invalid.write_text("invalid: yaml: content:")

        selector = QuickLLMSelector(config_dir=config_dir)
        providers = selector.discover_providers()

        # Should not crash, just skip the invalid file
        assert "invalid" not in providers

    def test_list_providers_filters_no_key(self, tmp_path):
        """Test that providers without API keys can be filtered."""
        config_dir = tmp_path / "agents"
        config_dir.mkdir()

        # Create provider with key
        with_key = config_dir / "with-key.yaml"
        with_key.write_text("""
model:
  provider: openai
  name: gpt-4
  api_key: sk-actual-key
""")

        # Create provider without key
        without_key = config_dir / "without-key.yaml"
        without_key.write_text("""
model:
  provider: anthropic
  name: claude-3
  api_key: ""
""")

        selector = QuickLLMSelector(config_dir=config_dir)
        selector.discover_providers()

        all_providers = selector.list_providers(show_all=True)
        with_key_providers = selector.list_providers(show_all=False)

        assert len(all_providers) == 2
        assert len(with_key_providers) == 1
        assert with_key_providers[0].config_path.name == "with-key.yaml"

    def test_select_provider_exists(self, tmp_path):
        """Test selecting an existing provider."""
        config_dir = tmp_path / "agents"
        config_dir.mkdir()

        config_file = config_dir / "test-provider.yaml"
        config_file.write_text("""
model:
  provider: openai
  name: gpt-4
""")

        selector = QuickLLMSelector(config_dir=config_dir)
        selector.discover_providers()

        provider = selector.select_provider("test-provider")
        assert provider is not None
        assert provider.provider_type == "openai"

    def test_select_provider_not_exists(self, tmp_path):
        """Test selecting a non-existent provider."""
        selector = QuickLLMSelector(config_dir=tmp_path)
        selector.discover_providers()

        provider = selector.select_provider("nonexistent")
        assert provider is None


class TestQuickLLMSelectorOutputFormatting:
    """Tests for output formatting."""

    @pytest.fixture
    def selector(self):
        return QuickLLMSelector()

    @pytest.fixture
    def success_result(self):
        return {
            "success": True,
            "provider": "GPT-4 (OpenAI)",
            "model": "gpt-4-turbo",
            "response": "Hello! How can I help you?",
            "latency": 1.5,
            "cost": 0.002,
            "tokens": 150,
            "finish_reason": "stop",
        }

    @pytest.fixture
    def failure_result(self):
        return {
            "success": False,
            "provider": "Claude 3 (Anthropic)",
            "error": "API rate limit exceeded",
        }

    def test_format_result_text_success(self, selector, success_result):
        """Test text formatting of successful result."""
        output = selector.format_result_text(success_result)

        assert "✅ GPT-4 (OpenAI)" in output
        assert "gpt-4-turbo" in output
        assert "1.50s" in output or "1.5" in output
        assert "0.002" in output
        assert "Hello! How can I help you?" in output

    def test_format_result_text_failure(self, selector, failure_result):
        """Test text formatting of failed result."""
        output = selector.format_result_text(failure_result)

        assert "❌ Claude 3 (Anthropic)" in output
        assert "API rate limit exceeded" in output

    def test_format_result_markdown_success(self, selector, success_result):
        """Test markdown formatting of successful result."""
        output = selector.format_result_markdown(success_result)

        assert "## ✅ GPT-4 (OpenAI)" in output
        assert "| Model |" in output
        assert "| Latency |" in output
        assert "### Response" in output
        assert "Hello! How can I help you?" in output

    def test_format_result_markdown_failure(self, selector, failure_result):
        """Test markdown formatting of failed result."""
        output = selector.format_result_markdown(failure_result)

        assert "## ❌ Claude 3 (Anthropic)" in output
        assert "**Error:**" in output

    def test_format_result_json(self, selector, success_result):
        """Test JSON formatting."""
        import json

        output = selector.format_result_json(success_result)
        parsed = json.loads(output)

        assert parsed["success"] is True
        assert parsed["provider"] == "GPT-4 (OpenAI)"
        assert parsed["model"] == "gpt-4-turbo"


class TestQuickLLMSelectorCLI:
    """Tests for CLI argument parsing."""

    def test_create_parser(self):
        """Test that parser is created successfully."""
        from src.tools.quick_llm_selector import create_parser

        parser = create_parser()

        assert parser is not None

    def test_parser_list_flag(self):
        """Test --list flag parsing."""
        from src.tools.quick_llm_selector import create_parser

        parser = create_parser()
        args = parser.parse_args(["--list"])

        assert args.list is True
        assert args.provider is None

    def test_parser_provider_and_prompt(self):
        """Test --provider and --prompt parsing."""
        from src.tools.quick_llm_selector import create_parser

        parser = create_parser()
        args = parser.parse_args(["--provider", "gpt-4", "--prompt", "Hello"])

        assert args.provider == "gpt-4"
        assert args.prompt == "Hello"

    def test_parser_compare_flag(self):
        """Test --compare flag parsing."""
        from src.tools.quick_llm_selector import create_parser

        parser = create_parser()
        args = parser.parse_args(["--compare", "gpt-4", "claude-3", "gemini"])

        assert args.compare == ["gpt-4", "claude-3", "gemini"]

    def test_parser_format_flag(self):
        """Test --format flag parsing."""
        from src.tools.quick_llm_selector import create_parser

        parser = create_parser()

        args = parser.parse_args(["--format", "markdown"])
        assert args.format == "markdown"

        args = parser.parse_args(["--format", "json"])
        assert args.format == "json"

        args = parser.parse_args(["--format", "text"])
        assert args.format == "text"

    def test_parser_short_flags(self):
        """Test short flag aliases."""
        from src.tools.quick_llm_selector import create_parser

        parser = create_parser()

        args = parser.parse_args(["-l"])
        assert args.list is True

        args = parser.parse_args(["-p", "gemini", "-f", "json"])
        assert args.provider == "gemini"
        assert args.format == "json"


class TestQuickLLMSelectorAsync:
    """Tests for async functionality."""

    @pytest.mark.asyncio
    async def test_test_provider_returns_result(self, tmp_path):
        """Test that test_provider returns a proper result dict."""
        from src.tools.quick_llm_selector import QuickLLMSelector

        # Create minimal config
        config_dir = tmp_path / "agents"
        config_dir.mkdir()

        config_file = config_dir / "mock-agent.yaml"
        config_file.write_text("""
model:
  provider: openai
  name: gpt-4
  api_key: ${NONEXISTENT_KEY}
agent:
  name: Mock Agent
""")

        selector = QuickLLMSelector(config_dir=config_dir)
        selector.discover_providers()

        provider = selector.select_provider("mock-agent")

        # This will fail due to missing API key, but should return error dict
        result = await selector.test_provider(provider, "Hello")

        assert "success" in result
        assert "provider" in result

    @pytest.mark.asyncio
    async def test_compare_providers_handles_missing(self, tmp_path):
        """Test that compare_providers handles missing providers."""
        from src.tools.quick_llm_selector import QuickLLMSelector

        config_dir = tmp_path / "agents"
        config_dir.mkdir()

        selector = QuickLLMSelector(config_dir=config_dir)
        selector.discover_providers()

        results = await selector.compare_providers(["nonexistent"], "Test prompt")

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "not found" in results[0]["error"]
