"""
OptiS Benchmark - Quick LLM Selector

A CLI tool for quickly testing and comparing different LLM providers.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


class OutputFormat(Enum):
    """Output format for results."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass
class ProviderInfo:
    """Information about an LLM provider."""

    name: str
    config_path: Path
    provider_type: str
    model_name: str
    has_api_key: bool
    tools_enabled: list[str]

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.model_name})"


class QuickLLMSelector:
    """
    Quick LLM selection and testing tool.

    Discovers available providers from configs and provides
    a simple interface to test them.
    """

    def __init__(self, config_dir: Path | str = "configs/agents"):
        self.config_dir = Path(config_dir)
        self.providers: dict[str, ProviderInfo] = {}

    def discover_providers(self) -> dict[str, ProviderInfo]:
        """Discover available providers from config files."""
        self.providers = {}

        if not self.config_dir.exists():
            return self.providers

        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.name == "template.yaml":
                continue

            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                # Extract provider info
                model_cfg = data.get("model", {})
                agent_cfg = data.get("agent", {})
                tools_cfg = data.get("tools", {})

                provider = model_cfg.get("provider", "unknown")
                model_name = model_cfg.get("name", "unknown")
                api_key = model_cfg.get("api_key", "")
                has_key = bool(api_key and not api_key.startswith("${"))

                # Get provider name from file
                name = config_file.stem  # filename without extension
                if name == "gpt-4":
                    display_name = "GPT-4 (OpenAI)"
                elif name == "claude-3":
                    display_name = "Claude 3 (Anthropic)"
                elif name == "gemini":
                    display_name = "Gemini (Google)"
                elif name == "groq":
                    display_name = "Groq (Free)"
                elif name == "ollama":
                    display_name = "Ollama (Local)"
                elif name == "bedrock":
                    display_name = "Bedrock (AWS)"
                elif name == "together":
                    display_name = "Together AI"
                else:
                    display_name = name.replace("-", " ").title()

                self.providers[name] = ProviderInfo(
                    name=display_name,
                    config_path=config_file,
                    provider_type=provider,
                    model_name=model_name,
                    has_api_key=has_key,
                    tools_enabled=tools_cfg.get("enabled", []),
                )
            except Exception as e:
                print(f"Warning: Failed to load {config_file}: {e}", file=sys.stderr)

        return self.providers

    def list_providers(self, show_all: bool = False) -> list[ProviderInfo]:
        """List available providers."""
        providers = list(self.providers.values())

        if not show_all:
            providers = [p for p in providers if p.has_api_key]

        return sorted(providers, key=lambda p: p.name)

    def select_provider(self, provider_id: str) -> Optional[ProviderInfo]:
        """Select a provider by ID."""
        return self.providers.get(provider_id)

    async def test_provider(
        self,
        provider: ProviderInfo,
        prompt: str,
        system_prompt: str = "",
    ) -> dict:
        """Test a single provider with a prompt."""
        from src.core.agent import AgentConfig, Message, create_agent

        # Load config
        config = AgentConfig.from_yaml(provider.config_path)

        # Override system prompt if provided
        if system_prompt:
            config.system_prompt = system_prompt

        # Create agent
        agent = create_agent(config)

        try:
            # Build messages
            messages = [Message(role="user", content=prompt)]

            # Send request
            start_time = time.time()
            response = await agent.chat(messages)
            latency = time.time() - start_time

            return {
                "success": True,
                "provider": provider.display_name,
                "model": provider.model_name,
                "response": response.content,
                "latency": latency,
                "cost": response.cost,
                "tokens": response.usage.get("total_tokens", 0) if response.usage else 0,
                "finish_reason": response.finish_reason,
            }
        except Exception as e:
            return {
                "success": False,
                "provider": provider.display_name,
                "error": str(e),
            }
        finally:
            await agent.close()

    async def compare_providers(
        self,
        provider_ids: list[str],
        prompt: str,
        system_prompt: str = "",
    ) -> list[dict]:
        """Compare multiple providers with the same prompt."""
        results = []

        for pid in provider_ids:
            provider = self.select_provider(pid)
            if not provider:
                results.append(
                    {
                        "success": False,
                        "provider": pid,
                        "error": f"Provider '{pid}' not found",
                    }
                )
                continue

            print(f"Testing {provider.display_name}...", file=sys.stderr)
            result = await self.test_provider(provider, prompt, system_prompt)  # type: ignore[arg-type]
            results.append(result)

        return results

    def format_result_text(self, result: dict) -> str:
        """Format result as plain text."""
        if not result["success"]:
            return f"❌ {result['provider']}: {result.get('error', 'Unknown error')}"

        lines = [
            f"✅ {result['provider']}",
            f"   Model: {result['model']}",
            f"   Latency: {result['latency']:.2f}s",
            f"   Cost: ${result['cost']:.6f}",
            f"   Tokens: {result['tokens']}",
            "",
            "Response:",
            "-" * 40,
            result["response"],
            "-" * 40,
        ]
        return "\n".join(lines)

    def format_result_markdown(self, result: dict) -> str:
        """Format result as markdown."""
        if not result["success"]:
            return f"## ❌ {result['provider']}\n\n**Error:** {result.get('error', 'Unknown')}\n"

        return f"""## ✅ {result["provider"]}

| Metric | Value |
|--------|-------|
| Model | {result["model"]} |
| Latency | {result["latency"]:.2f}s |
| Cost | ${result["cost"]:.6f} |
| Tokens | {result["tokens"]} |

### Response

{result["response"]}
"""

    def format_result_json(self, result: dict) -> str:
        """Format result as JSON."""
        import json

        return json.dumps(result, indent=2)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Quick LLM Selector - Test and compare LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available providers
  python -m src.tools.quick_llm_selector --list

  # Interactive mode
  python -m src.tools.quick_llm_selector

  # Test with specific provider
  python -m src.tools.quick_llm_selector --provider gpt-4 --prompt "Hello"

  # Compare multiple providers
  python -m src.tools.quick_llm_selector --compare gpt-4 claude-3 gemini --prompt "Explain optics"
        """,
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available providers and exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all providers including those without API keys",
    )
    parser.add_argument(
        "--provider",
        "-p",
        help="Provider to use (e.g., gpt-4, claude-3, gemini, groq, ollama, bedrock, together)",
    )
    parser.add_argument(
        "--compare",
        "-c",
        nargs="+",
        metavar="PROVIDER",
        help="Compare multiple providers with the same prompt",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt to send",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful AI assistant.",
        help="System prompt (default: You are a helpful AI assistant.)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/agents",
        help="Directory containing agent configs (default: configs/agents)",
    )

    return parser


async def interactive_mode(selector: QuickLLMSelector) -> None:
    """Run interactive selection mode."""
    providers = selector.list_providers(show_all=True)

    if not providers:
        print("No providers found. Please add configs in configs/agents/", file=sys.stderr)
        return

    print("\n🧪 OptiS Benchmark - Quick LLM Selector")
    print("=" * 50)
    print("\nAvailable providers:\n")

    for i, p in enumerate(providers, 1):
        status = "✅" if p.has_api_key else "⚠️ "
        tools = ", ".join(p.tools_enabled[:3]) if p.tools_enabled else "none"
        print(f"  {i}. {status} {p.display_name}")
        print(f"     Model: {p.model_name} | Tools: {tools}")

    print("\n" + "-" * 50)

    # Select provider
    while True:
        try:
            choice = input("\nSelect provider number (or 'q' to quit): ").strip()
            if choice.lower() == "q":
                return
            idx = int(choice) - 1
            if 0 <= idx < len(providers):
                selected = providers[idx]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
            continue

    if not selected.has_api_key:
        print(f"\n⚠️  Warning: {selected.display_name} may not have API key configured.")
        print("   Check configs/agents/*.yaml and set API keys.\n")

    # Get prompt
    prompt = input("\nEnter your prompt: ").strip()
    if not prompt:
        print("No prompt provided.")
        return

    # Run test
    print(f"\n🔄 Testing {selected.display_name}...\n")
    result = await selector.test_provider(selected, prompt)

    print(selector.format_result_text(result))


async def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Initialize selector
    selector = QuickLLMSelector(config_dir=args.config_dir)
    selector.discover_providers()

    # List mode
    if args.list:
        providers = selector.list_providers(show_all=args.all)
        if not providers:
            print("No providers found.")
            return

        print("\nAvailable LLM Providers:")
        print("-" * 60)
        for p in providers:
            status = "✅" if p.has_api_key else "⚠️ "
            print(f"  {status} {p.display_name}")
            print(f"      Config: {p.config_path.name}")
            print(f"      Model: {p.model_name}")
            if p.tools_enabled:
                print(f"      Tools: {', '.join(p.tools_enabled)}")
            print()
        return

    # Compare mode
    if args.compare:
        if not args.prompt:
            print("Error: --prompt is required with --compare", file=sys.stderr)
            sys.exit(1)

        print(f"\n🔄 Comparing {len(args.compare)} providers...\n")
        results = await selector.compare_providers(args.compare, args.prompt, args.system)

        for result in results:
            if args.format == "markdown":
                print(selector.format_result_markdown(result))
            elif args.format == "json":
                print(selector.format_result_json(result))
            else:
                print(selector.format_result_text(result))
            print()
        return

    # Single provider mode
    if args.provider:
        provider = selector.select_provider(args.provider)
        if not provider:
            print(f"Error: Provider '{args.provider}' not found.", file=sys.stderr)
            print("Use --list to see available providers.", file=sys.stderr)
            sys.exit(1)

        if not args.prompt:
            print("Error: --prompt is required", file=sys.stderr)
            sys.exit(1)

        print(f"\n🔄 Testing {provider.display_name}...\n")
        result = await selector.test_provider(provider, args.prompt, args.system)

        if args.format == "markdown":
            print(selector.format_result_markdown(result))
        elif args.format == "json":
            print(selector.format_result_json(result))
        else:
            print(selector.format_result_text(result))
        return

    # Interactive mode
    await interactive_mode(selector)


if __name__ == "__main__":
    asyncio.run(main())
