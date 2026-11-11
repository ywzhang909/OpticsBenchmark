#!/usr/bin/env python3
"""List the large language models a provider exposes and save them as Markdown.

The script covers every provider wired into ``src/llm``:

* OpenAI-compatible HTTP APIs (OpenAI, Qwen/DashScope compatible mode, DeepSeek,
  GLM, Kimi, Together AI) via the OpenAI SDK ``models.list()``;
* Anthropic Claude via the Anthropic SDK;
* Google Gemini via the google-genai SDK (only models that support
  ``generateContent`` are kept);
* Mistral AI via the official Mistral SDK;
* local Ollama via ``GET /api/tags``;
* AWS Bedrock via ``boto3`` ``bedrock.list_foundation_models()``.

The provider is selected on the command line; without arguments an interactive
menu is shown. Credentials are resolved from the CLI, from an existing
``configs/llm/*.yaml`` provider block (``--config``), or from the environment
variables listed in ``PROVIDER_PRESETS`` (``.env`` is loaded from the repo root).

Usage:
    uv run python utils/list_provider_support_models.py --list-providers
    uv run python utils/list_provider_support_models.py -p qwen
    uv run python utils/list_provider_support_models.py -p openai -o models/openai_models.md
    uv run python utils/list_provider_support_models.py -p custom \\
        --base-url http://127.0.0.1:8080/v1 --api-key EMPTY
    uv run python utils/list_provider_support_models.py -p claude \\
        --config configs/llm/claude_anthropic.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models"
DEFAULT_TIMEOUT = 30.0

ENV_REF_RE = re.compile(r"\$\{(\w+)\}")


# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderPreset:
    """Static description of a provider.

    Attributes:
        kind: Fetch protocol, one of ``openai``, ``anthropic``, ``google``,
            ``mistral``, ``ollama`` or ``bedrock``.
        base_url: Default API endpoint (empty for AWS Bedrock).
        api_key_env: Environment variable holding the API key.
        columns: Extra Markdown table columns, next to the model ID.
        description: Human-readable summary for the interactive menu.
        needs_api_key: Whether a missing API key should trigger a prompt.
    """

    kind: str
    base_url: str = ""
    api_key_env: str = ""
    columns: tuple[str, ...] = ("Created (UTC)", "Owned by")
    description: str = ""
    needs_api_key: bool = True


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "openai": ProviderPreset(
        kind="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="GPT_API_KEY",
        description="OpenAI official API (GPT family)",
    ),
    "qwen": ProviderPreset(
        kind="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="QWEN_API_KEY",
        description="Alibaba Cloud Model Studio Qwen (OpenAI-compatible mode)",
    ),
    "deepseek": ProviderPreset(
        kind="openai",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        description="DeepSeek official API",
    ),
    "glm": ProviderPreset(
        kind="openai",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
        description="Zhipu GLM official API",
    ),
    "kimi": ProviderPreset(
        kind="openai",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="KIMI_API_KEY",
        description="Moonshot Kimi China API",
    ),
    "together": ProviderPreset(
        kind="openai",
        base_url="https://api.together.ai/v1",
        api_key_env="TOGETHER_API_KEY",
        description="Together AI (open-model hosting)",
    ),
    "ollama": ProviderPreset(
        kind="ollama",
        base_url="http://localhost:11434",
        columns=("Parameters", "Size", "Modified"),
        description="Local Ollama service",
        needs_api_key=False,
    ),
    "anthropic": ProviderPreset(
        kind="anthropic",
        base_url="https://api.anthropic.com",
        api_key_env="CLAUDE_API_KEY",
        columns=("Display name", "Created (UTC)"),
        description="Anthropic Claude official API",
    ),
    "google": ProviderPreset(
        kind="google",
        base_url="https://generativelanguage.googleapis.com",
        api_key_env="GOOGLE_API_KEY",
        columns=("Display name", "Input token limit", "Output token limit"),
        description="Google Gemini official API",
    ),
    "mistral": ProviderPreset(
        kind="mistral",
        base_url="https://api.mistral.ai",
        api_key_env="MISTRAL_API_KEY",
        columns=("Created (UTC)", "Owned by", "Capabilities"),
        description="Mistral AI official API",
    ),
    "bedrock": ProviderPreset(
        kind="bedrock",
        columns=("Model", "Provider", "Input modalities", "Output modalities"),
        description="AWS Bedrock (AWS credential chain)",
        needs_api_key=False,
    ),
}

# Short aliases used by configs/llm and model type names.
PROVIDER_ALIASES: dict[str, str] = {
    "gpt": "openai",
    "claude": "anthropic",
    "gemini": "google",
    "llama": "together",
    "dashscope": "qwen",
}


@dataclass
class ModelInfo:
    """One model entry returned by a provider.

    Attributes:
        model_id: Model identifier passed to the API.
        extra: Extra column values keyed by ``ProviderPreset.columns`` names.
    """

    model_id: str
    extra: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_ts(value: Any) -> str:
    """Render a timestamp (unix seconds, datetime or string) as UTC text."""
    if value is None or value == "":
        return "-"
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, (int, float)):
        moment = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, str):
        return value
    else:
        return str(value)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_bytes(size: Any) -> str:
    """Render a byte count as a human-readable size."""
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return "-"


def _cell(value: Any) -> str:
    """Escape a value for a Markdown table cell."""
    text = "" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\n", " ").strip()
    return text or "-"


def _short_error(exc: Exception) -> str:
    """Render an exception as a single short line (drops HTML error pages)."""
    text = re.sub(r"<[^>]+>", " ", str(exc))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 300:
        text = text[:300] + "..."
    return f"{type(exc).__name__}: {text or '(no message)'}"


def _capabilities_text(caps: Any) -> str:
    """Summarize a Mistral capabilities object as enabled capability names."""
    if caps is None:
        return "-"
    if hasattr(caps, "model_dump"):
        data = caps.model_dump()
    elif hasattr(caps, "__dict__"):
        data = vars(caps)
    else:
        return str(caps)
    names = [key for key, enabled in data.items() if enabled is True]
    return ", ".join(names) or "-"


# ---------------------------------------------------------------------------
# Fetch implementations
# ---------------------------------------------------------------------------


def _fetch_openai(base_url: str, api_key: str, timeout: float) -> list[ModelInfo]:
    """Fetch models from an OpenAI-compatible ``/models`` endpoint."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url, timeout=timeout)
    try:
        page = client.models.list()
        return [
            ModelInfo(
                model_id=item.id,
                extra={
                    "Created (UTC)": _fmt_ts(getattr(item, "created", None)),
                    "Owned by": getattr(item, "owned_by", None) or "-",
                },
            )
            for item in page.data
        ]
    finally:
        client.close()


def _fetch_anthropic(api_key: str, timeout: float) -> list[ModelInfo]:
    """Fetch models from the Anthropic API, following pagination."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key, timeout=timeout)
    models: list[ModelInfo] = []
    try:
        page = client.models.list(limit=1000)
        while True:
            for item in page.data:
                models.append(
                    ModelInfo(
                        model_id=item.id,
                        extra={
                            "Display name": getattr(item, "display_name", None) or "-",
                            "Created (UTC)": _fmt_ts(getattr(item, "created_at", None)),
                        },
                    )
                )
            if not getattr(page, "has_more", False):
                break
            page = client.models.list(limit=1000, after_id=page.last_id)
    finally:
        client.close()
    return models


def _fetch_google(api_key: str, timeout: float) -> list[ModelInfo]:
    """Fetch Gemini models that support ``generateContent``."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout * 1000)),
    )
    models: list[ModelInfo] = []
    for item in client.models.list():
        actions = getattr(item, "supported_actions", None) or []
        if "generateContent" not in actions:
            continue
        models.append(
            ModelInfo(
                model_id=str(item.name).removeprefix("models/"),
                extra={
                    "Display name": getattr(item, "display_name", None) or "-",
                    "Input token limit": getattr(item, "input_token_limit", None) or "-",
                    "Output token limit": getattr(item, "output_token_limit", None) or "-",
                },
            )
        )
    return models


def _fetch_mistral(api_key: str, timeout: float) -> list[ModelInfo]:
    """Fetch models from the official Mistral SDK."""
    from mistralai.client import Mistral

    client = Mistral(api_key=api_key, timeout_ms=int(timeout * 1000))
    return [
        ModelInfo(
            model_id=item.id,
            extra={
                "Created (UTC)": _fmt_ts(getattr(item, "created", None)),
                "Owned by": getattr(item, "owned_by", None) or "-",
                "Capabilities": _capabilities_text(getattr(item, "capabilities", None)),
            },
        )
        for item in client.models.list().data
    ]


def _fetch_ollama(base_url: str, timeout: float) -> list[ModelInfo]:
    """Fetch locally installed models from Ollama's ``/api/tags``."""
    import httpx

    url = f"{base_url.rstrip('/')}/api/tags"
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    models: list[ModelInfo] = []
    for item in response.json().get("models", []):
        details = item.get("details") or {}
        models.append(
            ModelInfo(
                model_id=item.get("name") or item.get("model") or "-",
                extra={
                    "Parameters": details.get("parameter_size") or "-",
                    "Size": _fmt_bytes(item.get("size")),
                    "Modified": _fmt_ts(item.get("modified_at")),
                },
            )
        )
    return models


def _fetch_bedrock(timeout: float) -> list[ModelInfo]:
    """Fetch text LLMs from AWS Bedrock ``list_foundation_models``."""
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "bedrock",
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(
            connect_timeout=timeout,
            read_timeout=timeout,
            retries={"max_attempts": 2},
        ),
    )
    models: list[ModelInfo] = []
    kwargs: dict[str, Any] = {"byOutputModality": "TEXT"}
    while True:
        page = client.list_foundation_models(**kwargs)
        for item in page.get("modelSummaries", []):
            models.append(
                ModelInfo(
                    model_id=item.get("modelId", "-"),
                    extra={
                        "Model": item.get("modelName", "-"),
                        "Provider": item.get("providerName", "-"),
                        "Input modalities": ", ".join(item.get("inputModalities") or []) or "-",
                        "Output modalities": ", ".join(item.get("outputModalities") or []) or "-",
                    },
                )
            )
        token = page.get("nextToken")
        if not token:
            break
        kwargs["nextToken"] = token
    return models


def fetch_models(
    preset: ProviderPreset,
    base_url: str,
    api_key: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[ModelInfo]:
    """Fetch the model list for one provider.

    Args:
        preset: Provider preset describing the fetch protocol.
        base_url: API endpoint to query (ignored for Bedrock).
        api_key: API key (ignored by Ollama and Bedrock).
        timeout: Request timeout in seconds.

    Returns:
        Model entries sorted by model ID.

    Raises:
        RuntimeError: When the request fails or the protocol is unsupported.
    """
    try:
        if preset.kind == "openai":
            models = _fetch_openai(base_url, api_key, timeout)
        elif preset.kind == "anthropic":
            models = _fetch_anthropic(api_key, timeout)
        elif preset.kind == "google":
            models = _fetch_google(api_key, timeout)
        elif preset.kind == "mistral":
            models = _fetch_mistral(api_key, timeout)
        elif preset.kind == "ollama":
            models = _fetch_ollama(base_url, timeout)
        elif preset.kind == "bedrock":
            models = _fetch_bedrock(timeout)
        else:
            raise RuntimeError(f"Unsupported provider protocol: {preset.kind}")
    except Exception as exc:  # noqa: BLE001 - re-raised with provider context
        raise RuntimeError(_short_error(exc)) from exc
    return sorted(models, key=lambda item: item.model_id.lower())


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def render_markdown(
    provider: str,
    preset: ProviderPreset,
    base_url: str,
    models: list[ModelInfo],
) -> str:
    """Render the model list as a Markdown document with a summary table."""
    queried_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Models supported by {provider}",
        "",
        f"- Provider: `{provider}`",
        f"- Endpoint: {base_url or 'AWS Bedrock API'}",
        f"- Queried at: {queried_at}",
        f"- Model count: {len(models)}",
        "",
    ]
    if not models:
        lines.append(
            "> The endpoint returned no models; check base_url, the API key, or account permissions."
        )
        return "\n".join(lines) + "\n"

    lines.append("| # | Model ID | " + " | ".join(preset.columns) + " |")
    lines.append("|---|---------|" + "|".join("------" for _ in preset.columns) + "|")
    for index, model in enumerate(models, start=1):
        cells = [_cell(model.model_id)]
        cells += [_cell(model.extra.get(column)) for column in preset.columns]
        lines.append(f"| {index} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_markdown(text: str, output: Path) -> Path:
    """Write the Markdown document, creating parent directories as needed."""
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


# ---------------------------------------------------------------------------
# Config / argument resolution
# ---------------------------------------------------------------------------


def _expand_env(value: Any) -> str:
    """Expand ``${VAR}`` references in a config value."""
    if not isinstance(value, str):
        return ""
    return ENV_REF_RE.sub(lambda match: os.environ.get(match.group(1), ""), value)


def load_config_provider(config_path: Path) -> dict[str, str]:
    """Read the provider block of a ``configs/llm/*.yaml`` file."""
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    provider_cfg = ((data.get("llm") or {}).get("provider")) or {}
    return {
        "type": str(provider_cfg.get("type") or ""),
        "base_url": str(provider_cfg.get("base_url") or ""),
        "api_key": _expand_env(provider_cfg.get("api_key")),
    }


def resolve_provider(name: str) -> str:
    """Normalize a provider name through ``PROVIDER_ALIASES``."""
    key = name.strip().lower()
    return PROVIDER_ALIASES.get(key, key)


def infer_provider_from_base_url(base_url: str) -> str | None:
    """Return the preset name matching ``base_url``, when there is exactly one."""
    target = base_url.rstrip("/")
    if not target:
        return None
    matches = [
        name
        for name, preset in PROVIDER_PRESETS.items()
        if preset.base_url and preset.base_url.rstrip("/") == target
    ]
    return matches[0] if len(matches) == 1 else None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="List the LLMs a provider supports and save them as a Markdown file.",
    )
    parser.add_argument(
        "-p",
        "--provider",
        help="Provider name; omit to use the interactive menu",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT_DIR}/<provider>_models.md)",
    )
    parser.add_argument("--base-url", help="Override the provider API endpoint")
    parser.add_argument("--api-key", help="Override the API key (defaults to the environment variable)")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Read provider.base_url and api_key from configs/llm/*.yaml",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT:.0f})",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List built-in providers and default endpoints, then exit",
    )
    return parser.parse_args()


def print_providers() -> None:
    """Print the built-in provider presets."""
    print("Built-in providers:")
    for name, preset in PROVIDER_PRESETS.items():
        key_hint = f"${preset.api_key_env}" if preset.api_key_env else "no key required"
        endpoint = preset.base_url or "AWS Bedrock API"
        print(f"  - {name}: {preset.description}｜{endpoint}｜{key_hint}")
    print(f"Aliases: {', '.join(f'{k} -> {v}' for k, v in PROVIDER_ALIASES.items())}")
    print("Custom endpoint: --provider custom --base-url <url> --api-key <key>")


def interactive_provider() -> str:
    """Ask the user to choose a provider from a numbered menu."""
    names = list(PROVIDER_PRESETS) + ["custom"]
    print("Select a provider:")
    for index, name in enumerate(names, start=1):
        if name == "custom":
            print(f"  {index:>2}. custom: Custom OpenAI-compatible endpoint")
        else:
            preset = PROVIDER_PRESETS[name]
            print(f"  {index:>2}. {name}: {preset.description}")
    while True:
        raw = input(f"Enter a number [1-{len(names)}] (default 1): ").strip() or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        print("  Invalid number, try again.")


def main() -> int:
    """Entry point: resolve provider, fetch models, write the Markdown file."""
    args = parse_args()

    if args.list_providers:
        print_providers()
        return 0

    config: dict[str, str] = {}
    if args.config:
        if not args.config.exists():
            print(f"[error] Config file not found: {args.config}")
            return 1
        config = load_config_provider(args.config)

    provider_arg = args.provider
    if not provider_arg and config.get("type"):
        provider_arg = config["type"]
        if provider_arg == "openai":
            provider_arg = infer_provider_from_base_url(config.get("base_url", "")) or provider_arg
    if not provider_arg:
        if not sys.stdin.isatty():
            print("[error] In a non-interactive environment, specify --provider.")
            print("        See --list-providers for the available providers.")
            return 1
        provider_arg = interactive_provider()

    provider = resolve_provider(provider_arg)
    if provider == "custom":
        preset = ProviderPreset(
            kind="openai",
            base_url="",
            description="Custom OpenAI-compatible endpoint",
        )
    else:
        preset = PROVIDER_PRESETS.get(provider)
        if preset is None:
            available = ", ".join(PROVIDER_PRESETS)
            print(f"[error] Unknown provider: {provider_arg}; available: {available}")
            return 1

    base_url = args.base_url or config.get("base_url") or preset.base_url
    api_key = args.api_key or config.get("api_key") or ""

    if preset.needs_api_key and not api_key and preset.api_key_env:
        api_key = os.environ.get(preset.api_key_env, "")
    if preset.needs_api_key and not api_key:
        if not sys.stdin.isatty():
            key_name = preset.api_key_env or "API Key"
            print(f"[error] {key_name} not found; set the environment variable or pass --api-key.")
            return 1
        key_hint = preset.api_key_env or "API Key"
        api_key = getpass(f"{key_hint} not found; enter the API key: ").strip()

    if provider == "custom" and not base_url:
        if not sys.stdin.isatty():
            print("[error] The custom provider requires --base-url.")
            return 1
        base_url = input("Enter base_url (e.g. http://127.0.0.1:8080/v1): ").strip()
    if not base_url and preset.kind != "bedrock":
        print("[error] Missing base_url; pass --base-url.")
        return 1

    output = args.output or DEFAULT_OUTPUT_DIR / f"{provider}_models.md"

    print(f"Querying the model list of {provider} at {base_url or 'AWS Bedrock API'} ...")
    try:
        models = fetch_models(preset, base_url, api_key, args.timeout)
    except RuntimeError as exc:
        print(f"[error] Failed to fetch the model list: {exc}")
        print("        Check base_url, the API key, network connectivity and account permissions.")
        print("        Ollama requires the local service to be running.")
        if preset.kind == "bedrock":
            print("        Bedrock requires boto3: uv sync --extra providers")
        return 1

    text = render_markdown(provider, preset, base_url, models)
    path = save_markdown(text, output)

    print(f"\nModels available from {provider} ({len(models)}):")
    for model in models:
        print(f"  - {model.model_id}")
    print(f"\nSaved to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
