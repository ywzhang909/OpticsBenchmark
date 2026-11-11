"""
Optis Benchmark - Tests for LLM image input support

Covers ``src.utils.general.encode_image_data_url`` and the ``images`` message
key of ``QwenLLM``, i.e. how vision requests are assembled for the
OpenAI-compatible provider. No network access: the provider client is replaced
by a capture stub.
"""

import base64
from dataclasses import dataclass, field
from typing import Any

import pytest
from PIL import Image

from src.llm.models.qwen_llm import QwenLLM
from src.llm.providers.openai_provider import OpenAIProvider
from src.utils.general import IMAGE_MIME_TYPES, encode_image_data_url

# =============================================================================
# Stubs
# =============================================================================


@dataclass
class _FakeMessage:
    """Chat response message payload."""

    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str = "stop"


@dataclass
class _FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15

    def model_dump(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage = field(default_factory=_FakeUsage)


class _FakeCompletions:
    """Captures the kwargs of every chat completion call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content="{}"))])


class _FakeClient:
    """Minimal stand-in for ``AsyncOpenAI``."""

    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()

    @property
    def completions(self) -> _FakeCompletions:
        return self.chat.completions


@pytest.fixture
def capture_provider() -> tuple[OpenAIProvider, _FakeCompletions]:
    """An OpenAIProvider whose client only records requests."""
    provider = OpenAIProvider(api_key="test-key", base_url="http://127.0.0.1:9/v1")
    client = _FakeClient()
    provider._client = client
    return provider, client.chat.completions


def _write_image(tmp_path, name: str = "figure.jpeg", size=(200, 120), color="steelblue"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")
    return path


def _payload(url: str) -> bytes:
    """Return the decoded bytes of a base64 data URL."""
    return base64.b64decode(url.split(",", 1)[1])


# =============================================================================
# encode_image_data_url
# =============================================================================


class TestEncodeImageDataUrl:
    """Tests for the base64 data URL helper."""

    def test_roundtrip(self, tmp_path):
        path = _write_image(tmp_path)
        url = encode_image_data_url(path)
        assert url is not None
        header, payload = url.split(",", 1)
        assert header == "data:image/jpeg;base64"
        assert base64.b64decode(payload) == path.read_bytes()

    def test_supported_extensions(self, tmp_path):
        expected = {"a.png": "image/png", "b.webp": "image/webp", "c.GIF": "image/gif"}
        for name, mime in expected.items():
            path = tmp_path / name
            Image.new("RGB", (8, 8), "white").save(path, name.rsplit(".", 1)[-1].upper())
            url = encode_image_data_url(path)
            assert url is not None and url.startswith(f"data:{mime};base64,")

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")
        assert encode_image_data_url(path) is None

    def test_missing_file(self, tmp_path):
        assert encode_image_data_url(tmp_path / "absent.jpeg") is None

    def test_mime_table_covers_marker_output(self):
        assert ".jpeg" in IMAGE_MIME_TYPES and ".png" in IMAGE_MIME_TYPES


# =============================================================================
# QwenLLM images key
# =============================================================================


class TestQwenLLMImages:
    """Tests for the ``images`` message key of QwenLLM."""

    async def test_images_appended_after_text_in_order(self, tmp_path, capture_provider):
        provider, completions = capture_provider
        first = _write_image(tmp_path, "first.jpeg", size=(200, 120), color="red")
        second = _write_image(tmp_path, "second.jpeg", size=(90, 60), color="blue")

        llm = QwenLLM(model_name="vision-model")
        result = await llm.chat(
            messages=[
                {"prompt": "task"},
                {"prompt": "manuscript", "images": [str(first), str(second)]},
            ],
            provider=provider,
            setup={},
        )

        assert result["content"] == "{}"
        content = completions.calls[0]["messages"][0]["content"]
        assert [part["type"] for part in content] == ["text", "text", "image_url", "image_url"]
        assert content[0] == {"type": "text", "text": "task"}
        assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        # Images keep the order they were listed in.
        assert _payload(content[2]["image_url"]["url"]) == first.read_bytes()
        assert _payload(content[3]["image_url"]["url"]) == second.read_bytes()

    async def test_unreadable_image_is_skipped(self, tmp_path, capture_provider):
        provider, completions = capture_provider
        good = _write_image(tmp_path)

        await QwenLLM(model_name="vision-model").chat(
            messages=[{"prompt": "text", "images": [str(tmp_path / "missing.jpeg"), good]}],
            provider=provider,
            setup={},
        )

        content = completions.calls[0]["messages"][0]["content"]
        assert [part["type"] for part in content] == ["text", "image_url"]

    async def test_empty_images_keeps_previous_behaviour(self, capture_provider):
        provider, completions = capture_provider

        await QwenLLM(model_name="text-model").chat(
            messages=[{"prompt": "a"}, {"prompt": "b"}],
            provider=provider,
            setup={},
        )

        message = completions.calls[0]["messages"][0]
        assert message["role"] == "user"
        assert message["content"] == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    async def test_location_file_still_inlined(self, tmp_path, capture_provider):
        provider, completions = capture_provider
        document = tmp_path / "prompt.md"
        document.write_text("file contents", encoding="utf-8")

        await QwenLLM(model_name="text-model").chat(
            messages=[{"location": str(document)}],
            provider=provider,
            setup={},
        )

        assert completions.calls[0]["messages"][0]["content"] == [
            {"type": "text", "text": "file contents"}
        ]
