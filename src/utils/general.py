"""
Utils General - Common utility functions

Provides common helpers such as schema inference for LLM structured output,
text file reading for `location` inputs and image encoding for `images` inputs.
"""

import base64
from pathlib import Path
from typing import Any

#: Image extensions accepted by the `images` message input of the LLM layer.
IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def read_text_file(file_path: str | Path) -> str | None:
    """
    Read a local text file (the file a ``location`` message key points to, usually Markdown).

    Args:
        file_path: Path of the file

    Returns:
        The file's text content; None when the file is missing or cannot be read.
    """
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def encode_image_data_url(file_path: str | Path) -> str | None:
    """Read an image file and return it as a base64 data URL.

    Used by the LLM layer for `images` message inputs (vision requests).

    Args:
        file_path: Local image path (jpeg / png / webp / gif).

    Returns:
        A ``data:<mime>;base64,<payload>`` URL; None when the file is missing,
        has an unsupported extension, or cannot be read.
    """
    path = Path(file_path)
    mime = IMAGE_MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        return None
    try:
        payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None
    return f"data:{mime};base64,{payload}"


def _dict_to_response_format(d: dict[str, Any], strict: bool = True) -> dict[str, Any]:
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

    return schema
