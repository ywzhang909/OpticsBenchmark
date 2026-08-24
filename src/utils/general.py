"""
Utils General - 通用工具函数

提供 LLM 结构化输出的 schema 推导等通用辅助功能。
"""

from typing import Any


def _dict_to_response_format(
    d: dict[str, Any], strict: bool = True
) -> dict[str, Any]:
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
