"""
OptiS Benchmark — Prompt Manager

Centralised Jinja2 template loading for all LLM prompts.

Usage::

    from src.utils.prompt_manager import PromptManager

    pm = PromptManager.get_instance()
    prompt = pm.render(
        "evaluators/rubric_judge/field_prompt.jinja2",
        display="Keywords",
        predicted_value="...",
        expected_value="...",
        accuracy_rubric={5: "Perfect", ...},
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


class PromptManager:
    """Singleton that loads and renders Jinja2 prompt templates.

    Templates live under the project-level ``prompts/`` directory.
    Render them with a dotted path relative to that root::

        pm.render("evaluators/rubric_judge/field_prompt.jinja2", ...)
    """

    _instance: PromptManager | None = None

    def __init__(self, template_dir: str | os.PathLike[str] | None = None) -> None:
        if template_dir is None:
            template_dir = (
                Path(__file__).resolve().parent.parent.parent / "prompts"
            )
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            keep_trailing_newline=True,
            autoescape=False,
        )

    @classmethod
    def get_instance(
        cls, template_dir: str | os.PathLike[str] | None = None
    ) -> PromptManager:
        """Return the singleton instance.

        Args:
            template_dir: Optional override — only used when creating
                          the instance for the first time.
        """
        if cls._instance is None:
            cls._instance = cls(template_dir=template_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the singleton (useful in tests)."""
        cls._instance = None

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Render a Jinja2 template.

        Args:
            template_name: Path relative to ``prompts/``,
                           e.g. ``"evaluators/rubric_judge/field_prompt.jinja2"``.
            **kwargs: Template variables.

        Returns:
            Rendered string.
        """
        template = self._env.get_template(template_name)
        return template.render(**kwargs)
