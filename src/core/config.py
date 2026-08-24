"""
Optis Benchmark - Shared Configuration Types

Shared dataclasses used across agent, runner, and other modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TaskConfig:
    """Configuration for a task set."""

    task: dict[str, Any]
    dataset_config: dict[str, Any]
    prompt_config: dict[str, Any]
    max_samples: int | None = None
    shuffle: bool = False
    file_input: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskConfig:
        """Load task configuration from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        task_data = data.get("task", {})
        dataset_data = data.get("dataset", {})
        prompt_data = data.get("prompt", {})

        return cls(
            task=task_data,
            dataset_config=dataset_data,
            prompt_config=prompt_data,
            max_samples=dataset_data.get("num_samples"),
            shuffle=dataset_data.get("shuffle", False),
            file_input=task_data.get("file_input", False),
        )
