# Optis Benchmark - Coding Standards and Conventions

> This document defines the coding standards, naming conventions and best
> practices of the Optis Benchmark project.

---

## 📑 Table of Contents

1. [Python Coding Standards](#1-python-coding-standards)
2. [Naming Conventions](#2-naming-conventions)
3. [File Structure](#3-file-structure)
4. [Configuration Standards](#4-configuration-standards)
5. [Git Conventions](#5-git-conventions)
6. [Documentation Standards](#6-documentation-standards)
7. [Testing Standards](#7-testing-standards)

---

## 1. Python Coding Standards

### 1.1 Basics

- Follow **PEP 8**
- Line length limit: **100 characters**
- Use **type hints**
- Indentation: **4 spaces**

### 1.2 Import Order

```python
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import Any, Optional

# 2. Third-party libraries
import yaml
from pydantic import BaseModel

# 3. Project modules
from src.llm.base import BaseLLM
from src.utils.logger import get_logger

# 4. Relative imports (only within the same package)
from .base_env import BaseEnvironment
```

### 1.3 Docstrings

**All public functions, classes and modules must have docstrings.**

```python
def calculate_mtf(
    spatial_frequency: float,
    modulation_depth: float,
) -> float:
    """
    Compute the modulation transfer function (MTF) value.

    MTF describes how well an optical system transfers contrast,
    and is an important metric for evaluating lens performance.

    Args:
        spatial_frequency: Spatial frequency (lp/mm)
        modulation_depth: Input modulation depth (0-1)

    Returns:
        The MTF value, in the range [0, 1]

    Raises:
        ValueError: When spatial_frequency is negative

    Examples:
        >>> calculate_mtf(50.0, 0.5)
        0.85
    """
    if spatial_frequency < 0:
        raise ValueError("Spatial frequency must not be negative")
    
    # Compute the MTF
    mtf = modulation_depth * (1.0 / (1.0 + spatial_frequency / 100))
    return mtf
```

### 1.4 Type Annotations

```python
# ✅ Recommended
def process_data(data: dict[str, Any]) -> list[str]:
    ...

# ❌ Avoid
def process_data(data):
    ...
```

### 1.5 Constant Definitions

```python
# ✅ Use UPPER_SNAKE_CASE
DEFAULT_TIMEOUT = 300
MAX_RETRIES = 3

# ❌ Avoid
defaultTimeout = 300
```

---

## 2. Naming Conventions

### 2.1 Modules and Packages

| Type | Convention | Examples |
|------|------------|----------|
| Python module | lower case + underscores | `base_env.py`, `json_parser.py` |
| Package name | lower case + underscores | `src/core/`, `src/utils/` |
| Directory name | lower case + underscores | `configs/agents/` |

### 2.2 Class Names

| Type | Convention | Examples |
|------|------------|----------|
| Class name | CapWords (PascalCase) | `BaseLLM`, `MetricEvaluator` |
| Exception class | Suffix `Error` | `TimeoutError`, `ConfigError` |
| Base class | Prefix `Base` | `BaseEnvironment`, `BaseEvaluator` |

### 2.3 Functions and Variables

| Type | Convention | Examples |
|------|------------|----------|
| Function | lower case + underscores | `get_system_data()` |
| Variable | lower case + underscores | `total_cost`, `task_id` |
| Private method | double underscore prefix | `__init_connection()` |
| Protected method | single underscore prefix | `_validate_config()` |

### 2.4 Constants

```python
# Global constants: UPPER_SNAKE_CASE
MAX_CONCURRENT_TASKS = 10
DEFAULT_TIMEOUT_SECONDS = 300
SUPPORTED_PROVIDERS = ["openai", "anthropic", "groq"]
```

### 2.5 Configuration Fields

Field names in configuration YAML use **snake_case**:

```yaml
# ✅ Recommended
api_key: "${OPENAI_API_KEY}"
max_tokens: 4096
save_intermediate: true

# ❌ Avoid
apiKey: "..."
maxTokens: 4096
saveIntermediate: true
```

---

## 3. File Structure

### 3.1 Python Module Structure

```
src/core/
├── __init__.py          # exports the public API
├── config.py            # TaskConfig
├── llm_judge.py         # LLM Judge
├── llm_runner.py        # LLM Runner
└── runner.py            # Runner
```

### 3.2 `__init__.py` Convention

```python
"""
Core Module - core components

Provides the core components: Runner, Config, LLM Judge and so on.
"""

from .llm_judge import LLMJudge, JudgePromptBuilder, Rubric

from .llm_runner import LLMPredRunner, LLMRunnerConfig

from .runner import (
    AgentRunner,
    RunnerConfig,
    TaskConfig,
    TaskInstance,
)

__all__ = [
    "AgentRunner",
    "RunnerConfig",
    "TaskConfig",
    "TaskInstance",
    # ... other exports
]
```

### 3.3 Class File Structure

```python
"""
Module description
"""

from __future__ import annotations

import ...
from typing import ...

# =============================================================================
# Constants
# =============================================================================

DEFAULT_VALUE = 100

# =============================================================================
# Exceptions
# =============================================================================

class MyError(Exception):
    """Custom exception class"""
    pass

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MyConfig:
    """Configuration dataclass"""
    name: str
    value: int = 100

# =============================================================================
# Classes
# =============================================================================

class MyClass:
    """Main class"""
    
    def __init__(self, config: MyConfig):
        self.config = config
    
    def method(self) -> None:
        """Public method"""
        pass
    
    def _private_method(self) -> None:
        """Private method"""
        pass
```

---

## 4. Configuration Standards

### 4.1 YAML Configuration Structure

```yaml
# configs/llm/GPT_OpenAI.yaml

# =============================================================================
# LLM Configuration
# =============================================================================

llm:
  provider:
    type: "openai"
    api_key: "${OPENAI_API_KEY}"
  model:
    type: "gpt"
    name: "gpt-4o"
  setup:
    temperature: 0.0
    max_tokens: 4096
    top_p: 1.0

task:
  dataset_path: "dataset/paper_info_extract/dataset_json/dataset_v1.json"
  prompt_file: "prompts/paper_info_extract/zero-shot_v1.0.txt"

execution:
  timeout: 300
  max_retries: 3
```

### 4.2 Configuration Field Convention

| Field type | Format | Example |
|------------|--------|---------|
| API key | `${ENV_VAR}` | `${OPENAI_API_KEY}` |
| Boolean | `true`/`false` | `save_intermediate: true` |
| Number | unquoted | `timeout: 300` |
| String | unquoted | `provider: "openai"` |

### 4.3 Task Configuration Convention

```yaml
task:
  id: "lens_design"           # unique ID (snake_case)
  name: "Lens Design"         # display name
  description: "..."          # description
  category: "lens_design"     # category
  difficulty: 3               # difficulty 1-5

dataset:
  path: "dataset/processed/..."  # relative to the project root
  num_samples: 50
  shuffle: false

evaluation:
  scoring_method: "metric_based"  # exact_match, partial_match, metric_based
```

---

## 5. Git Conventions

### 5.1 Branch Naming

```
feature/<feature-name>      # new feature
fix/<bug-description>       # bug fix
docs/<doc-section>          # documentation update
task/<task-name>            # task related
dataset/<dataset-name>      # dataset related
```

Examples:
```
feature/multi-agent-support
fix/timeout-handling
docs/api-reference
```

### 5.2 Commit Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type:**
- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation
- `style`: formatting (no code change)
- `refactor`: refactoring
- `test`: tests
- `chore`: build/tooling

**Examples:**

```bash
# New feature
git commit -m "feat(agent): add Anthropic Claude 3 support

- Implement AnthropicAgent class
- Add Claude API integration
- Update cost calculation

Closes #123"

# Bug fix
git commit -m "fix(runner): handle timeout gracefully

The runner now catches TimeoutError and marks the task
as failed instead of crashing.

Fixes #456"

# Documentation
git commit -m "docs: add optical basics guide

Added docs/foundation/optical-basics.md covering:
- Geometric optics fundamentals
- Wave optics basics
- MTF and PSF concepts"
```

### 5.3 Pull Request Convention

```markdown
## Summary
<!-- briefly describe the change -->

## Motivation
<!-- why this change is needed -->

## Changes Made
<!-- what exactly was changed -->

## Testing
<!-- how it was tested -->

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
```

---

## 6. Documentation Standards

### 6.1 README Requirements

Every directory should contain a `README.md` that covers:
- The purpose of the directory
- A description of the files it contains
- Usage examples

### 6.2 Code Comments

```python
# ✅ Use explanatory comments
# Use the contrast transfer formula when computing the MTF
mtf = contrast_in / contrast_out

# ❌ Avoid meaningless comments
# Compute the MTF
mtf = contrast_in / contrast_out

# ✅ Use TODO/FIXME to mark pending work
# TODO(yourname): optimize performance
# FIXME: boundary conditions not handled
```

### 6.3 Markdown Formatting

```markdown
# Heading level (H3 at most)

## Major heading
### Minor heading
#### Third-level heading

**Bold** for emphasis

`Code` for commands or code references

| Table  | Format |
|--------|--------|
| Data   | Content|
```

---

## 7. Testing Standards

### 7.1 Test File Naming

```
tests/
├── test_evaluator_base.py      # BaseEvaluator tests
├── test_citation_evaluator.py  # CitationEvaluator tests
├── test_rouge_eval_utils.py    # ROUGE evaluation utilities
├── test_em_eval_utils.py       # Exact match utilities
├── test_bleu_eval_utils.py     # BLEU evaluation
├── test_hungarian_algorithm.py # Hungarian matching
├── test_sentence_similarity.py # Sentence similarity
├── test_bert_score_eval.py     # BERTScore evaluator
└── ...
```

### 7.2 Test Function Naming

```python
def test_agent_chat_success():
    """Test a successful Agent response"""
    ...

def test_agent_chat_timeout():
    """Test Agent timeout handling"""
    ...

def test_evaluator_metric_calculation():
    """Test the evaluation metric computation"""
    ...
```

### 7.3 Test Structure

```python
import pytest

class TestLLMModel:
    """Test class for the LLM model"""
    
    @pytest.fixture
    def llm(self):
        """Test fixture"""
        from src.llm import create_llm
        return create_llm("configs/llm/GPT_OpenAI.yaml")
    
    async def test_chat(self, llm):
        """Test the chat functionality"""
        from src.llm.base import Message
        messages = [Message(role="user", content="Hello")]
        response = await llm.chat(messages)
        
        assert response.content is not None
        assert response.cost > 0
    
    async def test_timeout(self, agent):
        """Test timeout handling"""
        ...
```

### 7.4 Coverage Requirements

| Module | Minimum coverage |
|--------|------------------|
| core/ | 80% |
| environments/ | 70% |
| utils/ | 80% |

---

## 📋 Checklist

Before committing, confirm:

- [ ] PEP 8 is followed
- [ ] Type annotations are included
- [ ] Docstrings are included
- [ ] Naming follows the conventions
- [ ] Tests cover the new code
- [ ] The commit message follows the convention
- [ ] No leftover debug code

---

*Last updated: 2026-08-19*
