# OptiS Benchmark - 代码规范与约定

> 本文档定义了 OptiS Benchmark 项目的代码规范、命名约定和最佳实践。

---

## 📑 目录

1. [Python 代码规范](#1-python-代码规范)
2. [命名约定](#2-命名约定)
3. [文件结构](#3-文件结构)
4. [配置规范](#4-配置规范)
5. [Git 约定](#5-git-约定)
6. [文档规范](#6-文档规范)
7. [测试规范](#7-测试规范)

---

## 1. Python 代码规范

### 1.1 基础规范

- 遵循 **PEP 8** 标准
- 行长度限制: **100 字符**
- 使用 **type hints** (类型提示)
- 缩进: **4 空格**

### 1.2 导入顺序

```python
# 1. 标准库
import os
import sys
from pathlib import Path
from typing import Any, Optional

# 2. 第三方库
import yaml
from pydantic import BaseModel

# 3. 本项目模块
from src.core.agent import BaseAgent
from src.utils.logger import get_logger

# 4. 相对导入 (仅在同一包内使用)
from .base_env import BaseEnvironment
```

### 1.3 文档字符串

**所有公共函数、类、模块必须包含文档字符串。**

```python
def calculate_mtf(
    spatial_frequency: float,
    modulation_depth: float,
) -> float:
    """
    计算调制传递函数 (MTF) 值。

    MTF 描述了光学系统传递对比度的能力，
    是评估镜头性能的重要指标。

    Args:
        spatial_frequency: 空间频率 (lp/mm)
        modulation_depth: 输入调制深度 (0-1)

    Returns:
        MTF 值，范围 [0, 1]

    Raises:
        ValueError: 当 spatial_frequency 为负数时

    Examples:
        >>> calculate_mtf(50.0, 0.5)
        0.85
    """
    if spatial_frequency < 0:
        raise ValueError("空间频率不能为负数")
    
    # 计算 MTF
    mtf = modulation_depth * (1.0 / (1.0 + spatial_frequency / 100))
    return mtf
```

### 1.4 类型注解

```python
# ✅ 推荐
def process_data(data: dict[str, Any]) -> list[str]:
    ...

# ❌ 避免
def process_data(data):
    ...
```

### 1.5 常量定义

```python
# ✅ 使用大写+下划线
DEFAULT_TIMEOUT = 300
MAX_RETRIES = 3

# ❌ 避免
defaultTimeout = 300
```

---

## 2. 命名约定

### 2.1 模块和包

| 类型 | 规范 | 示例 |
|------|------|------|
| Python 模块 | 小写 + 下划线 | `base_env.py`, `json_parser.py` |
| 包名 | 小写 + 下划线 | `src/core/`, `src/utils/` |
| 目录名 | 小写 + 下划线 | `configs/agents/` |

### 2.2 类名

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | CapWords (PascalCase) | `BaseAgent`, `MetricEvaluator` |
| 异常类 | 以 `Error` 结尾 | `TimeoutError`, `ConfigError` |
| 基类 | 以 `Base` 开头 | `BaseEnvironment`, `BaseEvaluator` |

### 2.3 函数和变量

| 类型 | 规范 | 示例 |
|------|------|------|
| 函数 | 小写 + 下划线 | `get_system_data()` |
| 变量 | 小写 + 下划线 | `total_cost`, `task_id` |
| 私有方法 | 前缀双下划线 | `__init_connection()` |
| 保护方法 | 前缀单下划线 | `_validate_config()` |

### 2.4 常量

```python
# 全局常量: 大写 + 下划线
MAX_CONCURRENT_TASKS = 10
DEFAULT_TIMEOUT_SECONDS = 300
SUPPORTED_PROVIDERS = ["openai", "anthropic", "groq"]
```

### 2.5 配置字段

配置 YAML 中的字段名使用 **snake_case**：

```yaml
# ✅ 推荐
api_key: "${OPENAI_API_KEY}"
max_tokens: 4096
save_intermediate: true

# ❌ 避免
apiKey: "..."
maxTokens: 4096
saveIntermediate: true
```

---

## 3. 文件结构

### 3.1 Python 模块结构

```
src/core/
├── __init__.py          # 导出公共 API
├── agent.py             # Agent 相关
├── evaluator.py         # Evaluator 相关
└── runner.py            # Runner 相关
```

### 3.2 __init__.py 规范

```python
"""
Core Module - 核心模块

提供 Agent、Evaluator、Runner 等核心组件。
"""

from .agent import (
    BaseAgent,
    AgentConfig,
    create_agent,
)

from .evaluator import (
    BaseEvaluator,
    EvaluationResult,
    create_evaluator,
)

from .runner import (
    EvaluationRunner,
    run_evaluation,
)

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "create_agent",
    # ... 其他导出
]
```

### 3.3 类文件结构

```python
"""
模块描述
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
    """自定义错误类"""
    pass

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MyConfig:
    """配置数据类"""
    name: str
    value: int = 100

# =============================================================================
# Classes
# =============================================================================

class MyClass:
    """主类"""
    
    def __init__(self, config: MyConfig):
        self.config = config
    
    def method(self) -> None:
        """公共方法"""
        pass
    
    def _private_method(self) -> None:
        """私有方法"""
        pass
```

---

## 4. 配置规范

### 4.1 YAML 配置结构

```yaml
# configs/agents/gpt-4.yaml

# =============================================================================
# Agent Configuration
# =============================================================================

agent:
  name: "optis-gpt4"
  version: "1.0.0"
  description: "GPT-4 Optical Design Agent"

model:
  provider: "openai"          # openai, anthropic, groq
  name: "gpt-4-turbo"
  api_key: "${OPENAI_API_KEY}"  # 使用环境变量
  temperature: 0.0
  max_tokens: 4096

execution:
  timeout: 300
  max_retries: 3
```

### 4.2 配置字段规范

| 字段类型 | 格式 | 示例 |
|----------|------|------|
| API 密钥 | `${ENV_VAR}` | `${OPENAI_API_KEY}` |
| 布尔值 | `true`/`false` | `save_intermediate: true` |
| 数值 | 不加引号 | `timeout: 300` |
| 字符串 | 不加引号 | `provider: "openai"` |

### 4.3 任务配置规范

```yaml
task:
  id: "lens_design"           # 唯一 ID (snake_case)
  name: "Lens Design"         # 显示名称
  description: "..."          # 描述
  category: "lens_design"     # 类别
  difficulty: 3               # 难度 1-5

dataset:
  path: "dataset/processed/..."  # 相对于项目根目录
  num_samples: 50
  shuffle: false

evaluation:
  scoring_method: "metric_based"  # exact_match, partial_match, metric_based
```

---

## 5. Git 约定

### 5.1 分支命名

```
feature/<feature-name>      # 新功能
fix/<bug-description>       # Bug 修复
docs/<doc-section>          # 文档更新
task/<task-name>            # 任务相关
dataset/<dataset-name>      # 数据集相关
```

示例:
```
feature/multi-agent-support
fix/timeout-handling
docs/api-reference
```

### 5.2 Commit 格式

遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type:**
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档
- `style`: 格式 (不影响代码)
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

**Examples:**

```bash
# 新功能
git commit -m "feat(agent): add Anthropic Claude 3 support

- Implement AnthropicAgent class
- Add Claude API integration
- Update cost calculation

Closes #123"

# Bug 修复
git commit -m "fix(runner): handle timeout gracefully

The runner now catches TimeoutError and marks the task
as failed instead of crashing.

Fixes #456"

# 文档
git commit -m "docs: add optical basics guide

Added docs/foundation/optical-basics.md covering:
- Geometric optics fundamentals
- Wave optics basics
- MTF and PSF concepts"
```

### 5.3 Pull Request 规范

```markdown
## Summary
<!-- 简要描述改动 -->

## Motivation
<!-- 为什么需要这个改动 -->

## Changes Made
<!-- 具体改了什么 -->

## Testing
<!-- 如何测试的 -->

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
```

---

## 6. 文档规范

### 6.1 README 要求

每个目录应包含 `README.md`，内容包括：
- 目录用途
- 包含文件说明
- 使用示例

### 6.2 代码注释

```python
# ✅ 使用说明性注释
# 计算 MTF 时使用对比度传递公式
mtf = contrast_in / contrast_out

# ❌ 避免无意义的注释
# 计算 MTF
mtf = contrast_in / contrast_out

# ✅ 使用 TODO/FIXME 标记待办
# TODO(yourname): 优化性能
# FIXME: 边界条件未处理
```

### 6.3 Markdown 格式

```markdown
# 标题层级 (最多 H3)

## 主要标题
### 次要标题
#### 三级标题

**加粗** 用于强调

`代码` 用于命令或代码引用

| 表格 | 格式 |
|------|------|
| 数据 | 内容 |
```

---

## 7. 测试规范

### 7.1 测试文件命名

```
tests/
├── test_agent.py           # 对应 src/core/agent.py
├── test_evaluator.py       # 对应 src/core/evaluator.py
├── test_runner.py          # 对应 src/core/runner.py
└── integration/
    └── test_full_pipeline.py
```

### 7.2 测试函数命名

```python
def test_agent_chat_success():
    """测试 Agent 成功响应"""
    ...

def test_agent_chat_timeout():
    """测试 Agent 超时处理"""
    ...

def test_evaluator_metric_calculation():
    """测试评估指标计算"""
    ...
```

### 7.3 测试结构

```python
import pytest

class TestAgent:
    """Agent 测试类"""
    
    @pytest.fixture
    def agent(self):
        """测试 fixture"""
        return create_agent("configs/agents/gpt-4.yaml")
    
    async def test_chat(self, agent):
        """测试聊天功能"""
        messages = [Message(role="user", content="Hello")]
        response = await agent.chat(messages)
        
        assert response.content is not None
        assert response.cost > 0
    
    async def test_timeout(self, agent):
        """测试超时处理"""
        ...
```

### 7.4 覆盖率要求

| 模块 | 最低覆盖率 |
|------|-----------|
| core/ | 80% |
| environments/ | 70% |
| utils/ | 80% |

---

## 📋 检查清单

提交代码前请确认：

- [ ] 遵循 PEP 8 规范
- [ ] 包含类型注解
- [ ] 包含文档字符串
- [ ] 命名符合规范
- [ ] 测试覆盖新增代码
- [ ] commit message 符合规范
- [ ] 没有调试代码遗留

---

*最后更新: 2024-XX-XX*
