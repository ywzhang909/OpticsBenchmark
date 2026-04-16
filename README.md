# OptiS Benchmark

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Topic](https://img.shields.io/badge/Topic-Optical%20Design-orange.svg)
![Stars](https://img.shields.io/github/stars/your-org/optis_benchmark?style=social)
![Forks](https://img.shields.io/github/forks/your-org/optis_benchmark?style=social)

**Open-Source Optical Design Agent Evaluation Framework**

*评估 LLM 在光学设计、光学仿真、光学检测等真实场景下的智能体能力*

[English](README.md) | [中文](README_zh.md)

</div>

---

## 📚 项目简介

OptiS Benchmark 是一个模块化、可扩展的开源评测框架，用于评估 LLM 在**光学设计**领域的智能体能力。

### 🎯 核心目标

1. **标准化评测** - 为光学领域 AI 智能体提供统一的评测标准和基准
2. **可复现性** - 确保评测结果可复现、可比较
3. **开放透明** - 开源数据集、评测代码和评分逻辑
4. **社区驱动** - 欢迎光学领域研究者贡献任务和数据集

### 🏛️ 设计理念

本项目借鉴 [AgentBench](https://github.com/OpenGVLab/AgentBench) 的架构设计，采用：

| 设计原则 | 实现方式 |
|----------|----------|
| **配置与代码分离** | `configs/` 目录集中管理所有配置 |
| **环境即代码** | `src/environments/` 模块化环境实现 |
| **评测可扩展** | 插件式 Agent、Evaluator、Task |

---

## ✨ 核心特性

| 特性 | 描述 |
|------|------|
| 🔬 **光学领域专用** | 针对 ZOS-API、CODE V、ASAP 等光学软件的专用环境 |
| 🏗️ **模块化架构** | Agent、Evaluator、Runner 完全解耦 |
| ⚙️ **配置中心化** | YAML 配置驱动，无代码修改即可切换模型/任务 |
| 📊 **多维度评测** | 支持指标评测、ROUGE、精确匹配等多种评测方式 |
| 🧪 **快速测试工具** | Quick LLM Selector 一键测试和对比多个模型 |
| 🌐 **开放数据集** | 公开的 JSONL 格式数据集，支持社区贡献 |
| 🔄 **并行评测** | 支持多并发、多任务并行执行 |
| 📈 **详细报告** | 自动生成 HTML/Markdown 评测报告 |

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆仓库
git clone https://github.com/your-org/optis_benchmark.git
cd optis_benchmark

# 使用 Conda (推荐)
conda env create -f environment.yml
conda activate optis_benchmark

# 或使用 pip
pip install -r requirements.txt
```

### 2. 环境变量配置

```bash
# 创建 .env 文件
cat > .env << EOF
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Google Gemini
GOOGLE_API_KEY=your-gemini-key

# Groq (免费)
GROQ_API_KEY=your-groq-key

# Together AI
TOGETHER_API_KEY=your-together-key

# AWS Bedrock (如需)
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
EOF

# 或导出到环境变量
export OPENAI_API_KEY=sk-your-key-here
```

---

## 🧪 快速测试 LLM 模型

使用 **Quick LLM Selector** 工具，无需编写代码即可快速测试和对比不同 LLM 提供商。

### 查看可用模型

```bash
python -m src.tools.quick_llm_selector --list
```

输出示例：
```
Available LLM Providers:
------------------------------------------------------------
  ✅ GPT-4 (OpenAI)
      Config: gpt-4.yaml
      Model: gpt-4-turbo
      Tools: file_read, bash_execute, python_execute

  ✅ Claude 3 (Anthropic)
      Config: claude-3.yaml
      Model: claude-3-5-sonnet

  ⚠️ Ollama (Local)
      Config: ollama.yaml
      Model: llama3
      Tools: none
```

### 交互式测试

```bash
# 启动交互式选择器
python -m src.tools.quick_llm_selector

# 输出示例：
# 🧪 OptiS Benchmark - Quick LLM Selector
# ==================================================
# 
# Available providers:
# 
#   1. ✅ GPT-4 (OpenAI)
#      Model: gpt-4-turbo | Tools: file_read, bash...
#   2. ✅ Claude 3 (Anthropic)
#      Model: claude-3-5-sonnet
#   ...
# 
# Select provider number (or 'q' to quit): 1
# Enter your prompt: Explain how a lens focuses light
```

### 快速测试单个模型

```bash
python -m src.tools.quick_llm_selector \
  --provider gpt-4 \
  --prompt "Explain optical refraction in simple terms"
```

### 对比多个模型

```bash
python -m src.tools.quick_llm_selector \
  --compare gpt-4 claude-3 gemini \
  --prompt "What is the difference between convex and concave lenses?"
```

### 输出格式

```bash
# 文本格式 (默认)
python -m src.tools.quick_llm_selector -p gpt-4 --prompt "Hi" -f text

# Markdown 格式 (适合复制到文档)
python -m src.tools.quick_llm_selector -p gpt-4 --prompt "Hi" -f markdown

# JSON 格式 (适合程序处理)
python -m src.tools.quick_llm_selector -p gpt-4 --prompt "Hi" -f json
```

### 自定义系统提示词

```bash
python -m src.tools.quick_llm_selector \
  --provider gpt-4 \
  --system "You are an expert optical engineer. Always include technical details." \
  --prompt "Design a lens system for a camera lens with 50mm focal length"
```

---

## 📂 项目结构

```
optis_benchmark/
├── configs/                        # ⚙️ 配置中心
│   ├── system.yaml                # 全局系统配置
│   ├── agents/                     # 智能体配置
│   │   ├── gpt-4.yaml            # OpenAI GPT-4
│   │   ├── claude-3.yaml         # Anthropic Claude
│   │   ├── gemini.yaml           # Google Gemini
│   │   ├── groq.yaml             # Groq (免费高速)
│   │   ├── ollama.yaml           # Ollama (本地部署)
│   │   ├── bedrock.yaml           # AWS Bedrock
│   │   └── together.yaml          # Together AI
│   └── tasks/                     # 任务配置
│       ├── lens_design.yaml       # 镜头设计
│       ├── system_analysis.yaml    # 系统分析
│       ├── paper_review.yaml      # 论文评审
│       ├── paper_retrieval_eval.yaml
│       ├── multi_doc_summary.yaml  # 多文档摘要
│       └── research_overview.yaml  # 研究概述
│
├── src/                           # 💻 核心源代码
│   ├── __init__.py
│   ├── main.py                    # CLI 入口
│   ├── core/                      # 核心模块
│   │   ├── __init__.py
│   │   ├── agent.py              # Agent 基类 + 7种 LLM Provider
│   │   ├── evaluator.py          # 评估器 (Metric/ROUGE/Exact)
│   │   └── runner.py             # 评测运行器 (并行执行)
│   ├── environments/              # 环境实现
│   │   ├── __init__.py
│   │   ├── base_env.py           # 环境基类
│   │   └── zos_env.py           # Zemax ZOS-API 集成
│   ├── utils/                     # 工具库
│   │   ├── __init__.py
│   │   ├── logger.py              # 日志工具
│   │   └── parser.py              # JSONL/YAML 解析器
│   └── tools/                     # 🛠️ 工具集
│       ├── __init__.py
│       └── quick_llm_selector.py  # Quick LLM Selector 工具
│
├── dataset/                        # 📚 评测数据集
├── prompts/                       # 🗣️ Prompt 模板
│   ├── system/                    # 系统提示词
│   │   ├── optical_agent.txt     # 光学设计智能体
│   │   └── research_agent.txt    # 研究任务智能体
│   └── templates/                 # 任务提示词模板
│
├── scripts/                       # 🛠️ 运维脚本
│   ├── download_data.sh
│   ├── run_eval.sh
│   └── generate_report.py
│
├── tests/                         # 🧪 测试套件
│   ├── conftest.py               # Pytest fixtures
│   ├── test_evaluator_base.py    # 评估器基础测试
│   ├── test_rouge_scorer.py      # ROUGE 评分测试
│   ├── test_citation_evaluator.py # 引用评估测试
│   ├── test_result_analyzer.py    # 结果分析测试
│   ├── test_report_generator.py   # 报告生成测试
│   ├── test_integration.py       # 集成测试
│   └── test_quick_llm_selector.py # Quick Selector 测试
│
├── docs/                          # 📖 文档
│   ├── foundation/                # 技术基础
│   │   ├── optical-basics.md
│   │   ├── agent-theory.md
│   │   └── evaluation.md
│   ├── contribution.md
│   └── README.md
│
├── website/                       # 🌐 排行榜页面
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 📋 支持的任务类型

### 🔬 光学设计任务

| 任务 ID | 描述 | 难度 | 指标 |
|---------|------|------|------|
| `lens_design` | 镜头设计与优化 | ⭐⭐⭐ | MTF, RMS Spot, Distortion |
| `system_analysis` | 光学系统性能分析 | ⭐⭐ | Completeness, Accuracy |
| `tolerance_analysis` | 公差分析与优化 | ⭐⭐⭐ | Sensitivity, Compensation |
| `surface_optimization` | 面型优化 | ⭐⭐⭐⭐ | Surface Error, Smoothness |
| `ray_tracing` | 光线追迹仿真 | ⭐⭐ | Accuracy, Performance |

### 📚 学术研究任务

| 任务 ID | 描述 | 难度 | 指标 |
|---------|------|------|------|
| `paper_review` | 学术论文评审 | ⭐⭐⭐ | Summary Quality, Technical Accuracy |
| `paper_retrieval_eval` | 论文检索与引用 | ⭐⭐ | Recall, Precision, Citation Accuracy |
| `multi_doc_summary` | 多文档综述生成 | ⭐⭐⭐ | ROUGE-L, Coverage, Coherence |
| `research_overview` | 领域内研究点概括 | ⭐⭐⭐⭐ | Coverage, Categorization, Trends |

---

## 🔧 二次开发指南

### 添加新的 LLM Provider

**步骤 1**: 创建 Provider 类

在 `src/core/agent.py` 中添加新类：

```python
class MyProviderAgent(BaseAgent):
    """My Custom LLM Provider."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # 初始化你的客户端
        self.client = MyProviderClient(api_key=config.api_key)
    
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
    ) -> AgentResponse:
        # 实现 chat 逻辑
        ...
        return AgentResponse(
            content=response_text,
            usage={"prompt_tokens": n1, "completion_tokens": n2},
            cost=calculate_cost(n1, n2),
            latency=elapsed_time,
        )
    
    async def close(self) -> None:
        await self.client.close()
```

**步骤 2**: 注册到 Factory

在 `create_agent()` 函数中添加映射：

```python
provider_map = {
    AgentProvider.OPENAI: OpenAIAgent,
    AgentProvider.ANTHROPIC: AnthropicAgent,
    # ... 其他 provider
    AgentProvider.MY_PROVIDER: MyProviderAgent,  # 添加这行
}
```

**步骤 3**: 创建配置文件

`configs/agents/my-provider.yaml`:

```yaml
model:
  provider: my_provider  # 与 AgentProvider 枚举值对应
  name: my-model-name
  api_key: ${MY_PROVIDER_API_KEY}
  temperature: 0.0
  max_tokens: 4096

agent:
  name: "My Provider Agent"

tools:
  enabled:
    - file_read
    - python_execute

execution:
  timeout: 300
  max_retries: 3
```

**步骤 4**: 测试新 Provider

```bash
python -m src.tools.quick_llm_selector --provider my-provider --prompt "Hello"
```

---

### 添加新的评测指标

**步骤 1**: 创建 Evaluator 类

在 `src/core/evaluator.py` 中添加：

```python
class MyMetricEvaluator(MetricBasedEvaluator):
    """Custom metric evaluator."""
    
    async def evaluate(
        self,
        task_id: str,
        predicted_output: Any,
        expected_output: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> EvaluationResult:
        # 实现评估逻辑
        ...
        return EvaluationResult(
            task_id=task_id,
            success=success,
            score=score,
            metrics={"my_metric": value},
        )
```

**步骤 2**: 注册到 Factory

在 `create_evaluator()` 函数中添加：

```python
def create_evaluator(config: dict[str, Any]) -> BaseEvaluator:
    scoring_method = config.get("scoring_method", "metric_based")
    
    if scoring_method == "my_metric":
        return MyMetricEvaluator(config)
    # ... 其他 evaluator
```

**步骤 3**: 在任务配置中使用

`configs/tasks/my_task.yaml`:

```yaml
evaluation:
  scoring_method: "my_metric"  # 与 factory 中的 key 对应
  # ... 其他配置
```

---

### 添加新的评测任务

**步骤 1**: 准备数据集 (JSONL 格式)

`dataset/processed/my_task.jsonl`:

```json
{"task_id": "task_001", "instruction": "...", "expected_output": {...}, "metadata": {...}}
{"task_id": "task_002", "instruction": "...", "expected_output": {...}, "metadata": {...}}
```

**步骤 2**: 创建任务配置

`configs/tasks/my_task.yaml`:

```yaml
task:
  id: "my_task"
  name: "My Task"
  category: "custom"
  difficulty: 3

dataset:
  path: "dataset/processed/my_task.jsonl"
  num_samples: 50

evaluation:
  scoring_method: "metric_based"  # 或 "summarization", "citation" 等
  metrics:
    - name: "accuracy"
      type: "numeric"
  success_criteria:
    - metric: "accuracy"
      operator: ">="
      value: 0.8
```

**步骤 3**: 创建 Prompt 模板

`prompts/templates/my_task.txt`:

```
## Task
{{instruction}}

## Expected Output Format
{{expected_format}}

## Output
```

**步骤 4**: 运行评测

```bash
python src/main.py \
  --agent-config configs/agents/gpt-4.yaml \
  --task-set my_task \
  --output results/my_task.jsonl
```

---

### 扩展 Quick LLM Selector

**添加新的输出格式**:

```python
# 在 quick_llm_selector.py 中添加

def format_result_csv(self, result: dict) -> str:
    """Format result as CSV row."""
    if not result["success"]:
        return f"{result['provider']},ERROR,{result.get('error', '')}"
    
    return f"{result['provider']},{result['model']},{result['latency']:.2f},{result['cost']:.6f}"
```

**添加批量测试模式**:

```python
async def batch_test(
    self,
    provider: ProviderInfo,
    prompts: list[str],
) -> list[dict]:
    """Test multiple prompts with the same provider."""
    results = []
    for prompt in prompts:
        result = await self.test_provider(provider, prompt)
        results.append(result)
    return results
```

---

### 编写测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_quick_llm_selector.py

# 运行带覆盖率的测试
pytest tests/ --cov=src --cov-report=html

# 运行特定测试函数
pytest tests/test_evaluator_base.py::TestMetricBasedEvaluator::test_evaluate_success -v
```

---

## 🛠️ 工具集

### Quick LLM Selector

快速测试和对比 LLM 模型。

```python
from src.tools.quick_llm_selector import QuickLLMSelector

# 初始化
selector = QuickLLMSelector(config_dir="configs/agents")
selector.discover_providers()

# 列出可用模型
providers = selector.list_providers()
for p in providers:
    print(f"{p.name}: {p.model_name}")

# 测试单个模型
import asyncio
result = asyncio.run(
    selector.test_provider(providers[0], "Hello, world!")
)
print(result["response"])
```

### Evaluator

灵活的评测框架。

```python
from src.core.evaluator import (
    create_evaluator,
    SummarizationEvaluator,
    CitationEvaluator,
)

# 使用 ROUGE 评估摘要
config = {"scoring_method": "summarization"}
evaluator = create_evaluator(config)

result = await evaluator.evaluate(
    task_id="test",
    predicted_output="The quick brown fox",
    expected_output="A fast brown fox jumps",
)
print(f"Score: {result.score}")

# 使用引用评估
config = {"scoring_method": "citation"}
evaluator = create_evaluator(config)
```

### Agent Factory

统一的 Agent 创建接口。

```python
from src.core.agent import AgentConfig, create_agent

# 从 YAML 加载
config = AgentConfig.from_yaml("configs/agents/gpt-4.yaml")
agent = create_agent(config)

# 或直接构建
from src.core.agent import AgentProvider
config = AgentConfig(
    name="my-agent",
    provider=AgentProvider.OPENAI,
    model_name="gpt-4",
    api_key="${OPENAI_API_KEY}",  # 支持环境变量
    ...
)
agent = create_agent(config)
```

---

## 📖 文档导航

| 文档 | 内容 |
|------|------|
| [评测理论](docs/theory.md) | 评测指标与方法的理论基础 |
| [贡献指南](docs/contribution.md) | 如何参与项目贡献 |
| [光学基础](docs/foundation/optical-basics.md) | 光学设计基本概念 |
| [智能体理论](docs/foundation/agent-theory.md) | AI 智能体架构与原理 |
| [评测方法论](docs/foundation/evaluation.md) | 评测设计原则与实践 |

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 贡献方式

| 类型 | 说明 | 链接 |
|------|------|------|
| 🐛 Bug | 报告问题 | [Issue](https://github.com/your-org/optis_benchmark/issues) |
| 💡 特性 | 提出建议 | [Discussion](https://github.com/your-org/optis_benchmark/discussions) |
| 📝 文档 | 完善文档 | PR: `docs/` |
| 🧪 测试 | 添加测试 | PR: `tests/` |
| 🔬 数据集 | 贡献数据集 | PR: `dataset/` |
| 💻 代码 | 修复/功能 | PR: `src/` |

### 开发流程

```bash
# 1. Fork 仓库
# 2. 创建分支
git checkout -b feature/your-feature

# 3. 安装开发依赖
pip install -e ".[dev]"

# 4. 开发 & 测试
pytest tests/

# 5. 提交 (遵循 Conventional Commits)
git commit -m "feat(tools): add quick LLM selector"

# 6. Push & PR
git push origin feature/your-feature
```

---

## 📊 评测结果

查看最新的评测结果和排行榜：[Leaderboard](website/index.html)

| 模型 | 成功率 | 平均分 | 平均时间 | 成本/任务 |
|------|--------|--------|----------|-----------|
| GPT-4 Turbo | 87.5% | 92.3 | 45.2s | $0.32 |
| Claude 3.5 Sonnet | 85.2% | 89.7 | 52.1s | $0.28 |
| GPT-4o | 83.8% | 88.1 | 38.9s | $0.25 |

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

本项目借鉴了以下开源项目的设计思想：

- [AgentBench](https://github.com/OpenGVLab/AgentBench) - 多领域智能体评测框架
- [ToolBench](https://github.com/OpenBMB/ToolBench) - 工具学习基准
- [MOSS](https://github.com/OpenMOSS/MOSS) - 开源智能体框架
- [Zemax OpticStudio](https://www.zemax.com/) - 光学设计软件

---

## 📬 联系方式

- 🌐 Website: https://optis-bench.org
- 💬 Discussion: [GitHub Discussions](https://github.com/your-org/optis_benchmark/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/optis_benchmark/issues)
- 📧 Email: contact@optis-bench.org

---

<div align="center">

**OptiS Benchmark** — 推动光学领域 AI 智能体评测的开放标准

⭐ Star us on GitHub!

</div>
