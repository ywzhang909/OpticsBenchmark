# AI 智能体理论基础

> 本文档介绍 AI 智能体的概念、架构和设计原理，为构建光学设计智能体提供理论基础。

---

## 📑 目录

1. [智能体概述](#1-智能体概述)
2. [智能体架构](#2-智能体架构)
3. [工具使用](#3-工具使用)
4. [规划与推理](#4-规划与推理)
5. [光学设计智能体](#5-光学设计智能体)

---

## 1. 智能体概述

### 1.1 什么是 AI 智能体

**AI 智能体 (Agent)** 是一个能够感知环境、做出决策并执行动作以实现目标的系统。

```
┌─────────────────────────────────────────────────────────────┐
│                        AI 智能体                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌───────────────┐    ┌─────────┐           │
│  │  感知   │ →  │     推理       │ →  │   行动   │           │
│  │Perceive│    │Reason/Plan    │    │  Act    │           │
│  └─────────┘    └───────────────┘    └─────────┘           │
│       ↑                                    │                 │
│       │         ┌───────────────┐         │                 │
│       └──────── │    记忆       │ ← ─ ─ ─ ┘                 │
│                 │   Memory      │                           │
│                 └───────────────┘                           │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    环 境 (Environment)                │    │
│  │         光学软件 / 文件系统 / API / 用户              │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 智能体的关键能力

| 能力 | 描述 | 在光学设计中的应用 |
|------|------|-------------------|
| **感知** | 理解任务和上下文 | 解析设计需求 |
| **推理** | 分析问题、制定策略 | 光学设计计算 |
| **规划** | 分解任务、确定步骤 | 设计流程规划 |
| **工具使用** | 调用外部工具 | 使用 ZOS-API |
| **学习** | 从反馈中改进 | 优化设计方案 |
| **记忆** | 保存上下文信息 | 跟踪设计历史 |

### 1.3 智能体 vs 传统程序

| 特性 | 传统程序 | AI 智能体 |
|------|----------|----------|
| 规则 | 显式编码 | 隐式学习 |
| 灵活性 | 低 | 高 |
| 泛化能力 | 有限 | 可泛化 |
| 交互方式 | 固定 API | 自然语言 |
| 错误处理 | 预定义 | 可适应 |

---

## 2. 智能体架构

### 2.1 ReAct 架构

ReAct (Reason + Act) 是一种流行的智能体架构：

```
┌─────────────────────────────────────────────────────────────┐
│                      ReAct 循环                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Thought → Action → Observation → Thought → ... → Final     │
│     ↓        ↓          ↓                                   │
│   思考      行动      观察结果                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class ReActAgent:
    """
    ReAct 架构的智能体实现
    
    循环执行:
    1. Thought: 分析当前状态，决定下一步
    2. Action: 执行选定的动作
    3. Observation: 获取执行结果
    """
    
    def __init__(self, llm, tools, max_steps: int = 10):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []
    
    async def run(self, task: str) -> str:
        """执行任务"""
        obs = ""  # 初始观察
        
        for step in range(self.max_steps):
            # 1. Thought: 生成思考
            thought = await self.think(task, obs)
            
            # 2. Action: 决定动作
            action = await self.decide_action(thought, self.tools)
            
            # 3. Execute: 执行动作
            result = await self.execute(action)
            
            # 4. 更新观察
            obs = result
            
            # 检查是否完成
            if self.is_complete(result):
                break
        
        return self.generate_final_response()
```

### 2.2 Chain-of-Thought (CoT)

链式思考是一种推理技术：

```
用户: 设计一个 50mm F/2 的消色差镜头

智能体推理链:
├─ Step 1: 分析需求
│   - 焦距: 50mm
│   - F数: F/2 → 入瞳直径: 25mm
│   - 消色差 → 需要至少两种玻璃材料
│
├─ Step 2: 初步选材
│   - 正透镜: N-BK7 (低色散)
│   - 负透镜: F2 (高色散)
│   - 组合可校正 486nm 和 656nm 的色差
│
├─ Step 3: 初始参数计算
│   - 使用薄透镜公式计算焦距分配
│   - 正透镜焦距: 约 80mm
│   - 负透镜焦距: 约 -130mm
│
├─ Step 4: 优化
│   - 设置变量: 曲率半径、透镜厚度
│   - 目标: 最小化 RMS 波前误差
│
└─ Step 5: 输出设计
    - 透镜1: R1=52mm, R2=-85mm, d=5mm, N-BK7
    - 透镜2: R1=-75mm, R2=120mm, d=2mm, F2
```

### 2.3 多智能体协作

```python
class MultiAgentSystem:
    """
    多智能体协作系统
    
    角色分工:
    - 光学专家: 负责光学设计计算
    - 代码工程师: 负责 API 调用和代码生成
    - 评审专家: 评估设计方案
    """
    
    def __init__(self):
        self.agents = {
            "optics_expert": OpticsExpertAgent(),
            "code_engineer": CodeEngineerAgent(),
            "reviewer": ReviewerAgent(),
        }
        self.coordinator = CoordinatorAgent(self.agents)
    
    async def solve(self, task: str) -> Solution:
        """协作解决问题"""
        
        # 1. 协调器分析任务
        plan = await self.coordinator.create_plan(task)
        
        # 2. 分配子任务
        subtasks = plan.decompose()
        
        # 3. 并行执行
        results = await asyncio.gather(*[
            self.execute_subtask(agent, subtask)
            for agent, subtask in subtasks
        ])
        
        # 4. 整合结果
        solution = await self.coordinator.integrate(results)
        
        # 5. 评审
        review = await self.agents["reviewer"].review(solution)
        
        return solution if review.approved else self.iterate(solution, review)
```

---

## 3. 工具使用

### 3.1 工具定义

```python
from pydantic import BaseModel
from typing import Callable, Any

class Tool(BaseModel):
    """工具定义"""
    name: str                          # 工具名称
    description: str                   # 工具描述
    parameters: dict[str, Any]         # 参数模式 (JSON Schema)
    execute: Callable[..., Any]        # 执行函数
    
    def to_openai_format(self) -> dict:
        """转换为 OpenAI tool format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
```

### 3.2 光学设计工具示例

```python
class OpticalTools:
    """光学设计工具集"""
    
    @staticmethod
    def load_lens(name: str, path: str) -> dict:
        """
        加载镜头文件
        
        Args:
            name: 镜头名称
            path: 文件路径
        
        Returns:
            镜头数据字典
        """
        pass  # 调用 ZOS-API 实现
    
    @staticmethod
    def get_system_data() -> dict:
        """
        获取当前系统数据
        
        Returns:
            {
                "focal_length": 50.0,
                "fnumber": 2.0,
                "wavelengths": [...],
                "fields": [...],
                "surfaces": 6,
            }
        """
        pass
    
    @staticmethod
    def analyze_mtf() -> dict:
        """
        运行 MTF 分析
        
        Returns:
            {
                "spatial_frequency": [10, 20, 30, ...],
                "mtf_tangential": [0.95, 0.88, 0.75, ...],
                "mtf_sagittal": [0.95, 0.90, 0.80, ...],
            }
        """
        pass
    
    @staticmethod
    def optimize(
        variables: list[str],
        targets: list[str],
        algorithm: str = "DLS",
    ) -> dict:
        """
        运行优化
        
        Args:
            variables: 优化变量列表
            targets: 优化目标列表
            algorithm: 优化算法
        
        Returns:
            优化结果
        """
        pass
    
    @staticmethod
    def calculate_raytrace(
        field_x: float,
        field_y: float,
        wavelength: float,
    ) -> list[dict]:
        """
        光线追迹
        
        Args:
            field_x, field_y: 视场坐标
            wavelength: 波长
        
        Returns:
            光线交点数据
        """
        pass
```

### 3.3 工具选择策略

```python
async def select_tool(
    task: str,
    available_tools: list[Tool],
    context: dict,
) -> Tool:
    """
    基于任务选择最合适的工具
    
    策略:
    1. 任务分解 → 选择主工具
    2. 参数匹配 → 筛选候选工具
    3. 上下文推断 → 最终选择
    """
    
    # LLM 决定使用哪个工具
    prompt = f"""
    任务: {task}
    
    可用工具:
    {format_tools(available_tools)}
    
    上下文:
    {format_context(context)}
    
    决定使用哪个工具，并说明原因。
    """
    
    response = await llm.chat(prompt)
    return parse_tool_selection(response)
```

---

## 4. 规划与推理

### 4.1 任务规划

```python
class TaskPlanner:
    """任务规划器"""
    
    def create_plan(self, goal: str) -> Plan:
        """
        创建任务执行计划
        
        1. 目标分解: 将大目标分解为子目标
        2. 依赖分析: 确定子目标间的依赖关系
        3. 资源分配: 分配执行所需的工具和知识
        """
        
        # 子目标分解
        subgoals = self.decompose(goal)
        
        # 构建执行图
        execution_graph = self.build_graph(subgoals)
        
        # 生成执行计划
        plan = Plan(
            steps=execution_graph.topological_sort(),
            estimated_time=self.estimate_time(execution_graph),
            required_tools=self.collect_tools(execution_graph),
        )
        
        return plan
```

### 4.2 自我反思

```python
class SelfReflection:
    """自我反思机制"""
    
    async def reflect(
        self,
        action: str,
        result: Any,
        expected: Any,
    ) -> Reflection:
        """
        反思行动结果
        
        1. 成功? 记录有效策略
        2. 失败? 分析原因，调整策略
        3. 改进? 更新内部知识
        """
        
        if self.is_success(result, expected):
            return Reflection(
                outcome="success",
                strategy=self.extract_strategy(action),
                confidence=1.0,
            )
        else:
            analysis = await self.analyze_failure(action, result, expected)
            return Reflection(
                outcome="failure",
                reason=analysis.cause,
                adjustment=analysis.adjustment,
                confidence=0.5,
            )
```

---

## 5. 光学设计智能体

### 5.1 智能体系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    OptiS Agent 系统架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    User Interface                        │    │
│  │              (自然语言设计需求)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  LLM Core (GPT-4 / Claude)               │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐               │    │
│  │  │Planner  │  │ Reasoner │  │Reflector│               │    │
│  │  └─────────┘  └─────────┘  └─────────┘               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     Tool Layer                           │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │    │
│  │  │ZOS-API  │ │ CODE V  │ │ Python  │ │  File   │     │    │
│  │  │  Tools  │ │  Tools  │ │  Tools  │ │  Tools  │     │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Optical Software Environment                │    │
│  │           (Zemax / CODE V / ASAP / Python)              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 光学智能体核心逻辑

```python
class OptiSAgent:
    """
    光学设计智能体
    
    能力:
    - 理解和解析光学设计需求
    - 使用光学设计软件 (ZOS-API)
    - 进行光学计算和优化
    - 生成设计报告
    """
    
    def __init__(self, llm_config: dict):
        self.llm = create_llm(llm_config)
        self.tools = self._load_tools()
        self.memory = Memory()
        self.context = {}
    
    async def design_lens(self, requirements: str) -> DesignResult:
        """
        设计镜头
        
        Args:
            requirements: 自然语言描述的设计需求
        
        Returns:
            设计结果
        """
        
        # 1. 解析需求
        specs = await self._parse_requirements(requirements)
        
        # 2. 创建设计计划
        plan = self._create_design_plan(specs)
        
        # 3. 执行计划
        for step in plan.steps:
            result = await self._execute_step(step)
            self.memory.add(result)
            
            # 自我检查
            if not self._check_result(result, specs):
                # 调整策略
                step = self._adjust_step(step, result)
                result = await self._execute_step(step)
        
        # 4. 优化设计
        optimized = await self._optimize_design(specs)
        
        # 5. 验证
        validated = await self._validate_design(optimized, specs)
        
        # 6. 生成报告
        report = self._generate_report(validated)
        
        return DesignResult(
            success=True,
            design=optimized,
            report=report,
            metrics=self._extract_metrics(validated),
        )
```

### 5.3 光学知识库

```python
OPTICAL_KNOWLEDGE = {
    # 光学设计规则
    "rules": [
        "消色差双合透镜需要选择色散差异大的玻璃",
        "F数越小，球差越严重",
        "透镜厚度一般为直径的 5-15%",
    ],
    
    # 材料选择指南
    "materials": {
        "visible": {
            "positive": ["N-BK7", "N-SSK8", "N-SK16"],
            "negative": ["F2", "N-SF11", "N-SF6"],
        },
        "achromats": {
            "WDM": ["N-BK7", "F2"],      # 可见光消色差
            "infrared": ["Ge", "ZnSe"],  # 红外消色差
        },
    },
    
    # 典型设计模板
    "templates": {
        "landscape_lens": {
            "type": "Doublet",
            "focal_length_range": [28, 135],
            "typical_config": "Gauss-type",
        },
        "portrait_lens": {
            "type": "Plano-convex + Meniscus",
            "focal_length_range": [50, 200],
            "typical_config": " reversed Gauss",
        },
    },
}
```

---

## 📚 参考资料

1. Wei et al. "Chain-of-Thought Prompting Elicits Reasoning" - NeurIPS 2022
2. Yao et al. "ReAct: Synergizing Reasoning and Acting" - ICLR 2023
3. OpenAI. "GPT-4 System Card"
4. Anthropic. "Claude Model Card"
5. Richards et al. "AgentBench: Evaluating LLMs as Agents" - 2023

---

*最后更新: 2024-XX-XX*
