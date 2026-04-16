# 评测方法论

> 本文档介绍 OptiS Benchmark 的评测设计原则、指标体系和最佳实践。

---

## 📑 目录

1. [评测设计原则](#1-评测设计原则)
2. [评测框架](#2-评测框架)
3. [评测指标](#3-评测指标)
4. [评测流程](#4-评测流程)
5. [结果分析](#5-结果分析)

---

## 1. 评测设计原则

### 1.1 核心原则

| 原则 | 说明 | 实施方式 |
|------|------|----------|
| **可复现性** | 相同输入产生相同输出 | 固定随机种子、环境隔离 |
| **公平性** | 相同条件比较不同系统 | 统一评测标准 |
| **全面性** | 多维度评估能力 | 多任务、多指标 |
| **实用性** | 反映真实场景需求 | 基于实际光学设计任务 |
| **可扩展性** | 易于添加新任务 | 模块化架构 |

### 1.2 评测维度

```
┌─────────────────────────────────────────────────────────────┐
│                    智能体能力评测维度                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │  准确性     │  │   效率       │  │   鲁棒性     │       │
│   ├─────────────┤  ├─────────────┤  ├─────────────┤       │
│   │ · MTF 达标  │  │ · 响应时间   │  │ · 边界处理  │       │
│   │ · 设计合规  │  │ · 迭代次数   │  │ · 错误恢复  │       │
│   │ · 计算准确  │  │ · API 调用   │  │ · 异常输入  │       │
│   └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │  完整性     │  │   可解释性   │  │   成本效益   │       │
│   ├─────────────┤  ├─────────────┤  ├─────────────┤       │
│   │ · 覆盖范围  │  │ · 设计说明   │  │ · API 成本  │       │
│   │ · 分析深度  │  │ · 推理过程   │  │ · 资源消耗  │       │
│   │ · 报告质量  │  │ · 透明度    │  │ · 时间成本   │       │
│   └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 评测框架

### 2.1 评测架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      OptiS 评测架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                              │
│  │   Task Set   │  ← 评测任务集合                               │
│  │              │     - lens_design                             │
│  │  ┌────────┐  │     - system_analysis                         │
│  │  │ Task 1 │  │     - tolerance_analysis                      │
│  │  ├────────┤  │                                              │
│  │  │ Task 2 │  │                                              │
│  │  ├────────┤  │                                              │
│  │  │  ...   │  │                                              │
│  │  └────────┘  │                                              │
│  └──────────────┘                                              │
│          ↓                                                      │
│  ┌──────────────┐                                              │
│  │    Agent     │  ← 被评测的智能体                             │
│  │              │     - OpenAI GPT-4                            │
│  │   ┌──────┐   │     - Anthropic Claude                       │
│  │   │  LLM │   │     - Google Gemini                          │
│  │   └──────┘   │     - ...                                     │
│  │      ↓       │                                              │
│  │   ┌──────┐   │                                              │
│  │   │Tools │   │                                              │
│  │   └──────┘   │                                              │
│  └──────────────┘                                              │
│          ↓                                                      │
│  ┌──────────────┐                                              │
│  │  Environment │  ← 执行环境                                 │
│  │              │     - ZOS-API (Zemax)                        │
│  │  ┌────────┐  │     - CODE V                                  │
│  │  │Software│  │     - ASAP                                    │
│  │  └────────┘  │     - Python Sandbox                         │
│  └──────────────┘                                              │
│          ↓                                                      │
│  ┌──────────────┐                                              │
│  │  Evaluator   │  ← 评测器                                    │
│  │              │     - MetricBased                             │
│  │  ┌────────┐  │     - ExactMatch                              │
│  │  │ Score  │  │     - LLMJudge                               │
│  │  └────────┘  │                                              │
│  └──────────────┘                                              │
│          ↓                                                      │
│  ┌──────────────┐                                              │
│  │   Results    │  ← 评测结果                                  │
│  │              │                                              │
│  │  ┌────────┐  │                                              │
│  │  │ Report │  │                                              │
│  │  └────────┘  │                                              │
│  └──────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 评测对象

| 对象 | 描述 | 评测什么 |
|------|------|----------|
| **LLM** | 底层语言模型 | 光学知识、推理能力 |
| **Agent** | 带工具的智能体 | 工具使用、规划能力 |
| **System** | 完整解决方案 | 端到端性能 |

### 2.3 任务分类

```python
TASK_CATEGORIES = {
    "lens_design": {
        "description": "镜头设计与优化",
        "difficulty_range": [2, 5],
        "typical_duration": "10-30 min",
        "evaluation_focus": ["design_quality", "mtf_performance"],
    },
    "system_analysis": {
        "description": "光学系统分析",
        "difficulty_range": [1, 3],
        "typical_duration": "5-15 min",
        "evaluation_focus": ["analysis_completeness", "metric_accuracy"],
    },
    "tolerance_analysis": {
        "description": "公差分析与优化",
        "difficulty_range": [3, 4],
        "typical_duration": "15-30 min",
        "evaluation_focus": ["sensitivity_prediction", "compensation"],
    },
}
```

---

## 3. 评测指标

### 3.1 指标体系

```
评测指标
├── 功能指标
│   ├── 任务成功率 (Success Rate)
│   ├── 完成度 (Completion Rate)
│   └── 正确性 (Correctness)
│
├── 性能指标
│   ├── MTF 达标率
│   ├── RMS Spot 达标率
│   └── 畸变达标率
│
├── 效率指标
│   ├── 响应时间 (Latency)
│   ├── 迭代次数 (Iterations)
│   ├── API 调用次数 (API Calls)
│   └── 令牌消耗 (Token Usage)
│
└── 成本指标
    ├── API 成本 (Cost)
    └── 综合成本效益 (Cost-Effectiveness)
```

### 3.2 核心指标定义

```python
class Metrics:
    """评测指标定义"""
    
    @staticmethod
    def success_rate(results: list[TaskResult]) -> float:
        """
        任务成功率
        
        定义: 成功完成任务数 / 总任务数
        
        成功标准:
        - 所有必需指标达标
        - 无致命错误
        - 在时间限制内完成
        """
        successful = sum(1 for r in results if r.success)
        return successful / len(results) if results else 0.0
    
    @staticmethod
    def mtf_attainment(
        predicted_mtf: float,
        target_mtf: float,
        tolerance: float = 0.05,
    ) -> float:
        """
        MTF 达标率
        
        Args:
            predicted_mtf: 预测 MTF 值
            target_mtf: 目标 MTF 值
            tolerance: 容差
        
        Returns:
            达标返回 1.0，否则返回实际比率
        """
        if predicted_mtf >= target_mtf:
            return 1.0
        
        ratio = predicted_mtf / target_mtf
        return max(0.0, ratio)  # 不返回负值
    
    @staticmethod
    def average_latency(results: list[TaskResult]) -> float:
        """
        平均响应时间
        
        定义: 总执行时间 / 任务数
        """
        if not results:
            return 0.0
        return sum(r.latency for r in results) / len(results)
    
    @staticmethod
    def cost_efficiency(
        success_rate: float,
        total_cost: float,
        baseline_cost: float = 1.0,
    ) -> float:
        """
        成本效益
        
        定义: 成功率 / 相对成本
        """
        if total_cost == 0:
            return 0.0
        relative_cost = total_cost / baseline_cost
        return success_rate / relative_cost
```

### 3.3 综合评分

```python
class CompositeScore:
    """综合评分计算"""
    
    WEIGHTS = {
        "success_rate": 0.30,      # 成功率权重
        "performance": 0.35,       # 性能指标权重
        "efficiency": 0.20,        # 效率权重
        "cost": 0.15,              # 成本权重
    }
    
    @classmethod
    def calculate(cls, results: list[TaskResult]) -> dict:
        """计算综合评分"""
        
        # 各维度得分
        success_score = cls.success_rate_score(results)
        performance_score = cls.performance_score(results)
        efficiency_score = cls.efficiency_score(results)
        cost_score = cls.cost_score(results)
        
        # 加权综合
        composite = (
            cls.WEIGHTS["success_rate"] * success_score +
            cls.WEIGHTS["performance"] * performance_score +
            cls.WEIGHTS["efficiency"] * efficiency_score +
            cls.WEIGHTS["cost"] * cost_score
        )
        
        return {
            "composite_score": composite,
            "success_rate": success_score,
            "performance": performance_score,
            "efficiency": efficiency_score,
            "cost": cost_score,
        }
```

---

## 4. 评测流程

### 4.1 完整评测流程

```
┌─────────────────────────────────────────────────────────────────┐
│                       完整评测流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  1. 准备     │ →  │  2. 执行     │ →  │  3. 收集     │        │
│  │  Preparation │   │  Execution  │   │  Collection │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│       ↓                   ↓                   ↓                │
│  · 加载配置          · 运行任务          · 记录结果              │
│  · 初始化环境        · 监控进度          · 收集指标              │
│  · 检查依赖          · 处理异常          · 保存日志              │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  4. 评估     │ →  │  5. 聚合     │ →  │  6. 报告     │        │
│  │  Evaluation  │   │  Aggregation │   │  Reporting   │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│       ↓                   ↓                   ↓                │
│  · 计算指标          · 汇总统计          · 生成报告              │
│  · 比对标准          · 置信区间          · 可视化              │
│  · 打分              · 排序              · 发布结果              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 并行评测

```python
async def parallel_evaluation(
    agent_config: AgentConfig,
    task_config: TaskConfig,
    max_concurrency: int = 4,
) -> list[TaskResult]:
    """
    并行评测
    
    Args:
        agent_config: 智能体配置
        task_config: 任务配置
        max_concurrency: 最大并发数
    
    Returns:
        任务结果列表
    """
    
    # 创建信号量控制并发
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def run_with_limit(task: Task):
        async with semaphore:
            return await run_single_task(agent_config, task)
    
    # 并行执行
    tasks = load_tasks(task_config)
    results = await asyncio.gather(*[
        run_with_limit(task) for task in tasks
    ])
    
    return results
```

### 4.3 质量保证

```python
class EvaluationQA:
    """评测质量保证"""
    
    @staticmethod
    def validate_results(results: list[TaskResult]) -> ValidationReport:
        """
        验证结果有效性
        """
        issues = []
        
        # 检查基本完整性
        for i, result in enumerate(results):
            if not result.task_id:
                issues.append(f"Task {i}: Missing task_id")
            
            if result.latency < 0:
                issues.append(f"Task {i}: Negative latency")
        
        # 检查统计异常
        latencies = [r.latency for r in results]
        outliers = detect_outliers(latencies)
        if outliers:
            issues.append(f"Found {len(outliers)} latency outliers")
        
        return ValidationReport(
            valid=len(issues) == 0,
            issues=issues,
        )
    
    @staticmethod
    def check_consistency(results: list[TaskResult]) -> bool:
        """
        检查结果一致性
        """
        # 同一任务多次运行应有相似结果
        # (需要实现重复运行机制)
        return True
```

---

## 5. 结果分析

### 5.1 统计分析

```python
class ResultAnalyzer:
    """结果分析器"""
    
    @staticmethod
    def compute_statistics(results: list[TaskResult]) -> dict:
        """
        计算统计指标
        """
        import numpy as np
        
        scores = [r.score for r in results]
        latencies = [r.latency for r in results]
        
        return {
            # 中心趋势
            "mean_score": np.mean(scores),
            "median_score": np.median(scores),
            
            # 离散程度
            "std_score": np.std(scores),
            "iqr_score": np.percentile(scores, 75) - np.percentile(scores, 25),
            
            # 范围
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            
            # 置信区间
            "ci_95": compute_confidence_interval(scores, 0.95),
            
            # 时间统计
            "mean_latency": np.mean(latencies),
            "p95_latency": np.percentile(latencies, 95),
        }
    
    @staticmethod
    def compare_models(
        results_a: list[TaskResult],
        results_b: list[TaskResult],
    ) -> dict:
        """
        模型对比分析
        
        使用统计检验判断差异是否显著
        """
        from scipy import stats
        
        scores_a = [r.score for r in results_a]
        scores_b = [r.score for r in results_b]
        
        # t 检验
        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
        
        return {
            "mean_a": np.mean(scores_a),
            "mean_b": np.mean(scores_b),
            "difference": np.mean(scores_a) - np.mean(scores_b),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "winner": "A" if np.mean(scores_a) > np.mean(scores_b) else "B",
        }
```

### 5.2 错误分析

```python
class ErrorAnalyzer:
    """错误分析器"""
    
    @staticmethod
    def categorize_errors(results: list[TaskResult]) -> dict:
        """
        错误分类统计
        """
        error_categories = {
            "timeout": [],
            "api_error": [],
            "invalid_output": [],
            "quality_below_threshold": [],
            "other": [],
        }
        
        for result in results:
            if not result.success:
                category = ErrorAnalyzer._classify_error(result)
                error_categories[category].append(result)
        
        return {
            category: len(errors)
            for category, errors in error_categories.items()
        }
    
    @staticmethod
    def _classify_error(result: TaskResult) -> str:
        """分类单个错误"""
        if "timeout" in result.error.lower():
            return "timeout"
        elif "api" in result.error.lower():
            return "api_error"
        elif "invalid" in result.error.lower():
            return "invalid_output"
        elif result.score < result.threshold:
            return "quality_below_threshold"
        else:
            return "other"
```

### 5.3 报告生成

```python
class ReportGenerator:
    """报告生成器"""
    
    @staticmethod
    def generate_html_report(
        results: list[TaskResult],
        model_name: str,
        task_name: str,
    ) -> str:
        """
        生成 HTML 评测报告
        """
        
        stats = ResultAnalyzer.compute_statistics(results)
        
        html = f"""
        <html>
        <head><title>Evaluation Report: {model_name}</title></head>
        <body>
            <h1>OptiS Benchmark Report</h1>
            <h2>Model: {model_name}</h2>
            <h2>Task: {task_name}</h2>
            
            <h3>Summary Statistics</h3>
            <ul>
                <li>Total Tasks: {len(results)}</li>
                <li>Success Rate: {stats['success_rate']:.1%}</li>
                <li>Mean Score: {stats['mean_score']:.3f}</li>
                <li>Std Dev: {stats['std_score']:.3f}</li>
            </ul>
            
            <!-- 更多内容... -->
        </body>
        </html>
        """
        
        return html
```

---

## 📚 参考资料

1. AgentBench: Evaluating LLMs as Agents - Liu et al., 2023
2. HELM: Holistic Evaluation of Language Models - Liang et al., 2022
3. BigBench: Beyond the Imitation Game - Srivastava et al., 2022
4. MMLU: Measuring Massive Multitask Language Understanding - Hendrycks et al., 2021

---

*最后更新: 2024-XX-XX*
