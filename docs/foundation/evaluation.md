# Evaluation Methodology

> This document introduces the design principles, metric system and best practices of the Optis Benchmark evaluation.

---

## 📑 Table of Contents

1. [Evaluation Design Principles](#1-evaluation-design-principles)
2. [Evaluation Framework](#2-evaluation-framework)
3. [Evaluation Metrics](#3-evaluation-metrics)
4. [Evaluation Workflow](#4-evaluation-workflow)
5. [Result Analysis](#5-result-analysis)

---

## 1. Evaluation Design Principles

### 1.1 Core Principles

| Principle | Description | How It Is Implemented |
|------|------|----------|
| **Reproducibility** | The same input produces the same output | Fixed random seeds, isolated environments |
| **Fairness** | Different systems are compared under identical conditions | Unified evaluation criteria |
| **Comprehensiveness** | Capability is assessed across multiple dimensions | Multiple tasks, multiple metrics |
| **Practicality** | Real-world requirements are reflected | Based on actual optical design tasks |
| **Extensibility** | New tasks are easy to add | Modular architecture |

### 1.2 Evaluation Dimensions

```
┌─────────────────────────────────────────────────────────────────────────┐
│          Dimensions for Evaluating Agent Capability                     │
│                                                                         │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐    │
│  │Accuracy           │  │Efficiency         │  │Robustness         │    │
│  ├───────────────────┤  ├───────────────────┤  ├───────────────────┤    │
│  │· MTF target met   │  │· Response time    │  │· Edge cases       │    │
│  │· Design valid     │  │· Iterations       │  │· Error recovery   │    │
│  │· Computation ok   │  │· API calls        │  │· Invalid input    │    │
│                                                                         │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐    │
│  │Completeness       │  │Explainability     │  │Cost-effectiveness │    │
│  ├───────────────────┤  ├───────────────────┤  ├───────────────────┤    │
│  │· Coverage         │  │· Design notes     │  │· API cost         │    │
│  │· Analysis depth   │  │· Reasoning        │  │· Resource use     │    │
│  │· Report quality   │  │· Transparency     │  │· Time cost        │    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Evaluation Framework

### 2.1 Evaluation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Optis Evaluation Architecture               │
│                                                                 │
│  ┌────────────┐                                                 │
│  │Task Set    │  ← task set                                     │
│  │            │  - lens_design                                  │
│  │            │  - system_analysis                              │
│  │            │  - tolerance_analysis                           │
│  │  ┌────────┐                                                  │
│  │  │Task 1  │                                                  │
│  │  ├────────┤                                                  │
│  │  │Task 2  │                                                  │
│  │  ├────────┤                                                  │
│  │  │...     │                                                  │
│  │  └────────┘                                                  │
│  │            │                                                 │
│  └────────────┘                                                 │
│         ▼                                                       │
│  ┌────────────┐                                                 │
│  │Agent       │  ← the agent under evaluation                   │
│  │            │  - OpenAI GPT-4                                 │
│  │            │  - Anthropic Claude                             │
│  │            │  - Google Gemini                                │
│  │            │  - ...                                          │
│  │  ┌───────┐                                                   │
│  │  │LLM    │                                                   │
│  │  ├───────┤                                                   │
│  │  │Tools  │                                                   │
│  │  └───────┘                                                   │
│  │            │                                                 │
│  └────────────┘                                                 │
│         ▼                                                       │
│  ┌────────────┐                                                 │
│  │Environment │  ← execution environment                        │
│  │            │  - ZOS-API (Zemax)                              │
│  │            │  - CODE V                                       │
│  │            │  - ASAP                                         │
│  │            │  - Python Sandbox                               │
│  │  ┌──────────┐                                                │
│  │  │Software  │                                                │
│  │  └──────────┘                                                │
│  │            │                                                 │
│  └────────────┘                                                 │
│         ▼                                                       │
│  ┌────────────┐                                                 │
│  │Evaluator   │  ← evaluator                                    │
│  │            │  - MetricBased                                  │
│  │            │  - ExactMatch                                   │
│  │            │  - LLMJudge                                     │
│  │  ┌───────┐                                                   │
│  │  │Score  │                                                   │
│  │  └───────┘                                                   │
│  │            │                                                 │
│  └────────────┘                                                 │
│         ▼                                                       │
│  ┌────────────┐                                                 │
│  │Results     │  ← evaluation results                           │
│  │  ┌────────┐                                                  │
│  │  │Report  │                                                  │
│  │  └────────┘                                                  │
│  │            │                                                 │
│  └────────────┘                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 What Is Evaluated

| Subject | Description | What Is Evaluated |
|------|------|----------|
| **LLM** | The underlying language model | Optical knowledge, reasoning ability |
| **Agent** | An agent equipped with tools | Tool usage, planning ability |
| **System** | The complete solution | End-to-end performance |

### 2.3 Task Categories

```python
TASK_CATEGORIES = {
    "lens_design": {
        "description": "lens design and optimization",
        "difficulty_range": [2, 5],
        "typical_duration": "10-30 min",
        "evaluation_focus": ["design_quality", "mtf_performance"],
    },
    "system_analysis": {
        "description": "optical system analysis",
        "difficulty_range": [1, 3],
        "typical_duration": "5-15 min",
        "evaluation_focus": ["analysis_completeness", "metric_accuracy"],
    },
    "tolerance_analysis": {
        "description": "tolerance analysis and optimization",
        "difficulty_range": [3, 4],
        "typical_duration": "15-30 min",
        "evaluation_focus": ["sensitivity_prediction", "compensation"],
    },
}
```

---

## 3. Evaluation Metrics

### 3.1 Metric System

```
Evaluation metrics
├── Functional metrics
│   ├── Task success rate (Success Rate)
│   ├── Completion rate (Completion Rate)
│   └── Correctness (Correctness)
│
├── Performance metrics
│   ├── MTF attainment rate
│   ├── RMS spot attainment rate
│   └── Distortion attainment rate
│
├── Efficiency metrics
│   ├── Response time (Latency)
│   ├── Number of iterations (Iterations)
│   ├── Number of API calls (API Calls)
│   └── Token usage (Token Usage)
│
└── Cost metrics
    ├── API cost (Cost)
    └── Overall cost-effectiveness (Cost-Effectiveness)
```

### 3.2 Definition of the Core Metrics

```python
class Metrics:
    """Definition of the evaluation metrics."""

    @staticmethod
    def success_rate(results: list[TaskResult]) -> float:
        """
        Task success rate.

        Definition: number of successfully completed tasks / total number of tasks

        Criteria for success:
        - all required metrics are met
        - no fatal errors
        - completed within the time limit
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
        MTF attainment rate.

        Args:
            predicted_mtf: predicted MTF value
            target_mtf: target MTF value
            tolerance: tolerance

        Returns:
            1.0 if the target is met, otherwise the actual ratio
        """
        if predicted_mtf >= target_mtf:
            return 1.0

        ratio = predicted_mtf / target_mtf
        return max(0.0, ratio)  # never returns a negative value

    @staticmethod
    def average_latency(results: list[TaskResult]) -> float:
        """
        Average response time.

        Definition: total execution time / number of tasks
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
        Cost-effectiveness.

        Definition: success rate / relative cost
        """
        if total_cost == 0:
            return 0.0
        relative_cost = total_cost / baseline_cost
        return success_rate / relative_cost
```

### 3.3 Composite Score

```python
class CompositeScore:
    """Composite score calculation."""

    WEIGHTS = {
        "success_rate": 0.30,      # weight of the success rate
        "performance": 0.35,       # weight of the performance metrics
        "efficiency": 0.20,        # weight of efficiency
        "cost": 0.15,              # weight of cost
    }

    @classmethod
    def calculate(cls, results: list[TaskResult]) -> dict:
        """Calculate the composite score."""

        # Score of each dimension
        success_score = cls.success_rate_score(results)
        performance_score = cls.performance_score(results)
        efficiency_score = cls.efficiency_score(results)
        cost_score = cls.cost_score(results)

        # Weighted combination
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

## 4. Evaluation Workflow

### 4.1 Complete Evaluation Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    Complete Evaluation Workflow                                     │
│                                                                                     │
│  ┌────────────────┐   →┌────────────────┐   →┌────────────────┐                     │
│  │1. Preparation  │   →│2. Execution    │   →│3. Collection   │                     │
│  │Preparation     │   →│Execution       │   →│Collection      │                     │
│  └────────────────┘   →└────────────────┘   →└────────────────┘                     │
│          ▼                   ▼                   ▼                                  │
│    · Load configuration       · Run tasks                · Record results           │
│    · Initialize environment   · Monitor progress         · Collect metrics          │
│    · Check dependencies       · Handle exceptions        · Save logs                │
│                                                                                     │
│  ┌────────────────┐   ←┌────────────────┐   ←┌────────────────┐                     │
│  │4. Evaluation   │   ←│5. Aggregation  │   ←│6. Reporting    │                     │
│  │Evaluation      │   ←│Aggregation     │   ←│Reporting       │                     │
│  └────────────────┘   ←└────────────────┘   ←└────────────────┘                     │
│          ▼                   ▼                   ▼                                  │
│    · Compute metrics          · Aggregate stats          · Generate report          │
│    · Compare with spec        · Confidence intervals     · Visualize                │
│    · Score                    · Rank                     · Publish results          │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Parallel Evaluation

```python
async def parallel_evaluation(
    agent_config: dict,
    task_config: TaskConfig,
    max_concurrency: int = 4,
) -> list[TaskResult]:
    """
    Parallel evaluation.

    Args:
        agent_config: agent configuration
        task_config: task configuration
        max_concurrency: maximum concurrency

    Returns:
        list of task results
    """

    # Create a semaphore to control concurrency
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_with_limit(task: Task):
        async with semaphore:
            return await run_single_task(agent_config, task)

    # Run in parallel
    tasks = load_tasks(task_config)
    results = await asyncio.gather(*[
        run_with_limit(task) for task in tasks
    ])

    return results
```

### 4.3 Quality Assurance

```python
class EvaluationQA:
    """Quality assurance for the evaluation."""

    @staticmethod
    def validate_results(results: list[TaskResult]) -> ValidationReport:
        """
        Validate the results.
        """
        issues = []

        # Check basic completeness
        for i, result in enumerate(results):
            if not result.task_id:
                issues.append(f"Task {i}: Missing task_id")

            if result.latency < 0:
                issues.append(f"Task {i}: Negative latency")

        # Check for statistical outliers
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
        Check the consistency of the results.
        """
        # Multiple runs of the same task should give similar results
        # (a repeat-run mechanism needs to be implemented)
        return True
```

---

## 5. Result Analysis

### 5.1 Statistical Analysis

```python
class ResultAnalyzer:
    """Result analyzer."""

    @staticmethod
    def compute_statistics(results: list[TaskResult]) -> dict:
        """
        Compute the statistical metrics.
        """
        import numpy as np

        scores = [r.score for r in results]
        latencies = [r.latency for r in results]

        return {
            # Central tendency
            "mean_score": np.mean(scores),
            "median_score": np.median(scores),

            # Dispersion
            "std_score": np.std(scores),
            "iqr_score": np.percentile(scores, 75) - np.percentile(scores, 25),

            # Range
            "min_score": np.min(scores),
            "max_score": np.max(scores),

            # Confidence interval
            "ci_95": compute_confidence_interval(scores, 0.95),

            # Timing statistics
            "mean_latency": np.mean(latencies),
            "p95_latency": np.percentile(latencies, 95),
        }

    @staticmethod
    def compare_models(
        results_a: list[TaskResult],
        results_b: list[TaskResult],
    ) -> dict:
        """
        Comparative analysis of two models.

        Uses a statistical test to decide whether the difference is significant.
        """
        from scipy import stats

        scores_a = [r.score for r in results_a]
        scores_b = [r.score for r in results_b]

        # t-test
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

### 5.2 Error Analysis

```python
class ErrorAnalyzer:
    """Error analyzer."""

    @staticmethod
    def categorize_errors(results: list[TaskResult]) -> dict:
        """
        Tally the errors by category.
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
        """Classify a single error."""
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

### 5.3 Report Generation

```python
class ReportGenerator:
    """Report generator."""

    @staticmethod
    def generate_html_report(
        results: list[TaskResult],
        model_name: str,
        task_name: str,
    ) -> str:
        """
        Generate the HTML evaluation report.
        """

        stats = ResultAnalyzer.compute_statistics(results)

        html = f"""
        <html>
        <head><title>Evaluation Report: {model_name}</title></head>
        <body>
            <h1>Optis Benchmark Report</h1>
            <h2>Model: {model_name}</h2>
            <h2>Task: {task_name}</h2>

            <h3>Summary Statistics</h3>
            <ul>
                <li>Total Tasks: {len(results)}</li>
                <li>Success Rate: {stats['success_rate']:.1%}</li>
                <li>Mean Score: {stats['mean_score']:.3f}</li>
                <li>Std Dev: {stats['std_score']:.3f}</li>
            </ul>

            <!-- more content... -->
        </body>
        </html>
        """

        return html
```

---

## 📚 References

1. AgentBench: Evaluating LLMs as Agents - Liu et al., 2023
2. HELM: Holistic Evaluation of Language Models - Liang et al., 2022
3. BigBench: Beyond the Imitation Game - Srivastava et al., 2022
4. MMLU: Measuring Massive Multitask Language Understanding - Hendrycks et al., 2021

---

*Last updated: 2024-XX-XX*
