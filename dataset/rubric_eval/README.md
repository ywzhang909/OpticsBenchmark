# OptiS Benchmark — Rubric 评估测试数据集

本目录包含 LLM-as-Judge Rubric 评估系统的测试数据，用于验证基于 Rubric 的光学领域评估系统。

---

## 目录结构

```
dataset/rubric_eval/
├── dataset_json/
│   ├── dataset_v1.json           # 测试用例数据集（7个用例）
│   ├── gold_answer_v1.json       # 专家标注的金标准答案
│   └── metadata.json             # 数据集元信息
├── rubrics/
│   ├── optical_experiment_review.yaml      # S1: 实验方案评审 Rubric (18条)
│   ├── optical_paper_review.yaml           # S2: 论文评审 Rubric (16条)
│   ├── optical_literature_understanding.yaml # S3: 文献理解 Rubric (14条)
│   ├── optical_system_design.yaml          # S4: 系统设计评估 Rubric (15条)
│   └── optical_agent_tool_use.yaml         # S5: Agent工具使用 Rubric (5条)
└── README.md                     # 本文件
```

---

## 数据集概览

| 项目 | 说明 |
|------|------|
| **数据集版本** | v1.0 |
| **创建日期** | 2026-07-27 |
| **测试用例数** | 7 |
| **评估场景** | S1-S5（5个光学领域场景） |
| **Rubric标准总数** | 68条（18+16+14+15+5） |
| **领域覆盖** | 自适应光学、激光物理、光学测量、超表面、光纤光学 |

---

## 评估场景

### S1: 光学实验方案评审 (Optical Experiment Design Review)

**场景描述：** 评估光学实验方案的物理一致性、可重复性、变量控制、创新贡献和实验设计完整性。

**Rubric标准：** 18条（10条原子级 + 8条分析级）

**测试用例：**
| ID | 标题 | 难度 | 领域 |
|----|------|------|------|
| S1_001 | 自适应光学系统波前校正实验方案 | Easy | 自适应光学 |
| S1_002 | 高功率激光相干合成实验方案 | Medium | 激光物理 |
| S1_003 | 超精密光学表面测量实验方案 | Hard | 光学测量 |

**关键检查项：**
- 能量守恒验证（C01, fail_hard）
- Nyquist采样验证（C03）
- 对照组设置（C08）
- 激光安全防护（C16）

---

### S2: 光学论文自动评审 (Optical Paper Review)

**场景描述：** 评估光学论文的技术新颖性、方法严谨性、实验充分性、理论深度和写作质量。

**Rubric标准：** 16条（6条原子级 + 10条分析级）

**测试用例：**
| ID | 标题 | 难度 | 领域 |
|----|------|------|------|
| S2_001 | 基于超表面的高效光束偏转器设计 | Easy | 超表面 |
| S2_002 | 少模光纤中模式耦合的实验研究 | Medium | 光纤光学 |

**关键检查项：**
- 方法描述可复现性（P03, fail_hard）
- 公式正确性（P04, fail_hard）
- 对照组设置（P07）
- 误差分析（P09）

---

### S3: 光学文献综合理解评估 (Optical Literature Understanding)

**场景描述：** 评估对光学领域文献的综合理解能力——事实准确性、技术深度、引用可追溯性、跨文献综合能力。

**Rubric标准：** 14条（7条原子级 + 7条分析级）

**测试用例：**
| ID | 标题 | 难度 | 领域 |
|----|------|------|------|
| S3_001 | 深度学习在波前传感中的应用综述 | Easy | 自适应光学 |

**关键检查项：**
- 事实准确性（L01, fail_hard）
- 无幻觉（L03, fail_hard）
- 跨文献综合（L08）
- 批判性思维（L11）

---

### S4: 光学系统设计参数评估 (Optical System Design Evaluation)

**场景描述：** 评估光学系统设计参数的合理性——参数范围、约束满足、性能预期、误差分析。

**Rubric标准：** 15条（9条原子级 + 6条分析级）

**测试用例：**
| ID | 标题 | 难度 | 领域 |
|----|------|------|------|
| S4_001 | 高功率激光系统光束质量评估 | Easy | 激光系统 |

**关键检查项：**
- 波长范围（D01）
- 功率处理（D02, fail_hard）
- 衍射极限（D05）
- 误差传播分析（D14）

---

### S5: 光学领域Agent工具使用评估 (Optical Agent Tool Use)

**场景描述：** 评估Agent在执行光学任务时的工具调用正确性、参数准确性和推理连贯性。

**Rubric标准：** 5条（2条原子级 + 3条分析级）

**测试用例：**
| ID | 标题 | 难度 | 领域 |
|----|------|------|------|
| S5_001 | 自适应光学系统仿真Agent交互评估 | Medium | 自适应光学 |

**关键检查项：**
- 工具选择正确性（T01, fail_hard）
- 工具参数正确性（T02）
- 结果解读准确性（T03）
- 工具链连贯性（T04）

---

## 数据格式

### 测试用例 (dataset_v1.json)

```json
{
  "id": "S1_001",
  "scenario": "S1",
  "scenario_name": "optical_experiment_design_review",
  "difficulty": "easy",
  "domain": "adaptive_optics",
  "title": "自适应光学系统波前校正实验方案",
  "content": "实验目标：...",
  "metadata": {
    "author": "测试用户",
    "date": "2026-07-27",
    "source": "synthetic",
    "word_count": 450,
    "language": "zh-CN"
  }
}
```

### 金标准答案 (gold_answer_v1.json)

```json
{
  "id": "S1_001",
  "scenario": "S1",
  "title": "自适应光学系统波前校正实验方案",
  "expert_annotations": {
    "C01_energy_conservation": {
      "verdict": "MET",
      "confidence": "HIGH",
      "evidence": "实验方案明确列出能量预算...",
      "line_ref": "能量预算部分"
    },
    "C04_approximation_validity": {
      "score": 4,
      "confidence": "HIGH",
      "evidence": "Kolmogorov湍流模型是标准模型..."
    }
  },
  "overall_score": 82,
  "pass_threshold": 80,
  "passed": true,
  "key_findings": [...]
}
```

### Rubric标准 (rubrics/*.yaml)

```yaml
rubric_v1:
  meta:
    task: "optical_experiment_design_review"
    version: "1.0"
    criteria_count: 18

  criteria:
    - id: C01_energy_conservation
      dimension: "物理一致性"
      type: binary
      question: "实验方案是否明确考虑了能量守恒..."
      evidence_types: [FORMULA, LOGIC, QUOTE]
      weight: 1.5
      fail_hard: true
```

---

## 使用方法

### 1. 加载测试数据

```python
import json

# 加载测试用例
with open('dataset/rubric_eval/dataset_json/dataset_v1.json') as f:
    test_cases = json.load(f)

# 加载金标准答案
with open('dataset/rubric_eval/dataset_json/gold_answer_v1.json') as f:
    gold_answers = json.load(f)

# 按场景筛选
s1_cases = [case for case in test_cases if case['scenario'] == 'S1']
```

### 2. 加载Rubric标准

```python
import yaml

# 加载S1场景的Rubric
with open('dataset/rubric_eval/rubrics/optical_experiment_review.yaml') as f:
    rubric = yaml.safe_load(f)

# 提取标准列表
criteria = rubric['rubric_v1']['criteria']
binary_criteria = [c for c in criteria if c['type'] == 'binary']
ordinal_criteria = [c for c in criteria if c['type'] == 'ordinal']
```

### 3. 运行评估流水线

```python
from rubric_eval.pipeline import FiveStagePipeline

# 初始化流水线
pipeline = FiveStagePipeline(
    rubric=rubric,
    deterministic_checker=deterministic_checker,
    judge_panel=judge_panel,
    calibrator=calibrator
)

# 运行评估
for case in test_cases:
    result = pipeline.evaluate(case['content'])
    print(f"Case {case['id']}: {result.overall_score}/100")
```

### 4. 验证评估结果

```python
# 对比金标准答案
for case_id, gold in gold_answers.items():
    predicted = results[case_id]
    
    # 计算Binary标准准确率
    binary_accuracy = calculate_binary_accuracy(
        predicted['binary_verdicts'],
        gold['binary_gold']
    )
    
    # 计算Ordinal标准准确率（允许±1偏差）
    ordinal_accuracy = calculate_ordinal_accuracy(
        predicted['ordinal_scores'],
        gold['ordinal_gold'],
        tolerance=1
    )
    
    print(f"{case_id}: Binary={binary_accuracy:.1%}, Ordinal={ordinal_accuracy:.1%}")
```

---

## 难度分层

| 难度 | 特征 | 用例数量 | 占比 |
|------|------|---------|------|
| **Easy** | 明确的事实性标准，答案确定 | 4 | 57% |
| **Medium** | 需要领域知识判断 | 2 | 29% |
| **Hard** | 边界案例，专家可能分歧 | 1 | 14% |

---

## 领域分布

| 领域 | 用例数量 | 占比 |
|------|---------|------|
| 自适应光学 | 3 | 43% |
| 激光物理 | 1 | 14% |
| 光学测量 | 1 | 14% |
| 超表面 | 1 | 14% |
| 光纤光学 | 1 | 14% |

---

## 评估指标

| 指标 | 阈值 | 说明 |
|------|------|------|
| Binary标准准确率 | >85% | MET/UNMET判断准确率 |
| Ordinal标准准确率 | >75% | 1-5评分准确率（允许±1偏差） |
| Cohen's κ | >0.67 | 与人类评估的一致性（substantial） |
| ECE | <0.05 | 概率校准误差 |
| Brier Score | <0.10 | 预测误差 |

---

## 关联配置

| 配置文件 | 路径 | 说明 |
|----------|------|------|
| 实验方案评审Rubric | `rubrics/optical_experiment_review.yaml` | S1场景18条标准 |
| 论文评审Rubric | `rubrics/optical_paper_review.yaml` | S2场景16条标准 |
| 文献理解Rubric | `rubrics/optical_literature_understanding.yaml` | S3场景14条标准 |
| 系统设计评估Rubric | `rubrics/optical_system_design.yaml` | S4场景15条标准 |
| Agent工具使用Rubric | `rubrics/optical_agent_tool_use.yaml` | S5场景5条标准 |

---

## 参考文献

- Chen et al. (2606.08625) — Rubric Survey
- RuVerBench — 评判忠实度评估
- RubricBench — 人类-模型Rubric差距分析
- Beyond Rating — 五维评估框架
- ReviewGuard — 评审质量控制
- AgentCompass (2607.13705) — 评估基础设施解耦
- Prometheus 2 (2405.01579) — 开源LLM judge

---

## 许可证

数据集采用 CC BY-SA 4.0 许可证。
