# LLM-as-Judge 实验方案：基于 Rubric 的光学领域评估系统

> **版本：** v16.0 — 2026年7月
> **定位：** 以光学领域为背景的可操作实验方案，覆盖 Rubric 设计、评估执行、可靠性验证和校准全链路
> **核心参考：** Chen et al. (2606.08625) 综述 + RuVerBench/RubricBench/Beyond Rating/ReviewGuard 等最新研究
> **前置依赖：** agent-evaluation-methodology skill, rubric-eval 实现框架

---

## 1. 实验目标与背景

### 1.1 问题定义

光学领域的实验方案生成、论文评审和技术报告评估具有高度专业性：涉及物理量纲（Strehl 比、RMS 波前误差）、实验约束（采样率、噪声底、相干时间）和领域惯例（对比实验设计、基线公平性）。通用 LLM-as-Judge 系统在此类任务中面临三重挑战：

| 挑战 | 具体表现 | 影响 |
|------|----------|------|
| **领域知识鸿沟** | LLM 对光学实验设计的专业判断与领域专家存在认知错位 | 评分与人类评估不一致 |
| **Rubric 执行漂移** | 同一 Rubric 在不同输入/提示下产生不同解释（RULERS, 2601.08654） | 结果不可复现 |
| **伪推理检测缺失** | 模型通过逻辑有缺陷的路径到达正确答案（Yuan et al., 2026） | 过程级评估盲区 |

### 1.2 实验目标

本实验方案旨在构建一个**端到端的 LLM-as-Judge 评估系统**，针对光学领域的具体评价场景，实现：

1. **高质量 Rubric 构建** — 专家编写 + 领域适配的原子/分析混合 Rubric
2. **结构化评估执行** — 五阶段流水线（锁定→上下文→执行→校准→完整性检查）
3. **可靠性验证** — 对标 RuVerBench 方法，量化评判忠实度
4. **生产级部署方案** — 从原型到生产的渐进式路径

### 1.3 评价场景定义（光学领域）

以下场景作为本实验方案的评估对象，对应原报告中"维度二及其以后"的内容：

| 编号 | 场景 | 评估类型 | 核心 Rubric 维度 |
|------|------|----------|-----------------|
| S1 | 光学实验方案生成与评审 | 过程级 + 输出级 | 物理一致性、可重复性、变量控制 |
| S2 | 光学论文自动评审 | 输出级 | 技术新颖性、方法严谨性、实验充分性 |
| S3 | 光学领域文献综合理解评估 | 输出级 | 事实准确性、技术深度、引用可追溯性 |
| S4 | 光学系统设计参数评估 | 输出级 + 过程级 | 参数合理性、约束满足、性能预期 |

---

## 2. Rubric 的设计与构建

### 2.1 Rubric 分类选择策略

基于 Chen et al. 的结构/内容分类矩阵，针对光学领域选择最优组合：

```
                    整体 (Holistic)     分析 (Analytic)        原子 (Atomic)
输出级 (Output)    ~~MT-Bench~~        ✅ 光学论文评审        ✅ 实验方案检查清单
过程级 (Process)   ~~MathShepherd~~    ✅ 推理步骤评估        ✅ 物理量纲验证
```

**选择原则：**

| 原则 | 说明 | 光学场景应用 |
|------|------|-------------|
| **原子性优先** | 二元命题（Yes/No）最可靠，减少主观性 | 物理量纲检查、公式验证、采样率是否满足 Nyquist |
| **分析维度补充** | 对需要程度的维度使用 3-5 级 Likert | 创新性（1-5）、实验充分性（1-5） |
| **混合策略** | 原子 + 分析混合，最大化可靠性 | 实验方案 = 原子检查清单 + 分析评分 |
| **排除连续值** | LLM 在未定界数值尺度上校准差 | ❌ 不使用 0-100 连续评分 |

### 2.2 光学领域 Rubric 构建流程

```
Step 1: 领域专家定义维度                Step 2: 原子化拆解                    Step 3: 可操作性验证
┌──────────────────────────┐       ┌──────────────────────────┐        ┌──────────────────────────┐
│ 1. 确定评估维度           │       │ 2. 每个维度拆分为         │        │ 3. 每条标准必须满足:     │
│    (4-8 个正交维度)       │ ───▶  │    不可再分的原子命题     │ ───▶  │    · 可独立验证           │
│                           │       │    (二元或 3-5 级)        │        │    · 有明确证据类型       │
│ 例: 实验方案评审          │       │                           │        │    · 无歧义性             │
│ · 物理一致性             │       │ 例: "物理一致性"维度       │        │                           │
│ · 实验可重复性           │       │   → 原子:                 │        │ 不合格:                  │
│ · 变量控制               │       │   · 能量守恒是否满足?     │        │   "物理分析是否合理"      │
│ · 创新贡献               │       │   · 采样是否满足 Nyquist? │        │ (过于模糊)                │
│ · 实验设计完整性         │       │   → 分析:                 │        │                           │
│                          │       │   · 近似假设合理程度(1-5) │        │ 合格:                     │
│                          │       │                           │        │ "能量守恒方程是否明确     │
│                          │       │                          │         │   列出且量纲一致?"        │
└──────────────────────────┘       └──────────────────────────┘        └──────────────────────────┘
```

### 2.3 光学领域 Rubric 示例（实验方案评审场景 S1）

```yaml
rubric_v1:
  meta:
    task: "optical_experiment_design_review"
    version: "1.0"
    hash: "sha256:..."
    criteria_count: 18

  criteria:
    # === 物理一致性 (原子 + 分析混合) ===
    - id: C01_energy_conservation
      dimension: "物理一致性"
      type: binary
      question: "实验方案是否明确考虑了能量守恒（输入功率 = 输出 + 损耗）？"
      evidence_types: [FORMULA, LOGIC, QUOTE]
      weight: 1.5
      fail_hard: true  # 硬性否决项

    - id: C02_diffraction_limit
      dimension: "物理一致性"
      type: binary
      question: "衍射极限计算是否与所用波长和孔径一致？"
      evidence_types: [FORMULA, DATA]
      weight: 1.2

    - id: C03_sampling_nyquist
      dimension: "物理一致性"
      type: binary
      question: "探测器采样是否满足 Nyquist 准则（至少 2 像素/空间频率）？"
      evidence_types: [DATA, FORMULA]
      weight: 1.2

    - id: C04_approximation_validity
      dimension: "物理一致性"
      type: ordinal
      scale: [1, 2, 3, 4, 5]
      question: "理论模型中的近似假设是否合理（1=完全不合理, 5=完全合理）？"
      evidence_types: [LOGIC, QUOTE]
      weight: 1.0

    # === 实验可重复性 (原子) ===
    - id: C05_equipment_specification
      dimension: "实验可重复性"
      type: binary
      question: "所有关键设备是否提供了完整规格参数（型号、关键指标）？"
      evidence_types: [DATA]
      weight: 1.0

    - id: C06_parameter_report
      dimension: "实验可重复性"
      type: binary
      question: "实验参数是否完整报告（波长、功率、曝光时间、环境条件等）？"
      evidence_types: [DATA]
      weight: 1.0

    - id: C07_measurement_conditions
      dimension: "实验可重复性"
      type: binary
      question: "测量条件和校准方法是否描述清晰？"
      evidence_types: [QUOTE, DATA]
      weight: 0.8

    # === 变量控制 (原子 + 分析) ===
    - id: C08_control_group
      dimension: "变量控制"
      type: binary
      question: "是否设置了合理的对照组或参考实验？"
      evidence_types: [LOGIC]
      weight: 1.0

    - id: C09_confounding_factors
      dimension: "变量控制"
      type: ordinal
      scale: [1, 2, 3, 4, 5]
      question: "混杂变量的控制程度如何（1=几乎无控制, 5=全面控制）？"
      evidence_types: [LOGIC, DATA]
      weight: 1.0

    - id: C10_uncertainty_quantification
      dimension: "变量控制"
      type: binary
      question: "是否提供了误差分析和不确定性量化？"
      evidence_types: [DATA, FORMULA]
      weight: 1.0

    # === 创新贡献 (分析) ===
    - id: C11_novelty
      dimension: "创新贡献"
      type: ordinal
      scale: [1, 2, 3, 4, 5]
      question: "方法/设计的新颖性程度（1=完全已有, 5=高度创新）？"
      evidence_types: [LOGIC, QUOTE]
      weight: 1.0

    - id: C12_contribution_clarity
      dimension: "创新贡献"
      type: ordinal
      scale: [1, 2, 3, 4, 5]
      question: "对已有工作的超越是否明确陈述且有证据支持？"
      evidence_types: [QUOTE, LOGIC]
      weight: 0.8

    # === 实验设计完整性 (原子) ===
    - id: C13_baseline_comparison
      dimension: "实验设计完整性"
      type: binary
      question: "是否包含与基线方法/现有设计的对比？"
      evidence_types: [DATA, LOGIC]
      weight: 0.8

    - id: C14_failure_analysis
      dimension: "实验设计完整性"
      type: binary
      question: "是否讨论了潜在失败模式或局限性？"
      evidence_types: [LOGIC, QUOTE]
      weight: 0.6

    - id: C15_scalability
      dimension: "实验设计完整性"
      type: ordinal
      scale: [1, 2, 3, 4, 5]
      question: "实验的可扩展性和推广性如何？"
      evidence_types: [LOGIC]
      weight: 0.6

    - id: C16_safety_consideration
      dimension: "实验设计完整性"
      type: binary
      question: "是否考虑了激光安全标准和防护措施？"
      evidence_types: [QUOTE]
      weight: 0.8

    # === 过程级评估 (针对推理链) ===
    - id: C17_reasoning_chain
      dimension: "过程级-推理链"
      type: ordinal
      scale: [1, 2, 3, 4, 5]
      question: "从实验目标到方案设计的推理链是否连贯、无逻辑跳跃？"
      evidence_types: [LOGIC]
      weight: 1.0

    - id: C18_parameter_derivation
      dimension: "过程级-参数推导"
      type: binary
      question: "关键参数的选择是否有推导过程或引用依据？"
      evidence_types: [FORMULA, QUOTE, DATA]
      weight: 1.0
```

### 2.4 Rubric 锁定制（Rubric Locking）

**问题：** Prompt 级别的 Rubric 容易被 LLM 重新解释（RULERS, 2601.08654）。

**解决方案：** 将 Rubric 编译为不可变规范。

```python
# Rubric 锁定流程
class SpecCompiler:
    """将 YAML Rubric 编译为不可变规范"""

    def compile(self, rubric_yaml: str) -> LockedRubricSpec:
        # 1. 验证：检查所有字段、类型、权重
        validated = self._validate(rubric_yaml)
        # 2. 标准化：统一格式、消除歧义
        standardized = self._standardize(validated)
        # 3. 哈希：生成 SHA-256 指纹
        spec_hash = hashlib.sha256(standardized).hexdigest()
        # 4. 锁定：返回不可变对象
        return LockedRubricSpec(
            criteria=standardized,
            hash=spec_hash,
            version=rubric_yaml["meta"]["version"]
        )
```

**实验要求：** 每次评估使用的 Rubric 必须通过锁定流程，记录 SHA-256 哈希值，确保评估可审计。

---

## 3. 评估执行流水线

### 3.1 五阶段流水线架构

```
┌───────────────────────────────────────────────────────────────────────┐
│                    光学领域 LLM-as-Judge 评估流水线                    │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Stage 1: RUBRIC LOCKING           Stage 2: CONTEXT BUILDING          │
│  ┌─────────────────────┐          ┌──────────────────────────┐       │
│  │ · YAML → SpecCompiler │         │ · Token-aware truncation │       │
│  │ · 验证 + 标准化      │         │ · 角色分离               │       │
│  │ · SHA-256 锁定       │         │   (作者/审稿人/专家)     │       │
│  │ · 不可变 LockedSpec  │         │ · 证据源映射             │       │
│  └─────────────────────┘          │ · 领域术语预处理         │       │
│          │                        └──────────────────────────┘       │
│          ▼                               │                           │
│  Stage 3: EVIDENCE-GROUNDED EXECUTION       ▼                        │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ 对每个提交 × 每个标准 × 每个评判者:                    │            │
│  │  ┌────────────────────────────────────────────────┐  │            │
│  │  │ · 结构化检查清单决策 (二元: MET/UNMET)          │  │            │
│  │  │ · 排序量表决策 (ordinal: 1-5)                  │  │            │
│  │  │ · 类型化证据提取 (QUOTE/DATA/FORMULA/LOGIC)     │  │            │
│  │  │ · 机械证据验证 (公式检查/数据验证/引用核对)     │  │            │
│  │  │ · 输出: CriterionVerdict                       │  │            │
│  │  └────────────────────────────────────────────────┘  │            │
│  └──────────────────────────────────────────────────────┘            │
│                              │                                       │
│                              ▼                                       │
│  Stage 4: CALIBRATION              Stage 5: INTEGRITY CHECK          │
│  ┌─────────────────────┐          ┌──────────────────────────┐       │
│  │ · Binary → Wasserstein│         │ · Adequacy: 充分证据     │       │
│  │ · Ordinal → Platt    │         │ · Provenance: 可追溯     │       │
│  │ · Skewed → Stratified│         │ · Safety: 无注入攻击     │       │
│  │ · 自动选择校准器      │         │ · Responsible Use: 偏倚  │       │
│  └─────────────────────┘          └──────────────────────────┘       │
│                                                                       │
│                              ▼                                       │
│                   Final EvaluationReport                              │
│                   (Calibrated Scores + Evidence + Integrity Flags)   │
└───────────────────────────────────────────────────────────────────────┘
```

### 3.2 阶段详解

#### Stage 3: 证据 grounding 执行（核心阶段）

**评判策略：每标准独立调用（Per-Criterion Independent Calls）**

| 设计选择 | 方案 | 原因 |
|----------|------|------|
| 单调用 vs 多调用 | 每个标准独立一次 LLM 调用 | 防止维度间污染（halo effect） |
| 评判者数量 | 3-5 个评判者（panel） | 多数投票降噪，收益递减在 5 之后 |
| 选项洗牌 | 对 ordinal/nominal 标准洗牌选项 | 缓解位置偏见 |
| 提示模板 | 包含 Rubric 标准 + 证据类型要求 + 领域上下文 | 结构化输出 |

**评判提示模板（光学实验方案评审）：**

````
You are an expert reviewer in optical physics. Evaluate the following
experiment design against ONE criterion. Provide a verdict with evidence.

=== RUBRIC CRITERION ===
ID: C01_energy_conservation
Type: binary (MET / UNMET)
Question: 实验方案是否明确考虑了能量守恒（输入功率 = 输出 + 损耗）？
Evidence types required: FORMULA, LOGIC, QUOTE

=== SUBMISSION ===
{experiment_design_text}

=== RESPONSE FORMAT ===
verdict: MET | UNMET
confidence: HIGH | MEDIUM | LOW
evidence:
  - type: FORMULA | LOGIC | QUOTE | DATA
    text: "<specific evidence from the submission>"
    line_ref: "<line number or section>"
    verified: true | false
reasoning: "<brief explanation linking evidence to verdict>"
````

**证据类型与光学领域映射：**

| 证据类型 | 光学领域示例 | 验证方式 |
|----------|-------------|----------|
| `FORMULA` | 公式推导、量纲分析 | 数学一致性检查 |
| `DATA` | 参数值、规格表 | 数值范围合理性 |
| `QUOTE` | 引用原文 | 子串匹配 |
| `LOGIC` | 推理链、因果关系 | 逻辑连贯性 |
| `ABSENCE` | 缺失的关键信息 | 用于幻觉/遗漏检测 |
| `FORMULA_REF` | 公式引用（如 Fresnel 方程） | 公式正确性验证 |

#### Stage 4: 校准策略

| 标准类型 | 校准器 | 最小校准样本 | 光学场景适用 |
|----------|--------|-------------|-------------|
| Binary (MET/UNMET) | Wasserstein OT | 20+ | 物理量纲检查、公式验证 |
| Ordinal (1-5) | Platt sigmoid | 30+ | 创新性、充分性评分 |
| Skewed Ordinal | StratifiedPlatt | 50+ | 偏态分布的维度 |

**校准数据构建（光学领域）：**

```
1. 选取 50+ 光学实验方案/论文
2. 邀请 3 位领域专家独立评分
3. 运行 LLM-as-Judge 得到原始评分
4. 构建 (LLM_score, human_score) 配对
5. 按标准类型分组 → 训练对应校准器
```

#### Stage 5: 完整性检查（不可跳过）

| 检查项 | 检查内容 | 光学场景示例 |
|--------|----------|-------------|
| Adequacy | 每个 verdict 有足够证据 | 物理一致性 verdict 必须有 FORMULA 证据 |
| Provenance | 分数可追溯到具体评判者+标准 | 追踪到 C01 + Judge_A + submission_v3 |
| Safety | 无提示注入或角色混淆 | 检查评判输出是否被输入中的指令干扰 |
| Responsible Use | 偏倚和公平性标志 | 不同子领域评分是否存在系统性偏差 |

---

## 4. 可靠性验证实验

### 4.1 评判忠实度评估（Rubric Faithfulness）

**参考 RuVerBench 方法 (2606.29920)，构建光学领域的忠实度基准。**

#### 4.1.1 忠实度数据集构建

```
构建流程:
1. 收集 200+ 光学实验方案/论文片段
2. 专家标注每个 Rubric 标准的正确标签 (MET/UNMET 或 1-5 分数)
3. 记录标注依据和边界案例
4. 按难度分层: 简单(60%) / 中等(30%) / 困难(10%)
```

**难度定义：**

| 难度 | 特征 | 示例 |
|------|------|------|
| 简单 | 明确的事实性标准，答案确定 | "是否提供了设备型号？" |
| 中等 | 需要领域知识判断 | "近似假设是否合理？" |
| 困难 | 边界案例，专家可能分歧 | "创新贡献程度（1-5）" |

#### 4.1.2 忠实度评估指标

| 指标 | 公式 | 阈值 |
|------|------|------|
| **准确率 (Accuracy)** | TP+TN / Total | > 85% (binary), > 75% (ordinal) |
| **精确率 (Precision)** | TP / (TP+FP) | > 85% |
| **召回率 (Recall)** | TP / (TP+FN) | > 80% |
| **Cohen's κ** | (Po - Pe) / (1 - Pe) | > 0.67 (substantial) |
| **置信区间** | 95% bootstrap CI (1000 resamples) | 宽度 < 0.10 |

#### 4.1.3 关键实验变量

| 变量 | 水平 | 预期影响 |
|------|------|----------|
| 评判者数量 | 1 / 3 / 5 | 多数投票降噪，收益递减 |
| Prompt 变体 | 标准 / 简化 / 详细 | 弱模型更敏感（RuVerBench 发现） |
| 批量 vs 逐条 | batched / individual | 准确率和效率的权衡 |
| 标准粒度 | 原子 / 分析 / 混合 | 混合策略可能最优 |

### 4.2 偏见检测实验

**参考 FairJudge (Yang et al., 2026) 和 Li et al. (2025a) 的偏见分析方法。**

#### 4.2.1 已识别的偏见类型

| 偏见类型 | 检测方法 | 缓解策略 |
|----------|----------|----------|
| 位置偏见 | 选项顺序 A/B → 分数差异 | 选项洗牌 + 按值合并 |
| 分数 ID 偏见 | 重新编号选项 → 分数变化 | 随机化标签映射 |
| 标准顺序偏见 | 改变标准排列 → 分数变化 | 随机化标准顺序 |
| 标准间隙偏见 | 标准未覆盖维度是否被忽略 | 显式"未覆盖维度"检查 |
| 标准纠缠偏见 | 维度间相关性 vs 评分独立性 | 独立调用 + 相关性分析 |

#### 4.2.2 偏见量化实验

```
实验设计:
1. 选取 100 个光学实验方案
2. 对每个方案运行 4 次评估（不同选项顺序/标准顺序）
3. 计算同一方案不同运行的分数差异
4. 统计: 平均差异、标准差、最大差异
5. 阈值: 平均差异 < 0.2 (ordinal) 或 翻转率 < 10% (binary)
```

### 4.3 人类一致性评估

**参考 Beyond Rating (2604.19502) 的五维评估框架。**

| 维度 | Beyond Rating 定义 | 光学领域适配 |
|------|-------------------|-------------|
| Content Faithfulness | 内容忠实于原文 | 光学参数/公式忠实于实验方案 |
| Argumentative Alignment | 论证方向与专家一致 | 评审意见方向（优点/缺点）与专家一致 |
| Focus Consistency | 关注点与专家一致 | 是否关注关键维度（如物理一致性） |
| Question Constructiveness | 问题的建设性 | 追问是否帮助改进实验设计 |
| AI-Likelihood | 是否像 AI 生成 | 评审意见的自然程度 |

**Max-Recall 策略（处理专家分歧）：**

```
当多位专家对同一标准评分不一致时:
1. 计算专家标注的并集（max-recall）
2. LLM 评判与并集比较 → 更宽容的评估
3. 同时记录与每个专家个体的分歧 → 分析 LLM 更接近哪位专家
```

---

## 5. 实验方案：端到端执行计划

### 5.1 实验阶段总览

```
Phase 1: 基础建设 (Week 1-2)           Phase 2: 评估执行 (Week 3-4)
┌──────────────────────────┐          ┌──────────────────────────┐
│ · Rubric v1 构建         │          │ · 运行五阶段流水线        │
│ · 数据集准备 (200+ 样本)  │          │ · 3/5 评判者 panel        │
│ · 校准集构建 (50+ 配对)   │ ───▶    │ · 偏见检测实验            │
│ · 基础设施部署            │          │ · 忠实度评估              │
│                          │          │ · 人类一致性评估          │
│ 交付物: LockedRubricSpec │          │ 交付物: Raw evaluation    │
│ + CalibrationDataset     │          │        dataset + bias report│
└──────────────────────────┘          └──────────────────────────┘

Phase 3: 分析与迭代 (Week 5-6)         Phase 4: 优化与部署 (Week 7-8)
┌──────────────────────────┐          ┌──────────────────────────┐
│ · 校准模型训练           │          │ · Rubric v2 修订         │
│ · 性能分析报告           │ ───▶    │ · 最优参数组合确定        │
│ · Rubric 修订 (基于发现) │          │ · 自动化 Pipeline 部署    │
│ · 边界案例分析           │          │ · 文档与工具链            │
│                          │          │ 交付物: Production-ready  │
│ 交付物: CalibrationModel │          │        system + final report│
│ + AnalysisReport         │          │                          │
└──────────────────────────┘          └──────────────────────────┘
```

### 5.2 Phase 1: 基础建设（详细步骤）

#### Step 1.1: Rubric 构建

```python
# 使用 rubric-eval 框架
from rubric_eval.core.rubric import RubricBuilder

builder = RubricBuilder()
# 1. 从 YAML 加载
rubric = builder.from_yaml("rubrics/optical_experiment_review.yaml")
# 2. 编译锁定
locked_spec = builder.compile(rubric)
# 3. 验证
assert locked_spec.is_valid()
# 4. 记录哈希
print(f"Rubric hash: {locked_spec.hash}")
```

#### Step 1.2: 数据集准备

| 数据类型 | 数量 | 来源 |
|----------|------|------|
| 光学实验方案 | 150 | 已发表论文 + 内部文档 |
| 光学论文 | 50 | Optics Express, JOSA, Applied Optics |
| 专家标注 | 50 (配对) | 3 位领域专家 × 50 样本 |

**数据分层策略：**

```
按难度:
· 简单 (60%): 事实性检查可判定
· 中等 (30%): 需要领域知识
· 困难 (10%): 边界案例

按领域:
· 自适应光学 (30%)
· 激光物理 (25%)
· 光学测量 (25%)
· 光子学/纳米光学 (20%)
```

#### Step 1.3: 基础设施部署

```bash
# 初始化评估项目
rubric-eval init --output ./optical-eval/ --domain physics

# 配置评判者面板
# config.yaml
judge_panel:
  judges:
    - model: gpt-4o
      role: primary_judge
    - model: claude-sonnet-4
      role: secondary_judge
    - model: qwen-max
      role: tertiary_judge
  strategy: majority_vote
  option_shuffling: true
```

### 5.3 Phase 2: 评估执行（详细步骤）

#### Step 2.1: 运行评估流水线

```python
from rubric_eval.pipeline import FiveStagePipeline
from rubric_eval.judges import JudgePanel
from rubric_eval.calibrators import CalibrationRunner

# 1. 构建 pipeline
pipeline = FiveStagePipeline(
    locked_spec=locked_spec,
    judge_panel=JudgePanel(models=["gpt-4o", "claude-sonnet-4", "qwen-max"]),
    calibrator=CalibrationRunner(calibration_data=calibration_dataset),
    integrity_checks=["adequacy", "provenance", "safety", "responsible_use"]
)

# 2. 运行评估
reports = pipeline.evaluate(submissions=test_dataset)

# 3. 导出结果
for report in reports:
    report.export_json(f"results/{report.submission_id}.json")
```

#### Step 2.2: 偏见检测实验

```python
# 同一方案，不同选项顺序 × 4 次
from rubric_eval.reliability import BiasAnalyzer

analyzer = BiasAnalyzer(pipeline, test_dataset)

# 位置偏见
position_bias = analyzer.detect_position_bias(n_permutations=8)
print(f"Position bias: mean_diff={position_bias.mean_diff:.3f}")

# 标准顺序偏见
criterion_order_bias = analyzer.detect_criterion_order_bias(n_permutations=8)
print(f"Criterion order bias: flip_rate={criterion_order_bias.flip_rate:.3f}")

# 标准纠缠检测
entanglement = analyzer.detect_criterion_entanglement()
print(f"Criterion entanglement: avg_correlation={entanglement.avg_correlation:.3f}")
```

#### Step 2.3: 忠实度评估

```python
from rubric_eval.reliability import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(
    ground_truth=expert_annotations,
    predictions=llm_reports
)

# 准确率
accuracy = evaluator.accuracy()
print(f"Overall accuracy: {accuracy:.3f}")

# 按标准类型拆分
for criterion in locked_spec.criteria:
    acc = evaluator.accuracy_by_criterion(criterion.id)
    print(f"  {criterion.id} ({criterion.type}): {acc:.3f}")

# Cohen's kappa
kappa = evaluator.cohens_kappa()
print(f"Cohen's κ: {kappa:.3f} (threshold: 0.67)")

# 置信区间
ci = evaluator.bootstrap_ci(n_resamples=1000)
print(f"95% CI: [{ci.lower:.3f}, {ci.upper:.3f}]")
```

### 5.4 Phase 3: 分析与迭代

#### Step 3.1: 校准模型训练

```python
from rubric_eval.calibrators import CalibrationRunner

runner = CalibrationRunner(calibration_data=calibration_dataset)

# 自动选择并训练校准器
calibrators = runner.train(locked_spec)

# 评估校准效果
for calibrator in calibrators:
    report = calibrator.quality_report()
    print(f"Criterion {calibrator.criterion_id}:")
    print(f"  ECE: {report.ece:.3f}")      # Expected Calibration Error
    print(f"  Brier: {report.brier_score:.3f}")  # Brier Score
    print(f"  Method: {report.method}")
```

**校准质量阈值：**

| 指标 | 合格阈值 | 说明 |
|------|----------|------|
| ECE (Expected Calibration Error) | < 0.05 | 概率校准误差 |
| Brier Score | < 0.10 | 预测误差 |
| Calibration R² | > 0.70 | 校准后与人类的一致性 |

#### Step 3.2: 性能分析报告

```
报告结构:
1. 总体性能
   - 准确率 (binary / ordinal 分别报告)
   - Cohen's κ
   - 95% 置信区间

2. 按标准类型分析
   - 每个标准的准确率
   - 最难/最易标准
   - 标准间相关性

3. 偏见分析
   - 位置偏见量化
   - 标准顺序偏见
   - 标准纠缠程度

4. 人类一致性
   - Beyond Rating 五维评分
   - Max-Recall 召回率
   - 与每位专家的分歧分析

5. 边界案例
   - 低置信度评判 (confidence=LOW)
   - 评判者间分歧案例
   - 校准偏差最大的样本

6. 改进建议
   - Rubric 修订建议
   - 评判策略调整
   - 数据增强方向
```

### 5.5 Phase 4: 优化与部署

#### Step 4.1: Rubric v2 修订

基于 Phase 3 的发现，对 Rubric 进行修订：

| 修订类型 | 触发条件 | 操作 |
|----------|----------|------|
| 标准模糊 | 准确率 < 70% | 重述标准，增加可操作性 |
| 标准冗余 | 两个标准相关性 > 0.80 | 合并或移除 |
| 标准遗漏 | 人类专家关注但 Rubric 未覆盖 | 新增标准 |
| 权重不当 | 高权重标准与人类重要性不匹配 | 调整权重 |

#### Step 4.2: 最优参数确定

```
消融实验矩阵:
· 评判者数量: 1 vs 3 vs 5
· 提示变体: 标准 vs 详细 vs CoT
· 批量策略: 逐条 vs 批量(5) vs 批量(10)
· 校准策略: 无 vs Platt vs Wasserstein

指标: 准确率、成本、延迟
目标: Pareto 最优（准确率高、成本低、延迟可接受）
```

#### Step 4.3: 生产部署

```yaml
# 生产配置
production:
  judge_panel:
    strategy: majority_vote  # 3-judge panel
    models: [gpt-4o, claude-sonnet-4, qwen-max]
    option_shuffling: true
    temperature: 0.0
  pipeline:
    stage_1_lock: true
    stage_2_token_budget: 8192
    stage_3_per_criterion_call: true
    stage_4_auto_calibrate: true
    stage_5_integrity_checks: ["adequacy", "provenance", "safety"]
  monitoring:
    drift_detection: true       # 评分分布漂移
    human_review_threshold: 0.3 # 低置信度自动标记人工审核
    feedback_loop: true         # 人工审核结果更新校准集
```

---

## 6. 预期结果与评估标准

### 6.1 成功标准

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Binary 标准准确率 | > 85% | 对标 RuVerBench 最优模型 |
| Ordinal 标准准确率 | > 75% | 允许 ±1 级偏差 |
| Cohen's κ (vs 人类) | > 0.67 | substantial agreement |
| 位置偏见 | 翻转率 < 10% | 选项顺序影响 |
| 标准纠缠 | 相关性 < 0.50 | 维度间独立性 |
| 校准 ECE | < 0.05 | 概率校准质量 |
| 完整性通过率 | > 95% | Stage 5 检查 |

### 6.2 预期发现

基于已有研究，我们预期：

1. **原子标准显著优于分析标准** — 二元检查清单的准确率将比 1-5 量表高 10-15%
2. **3 评判者 panel 是成本/性能的甜点** — RuVerBench 显示多数投票收益递减
3. **校准是必要的** — 原始 LLM 评分分布与人类分布显著不同
4. **某些维度存在固有噪声** — 创新性和影响力评估的专家一致性本身就低
5. **领域专家 Rubric vs 通用 Rubric 差距显著** — RubricBench 显示 27% 的准确性差距

---

## 7. 风险与缓解

### 7.1 技术风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| LLM 评判者表现低于预期 | 中 | 高 | 多模型面板 + 回退到人工评审 |
| 校准数据不足 | 中 | 中 | 最小样本检查，回退到 identity 校准 |
| Prompt 注入攻击 | 低 | 高 | Stage 5 Safety 检查 + system prompt 隔离 |
| 领域知识不足 | 中 | 中 | 领域术语预处理 + 参考文档注入 |

### 7.2 方法论风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| Rubric 定义存在盲区 | 高 | 高 | 多专家评审 + 边界案例分析 |
| 专家标注一致性差 | 中 | 中 | Max-Recall 策略 + 标注质量评估 |
| 标准间存在隐含依赖 | 中 | 中 | 标准纠缠检测 + GEAR 加权调整 |
| 数据集代表性不足 | 中 | 高 | 领域分层采样 + 持续扩充 |

---

## 8. 关键参考文献

### 8.1 核心论文

| 论文 | arXiv ID | 核心贡献 | 本方案中的应用 |
|------|----------|----------|---------------|
| Chen et al. — Rubric 综述 | 2606.08625 | 统一框架: Rubric 从评估到训练到内化 | Rubric 分类体系 + 演进阶段 |
| Peng et al. — RuVerBench | 2606.29920 | LLM 评判者忠实度基准 (2458 实例) | 忠实度评估方法 + 指标 |
| Zhang et al. — RubricBench | 2603.01562 | 人类 vs 模型 Rubric 差距 (27%) | Rubric 构建质量意识 |
| Li et al. — Beyond Rating | 2604.19502 | 五维评估 + Max-Recall 策略 | 人类一致性评估框架 |
| Rasool et al. — ReviewGuard | 2606.24892 | 引用对齐的两阶段框架 | 长期影响力评估 |

### 8.2 方法论文

| 论文 | 核心贡献 | 本方案中的应用 |
|------|----------|---------------|
| RULERS (2601.08654) | 三阶段流水线: 锁定→证据→校准 | Stage 1 + Stage 3 + Stage 4 |
| FairJudge (Yang et al., 2026) | 多级偏见缓解: SFT + DPO + GRPO | 偏见检测与缓解 |
| AutoRubric (2603.00077) | 混合标准类型 + 面板评判 | Rubric 类型选择 + 评判面板 |
| GEAR (Lv et al., 2026) | 标准间干扰建模 | 标准纠缠检测 |
| Tournament-GRPO (Yang et al., 2026) | 绝对评分的结构性失败 | 何时使用 pairwise vs rubric |
| iRULER | Rubric-of-Rubrics 元评估 | Rubric 迭代改进 |

### 8.3 开源工具

| 工具 | 用途 | 本方案中的应用 |
|------|------|---------------|
| rubric-eval (本项目) | 五阶段流水线实现 | 核心评估框架 |
| Prometheus | 微调评判者模型 | 备选评判模型 |
| LlamaArena Judge | 配对比较框架 | 配对比较实验 |
| RAGAS | RAG 评估框架 | 文献检索质量评估 |
| OpenCompass/LLMRuler | 自校准 LLM 评判 | 校准方法参考 |

---

## 9. 实施检查清单

### Phase 1 完成标准

- [ ] Rubric YAML 文件编写并通过专家审核
- [ ] LockedRubricSpec 生成，SHA-256 哈希记录
- [ ] 200+ 光学实验方案/论文数据集准备完成
- [ ] 50+ 专家标注配对完成 (3 位专家)
- [ ] rubric-eval 项目初始化，评判者面板配置
- [ ] 环境依赖验证 (scipy, litellm 等)

### Phase 2 完成标准

- [ ] 所有样本通过五阶段流水线评估
- [ ] 偏见检测实验完成，结果记录
- [ ] 忠实度评估完成 (准确率 + κ + CI)
- [ ] Beyond Rating 五维人类一致性评估完成
- [ ] 原始结果导出 (JSON/YAML)

### Phase 3 完成标准

- [ ] 校准模型训练完成，ECE < 0.05
- [ ] 性能分析报告编写完成
- [ ] Rubric v2 修订完成，通过专家审核
- [ ] 边界案例分析文档完成

### Phase 4 完成标准

- [ ] 消融实验完成，最优参数确定
- [ ] 生产配置文件就绪
- [ ] 监控和反馈循环部署
- [ ] 最终实验报告完成

---

## 10. 与已有工作的关联

### 10.1 与 agent-evaluation-methodology skill 的关系

本实验方案是 agent-evaluation-methodology skill 的**具体化和扩展**：

| agent-evaluation-methodology | 本方案 |
|------------------------------|--------|
| 通用 10 门评估检查 | 光学领域 18 条具体 Rubric 标准 |
| 15 个问题模式 | 扩展为可操作的证据 grounding 流程 |
| 4 位审稿人角色 | 扩展为 LLM-as-Judge 评判面板 |
| 校准策略概述 | 详细校准器选择、训练和评估流程 |
| 通用方法论 | 端到端实验方案 + 实施检查清单 |

### 10.2 与 rubric-eval 实现的关系

本方案依赖 rubric-eval 项目的生产实现 (位于 `projects/ai-agents/rubric_eval/`)：

- 五阶段流水线: `pipeline.py`
- 评判者面板: `judges/panel.py`
- 校准器: `calibrators/calibration_runner.py`
- 完整性检查: `integrity/checker.py`
- 可靠性指标: `reliability/metrics.py`

### 10.3 与光学领域评估的关系

光学领域评估门和标准在 `mlops/agent-evaluation-methodology/references/optical-agent-evaluation.md` 中定义，本方案将其从 agent 评估扩展到 LLM-as-Judge 评估：

- 10 个评估门 → 18 条 Rubric 标准
- 15 个问题模式 → 证据类型 + 验证规则
- 4 位审稿人 → LLM 评判面板角色配置

---

*本实验方案基于以下核心文献和工具构建：*
*Chen et al. (2606.08625) — Rubric 统一综述*
*RuVerBench (2606.29920) — 评判忠实度基准*
*RubricBench (2603.01562) — Rubric 质量差距*
*Beyond Rating (2604.19502) — 五维人类一致性评估*
*ReviewGuard (2606.24892) — 引用对齐评估*
*rubric-eval 实现框架 (projects/ai-agents/rubric_eval/)*
*agent-evaluation-methodology skill (光学领域适配)*
*文档版本: v16.0*
