# OptiS Benchmark - Evaluation Theory

## Overview

This document outlines the theoretical foundations and methodology behind the OptiS Benchmark evaluation system.

## 1. Agent Evaluation Framework

### 1.1 What is an AI Agent?

An AI agent in the context of optical design is an AI system that can:
- **Perceive** optical design requirements and constraints
- **Plan** solutions using domain knowledge
- **Execute** actions (create designs, run analyses, modify parameters)
- **Learn** from feedback and improve

### 1.2 Agent Architecture

Our benchmark evaluates agents following a standard architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Agent System                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   LLM Core  │───▶│   Planner   │───▶│    Tools    │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
│         ▲                  │                  │         │
│         │                  ▼                  │         │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Memory / Context                        ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## 2. Task Categories

### 2.1 Lens Design Tasks

**Objective**: Evaluate agent's ability to design optical lens systems.

**Metrics**:
- MTF (Modulation Transfer Function) performance
- RMS spot size
- Distortion
- Transmission efficiency
- Design feasibility

**Example Task**:
> Design an achromatic doublet lens with focal length 100mm, F/5, for visible wavelengths (486nm-656nm).

### 2.2 System Analysis Tasks

**Objective**: Evaluate agent's ability to analyze existing optical systems.

**Metrics**:
- Analysis completeness
- Metric accuracy
- Report quality (via LLM judge)

**Example Task**:
> Analyze the attached lens file and provide a comprehensive performance report including MTF, spot diagram, and aberration analysis.

### 2.3 Tolerance Analysis Tasks

**Objective**: Evaluate agent's ability to perform tolerance analysis and optimization.

**Metrics**:
- Tolerance sensitivity prediction accuracy
- Compensation strategy effectiveness
- Manufacturing cost estimates

## 3. Evaluation Methodology

### 3.1 Evaluation Pipeline

```
Task Instance ──▶ Agent ──▶ Response ──▶ Evaluator ──▶ Score
     │                                      │
     │          Ground Truth ───────────────▶│
     ▼                                      ▼
Metadata                               Aggregated
                                       Results
```

### 3.2 Scoring Methods

#### Metric-Based Scoring

For tasks with quantifiable outputs:
- Compare numerical outputs against ground truth
- Apply success criteria (thresholds)

```
Score = Σ (w_i × metric_score_i) / Σ w_i
```

Where `metric_score_i` is calculated as:
```
if operator == ">=":
    score = 1 if value >= threshold else 0
elif operator == "<=":
    score = 1 if value <= threshold else 0
```

#### LLM Judge Scoring

For subjective evaluations:
- Use a separate LLM to evaluate response quality
- Provide rubric and examples

### 3.3 Success Criteria

A task is considered **successful** if:
1. All mandatory metrics meet their thresholds
2. No critical errors occurred during execution
3. Execution completed within time limit

## 4. Metrics

### 4.1 Optical Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| MTF @ 50 lp/mm | Modulation transfer function | ≥ 0.7 |
| RMS Spot Size | Root-mean-square spot radius | ≤ 0.05 mm |
| Distortion | Maximum distortion | ≤ 2% |
| Transmission | System transmission | ≥ 0.85 |
| Wavefront Error | RMS wavefront error | ≤ 0.07 λ |

### 4.2 Execution Metrics

| Metric | Description |
|--------|-------------|
| Success Rate | Percentage of successful tasks |
| Average Steps | Average number of tool calls |
| Average Time | Average execution time |
| Cost per Task | API cost per task |

## 5. Benchmark Design Principles

### 5.1 Reproducibility

- All task instances are fixed and versioned
- Random seeds are controlled
- Evaluation environment is containerized

### 5.2 Fairness

- Same tasks for all agents
- Cost tracking for all providers
- Time limits applied uniformly

### 5.3 Comprehensiveness

- Multiple task categories
- Various difficulty levels
- Real-world complexity

### 5.4 Practicality

- Focus on real optical design challenges
- Use actual optical software APIs
- Evaluate end-to-end capability

## 6. Limitations and Future Work

### 6.1 Current Limitations

- Limited to simulated/ idealized optical environments
- Ground truth may not represent all valid solutions
- LLM judge scoring has inherent subjectivity

### 6.2 Future Enhancements

- Real Zemax/CODE V integration
- Multi-agent collaboration evaluation
- Dynamic task generation
- Cross-domain transfer learning assessment

## References

1. Liu et al. "AgentBench: Evaluating LLMs as Agents" - arXiv 2023
2. Zemax OpticStudio Documentation
3. ISO 10110: Optics and photonics - Preparation of drawings for optical elements and systems
