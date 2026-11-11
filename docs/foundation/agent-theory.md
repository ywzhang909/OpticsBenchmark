# AI Agent Theory

> This document introduces the concepts, architecture and design principles of AI
> agents, providing the theoretical foundation for building optical-design agents.

---

## 📑 Table of Contents

1. [Agent Overview](#1-agent-overview)
2. [Agent Architecture](#2-agent-architecture)
3. [Tool Use](#3-tool-use)
4. [Planning and Reasoning](#4-planning-and-reasoning)
5. [Optical Design Agents](#5-optical-design-agents)

---

## 1. Agent Overview

### 1.1 What is an AI Agent?

An **AI agent** is a system that perceives its environment, makes decisions and
executes actions in order to achieve a goal.

```
┌─────────────────────────────────────────────────────────────┐
│                        AI Agent                             │
│                                                             │
│  ┌─────────┐    ┌───────────────┐    ┌─────────┐            │
│  │Perceive │ ──►│ Reason / Plan │──► │   Act   │            │
│  └─────────┘    └───────────────┘    └─────────┘            │
│       ▲                                                     │
│       │         ┌───────────────┐                           │
│       └──────── │    Memory     │ ◀ ─ ─ ─ ┘                 │
│                 └───────────────┘                           │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Environment                        │    │
│  │     Optical software / file system / API / user     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Key Capabilities of an Agent

| Capability | Description | Application in optical design |
|------------|-------------|------------------------------|
| **Perception** | Understands the task and its context | Parses design requirements |
| **Reasoning** | Analyzes the problem and devises a strategy | Performs optical design computations |
| **Planning** | Decomposes the task and fixes the steps | Plans the design workflow |
| **Tool use** | Calls external tools | Uses ZOS-API |
| **Learning** | Improves from feedback | Optimizes the design |
| **Memory** | Retains contextual information | Tracks the design history |

### 1.3 Agent vs Traditional Program

| Property | Traditional program | AI agent |
|----------|---------------------|----------|
| Rules | Explicitly coded | Implicitly learned |
| Flexibility | Low | High |
| Generalization | Limited | Generalizes |
| Interaction | Fixed API | Natural language |
| Error handling | Predefined | Adaptive |

---

## 2. Agent Architecture

### 2.1 The ReAct Architecture

ReAct (Reason + Act) is a popular agent architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     The ReAct Loop                          │
│                                                             │
│  Thought ──► Action ──► Observation ──► Thought ──► ...     │
│     │           │            │                              │
│   Think        Act        Observe                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class ReActAgent:
    """
    An agent implementing the ReAct architecture.

    Loops through:
    1. Thought: analyse the current state and decide the next step
    2. Action: execute the chosen action
    3. Observation: collect the result of the execution
    """
    
    def __init__(self, llm, tools, max_steps: int = 10):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []
    
    async def run(self, task: str) -> str:
        """Execute the task"""
        obs = ""  # initial observation
        
        for step in range(self.max_steps):
            # 1. Thought: generate the reasoning
            thought = await self.think(task, obs)
            
            # 2. Action: decide on the action
            action = await self.decide_action(thought, self.tools)
            
            # 3. Execute: perform the action
            result = await self.execute(action)
            
            # 4. Update the observation
            obs = result
            
            # Check whether we are done
            if self.is_complete(result):
                break
        
        return self.generate_final_response()
```

### 2.2 Chain-of-Thought (CoT)

Chain-of-thought is a reasoning technique:

```
User: design a 50mm F/2 achromatic lens

Agent reasoning chain:
├─ Step 1: analyse the requirements
│   - focal length: 50mm
│   - F-number: F/2 → entrance pupil diameter: 25mm
│   - achromatic → at least two glass materials are needed
│                                                                    │
├─ Step 2: initial material selection
│   - positive lens: N-BK7 (low dispersion)
│   - negative lens: F2 (high dispersion)
│   - the combination corrects chromatic aberration at 486nm and 656nm
│                                                                    │
├─ Step 3: initial parameter computation
│   - use the thin-lens formula to split the focal length
│   - positive lens focal length: about 80mm
│   - negative lens focal length: about -130mm
│                                                                    │
├─ Step 4: optimization
│   - variables to set: curvature radii, lens thicknesses
│   - objective: minimize the RMS wavefront error
│                                                                    │
└─ Step 5: output the design
    - lens 1: R1=52mm, R2=-85mm, d=5mm, N-BK7
    - lens 2: R1=-75mm, R2=120mm, d=2mm, F2
```

### 2.3 Multi-Agent Collaboration

```python
class MultiAgentSystem:
    """
    A multi-agent collaboration system.

    Role division:
    - Optics expert: responsible for the optical design computations
    - Code engineer: responsible for API calls and code generation
    - Review expert: evaluates the design proposal
    """
    
    def __init__(self):
        self.agents = {
            "optics_expert": OpticsExpertAgent(),
            "code_engineer": CodeEngineerAgent(),
            "reviewer": ReviewerAgent(),
        }
        self.coordinator = CoordinatorAgent(self.agents)
    
    async def solve(self, task: str) -> Solution:
        """Solve the task collaboratively"""
        
        # 1. The coordinator analyses the task
        plan = await self.coordinator.create_plan(task)
        
        # 2. Assign the subtasks
        subtasks = plan.decompose()
        
        # 3. Execute in parallel
        results = await asyncio.gather(*[
            self.execute_subtask(agent, subtask)
            for agent, subtask in subtasks
        ])
        
        # 4. Integrate the results
        solution = await self.coordinator.integrate(results)
        
        # 5. Review
        review = await self.agents["reviewer"].review(solution)
        
        return solution if review.approved else self.iterate(solution, review)
```

---

## 3. Tool Use

### 3.1 Tool Definition

```python
from pydantic import BaseModel
from typing import Callable, Any

class Tool(BaseModel):
    """Tool definition"""
    name: str                          # tool name
    description: str                   # tool description
    parameters: dict[str, Any]         # parameter schema (JSON Schema)
    execute: Callable[..., Any]        # execution function
    
    def to_openai_format(self) -> dict:
        """Convert to the OpenAI tool format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
```

### 3.2 Example Optical Design Tools

```python
class OpticalTools:
    """A set of optical design tools"""
    
    @staticmethod
    def load_lens(name: str, path: str) -> dict:
        """
        Load a lens file.

        Args:
            name: lens name
            path: file path

        Returns:
            the lens data dict
        """
        pass  # implemented via the ZOS-API
    
    @staticmethod
    def get_system_data() -> dict:
        """
        Get the current system data.

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
        Run the MTF analysis.

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
        Run the optimization.

        Args:
            variables: list of optimization variables
            targets: list of optimization targets
            algorithm: optimization algorithm

        Returns:
            the optimization result
        """
        pass
    
    @staticmethod
    def calculate_raytrace(
        field_x: float,
        field_y: float,
        wavelength: float,
    ) -> list[dict]:
        """
        Ray trace.

        Args:
            field_x, field_y: field coordinates
            wavelength: wavelength

        Returns:
            the ray intersection data
        """
        pass
```

### 3.3 Tool Selection Strategy

```python
async def select_tool(
    task: str,
    available_tools: list[Tool],
    context: dict,
) -> Tool:
    """
    Pick the most suitable tool for the task.

    Strategy:
    1. Task decomposition -> choose the primary tool
    2. Parameter matching -> filter the candidate tools
    3. Context inference -> make the final choice
    """
    
    # The LLM decides which tool to use
    prompt = f"""
    Task: {task}
    
    Available tools:
    {format_tools(available_tools)}
    
    Context:
    {format_context(context)}
    
    Decide which tool to use, and explain why.
    """
    
    response = await llm.chat(prompt)
    return parse_tool_selection(response)
```

---

## 4. Planning and Reasoning

### 4.1 Task Planning

```python
class TaskPlanner:
    """Task planner"""
    
    def create_plan(self, goal: str) -> Plan:
        """
        Create a task execution plan.

        1. Goal decomposition: split the big goal into subgoals
        2. Dependency analysis: determine the dependencies between subgoals
        3. Resource allocation: assign the tools and knowledge needed
        """
        
        # Decompose into subgoals
        subgoals = self.decompose(goal)
        
        # Build the execution graph
        execution_graph = self.build_graph(subgoals)
        
        # Generate the execution plan
        plan = Plan(
            steps=execution_graph.topological_sort(),
            estimated_time=self.estimate_time(execution_graph),
            required_tools=self.collect_tools(execution_graph),
        )
        
        return plan
```

### 4.2 Self-Reflection

```python
class SelfReflection:
    """Self-reflection mechanism"""
    
    async def reflect(
        self,
        action: str,
        result: Any,
        expected: Any,
    ) -> Reflection:
        """
        Reflect on the outcome of an action.

        1. Success? Record the effective strategy
        2. Failure? Analyse the cause and adjust the strategy
        3. Improvement? Update the internal knowledge
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

## 5. Optical Design Agents

### 5.1 Agent System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Optis Agent System Architecture                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   User Interface                        │    │
│  │          (design requirements in natural language)      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                               │                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  LLM Core (GPT-4 / Claude)              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │
│  │  │ Planner  │  │ Reasoner │  │Reflector │                    │
│  │  └──────────┘  └──────────┘  └──────────┘                    │
│  └─────────────────────────────────────────────────────────┘    │
│                               │                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     Tool Layer                          │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐             │
│  │  │ZOS-API  │ │CODE V   │ │ Python  │ │  File   │             │
│  │  │ Tools   │ │ Tools   │ │ Tools   │ │ Tools   │             │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘             │
│  └─────────────────────────────────────────────────────────┘    │
│                               │                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Optical Software Environment               │    │
│  │           (Zemax / CODE V / ASAP / Python)              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Core Logic of the Optics Agent

```python
class OptiSAgent:
    """
    The optical design agent.

    Capabilities:
    - Understand and parse optical design requirements
    - Use optical design software (ZOS-API)
    - Perform optical computations and optimization
    - Generate design reports
    """
    
    def __init__(self, llm_config: dict):
        self.llm = create_llm(llm_config)
        self.tools = self._load_tools()
        self.memory = Memory()
        self.context = {}
    
    async def design_lens(self, requirements: str) -> DesignResult:
        """
        Design a lens.

        Args:
            requirements: the design requirements described in natural language

        Returns:
            the design result
        """
        
        # 1. Parse the requirements
        specs = await self._parse_requirements(requirements)
        
        # 2. Create the design plan
        plan = self._create_design_plan(specs)
        
        # 3. Execute the plan
        for step in plan.steps:
            result = await self._execute_step(step)
            self.memory.add(result)
            
            # Self-check
            if not self._check_result(result, specs):
                # Adjust the strategy
                step = self._adjust_step(step, result)
                result = await self._execute_step(step)
        
        # 4. Optimize the design
        optimized = await self._optimize_design(specs)
        
        # 5. Validate
        validated = await self._validate_design(optimized, specs)
        
        # 6. Generate the report
        report = self._generate_report(validated)
        
        return DesignResult(
            success=True,
            design=optimized,
            report=report,
            metrics=self._extract_metrics(validated),
        )
```

### 5.3 Optical Knowledge Base

```python
OPTICAL_KNOWLEDGE = {
    # Optical design rules
    "rules": [
        "An achromatic doublet requires glasses with a large dispersion difference",
        "The smaller the F-number, the worse the spherical aberration",
        "Lens thickness is typically 5-15% of the diameter",
    ],
    
    # Material selection guide
    "materials": {
        "visible": {
            "positive": ["N-BK7", "N-SSK8", "N-SK16"],
            "negative": ["F2", "N-SF11", "N-SF6"],
        },
        "achromats": {
            "WDM": ["N-BK7", "F2"],      # visible-light achromat
            "infrared": ["Ge", "ZnSe"],  # infrared achromat
        },
    },
    
    # Typical design templates
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

## 📚 References

1. Wei et al. "Chain-of-Thought Prompting Elicits Reasoning" - NeurIPS 2022
2. Yao et al. "ReAct: Synergizing Reasoning and Acting" - ICLR 2023
3. OpenAI. "GPT-4 System Card"
4. Anthropic. "Claude Model Card"
5. Richards et al. "AgentBench: Evaluating LLMs as Agents" - 2023

---

*Last updated: 2024-XX-XX*
