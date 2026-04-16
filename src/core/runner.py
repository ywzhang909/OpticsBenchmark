"""
OptiS Benchmark - Runner Module

This module defines the main evaluation runner that coordinates
agents, environments, and evaluators.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from tqdm import tqdm

from .agent import AgentConfig, BaseAgent, create_agent, Message
from .evaluator import (
    AggregatedResults,
    BaseEvaluator,
    EvaluationResult,
    create_evaluator,
)


@dataclass
class TaskInstance:
    """A single task instance to be evaluated."""
    task_id: str
    instruction: str
    expected_output: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a task set."""
    task_id: str
    name: str
    dataset_path: str
    evaluation_config: dict[str, Any]
    max_samples: Optional[int] = None
    shuffle: bool = False
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskConfig":
        """Load task configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        task_data = data.get("task", {})
        dataset_data = data.get("dataset", {})
        eval_data = data.get("evaluation", {})
        
        return cls(
            task_id=task_data.get("id", "unknown"),
            name=task_data.get("name", "Unknown Task"),
            dataset_path=dataset_data.get("path", ""),
            evaluation_config=eval_data,
            max_samples=dataset_data.get("num_samples"),
            shuffle=dataset_data.get("shuffle", False),
        )


@dataclass
class RunnerConfig:
    """Configuration for the evaluation runner."""
    agent_config: AgentConfig
    task_config: TaskConfig
    output_path: str
    max_concurrency: int = 1
    timeout: int = 300
    save_intermediate: bool = True
    verbose: bool = True
    
    @classmethod
    def from_files(
        cls,
        agent_config_path: str | Path,
        task_config_path: str | Path,
        output_path: str = "results/output.jsonl",
        **kwargs,
    ) -> "RunnerConfig":
        """Create runner config from files."""
        agent_config = AgentConfig.from_yaml(agent_config_path)
        task_config = TaskConfig.from_yaml(task_config_path)
        
        return cls(
            agent_config=agent_config,
            task_config=task_config,
            output_path=output_path,
            **kwargs,
        )


class EvaluationRunner:
    """
    Main evaluation runner that coordinates the evaluation process.
    
    The runner:
    1. Loads task instances from the dataset
    2. Creates agent and evaluator instances
    3. Runs evaluations in parallel (with configurable concurrency)
    4. Aggregates and saves results
    """
    
    def __init__(self, config: RunnerConfig):
        """Initialize the runner with configuration."""
        self.config = config
        self.agent: Optional[BaseAgent] = None
        self.evaluator: Optional[BaseEvaluator] = None
        self.results: list[EvaluationResult] = []
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    async def setup(self) -> None:
        """Set up agent and evaluator."""
        self.agent = create_agent(self.config.agent_config)
        self.evaluator = create_evaluator(self.config.task_config.evaluation_config)
        
        # Add system prompt
        if self.config.agent_config.system_prompt:
            self.agent.add_system_message(self.config.agent_config.system_prompt)
        
        # Set up concurrency limiter
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
    
    async def teardown(self) -> None:
        """Clean up resources."""
        if self.agent:
            await self.agent.close()
    
    def load_tasks(self) -> list[TaskInstance]:
        """Load task instances from dataset file."""
        dataset_path = Path(self.config.task_config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        tasks = []
        dataset_format = self.config.task_config.evaluation_config.get(
            "dataset", {}
        ).get("format", {})
        
        input_field = dataset_format.get("input_field", "instruction")
        output_field = dataset_format.get("output_field", "expected_output")
        metadata_fields = dataset_format.get("metadata_fields", [])
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    metadata = {
                        "task_id": data.get("task_id", f"task_{line_num}"),
                    }
                    for field in metadata_fields:
                        if field in data:
                            metadata[field] = data[field]
                    
                    tasks.append(TaskInstance(
                        task_id=metadata["task_id"],
                        instruction=data.get(input_field, ""),
                        expected_output=data.get(output_field, ""),
                        metadata=metadata,
                    ))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue
        
        # Limit number of samples
        if self.config.task_config.max_samples:
            tasks = tasks[:self.config.task_config.max_samples]
        
        # Shuffle if requested
        if self.config.task_config.shuffle:
            import random
            random.shuffle(tasks)
        
        return tasks
    
    async def run_single_task(self, task: TaskInstance) -> EvaluationResult:
        """Run a single task evaluation."""
        async with self._semaphore:
            return await self._evaluate_task(task)
    
    async def _evaluate_task(self, task: TaskInstance) -> EvaluationResult:
        """Evaluate a single task."""
        from loguru import logger
        
        try:
            # Create user message with task instruction
            user_message = task.instruction
            
            # Add metadata if available
            if task.metadata:
                metadata_str = "\n\nTask Metadata:\n"
                for key, value in task.metadata.items():
                    if key != "task_id":
                        metadata_str += f"- {key}: {value}\n"
                user_message += metadata_str
            
            # Reset agent conversation and add user message
            self.agent.reset()
            self.agent.add_user_message(user_message)
            
            # Get response from agent
            messages = self.agent.conversation_history.copy()
            response = await self.agent.chat(messages)
            
            # Evaluate the response
            result = await self.evaluator.evaluate(
                task_id=task.task_id,
                predicted_output=response.content,
                expected_output=task.expected_output,
                metadata=task.metadata,
            )
            
            # Add cost information
            result.cost = response.cost
            
            return result
            
        except asyncio.TimeoutError:
            return EvaluationResult(
                task_id=task.task_id,
                success=False,
                score=0.0,
                error="Task timeout",
            )
        except Exception as e:
            from loguru import logger
            logger.error(f"Error evaluating task {task.task_id}: {e}")
            return EvaluationResult(
                task_id=task.task_id,
                success=False,
                score=0.0,
                error=str(e),
            )
    
    async def run(self) -> AggregatedResults:
        """
        Run the full evaluation.
        
        Returns:
            AggregatedResults with summary statistics
        """
        from loguru import logger
        
        # Set up
        await self.setup()
        
        # Load tasks
        tasks = self.load_tasks()
        logger.info(f"Loaded {len(tasks)} task instances")
        
        if not tasks:
            raise ValueError("No tasks loaded")
        
        # Create output directory
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run evaluations
        self.results = []
        
        if self.config.verbose:
            pbar = tqdm(total=len(tasks), desc="Evaluating")
        else:
            pbar = None
        
        # Create tasks for parallel execution
        async def run_and_update(task: TaskInstance) -> EvaluationResult:
            result = await self.run_single_task(task)
            if pbar:
                pbar.update(1)
            
            # Save intermediate result
            if self.config.save_intermediate:
                self._save_result(result)
            
            return result
        
        # Execute all tasks
        task_coroutines = [run_and_update(task) for task in tasks]
        
        # Use semaphore-controlled execution
        results = await asyncio.gather(*task_coroutines)
        
        if pbar:
            pbar.close()
        
        self.results = results
        
        # Aggregate results
        aggregated = await self.evaluator.aggregate(self.results)
        
        # Save final results
        self._save_final_results(aggregated)
        
        # Clean up
        await self.teardown()
        
        return aggregated
    
    def _save_result(self, result: EvaluationResult) -> None:
        """Save an individual result to the output file."""
        output_path = Path(self.config.output_path)
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "task_id": result.task_id,
                "success": result.success,
                "score": result.score,
                "metrics": result.metrics,
                "details": result.details,
                "error": result.error,
                "execution_time": result.execution_time,
                "cost": result.cost,
            }, ensure_ascii=False) + "\n")
    
    def _save_final_results(self, aggregated: AggregatedResults) -> None:
        """Save aggregated results."""
        output_path = Path(self.config.output_path)
        
        # Save as JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "task_id": self.config.task_config.task_id,
                "agent_name": self.config.agent_config.name,
                "model": self.config.agent_config.model_name,
                "total_tasks": aggregated.total_tasks,
                "successful_tasks": aggregated.successful_tasks,
                "success_rate": aggregated.success_rate,
                "avg_score": aggregated.avg_score,
                "avg_execution_time": aggregated.avg_execution_time,
                "total_cost": aggregated.total_cost,
                "metrics_summary": aggregated.metrics_summary,
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to:")
        print(f"  - Individual results: {output_path}")
        print(f"  - Aggregated results: {json_path}")
    
    def get_statistics(self) -> dict[str, Any]:
        """Get run statistics."""
        if not self.agent:
            return {}
        
        return {
            "agent_stats": self.agent.get_statistics(),
            "num_results": len(self.results),
        }


async def run_evaluation(
    agent_config_path: str | Path,
    task_config_path: str | Path,
    output_path: str = "results/output.jsonl",
    max_concurrency: int = 1,
    timeout: int = 300,
    verbose: bool = True,
) -> AggregatedResults:
    """
    Convenience function to run an evaluation.
    
    Args:
        agent_config_path: Path to agent configuration file
        task_config_path: Path to task configuration file
        output_path: Path for output results
        max_concurrency: Maximum parallel tasks
        timeout: Task timeout in seconds
        verbose: Show progress bar
        
    Returns:
        AggregatedResults with summary statistics
    """
    config = RunnerConfig.from_files(
        agent_config_path=agent_config_path,
        task_config_path=task_config_path,
        output_path=output_path,
        max_concurrency=max_concurrency,
        timeout=timeout,
        verbose=verbose,
    )
    
    runner = EvaluationRunner(config)
    return await runner.run()
