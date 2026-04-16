#!/usr/bin/env python3
"""
OptiS Benchmark - Main Entry Point

Command-line interface for running optical design agent evaluations.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.runner import run_evaluation
from src.utils.logger import setup_logger
from src.utils.parser import ConfigParser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OptiS Benchmark - Optical Design Agent Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default settings
  python src/main.py --agent-config configs/agents/gpt-4.yaml --task-set lens_design

  # Run with custom output and concurrency
  python src/main.py -a configs/agents/claude-3.yaml -t lens_design -o results/my_eval.jsonl -c 4

  # Run all task sets with a specific agent
  python src/main.py -a configs/agents/gpt-4.yaml --all-tasks -o results/all_results.jsonl
        """,
    )
    
    # Agent configuration
    agent_group = parser.add_argument_group("Agent Configuration")
    agent_group.add_argument(
        "-a", "--agent-config",
        type=str,
        required=True,
        help="Path to agent configuration file (YAML)",
    )
    agent_group.add_argument(
        "--agent-overrides",
        type=str,
        nargs="+",
        help="Override agent config values (format: key=value)",
    )
    
    # Task configuration
    task_group = parser.add_argument_group("Task Configuration")
    task_group.add_argument(
        "-t", "--task-set",
        type=str,
        help="Task set name (without .yaml extension) or path to task config",
    )
    task_group.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all task sets in configs/tasks/",
    )
    task_group.add_argument(
        "--task-overrides",
        type=str,
        nargs="+",
        help="Override task config values (format: key=value)",
    )
    
    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "-o", "--output",
        type=str,
        default="results/output.jsonl",
        help="Output path for results (default: results/output.jsonl)",
    )
    output_group.add_argument(
        "--save-intermediate",
        action="store_true",
        default=True,
        help="Save intermediate results (default: True)",
    )
    output_group.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Disable intermediate result saving",
    )
    
    # Execution configuration
    exec_group = parser.add_argument_group("Execution Configuration")
    exec_group.add_argument(
        "-c", "--concurrency",
        type=int,
        default=1,
        help="Maximum parallel evaluation tasks (default: 1)",
    )
    exec_group.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Task timeout in seconds (default: 300)",
    )
    exec_group.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples per task (for testing)",
    )
    
    # Logging configuration
    log_group = parser.add_argument_group("Logging Configuration")
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    log_group.add_argument(
        "--log-file",
        type=str,
        help="Log file path (if not specified, only console logging)",
    )
    
    # System configuration
    sys_group = parser.add_argument_group("System Configuration")
    sys_group.add_argument(
        "--system-config",
        type=str,
        default="configs/system.yaml",
        help="Path to system configuration file (default: configs/system.yaml)",
    )
    
    # Misc
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    misc_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running evaluation",
    )
    misc_group.add_argument(
        "--version",
        action="version",
        version="OptiS Benchmark v1.0.0",
    )
    
    return parser.parse_args()


def load_system_config(system_config_path: str) -> dict:
    """Load system configuration."""
    try:
        return ConfigParser.load_config(system_config_path)
    except FileNotFoundError:
        print(f"Warning: System config not found: {system_config_path}")
        return {}


def resolve_task_configs(
    task_set: str | None,
    all_tasks: bool,
) -> list[Path]:
    """Resolve task configuration file paths."""
    configs_dir = Path("configs/tasks")
    
    if all_tasks:
        return list(configs_dir.glob("*.yaml"))
    
    if task_set:
        task_path = Path(task_set)
        if task_path.exists():
            return [task_path]
        else:
            # Try configs/tasks/ directory
            task_yaml = configs_dir / f"{task_set}.yaml"
            if task_yaml.exists():
                return [task_yaml]
            else:
                raise FileNotFoundError(
                    f"Task config not found: {task_set} or {task_yaml}"
                )
    
    return []


async def run_single_evaluation(
    agent_config_path: Path,
    task_config_path: Path,
    output_path: str,
    concurrency: int,
    timeout: int,
    verbose: bool,
    max_samples: int | None,
) -> int:
    """Run a single evaluation task."""
    print(f"\n{'='*60}")
    print(f"Running Evaluation")
    print(f"{'='*60}")
    print(f"Agent:  {agent_config_path}")
    print(f"Task:   {task_config_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    try:
        results = await run_evaluation(
            agent_config_path=agent_config_path,
            task_config_path=task_config_path,
            output_path=output_path,
            max_concurrency=concurrency,
            timeout=timeout,
            verbose=verbose,
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total Tasks:       {results.total_tasks}")
        print(f"Successful:        {results.successful_tasks} ({results.success_rate*100:.1f}%)")
        print(f"Average Score:    {results.avg_score*100:.1f}%")
        print(f"Total Cost:       ${results.total_cost:.4f}")
        print(f"Avg Time/Task:    {results.avg_execution_time:.1f}s")
        print(f"{'='*60}\n")
        
        return 0 if results.success_rate >= 0.5 else 1
        
    except Exception as e:
        print(f"\nError during evaluation: {e}", file=sys.stderr)
        return 1


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    # Setup logging
    setup_logger(
        log_file=args.log_file,
        level=args.log_level,
    )
    
    # Load system config
    system_config = load_system_config(args.system_config)
    
    # Get agent config path
    agent_config_path = Path(args.agent_config)
    if not agent_config_path.exists():
        print(f"Error: Agent config not found: {agent_config_path}", file=sys.stderr)
        return 1
    
    # Get task config paths
    try:
        task_config_paths = resolve_task_configs(args.task_set, args.all_tasks)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if not task_config_paths:
        print("Error: No task configurations specified", file=sys.stderr)
        return 1
    
    # Apply overrides
    # TODO: Implement config overrides
    
    # Dry run mode
    if args.dry_run:
        print("Dry Run Mode - Configuration:")
        print(f"  Agent Config: {agent_config_path}")
        print(f"  Task Configs: {task_config_paths}")
        print(f"  Output: {args.output}")
        print(f"  Concurrency: {args.concurrency}")
        return 0
    
    # Run evaluations
    exit_code = 0
    for i, task_path in enumerate(task_config_paths):
        # Generate output path for each task
        if len(task_config_paths) > 1:
            task_name = task_path.stem
            output_path = Path(args.output)
            output_path = output_path.parent / f"{output_path.stem}_{task_name}.jsonl"
        else:
            output_path = args.output
        
        code = await run_single_evaluation(
            agent_config_path=agent_config_path,
            task_config_path=task_path,
            output_path=str(output_path),
            concurrency=args.concurrency,
            timeout=args.timeout,
            verbose=args.verbose or True,
            max_samples=args.max_samples,
        )
        
        if code != 0:
            exit_code = code
    
    return exit_code


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    try:
        exit_code = asyncio.run(main_async(args))
        return exit_code
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
