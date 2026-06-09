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
from src.utils.logger import logger, setup_logger
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
        "-a",
        "--agent-config",
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
        "-t",
        "--task-set",
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
        "-o",
        "--output",
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
        "-c",
        "--concurrency",
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
        "-v",
        "--verbose",
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
                raise FileNotFoundError(f"Task config not found: {task_set} or {task_yaml}")

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
    logger.info("Running Evaluation")
    logger.info(f"Agent:  {agent_config_path}")
    logger.info(f"Task:   {task_config_path}")
    logger.info(f"Output: {output_path}")

    try:
        results = await run_evaluation(
            agent_config_path=agent_config_path,
            task_config_path=task_config_path,
            output_path=output_path,
            max_concurrency=concurrency,
            timeout=timeout,
            verbose=verbose,
        )

        logger.info("EVALUATION COMPLETE")
        logger.info(f"Total Tasks:       {results.total_tasks}")
        logger.info(f"Successful:        {results.successful_tasks} ({results.success_rate * 100:.1f}%)")
        logger.info(f"Average Score:    {results.avg_score * 100:.1f}%")
        logger.info(f"Total Cost:       ${results.total_cost:.4f}")
        logger.info(f"Avg Time/Task:    {results.avg_execution_time:.1f}s")

        return 0 if results.success_rate >= 0.5 else 1

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return 1


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    # Load system config first
    sys_configs = load_system_config(args.system_config)
    logging_config = sys_configs.get("logging", {})

    # Then setup logger with system config + CLI overrides
    setup_logger(
        level=args.log_level or logging_config.get("level", "INFO"),
        log_file=args.log_file or logging_config.get("file"),
        console=logging_config.get("console", True),
        format=logging_config.get("format"),
        rotation=logging_config.get("rotation", "100 MB"),
        retention=logging_config.get("retention", "30 days"),
        compression=logging_config.get("compression", "zip"),
    )

    # Get agent config path
    agent_config_path = Path(args.agent_config)
    if not agent_config_path.exists():
        logger.error(f"Error: Agent config not found: {agent_config_path}")
        return 1

    # Get task config paths
    try:
        task_config_paths = resolve_task_configs(args.task_set, args.all_tasks)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1

    if not task_config_paths:
        logger.error("Error: No task configurations specified")
        return 1

    # Apply overrides
    # TODO: Implement config overrides

    # Dry run mode
    if args.dry_run:
        logger.info("Dry Run Mode - Configuration:")
        logger.info(f"  Agent Config: {agent_config_path}")
        logger.info(f"  Task Configs: {task_config_paths}")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Concurrency: {args.concurrency}")
        return 0

    # Run evaluations
    exit_code = 0
    for _i, task_path in enumerate(task_config_paths):
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
        logger.warning("Evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
