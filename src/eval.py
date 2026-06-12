#!/usr/bin/env python3
"""
OptiS Benchmark - Evaluation Entry Point (Phase 2)

Load agent outputs and compute evaluation metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.runner import AgentRunner, TaskConfig
from src.utils.logger import logger, setup_logger
from src.utils.parser import ConfigParser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OptiS Benchmark - Evaluation Engine (Phase 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate agent outputs with task config
  python src/eval.py -i results/agent_outputs.jsonl -t configs/tasks/lens_design.yaml

  # Evaluate with custom output path
  python src/eval.py -i results/outputs.jsonl -t configs/tasks/lens_design.yaml -o results/eval_results.json
        """,
    )

    # Input/Output
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to agent outputs JSONL file (from Phase 1)",
    )
    parser.add_argument(
        "-t",
        "--task-config",
        type=str,
        required=True,
        help="Path to task configuration file (YAML)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results/eval_results.json",
        help="Output path for evaluation results (default: results/eval_results.json)",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (if not specified, only console logging)",
    )

    # System configuration
    parser.add_argument(
        "--system-config",
        type=str,
        default="configs/system.yaml",
        help="Path to system configuration file (default: configs/system.yaml)",
    )

    return parser.parse_args()


def load_system_config(system_config_path: str) -> dict:
    """Load system configuration."""
    try:
        return ConfigParser.load_config(system_config_path)
    except FileNotFoundError:
        logger.warning(f"System config not found: {system_config_path}")
        return {}


async def run_evaluation(
    input_path: str,
    task_config_path: str,
    output_path: str,
) -> int:
    """Phase 2: Evaluate agent outputs and compute metrics."""
    logger.info("Evaluation (Phase 2 - Metric Scoring)")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Task:   {task_config_path}")
    logger.info(f"Output: {output_path}")

    try:
        # Load agent outputs
        agent_outputs = AgentRunner.load_agent_outputs(input_path)
        logger.info(f"Loaded {len(agent_outputs)} agent outputs")

        if not agent_outputs:
            logger.error("No agent outputs found")
            return 1

        # Load task config and create evaluator
        task_config = TaskConfig.from_yaml(task_config_path)

        from src.core.evaluator import create_evaluator

        evaluator = create_evaluator(task_config.evaluation_config)

        # Evaluate each output
        results = []
        for ao in agent_outputs:
            result = await evaluator.evaluate(
                task_id=ao.task_id,
                predicted_output=ao.response,
                expected_output=ao.expected_output,
                metadata=ao.metadata,
            )
            result.cost = ao.cost
            results.append(result)

        # Aggregate results
        aggregated = await evaluator.aggregate(results)

        # Save results
        _save_evaluation_results(aggregated, output_path, task_config)

        # Log summary
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Total Tasks:       {aggregated.total_tasks}")
        logger.info(f"Successful:        {aggregated.successful_tasks} ({aggregated.success_rate * 100:.1f}%)")
        logger.info(f"Average Score:    {aggregated.avg_score * 100:.1f}%")
        logger.info(f"Total Cost:       ${aggregated.total_cost:.4f}")
        logger.info(f"Avg Time/Task:    {aggregated.avg_execution_time:.1f}s")

        return 0

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return 1


def _save_evaluation_results(aggregated, output_path: str, task_config: TaskConfig) -> None:
    """Save aggregated evaluation results."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "task_id": task_config.task_id,
        "task_name": task_config.name,
        "total_tasks": aggregated.total_tasks,
        "successful_tasks": aggregated.successful_tasks,
        "success_rate": aggregated.success_rate,
        "avg_score": aggregated.avg_score,
        "avg_execution_time": aggregated.avg_execution_time,
        "total_cost": aggregated.total_cost,
        "metrics_summary": aggregated.metrics_summary,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Also save per-task results as JSONL
    jsonl_path = out_path.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in aggregated.per_task_results:
            f.write(
                json.dumps(
                    {
                        "task_id": r.task_id,
                        "success": r.success,
                        "score": r.score,
                        "metrics": r.metrics,
                        "details": r.details,
                        "error": r.error,
                        "execution_time": r.execution_time,
                        "cost": r.cost,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info(f"Results saved to: {out_path}")
    logger.info(f"Per-task results: {jsonl_path}")


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    # Load system config first
    sys_configs = load_system_config(args.system_config)
    logging_config = sys_configs.get("logging", {})

    # Setup logger
    setup_logger(
        level=args.log_level or logging_config.get("level", "INFO"),
        log_file=args.log_file or logging_config.get("file"),
        console=logging_config.get("console", True),
        format=logging_config.get("format"),
        rotation=logging_config.get("rotation", "100 MB"),
        retention=logging_config.get("retention", "30 days"),
        compression=logging_config.get("compression", "zip"),
    )

    return await run_evaluation(
        input_path=args.input,
        task_config_path=args.task_config,
        output_path=args.output,
    )


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
