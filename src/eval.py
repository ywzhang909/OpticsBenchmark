#!/usr/bin/env python3
"""
OptiS Benchmark - Evaluation Entry Point (Phase 2)

Load agent outputs and compute evaluation metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.runner import AgentRunner
from src.utils.logger import logger, setup_logger
from src.utils.parser import ConfigParser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OptiS Benchmark - Evaluation Engine (Phase 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate agent outputs with eval config and gold answers
  python src/eval.py -i results/agent_outputs.jsonl -g dataset/gold.json -e configs/eval/lens_design.yaml

  # Evaluate with custom output path
  python src/eval.py -i results/outputs.jsonl -g dataset/gold.json -e configs/eval/lens_design.yaml -o results/eval_results.json
        """,
    )

    # Input/Output
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        # required=True,
        default="self_test/dataset/paper_info_extract/test_v1.json",
        help="Path to agent outputs JSONL file (from Phase 1)",
    )
    parser.add_argument(
        "-e",
        "--eval-config",
        type=str,
        # required=True,
        default="configs/evaluations/paper_info_extract.yaml",
        help="Path to evaluation configuration file (YAML)",
    )
    parser.add_argument(
        "-g",
        "--gold",
        type=str,
        # required=True,
        default="dataset/paper_info_extract/dataset_json/gold_answer_v1.json",
        help="Path to gold standard answer dataset file (JSON)",
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


def _load_gold_data(gold_path: str) -> dict[str, Any]:
    """Load gold standard answers from JSON file and build task_id -> data map.

    Expected JSON format: [{"task_id": "...", "data": {...}}, ...]
    """
    path = Path(gold_path)
    if not path.exists():
        logger.error(f"Gold data file not found: {gold_path}")
        return {}

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        logger.error("Gold data must be a JSON array of {task_id, data} objects")
        return {}

    gold_map: dict[str, Any] = {}
    for item in raw:
        tid = item.get("id")
        if not tid:
            logger.warning(f"Skipping gold item without task_id: {item}")
            continue
        gold_map[tid] = item.get("data")

    logger.info(f"Loaded {len(gold_map)} gold answers from {gold_path}")
    return gold_map


async def run_evaluation(
    input_path: str,
    gold_path: str,
    eval_config_path: str,
    output_path: str,
) -> int:
    """Phase 2: Evaluate agent outputs and compute metrics."""
    logger.info("Evaluation (Phase 2 - Metric Scoring)")
    logger.info(f"Input:       {input_path}")
    logger.info(f"Gold:        {gold_path}")
    logger.info(f"Eval Config: {eval_config_path}")
    logger.info(f"Output:      {output_path}")

    try:
        # Load gold standard answers
        gold_map = _load_gold_data(gold_path)
        if not gold_map:
            logger.error("No gold answers loaded")
            return 1

        # Load agent outputs
        agent_outputs = AgentRunner.load_agent_outputs(input_path)
        logger.info(f"Loaded {len(agent_outputs)} agent outputs")

        if not agent_outputs:
            logger.error("No agent outputs found")
            return 1

        # Load eval config and create evaluators
        eval_config = ConfigParser.load_config(eval_config_path)

        from src.core.evaluator import create_evaluator, AggregatedResults

        evaluators = create_evaluator(eval_config)
        evaluator_names = list(eval_config.get("eval_metrics", {}).keys())

        if not evaluators:
            logger.error("No evaluators created from config")
            return 1

        # Evaluate each output against each evaluator, matched by task_id
        per_evaluator_results: dict[str, list] = {name: [] for name in evaluator_names}
        for ao in agent_outputs:
            if ao.task_id not in gold_map:
                logger.warning(f"Gold answer not found for task_id: {ao.task_id}, skipping")
                continue

            for i, ev in enumerate(evaluators):
                result = await ev.evaluate(
                    task_id=ao.task_id,
                    predicted_output=ao.response,
                    expected_output=gold_map[ao.task_id],
                    metadata=None,
                )
                per_evaluator_results[evaluator_names[i]].append(result)

        all_empty = all(len(v) == 0 for v in per_evaluator_results.values())
        if all_empty:
            logger.error("No results after matching agent outputs with gold answers")
            return 1

        # Aggregate per evaluator
        aggregated_by_name: dict[str, AggregatedResults] = {}
        for i, ev in enumerate(evaluators):
            name = evaluator_names[i]
            aggregated_by_name[name] = await ev.aggregate(per_evaluator_results[name])

        # Save results
        _save_evaluation_results(aggregated_by_name, output_path, eval_config)

        # Log summary
        logger.info("EVALUATION COMPLETE")
        for name, agg in aggregated_by_name.items():
            logger.info(f"[{name}] Tasks: {agg.total_tasks} | Metrics: {agg.metrics_summary} | AvgTime: {agg.avg_execution_time:.1f}s")

        return 0

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return 1


def _save_evaluation_results(
    aggregated_by_name: dict[str, AggregatedResults],
    output_path: str,
    eval_config: dict,
) -> None:
    """Save aggregated evaluation results, grouped by evaluator name."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    evaluators_out = {}
    for name, agg in aggregated_by_name.items():
        evaluators_out[name] = {
            "total_tasks": agg.total_tasks,
            "avg_execution_time": agg.avg_execution_time,
            "metrics_summary": agg.metrics_summary,
            "per_task_results": [
                {
                    "task_id": r.task_id,
                    "metrics": r.metrics,
                    "execution_time": r.execution_time,
                }
                for r in agg.per_task_results
            ],
        }

    data = {
        "timestamp": datetime.now().isoformat(),
        "eval_config": eval_config,
        "evaluators": evaluators_out,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Also save per-task results as JSONL with evaluator tag
    jsonl_path = out_path.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for name, agg in aggregated_by_name.items():
            for r in agg.per_task_results:
                f.write(
                    json.dumps(
                        {
                            "evaluator": name,
                            "task_id": r.task_id,
                            "metrics": r.metrics,
                            "execution_time": r.execution_time,
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
        gold_path=args.gold,
        eval_config_path=args.eval_config,
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
