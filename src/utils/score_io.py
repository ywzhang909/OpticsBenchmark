"""
Optis Benchmark - Score File Loading and Parsing

Shared loading and parsing of judge score records for the consistency and bias
evaluations.

The canonical record is a judge score record::

    {"id": int, "output_model": str, "context": str, "judge_model": str, "score": float}

``id`` is unique within a single file, while ``output_model`` / ``context`` /
``judge_model`` may repeat (one answer can be scored by several judges, and one
``id`` can map to different ``output_model`` values); ``score`` is a scalar. The
bias evaluation instead reads one judge's complete grid over n samples x m model
outputs, where ``id`` repeats and the pairing key is ``(id, output_model)``;
use :func:`records_by_key` for that two-level index.

The evaluations depend only on ``id`` and ``score``; every other key is kept
verbatim as metadata for callers to summarize (:func:`summarize_records`) and
cross-check (:func:`cross_check_metadata`). Two legacy shapes are also accepted:
score files holding only ``{"id", "score"}``, and the benchmark's own evaluation
result records ``{"task_id": int, "metrics": {...}}``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .logger import logger

# =============================================================================
# Constants
# =============================================================================

_LIST_CONTAINER_KEYS = ("scores", "data", "results", "items")

# =============================================================================
# Private Functions
# =============================================================================


def _normalize_record(record: Any, metric: str | None) -> Any:
    """Normalize one record to the ``{"id": ..., "score": ...}`` shape.

    Also accepts the benchmark's own evaluation result records::

        {"evaluator": "exact_match", "task_id": 1, "metrics": {"exact_match_avg": 1.0}}

    Args:
        record: A single record
        metric: Metric to read; the first numeric entry of ``metrics`` when ``None``

    Returns:
        The normalized record, or the input unchanged when it cannot be
        normalized (the computation modules then skip it)
    """
    if not isinstance(record, dict):
        return record

    if "id" in record and "score" in record:
        return record

    entry_id = record.get("id", record.get("task_id"))
    metrics = record.get("metrics")
    if entry_id is None or not isinstance(metrics, dict) or not metrics:
        return record

    if metric is not None:
        if metric not in metrics:
            return record
        score = metrics[metric]
    else:
        score = next(
            (v for v in metrics.values() if isinstance(v, (int, float))),
            None,
        )
    if score is None:
        return record

    return {"id": entry_id, "score": score}


def _load_json(path: Path, metric: str | None) -> list[Any]:
    """Load a single JSON document, either a list or a dict wrapping a list."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [_normalize_record(record, metric) for record in data]

    if isinstance(data, dict):
        for key in _LIST_CONTAINER_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return [_normalize_record(record, metric) for record in value]

    raise ValueError(
        f"{path} is neither a top-level list nor a dict with a "
        f"{'/'.join(_LIST_CONTAINER_KEYS)} list field"
    )


def _unique_values(records: Iterable[dict[str, Any]], key: str) -> list[str]:
    """Collect every non-blank value of one field across records, deduplicated and sorted."""
    values = {str(entry[key]).strip() for entry in records if entry.get(key)}
    return sorted(value for value in values if value)


def _conflicting_ids(
    indexes: dict[str, dict[int, dict[str, Any]]],
    key: str,
) -> list[int]:
    """Find the ids whose field differs between record sets.

    Only non-blank values are compared: legacy ``{"id", "score"}`` records lack
    the field and are therefore not reported as inconsistent.
    """
    seen: dict[int, set[str]] = {}
    for indexed in indexes.values():
        for entry_id, entry in indexed.items():
            value = entry.get(key)
            if value:
                seen.setdefault(entry_id, set()).add(str(value).strip())
    return sorted(entry_id for entry_id, values in seen.items() if len(values) > 1)


# =============================================================================
# Functions
# =============================================================================


def load_scores(path: str | Path, metric: str | None = None) -> list[Any]:
    """Load a score list from a ``.json`` / ``.jsonl`` file.

    Args:
        path: File path; the suffix decides the format
        metric: Metric to read from evaluation result records (``{task_id, metrics}``)

    Returns:
        List of normalized records shaped like ``{"id": int, "score": float}``

    Raises:
        FileNotFoundError: The file does not exist
        ValueError: JSON parsing failed or the top-level structure is unsupported
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")

    if file_path.suffix == ".jsonl":
        records: list[Any] = []
        with open(file_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(_normalize_record(json.loads(line), metric))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{file_path} line {line_no}: invalid JSON: {e}") from e
        return records

    return _load_json(file_path, metric)


def extract_scores(data: list[Any]) -> tuple[dict[int, float], int]:
    """Turn ``[{"id": int, "score": float}, ...]`` into an ``{id: score}`` mapping.

    Args:
        data: Score list to parse

    Returns:
        Tuple ``(id -> score mapping, number of skipped invalid entries)``.
        A repeated id keeps its last occurrence.
    """
    scores: dict[int, float] = {}
    skipped = 0

    if not isinstance(data, (list, tuple)):
        logger.warning(f"score input should be a list, got {type(data).__name__}")
        return scores, skipped + 1

    for entry in data:
        if not isinstance(entry, dict) or "id" not in entry or "score" not in entry:
            skipped += 1
            continue
        try:
            entry_id = int(entry["id"])
            entry_score = float(entry["score"])
        except (TypeError, ValueError):
            skipped += 1
            continue
        scores[entry_id] = entry_score

    return scores, skipped


def records_by_id(data: list[Any]) -> dict[int, dict[str, Any]]:
    """Index a record list by ``id``, keeping every field of the record.

    Like :func:`extract_scores` it only relies on the ``id`` key and a repeated id
    keeps its last occurrence; the extra ``output_model`` / ``context`` /
    ``judge_model`` metadata is preserved so that two score files can be compared
    field by field on id.

    Args:
        data: Record list to index

    Returns:
        ``id -> full record`` mapping; entries whose ``id`` cannot be parsed are skipped.
    """
    indexed: dict[int, dict[str, Any]] = {}

    if not isinstance(data, (list, tuple)):
        logger.warning(f"score input should be a list, got {type(data).__name__}")
        return indexed

    for entry in data:
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        try:
            entry_id = int(entry["id"])
        except (TypeError, ValueError):
            continue
        indexed[entry_id] = entry

    return indexed


def records_by_key(data: list[Any], key: str) -> dict[int, dict[str, dict[str, Any]]]:
    """Index a record list by ``id`` plus a given field, keeping every field.

    Use this when several records share one ``id`` (e.g. one judge scoring the
    outputs of several models), where :func:`records_by_id` would collapse them
    into a single entry. A repeated ``(id, field value)`` pair keeps its last
    occurrence and is reported with a warning.

    Args:
        data: Record list to index
        key: Field distinguishing the records of one sample, e.g. ``"output_model"``

    Returns:
        ``id -> {field value: full record}`` mapping; entries whose ``id`` cannot be
        parsed, or whose field is blank, are skipped.
    """
    indexed: dict[int, dict[str, dict[str, Any]]] = {}

    if not isinstance(data, (list, tuple)):
        logger.warning(f"score input should be a list, got {type(data).__name__}")
        return indexed

    for entry in data:
        if not isinstance(entry, dict) or "id" not in entry or entry.get(key) is None:
            continue
        field = str(entry[key]).strip()
        if not field:
            continue
        try:
            entry_id = int(entry["id"])
        except (TypeError, ValueError):
            continue
        models = indexed.setdefault(entry_id, {})
        if field in models:
            logger.warning(f"sample {entry_id} repeats {key}={field}; keeping the last one")
        models[field] = entry

    return indexed


def summarize_records(data: list[Any], key: str | None = None) -> dict[str, Any]:
    """Summarize the judge score metadata of a record list.

    Args:
        data: Record list to summarize
        key: Optional two-level index field. It must be given when several records
            share one ``id`` (e.g. one judge scoring several models' outputs),
            otherwise :func:`records_by_id` keeps only the last record per ``id``
            and the summary misses the remaining models

    Returns:
        Dict with the following keys:
        - ``num_records``: number of records that took part in the summary
        - ``judge_models``: judge names seen in the file (deduplicated, blanks dropped, sorted)
        - ``output_models``: scored model names seen in the file (deduplicated, blanks dropped,
          sorted)
        - ``has_context``: whether any non-blank ``context`` field is present

        For legacy records (``{"id", "score"}`` only) both lists are empty and
        ``has_context`` is ``False``, so callers can tell whether the file carries
        metadata at all.
    """
    if key:
        rows = [
            record for per_key in records_by_key(data, key).values() for record in per_key.values()
        ]
    else:
        rows = list(records_by_id(data).values())
    valid = [entry for entry in data if isinstance(entry, dict)]

    return {
        "num_records": len(rows),
        "judge_models": _unique_values(valid, "judge_model"),
        "output_models": _unique_values(valid, "output_model"),
        "has_context": any(entry.get("context") for entry in valid),
    }


def cross_check_metadata(record_sets: dict[str, list[Any]]) -> dict[str, Any]:
    """Check across several score record sets whether their metadata is comparable.

    Consistency evaluation compares two judges and bias evaluation a set of
    judges; both need the same comparability check: one ``id`` must point at the
    same output in every file (same ``output_model`` and ``context``), and one
    judge must not appear in several files.

    Args:
        record_sets: ``{name: record list}``, names usually being file or judge names

    Returns:
        Dict with the following keys:
        - ``judges``: metadata summary of each record set (see :func:`summarize_records`)
        - ``duplicate_judges``: ``judge_model`` values appearing in several record sets
          (sorted); what that means depends on the caller, so nothing is warned here
        - ``output_model_mismatch_ids``: ids whose ``output_model`` differs between sides
        - ``context_mismatch_ids``: ids whose ``context`` differs between sides

        The latter two feed ``logger.warning`` calls directly.
    """
    judges = {name: summarize_records(records) for name, records in record_sets.items()}
    indexes = {name: records_by_id(records) for name, records in record_sets.items()}

    judge_sources: dict[str, list[str]] = defaultdict(list)
    for name, summary in judges.items():
        for judge_model in summary["judge_models"]:
            judge_sources[judge_model].append(name)
    duplicate_judges = sorted(
        judge_model for judge_model, sources in judge_sources.items() if len(sources) > 1
    )

    output_model_mismatches = _conflicting_ids(indexes, "output_model")
    context_mismatches = _conflicting_ids(indexes, "context")

    if output_model_mismatches:
        logger.warning(
            f"{len(output_model_mismatches)} ids have a different output_model per side: "
            f"{output_model_mismatches}"
        )
    if context_mismatches:
        logger.warning(
            f"{len(context_mismatches)} ids have a different context per side: {context_mismatches}"
        )

    return {
        "judges": judges,
        "duplicate_judges": duplicate_judges,
        "output_model_mismatch_ids": output_model_mismatches,
        "context_mismatch_ids": context_mismatches,
    }
