#!/usr/bin/env python3
"""Generate a Markdown statistics report for results/paper_review_filter.jsonl.

Reads the JSONL produced by utils/paper_review_filter.py and writes a Markdown
report that answers four questions:

1. Which files belong to the optics field, and why?
2. Which files do not belong to the optics field, and why?
3. Which files are initial (first-round) review comments, and why?
4. Which files are not initial review comments, and why?

Records with missing boolean fields are reported separately as Unknown
instead of being silently treated as false.

Usage:
    uv run python tests/report_paper_review_filter.py
    uv run python tests/report_paper_review_filter.py \
        --input results/paper_review_filter.jsonl \
        --output results/paper_review_filter_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = REPO_ROOT / "results" / "paper_review_filter.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "paper_review_filter_report.md"

REQUIRED_FIELDS = (
    "title",
    "is_optics_paper",
    "optics_judge_reason",
    "is_have_images",
    "images_document",
    "is_first_reviews",
    "reviews_judge_reason",
)

MISSING = object()

YES = "Yes"
NO = "No"
UNKNOWN = "Unknown"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Markdown statistics report for the "
        "paper review filter results.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the terminal summary",
    )
    return parser.parse_args()


def load_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Load JSONL records, returning (records, malformed_line_count)."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []
    malformed = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                print(
                    f"[warn] line {line_no} is not valid JSON, skipped",
                    file=sys.stderr,
                )
                continue
            if not isinstance(record, dict):
                malformed += 1
                print(
                    f"[warn] line {line_no} is not a JSON object, skipped",
                    file=sys.stderr,
                )
                continue
            records.append(record)
    return records, malformed


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def tri_state(value: Any) -> str:
    """Map a boolean field to Yes/No/Unknown without truthiness coercion."""
    if value is True:
        return YES
    if value is False:
        return NO
    return UNKNOWN


def missing_fields(data: dict[str, Any]) -> list[str]:
    """Return required fields absent from the record's data object."""
    return [name for name in REQUIRED_FIELDS if name not in data]


def split_by(
    records: list[dict[str, Any]], field_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records by a boolean field into (yes, no, unknown) groups."""
    yes: list[dict[str, Any]] = []
    no: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    for record in records:
        state = tri_state(record.get("data", {}).get(field_name, MISSING))
        if state == YES:
            yes.append(record)
        elif state == NO:
            no.append(record)
        else:
            unknown.append(record)
    return yes, no, unknown


def record_id(record: dict[str, Any]) -> int:
    """Return a sortable integer id for a record."""
    try:
        return int(record.get("id"))
    except (TypeError, ValueError):
        return 0


def record_file(record: dict[str, Any]) -> str:
    """Return the file field of a record, or a placeholder."""
    value = record.get("file")
    return str(value) if value not in (None, "") else "(missing file name)"


def record_title(record: dict[str, Any]) -> str:
    """Return the title field of a record, or a placeholder."""
    value = record.get("data", {}).get("title")
    return str(value).strip() if value else "(missing title)"


def get_reason(record: dict[str, Any], field_name: str) -> str:
    """Return a reason field, or a placeholder when absent/empty."""
    value = record.get("data", {}).get(field_name)
    if value is None or str(value).strip() == "":
        return "(missing reason)"
    return str(value).strip()


def md_code(value: str) -> str:
    """Render a value as inline code, avoiding broken backticks."""
    return "`" + value.replace("`", "'") + "`"


def md_quote(text: str) -> str:
    """Render multi-line text as a Markdown blockquote."""
    lines = text.splitlines() or [""]
    return "\n".join("> " + line if line else ">" for line in lines)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_table(title: str, total: int, yes: int, no: int, unknown: int) -> list[str]:
    """Render one summary table row set as Markdown lines."""
    yes_pct = f"{yes / total:.1%}" if total else "-"
    no_pct = f"{no / total:.1%}" if total else "-"
    unknown_pct = f"{unknown / total:.1%}" if total else "-"
    return [
        f"### {title}",
        "",
        "| Verdict | Count | Share |",
        "|---|---:|---:|",
        f"| Yes | {yes} | {yes_pct} |",
        f"| No | {no} | {no_pct} |",
        f"| Unknown (missing field) | {unknown} | {unknown_pct} |",
        f"| Total | {yes + no + unknown} | 100% |",
        "",
    ]


def render_entries(
    heading: str,
    records: list[dict[str, Any]],
    reason_field: str,
    description: str,
) -> list[str]:
    """Render a detailed section with full reasons for every record."""
    lines = [f"## {heading} ({len(records)} files)", "", description, ""]
    if not records:
        lines.extend(["(none)", ""])
        return lines

    for record in sorted(records, key=record_id):
        lines.append(f"### #{record_id(record)} {record_title(record)}")
        lines.append("")
        lines.append(f"- File: {md_code(record_file(record))}")
        lines.extend(["", "**Reasoning:**", "", md_quote(get_reason(record, reason_field))])
        lines.append("")
    return lines


def render_unknown_section(records: list[dict[str, Any]]) -> list[str]:
    """Render the section for records whose boolean fields are missing."""
    lines = [
        "## 5. Unknown (missing fields)",
        "",
        "Records below lack the boolean fields required for classification and are "
        "therefore excluded from the Yes / No counts. A value of `null` or an absent "
        "field both count as Unknown.",
        "",
    ]
    if not records:
        lines.extend(["(none)", ""])
        return lines

    for record in sorted(records, key=record_id):
        data = record.get("data") or {}
        missing = missing_fields(data)
        lines.append(f"### #{record_id(record)} {record_title(record)}")
        lines.append("")
        lines.append(f"- File: {md_code(record_file(record))}")
        lines.append(f"- Missing fields: {', '.join(f'`{name}`' for name in missing)}")
        optics_state = tri_state(data.get("is_optics_paper", MISSING))
        first_state = tri_state(data.get("is_first_reviews", MISSING))
        lines.append(f"- Optics-field verdict: {optics_state}")
        lines.append(f"- First-round verdict: {first_state}")
        lines.append("")
    return lines


def render_image_appendix(records: list[dict[str, Any]]) -> list[str]:
    """Render the supplementary image-field statistics appendix."""
    have_images, no_images, unknown_images = split_by(records, "is_have_images")
    lines = [
        "## Appendix A: image-field supplementary statistics",
        "",
        "> Note: image information is outside the four required questions and is listed here only as a supplement.",
        "",
        "| is_have_images | Count |",
        "|---|---:|",
        f"| Yes | {len(have_images)} |",
        f"| No | {len(no_images)} |",
        f"| Unknown (missing field) | {len(unknown_images)} |",
        f"| Total | {len(records)} |",
        "",
        "### Files that contain images (sorted by id)",
        "",
    ]
    if not have_images:
        lines.extend(["(none)", ""])
    else:
        for record in sorted(have_images, key=record_id):
            lines.append(f"- #{record_id(record)} {md_code(record_file(record))}")
        lines.append("")

    if unknown_images:
        lines.extend(["### Files missing the is_have_images field", ""])
        for record in sorted(unknown_images, key=record_id):
            lines.append(f"- #{record_id(record)} {md_code(record_file(record))}")
        lines.append("")
    return lines


def render_data_quality(
    records: list[dict[str, Any]], malformed: int, input_path: Path
) -> list[str]:
    """Render the data quality appendix."""
    errors = [r for r in records if r.get("error") is not None]
    null_data = [r for r in records if r.get("data") is None]
    files = [record_file(r) for r in records]
    duplicate_files = sorted({name for name in files if files.count(name) > 1})
    ids = [record_id(r) for r in records]
    unique_ids = len(set(ids))

    lines = [
        "## Appendix B: data quality",
        "",
        f"- Source: {md_code(str(input_path))}",
        f"- Valid records: {len(records)}",
        f"- Malformed lines: {malformed}",
        f"- Records with non-null `error`: {len(errors)}",
        f"- Records with empty `data`: {len(null_data)}",
        f"- Duplicated file names: {len(duplicate_files)}",
        f"- id range: {min(ids) if ids else '-'} ~ {max(ids) if ids else '-'}, "
        f"unique ids: {unique_ids}",
        "",
        "### Missing required fields",
        "",
        "| Field | Records missing |",
        "|---|---:|",
    ]
    for name in REQUIRED_FIELDS:
        count = sum(1 for r in records if name not in (r.get("data") or {}))
        lines.append(f"| `{name}` | {count} |")
    lines.append("")

    if errors:
        lines.extend(["### Records with non-null error", ""])
        for record in errors:
            lines.append(
                f"- #{record_id(record)} {md_code(record_file(record))}: "
                f"{record.get('error')}"
            )
        lines.append("")
    if duplicate_files:
        lines.extend(["### Duplicated file names", ""])
        lines.extend(f"- {md_code(name)}" for name in duplicate_files)
        lines.append("")
    return lines


def build_report(
    records: list[dict[str, Any]], malformed: int, input_path: Path
) -> str:
    """Build the full Markdown report."""
    optics_yes, optics_no, optics_unknown = split_by(records, "is_optics_paper")
    first_yes, first_no, first_unknown = split_by(records, "is_first_reviews")

    models = sorted({str(r.get("model") or "unknown") for r in records})

    lines: list[str] = [
        "# Paper review filter statistics report",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Source: {md_code(str(input_path))}",
        f"- Total records: {len(records)}",
        f"- Model(s): {', '.join(models)}",
        "",
        "## Overview",
        "",
        "> Note: the optics field and the first-round flag are independent dimensions; a "
        "file may appear in the Yes group of both, and the four groups are not mutually "
        "exclusive.",
        "",
        "| Question | Yes | No | Unknown |",
        "|---|---:|---:|---:|",
        f"| Is an optics paper | {len(optics_yes)} | {len(optics_no)} | "
        f"{len(optics_unknown)} |",
        f"| Is a first-round review | {len(first_yes)} | {len(first_no)} | "
        f"{len(first_unknown)} |",
        "",
        "### Breakdown",
        "",
    ]
    lines.extend(
        render_table(
            "Optics-field verdict",
            len(records),
            len(optics_yes),
            len(optics_no),
            len(optics_unknown),
        )
    )
    lines.extend(
        render_table(
            "First-round verdict",
            len(records),
            len(first_yes),
            len(first_no),
            len(first_unknown),
        )
    )

    lines.extend(
        render_entries(
            "1. Papers in the optics field",
            optics_yes,
            "optics_judge_reason",
            "Classified as optics; reasoning quoted verbatim from `optics_judge_reason`.",
        )
    )
    lines.extend(
        render_entries(
            "2. Papers outside the optics field",
            optics_no,
            "optics_judge_reason",
            "Classified as non-optics; reasoning quoted verbatim from `optics_judge_reason`.",
        )
    )
    lines.extend(
        render_entries(
            "3. Files with first-round review comments",
            first_yes,
            "reviews_judge_reason",
            "Classified as first-round reviews; reasoning quoted verbatim from `reviews_judge_reason`.",
        )
    )
    lines.extend(
        render_entries(
            "4. Files without first-round review comments",
            first_no,
            "reviews_judge_reason",
            "Classified as not first-round (e.g. revision rounds / re-review comments); "
            "reasoning quoted verbatim from `reviews_judge_reason`.",
        )
    )
    unknown_section: dict[int, dict[str, Any]] = {}
    for record in optics_unknown + first_unknown:
        unknown_section.setdefault(record_id(record), record)
    lines.extend(render_unknown_section(list(unknown_section.values())))
    lines.extend(render_image_appendix(records))
    lines.extend(render_data_quality(records, malformed, input_path))
    return "\n".join(lines)


def print_summary(
    output_path: Path,
    records: list[dict[str, Any]],
    optics_yes: int,
    optics_no: int,
    optics_unknown: int,
    first_yes: int,
    first_no: int,
    first_unknown: int,
) -> None:
    """Print a compact terminal summary."""
    print("=" * 52)
    print("Paper review filter statistics report")
    print("=" * 52)
    print(f"Total records: {len(records)}")
    print(f"Optics field: yes {optics_yes} / no {optics_no} / unknown {optics_unknown}")
    print(f"First-round: yes {first_yes} / no {first_no} / unknown {first_unknown}")
    print(f"Report written to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Load the JSONL, write the Markdown report, and print a summary."""
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    try:
        records, malformed = load_records(input_path)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    report = build_report(records, malformed, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    optics_yes, optics_no, optics_unknown = split_by(records, "is_optics_paper")
    first_yes, first_no, first_unknown = split_by(records, "is_first_reviews")
    if not args.quiet:
        print_summary(
            output_path,
            records,
            len(optics_yes),
            len(optics_no),
            len(optics_unknown),
            len(first_yes),
            len(first_no),
            len(first_unknown),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
