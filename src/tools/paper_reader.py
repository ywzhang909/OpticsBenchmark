"""
OptiS Benchmark - Paper Reader

PDF and text file reading utilities for the evaluation pipeline.

Supports:
- PDF text extraction via PyPDF2
- Plain text file reading
- AO analysis TXT format parsing
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_pdf(path: str | Path) -> str:
    """Extract text from a PDF file using PyPDF2.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text content.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ImportError: If PyPDF2 is not installed.
        ValueError: If the PDF has no extractable text.
    """
    import PyPDF2

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())

    if not pages:
        raise ValueError(f"No extractable text found in PDF: {path}")

    return "\n\n".join(pages)


def read_text(path: str | Path) -> str:
    """Read a plain text file (UTF-8).

    Args:
        path: Path to the text file.

    Returns:
        File contents as string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    return path.read_text(encoding="utf-8")


def read_file(path: str | Path) -> str:
    """Auto-detect file type and read contents.

    Supports ``.pdf`` (via PyPDF2) and ``.txt`` (plain text).

    Args:
        path: Path to the file.

    Returns:
        File text content.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return read_pdf(path)
    elif suffix == ".txt":
        return read_text(path)
    else:
        # Fallback — try as plain text
        return read_text(path)


def parse_ao_analysis(text: str) -> dict[str, Any]:
    """Parse an AO analysis TXT file into structured fields.

    The AO analysis files follow the format::

        ====== ... ======
        文献调研辅助工具 - 分析报告
        ====== ... ======

        分析结果:

        ## 论文基本信息
        - **标题**：...
        - **出版年份**：...
        - **DOI**：...
        ...

        ## 核心内容

        ### 背景与目标
        - **研究背景**：...
        - **本文目标**：...

        ### 研究内容
        - **技术路线**：...
        - **关键方法**：...

    Args:
        text: Raw text of an AO analysis file.

    Returns:
        Dict with keys: ``title``, ``year``, ``doi``, ``journal``,
        ``keywords``, ``authors``, ``objective``, ``novelty``,
        ``method``, ``performance_metrics``. Missing fields are ``""``.
    """
    result: dict[str, Any] = {
        "title": "",
        "year": "",
        "doi": "",
        "journal": "",
        "keywords": "",
        "authors": "",
        "objective": "",
        "novelty": "",
        "method": "",
        "performance_metrics": "",
    }

    # --- metadata section (## 论文基本信息) ---
    meta_match = re.search(
        r"## 论文基本信息\s*\n(.*?)(?=\n## |\Z)",
        text, re.DOTALL,
    )
    if meta_match:
        meta_block = meta_match.group(1)

        def _extract(label: str) -> str:
            m = re.search(
                rf"- \*\*{re.escape(label)}\*\*[：:]\s*(.*?)(?=\n|$)",
                meta_block,
            )
            return m.group(1).strip() if m else ""

        result["title"] = _extract("标题")
        result["year"] = _extract("出版年份")
        result["doi"] = _extract("DOI")
        result["journal"] = _extract("期刊名称")

        # keywords — may be semicolon or comma separated
        kw_raw = _extract("10个关键词")
        result["keywords"] = kw_raw

        authors_raw = _extract("作者")
        result["authors"] = authors_raw

    # --- core content sections ---
    core_match = re.search(
        r"## 核心内容\s*\n(.*?)(?=\n## |\Z)",
        text, re.DOTALL,
    )
    if core_match:
        core_block = core_match.group(1)

        # Objective (背景与目标 → 本文目标)
        obj_match = re.search(r"- \*\*本文目标\*\*[：:]\s*(.*?)(?=\n|$)", core_block)
        if obj_match:
            result["objective"] = obj_match.group(1).strip()

        # Novelty (关键方法 or 创新点)
        for label in ("创新点", "关键方法"):
            nov_match = re.search(
                rf"- \*\*{re.escape(label)}\*\*[：:]\s*(.*?)(?=\n|$)",
                core_block,
            )
            if nov_match:
                result["novelty"] = nov_match.group(1).strip()
                break

        # Method (技术路线 or 核心方法)
        for label in ("技术路线", "核心方案"):
            meth_match = re.search(
                rf"- \*\*{re.escape(label)}\*\*[：:]\s*(.*?)(?=\n|$)",
                core_block,
            )
            if meth_match:
                result["method"] = meth_match.group(1).strip()
                break

        # Performance metrics (实验结果 → look for numbers)
        perf_match = re.search(
            r"- \*\*实验结果\*\*[：:]\s*(.*?)(?=\n\n|\n###|\Z)",
            core_block, re.DOTALL,
        )
        if perf_match:
            result["performance_metrics"] = perf_match.group(1).strip()

    return result
