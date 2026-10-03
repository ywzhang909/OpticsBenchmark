"""
OptiS 文献处理工作台 — 文档处理服务

封装 ``src/`` 下的工具函数，为 Streamlit 界面提供文档加载、信息提取、
论文评审与评估报告等处理能力。

- :func:`load_document` — 上传文件（PDF/TXT）→ 文本（写入临时文件后调用
  ``src.tools.paper_reader.read_file``）
- :func:`extract_paper_info` — 基于 ``prompts/paper_info_extract`` 的 13 字段信息提取
- :func:`review_paper` — 基于 ``prompts/templates/paper_review.txt`` 的论文评审
- :func:`eval_paper_report` — 复用 ``src.tools.paper_eval_report`` 的
  Rubric 评估 + Markdown 报告生成
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from src.tools.paper_eval_report import evaluate_paper, generate_report
from src.tools.paper_reader import parse_ao_analysis, read_file

from .llm_client import ChatResult, LLMClient

#: 送入 LLM 的文档最大字符数（超出截断并提示）
MAX_DOC_CHARS: int = 30000


# ---------------------------------------------------------------------------
# 文档加载
# ---------------------------------------------------------------------------


def load_document(uploaded_bytes: bytes, filename: str) -> tuple[str, bool]:
    """将上传文件（PDF/TXT）解析为文本。

    上传的字节流先写入临时文件，再调用 ``src.tools.paper_reader.read_file``
    （自动识别 PDF / 纯文本），最后按 :data:`MAX_DOC_CHARS` 截断。

    Args:
        uploaded_bytes: 上传文件的原始字节。
        filename: 原始文件名（用于推断扩展名）。

    Returns:
        ``(文本, 是否被截断)``。

    Raises:
        FileNotFoundError / ValueError / ImportError: 透传自 ``read_file``
            （无法解析的 PDF、空文本等），由调用方捕获并展示。
    """
    suffix = Path(filename).suffix.lower() or ".txt"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_bytes)
            tmp_path = tmp.name
        text = read_file(tmp_path)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
    return truncate_text(text, MAX_DOC_CHARS)


def truncate_text(text: str, max_chars: int = MAX_DOC_CHARS) -> tuple[str, bool]:
    """截断超长文本，返回 ``(截断后文本, 是否发生截断)``。"""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中鲁棒地解析 JSON 对象。

    处理三种常见情况：
    1. 输出被 ```json ... ``` 代码围栏包裹；
    2. 输出前后夹杂解释性文字；
    3. 输出本身就是 JSON。

    Returns:
        解析成功返回 dict，失败返回 ``None``。
    """
    cleaned = (text or "").strip()
    # 剥离代码围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    # 截取首尾大括号之间的部分
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# 文献信息提取
# ---------------------------------------------------------------------------


def build_extraction_prompt(
    paper_text: str,
    prompt_path: str | Path = "prompts/paper_info_extract/zero-shot_v1.0.txt",
) -> str:
    """加载信息提取提示词模板并拼接论文全文。"""
    template = Path(prompt_path).read_text(encoding="utf-8")
    return f"{template}\n\n## 论文全文\n\n{paper_text}"


def extract_paper_info(
    paper_text: str,
    llm_client: LLMClient,
    prompt_path: str | Path = "prompts/paper_info_extract/zero-shot_v1.0.txt",
) -> tuple[dict[str, Any] | None, ChatResult]:
    """提取论文 13 个结构化字段。

    Args:
        paper_text: 论文全文文本。
        llm_client: 已配置的 :class:`LLMClient`。
        prompt_path: 提取提示词模板路径。

    Returns:
        ``(解析后的字段 dict 或 None, ChatResult)``。LLM 返回非 JSON 内容时
        dict 为 ``None``（原始内容可从 ``ChatResult.content`` 查看）。
    """
    prompt = build_extraction_prompt(paper_text, prompt_path)
    result = llm_client.chat([{"role": "user", "content": prompt}])
    if result.error or not result.content:
        return None, result
    data = extract_json(result.content)
    return data, result


# ---------------------------------------------------------------------------
# 论文评审
# ---------------------------------------------------------------------------


def build_review_prompt(
    paper_text: str,
    template_path: str | Path = "prompts/templates/paper_review.txt",
    title: str = "",
) -> str:
    """渲染论文评审提示词（替换 ``{{paper_content}}`` / ``{{title}}``）。"""
    template = Path(template_path).read_text(encoding="utf-8")
    rendered = template.replace("{{paper_content}}", paper_text)
    if title:
        rendered = rendered.replace("{{title}}", title)
    return rendered


def review_paper(
    paper_text: str,
    llm_client: LLMClient,
    template_path: str | Path = "prompts/templates/paper_review.txt",
    system_prompt_path: str | Path = "prompts/system/research_agent.txt",
) -> tuple[str | None, ChatResult]:
    """生成论文评审意见（Markdown）。

    Args:
        paper_text: 论文全文文本。
        llm_client: 已配置的 :class:`LLMClient`。
        template_path: 评审提示词模板路径。
        system_prompt_path: 系统提示词路径（不存在时跳过）。

    Returns:
        ``(评审 Markdown 或 None, ChatResult)``。
    """
    system_file = Path(system_prompt_path)
    system = system_file.read_text(encoding="utf-8") if system_file.is_file() else ""
    prompt = build_review_prompt(paper_text, template_path)

    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": prompt})

    result = llm_client.chat(messages)
    if result.error or not result.content:
        return None, result
    return result.content, result


# ---------------------------------------------------------------------------
# 评估报告（Rubric LLM-as-Judge）
# ---------------------------------------------------------------------------


def eval_paper_report(
    paper_text: str,
    judge_config: dict[str, Any] | None,
    lang: str = "zh_CN",
    commit_id: str = "streamlit",
    source_file: str = "uploaded-document",
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """对解析后的论文字段执行 Rubric 评估并生成 Markdown 报告。

    复用 ``src.tools.paper_eval_report.evaluate_paper`` 与 ``generate_report``：
    读取 → ``parse_ao_analysis`` 解析 → RubricBasedEvaluator 评估 →
    Markdown 报告（含 Mermaid 图表）。

    Args:
        paper_text: 文档文本（AO 分析格式）。
        judge_config: LLM 评委配置；``None`` 时离线模式（得分为 0）。
        lang: 报告语言（``"zh_CN"`` / ``"en"``）。
        commit_id: 报告头部展示的提交标识。
        source_file: 报告头部展示的来源文件名。

    Returns:
        ``(parsed 字段 dict, metrics dict, Markdown 报告字符串)``。
        评估异常不会抛出——``evaluate_paper`` 内部已降级为零分结果。
    """
    parsed = parse_ao_analysis(paper_text)
    if not parsed.get("title"):
        parsed["title"] = source_file

    predicted = {
        "ten keywords": parsed.get("keywords", ""),
        "objective": parsed.get("objective", ""),
        "novelty": parsed.get("novelty", ""),
        "method": parsed.get("method", ""),
        "performance metrics": parsed.get("performance_metrics", ""),
    }
    expected = dict(predicted)

    result = asyncio.run(evaluate_paper(predicted, expected, judge_config))
    report_md = generate_report(
        parsed=parsed,
        result=result,
        commit_id=commit_id,
        source_file=source_file,
        duration_sec=0.0,
        lang=lang,
    )
    return parsed, result.metrics, report_md


__all__ = [
    "MAX_DOC_CHARS",
    "load_document",
    "truncate_text",
    "extract_json",
    "build_extraction_prompt",
    "extract_paper_info",
    "build_review_prompt",
    "review_paper",
    "eval_paper_report",
]
