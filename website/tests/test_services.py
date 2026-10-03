"""离线测试 — 文档处理服务（假 LLM 客户端，无网络）。

覆盖：文档加载/截断、JSON 鲁棒解析、信息提取、论文评审、
以及评估报告（离线模式，RubricBasedEvaluator 不调用 LLM 评委）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from website.llm_client import ChatResult
from website.services import (
    MAX_DOC_CHARS,
    build_review_prompt,
    eval_paper_report,
    extract_json,
    extract_paper_info,
    load_document,
    review_paper,
    truncate_text,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#: 一段 AO 分析格式的样例文本（parse_ao_analysis 可解析）
AO_SAMPLE = """====== 文献调研辅助工具 - 分析报告 ======

## 论文基本信息
- **标题**：基于深度学习的波前传感方法
- **出版年份**：2023
- **DOI**：10.1000/example
- **期刊名称**：Optics Letters
- **10个关键词**：wavefront sensing; deep learning; adaptive optics
- **作者**：张三; 李四

## 核心内容

### 背景与目标
- **研究背景**：传统波前传感速度慢。
- **本文目标**：提出深度学习方法提升波前重建速度。

### 研究内容
- **技术路线**：使用 CNN 网络对 Shack-Hartmann 光斑图直接回归。
- **创新点**：首次将端到端深度学习用于波前重建。

### 实验结果
- **实验结果**：重建误差 RMS 小于 0.05λ，速度提升 10 倍。
"""


class FakeLLMClient:
    """鸭子类型替代 LLMClient —— 按队列返回预设回复，不发起网络请求。"""

    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies or [])
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs) -> ChatResult:
        self.calls.append(messages)
        content = self.replies.pop(0) if self.replies else ""
        return ChatResult(
            content=content,
            usage={"total_tokens": 10},
            cost=0.001,
            latency=0.5,
        )


# ---------------------------------------------------------------------------
# 文本处理
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_untouched(self):
        text, truncated = truncate_text("short")
        assert text == "short"
        assert truncated is False

    def test_long_text_truncated(self):
        text, truncated = truncate_text("a" * (MAX_DOC_CHARS + 100))
        assert truncated is True
        assert len(text) == MAX_DOC_CHARS

    def test_custom_max(self):
        text, truncated = truncate_text("abcdef", max_chars=3)
        assert truncated is True
        assert text == "abc"


class TestLoadDocument:
    def test_txt_upload(self, tmp_path):
        source = tmp_path / "doc.txt"
        source.write_text("hello 文献", encoding="utf-8")
        text, truncated = load_document(source.read_bytes(), "doc.txt")
        assert text == "hello 文献"
        assert truncated is False

    def test_unknown_suffix_falls_back_to_text(self, tmp_path):
        source = tmp_path / "doc.md"
        source.write_text("# title\nbody", encoding="utf-8")
        text, _ = load_document(source.read_bytes(), "doc.md")
        assert "title" in text and "body" in text

    def test_unreadable_pdf_raises(self, tmp_path):
        bad_pdf = b"%PDF-1.4\nnot really a pdf content"
        with pytest.raises((ValueError, Exception)):
            load_document(bad_pdf, "bad.pdf")


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self):
        data = extract_json('{"a": 1, "b": "x"}')
        assert data == {"a": 1, "b": "x"}

    def test_fenced_json(self):
        data = extract_json('```json\n{"a": 1}\n```')
        assert data == {"a": 1}

    def test_fence_without_lang(self):
        data = extract_json('```\n{"a": 1}\n```')
        assert data == {"a": 1}

    def test_prose_wrapped(self):
        data = extract_json('以下是提取结果：\n{"a": 1}\n希望对你有帮助。')
        assert data == {"a": 1}

    def test_invalid_returns_none(self):
        assert extract_json("no json here") is None
        assert extract_json("") is None
        assert extract_json("[]") is None  # 顶层数组不属于论文字段对象


# ---------------------------------------------------------------------------
# 信息提取
# ---------------------------------------------------------------------------


class TestExtractPaperInfo:
    def test_returns_parsed_dict(self):
        client = FakeLLMClient(['```json\n{"title": "T", "Publication Year": "2023"}\n```'])
        data, result = extract_paper_info("paper text", client)
        assert data == {"title": "T", "Publication Year": "2023"}
        assert result.ok

    def test_prompt_contains_template_and_text(self):
        client = FakeLLMClient([""])
        extract_paper_info("THE-PAPER-BODY", client)
        prompt = client.calls[0][0]["content"]
        assert "论文全文" in prompt
        assert "THE-PAPER-BODY" in prompt

    def test_llm_error_returns_none(self):
        class BrokenClient:
            def chat(self, messages, **kwargs):
                return ChatResult(error="connection refused")

        data, result = extract_paper_info("text", BrokenClient())
        assert data is None
        assert result.error == "connection refused"

    def test_non_json_content_returns_none(self):
        client = FakeLLMClient(["抱歉，无法提取。"])
        data, result = extract_paper_info("text", client)
        assert data is None
        assert result.content == "抱歉，无法提取。"


# ---------------------------------------------------------------------------
# 论文评审
# ---------------------------------------------------------------------------


class TestReviewPaper:
    def test_build_review_prompt_substitutes_content(self):
        prompt = build_review_prompt("BODY-OF-PAPER")
        assert "BODY-OF-PAPER" in prompt
        assert "{{paper_content}}" not in prompt
        assert "论文评审" in prompt

    def test_review_returns_markdown(self):
        client = FakeLLMClient(["## 论文评审\n\n### 综合评分\n| 维度 | 评分 |"])
        review_md, result = review_paper("paper", client)
        assert review_md.startswith("## 论文评审")
        assert result.ok
        # 系统提示词应被注入
        assert client.calls[0][0]["role"] == "system"

    def test_review_llm_error(self):
        class BrokenClient:
            def chat(self, messages, **kwargs):
                return ChatResult(error="rate limited")

        review_md, result = review_paper("paper", BrokenClient())
        assert review_md is None
        assert "rate limited" in result.error


# ---------------------------------------------------------------------------
# 评估报告（离线模式）
# ---------------------------------------------------------------------------


class TestEvalReport:
    def test_offline_report_returns_zero_scores(self):
        parsed, metrics, report_md = eval_paper_report(AO_SAMPLE, judge_config=None)
        assert parsed["title"] == "基于深度学习的波前传感方法"
        assert metrics["accuracy"] == 0.0
        assert metrics["completeness"] == 0.0
        assert metrics["readability"] == 0.0
        assert "论文评估报告" in report_md
        assert "评估指标汇总" in report_md
        assert "基于深度学习的波前传感方法" in report_md

    def test_parses_ao_fields(self):
        parsed, _, _ = eval_paper_report(AO_SAMPLE, judge_config=None)
        assert parsed["keywords"] == "wavefront sensing; deep learning; adaptive optics"
        assert parsed["objective"] == "提出深度学习方法提升波前重建速度。"
        assert "CNN" in parsed["method"]
        assert "0.05" in parsed["performance_metrics"]
