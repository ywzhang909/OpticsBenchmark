"""
OptiS Benchmark — MdReportBuilder Tests

Tests for the Markdown report builder utility class.
All pure unit tests — no external dependencies.
"""

from __future__ import annotations

import pytest

from src.tools import MdReportBuilder


# ===========================================================================
# Basic text building
# ===========================================================================


class TestBasicBuilding:
    """Headers, paragraphs, raw content, blank lines."""

    def test_h1(self):
        md = MdReportBuilder()
        md.h1("Title")
        result = md.build()
        assert "# Title" in result

    def test_h2(self):
        md = MdReportBuilder()
        md.h2("Section")
        result = md.build()
        assert "## Section" in result

    def test_h3(self):
        md = MdReportBuilder()
        md.h3("Sub")
        result = md.build()
        assert "### Sub" in result

    def test_h4_h5_h6(self):
        md = MdReportBuilder()
        md.h4("a").h5("b").h6("c")
        result = md.build()
        assert "#### a" in result
        assert "##### b" in result
        assert "###### c" in result

    def test_p(self):
        md = MdReportBuilder()
        md.p("Hello world")
        result = md.build()
        assert "Hello world" in result

    def test_raw(self):
        md = MdReportBuilder()
        md.raw("line1\nline2")
        lines = md._lines
        assert "line1" in lines
        assert "line2" in lines

    def test_blank_adds_empty_line(self):
        md = MdReportBuilder()
        md.p("a")
        md.blank()
        md.p("b")
        result = md.build()
        assert "a\n\nb" in result or "a\n\n\nb" in result

    def test_reset(self):
        md = MdReportBuilder()
        md.h1("A")
        md.reset()
        assert md.build().strip() == ""
        assert len(md) == 0

    def test_len(self):
        md = MdReportBuilder()
        assert len(md) == 0
        md.p("hello")
        assert len(md) == 2  # one line + trailing blank

    def test_str(self):
        md = MdReportBuilder()
        md.h1("X")
        assert isinstance(str(md), str)
        assert "# X" in str(md)


# ===========================================================================
# Horizontal rules & breaks
# ===========================================================================


class TestHorizontalRules:
    def test_hr(self):
        md = MdReportBuilder()
        md.hr()
        result = md.build()
        assert "---" in result

    def test_br(self):
        md = MdReportBuilder()
        md.br()
        result = md.build()
        assert "<br>" in result


# ===========================================================================
# Blockquotes
# ===========================================================================


class TestBlockquote:
    def test_simple_blockquote(self):
        md = MdReportBuilder()
        md.blockquote("Warning: something")
        result = md.build()
        assert "> Warning: something" in result

    def test_multiline_blockquote(self):
        md = MdReportBuilder()
        md.blockquote("line1\nline2")
        result = md.build()
        assert "> line1" in result
        assert "> line2" in result


# ===========================================================================
# Lists
# ===========================================================================


class TestLists:
    def test_bullet_list(self):
        md = MdReportBuilder()
        md.bullet_list(["a", "b", "c"])
        result = md.build()
        assert "- a" in result
        assert "- b" in result
        assert "- c" in result

    def test_numbered_list(self):
        md = MdReportBuilder()
        md.numbered_list(["first", "second"])
        result = md.build()
        assert "1. first" in result
        assert "2. second" in result

    def test_empty_list(self):
        md = MdReportBuilder()
        md.bullet_list([])
        md.numbered_list([])
        # Should not crash
        assert True


# ===========================================================================
# Tables
# ===========================================================================


class TestTable:
    def test_basic_table(self):
        md = MdReportBuilder()
        md.table(["A", "B"], [["1", "2"], ["3", "4"]])
        result = md.build()
        assert "| A | B |" in result
        assert "| 1 | 2 |" in result
        assert "| 3 | 4 |" in result

    def test_table_left_align(self):
        md = MdReportBuilder()
        md.table(["X"], [["y"]], align="left")
        result = md.build()
        assert "|---|" in result

    def test_table_center_align(self):
        md = MdReportBuilder()
        md.table(["X"], [["y"]], align="center")
        result = md.build()
        assert "|:---:|" in result

    def test_table_right_align(self):
        md = MdReportBuilder()
        md.table(["X"], [["y"]], align="right")
        result = md.build()
        assert "|---:|" in result

    def test_table_pads_shorter_rows(self):
        md = MdReportBuilder()
        md.table(["A", "B", "C"], [["1", "2"]])  # missing C
        result = md.build()
        assert "| 1 | 2 |  |" in result

    def test_table_single_row(self):
        md = MdReportBuilder()
        md.table(["H"], [["v"]])
        result = md.build()
        assert "| H |" in result
        assert "| v |" in result


# ===========================================================================
# Code blocks
# ===========================================================================


class TestCodeBlock:
    def test_code_block_no_lang(self):
        md = MdReportBuilder()
        md.code_block("print('hello')")
        result = md.build()
        assert "```" in result
        assert "print('hello')" in result

    def test_code_block_with_lang(self):
        md = MdReportBuilder()
        md.code_block('{"a": 1}', lang="json")
        result = md.build()
        assert "```json" in result
        assert '{"a": 1}' in result


# ===========================================================================
# Inline formatting
# ===========================================================================


class TestInlineFormatting:
    def test_bold(self):
        assert MdReportBuilder.bold("text") == "**text**"

    def test_italic(self):
        assert MdReportBuilder.italic("text") == "*text*"

    def test_code(self):
        assert MdReportBuilder.code("var") == "``var``"

    def test_link(self):
        result = MdReportBuilder.link("GitHub", "https://github.com")
        assert result == "[GitHub](https://github.com)"

    def test_image(self):
        result = MdReportBuilder.image("alt", "img.png")
        assert result == "![alt](img.png)"

    def test_image_with_title(self):
        result = MdReportBuilder.image("alt", "img.png", title="Caption")
        assert "Caption" in result


# ===========================================================================
# Mermaid helpers
# ===========================================================================


class TestMermaid:
    def test_mermaid_generic(self):
        md = MdReportBuilder()
        md.mermaid("flowchart LR\n    A-->B")
        result = md.build()
        assert "```mermaid" in result
        assert "flowchart LR" in result
        assert "A-->B" in result

    def test_mermaid_bar(self):
        md = MdReportBuilder()
        md.mermaid_bar("Chart", ["A", "B", "C"], [3.0, 4.5, 2.0])
        result = md.build()
        assert "```mermaid" in result
        assert "xychart" in result
        assert 'title "Chart"' in result
        assert '"A", "B", "C"' in result
        assert "bar [" in result
        assert "3.00" in result

    def test_mermaid_bar_custom_axis(self):
        md = MdReportBuilder()
        md.mermaid_bar("T", ["X"], [2.5], y_label="Custom", y_min=0, y_max=10)
        result = md.build()
        assert '"Custom"' in result
        assert "0 --> 10" in result

    def test_mermaid_pie(self):
        md = MdReportBuilder()
        md.mermaid_pie("Breakdown", {"A": 3, "B": 7})
        result = md.build()
        assert "pie title Breakdown" in result
        assert '"A" : 3' in result
        assert '"B" : 7' in result

    def test_mermaid_flowchart(self):
        md = MdReportBuilder()
        md.mermaid_flowchart(
            "LR",
            ["A[Start]", "B[End]"],
            [("A", "B", "process")],
        )
        result = md.build()
        assert "flowchart LR" in result
        assert "A[Start]" in result
        assert "A -->|process| B" in result


# ===========================================================================
# Metadata
# ===========================================================================


class TestMetadata:
    def test_meta(self):
        md = MdReportBuilder()
        md.meta("Generated", "2025-07-25")
        result = md.build()
        assert "**Generated:** 2025-07-25" in result

    def test_meta_block(self):
        md = MdReportBuilder()
        md.meta_block({"Key1": "Val1", "Key2": "Val2"})
        result = md.build()
        assert "**Key1:** Val1" in result
        assert "**Key2:** Val2" in result


# ===========================================================================
# Figure captions
# ===========================================================================


class TestFigureCaption:
    def test_figure_caption_default_lang(self):
        md = MdReportBuilder()
        md.figure_caption(1, "Overview")
        result = md.build()
        assert "*Figure 1: Overview.*" in result

    def test_figure_caption_zh(self):
        md = MdReportBuilder(lang="zh_CN")
        md.figure_caption(2, "图表")
        result = md.build()
        assert "*图 2: 图表.*" in result


# ===========================================================================
# i18n
# ===========================================================================


class TestI18n:
    def test_default_lang_is_en(self):
        md = MdReportBuilder()
        assert md.t("report_title") == "Paper Evaluation Report"
        assert md.t("generated") == "Generated"

    def test_zh_CN_translation(self):
        md = MdReportBuilder(lang="zh_CN")
        assert md.t("report_title") == "论文评估报告"
        assert md.t("generated") == "生成时间"

    def test_unknown_key_falls_back(self):
        md = MdReportBuilder()
        assert md.t("nonexistent_key") == "nonexistent_key"

    def test_unknown_key_with_default(self):
        md = MdReportBuilder()
        assert md.t("nonexistent", "fallback") == "fallback"

    @property
    def lang(self):
        md = MdReportBuilder()
        assert md.lang == "en"
        md_zh = MdReportBuilder(lang="zh_CN")
        assert md_zh.lang == "zh_CN"


# ===========================================================================
# build() output quality
# ===========================================================================


class TestBuildOutput:
    def test_trailing_newline(self):
        md = MdReportBuilder()
        md.p("hello")
        result = md.build()
        assert result.endswith("\n")

    def test_no_stray_blank_lines_at_end(self):
        md = MdReportBuilder()
        md.h1("Title")
        md.p("Body")
        result = md.build()
        # Last non-empty should be the paragraph content
        lines = [l for l in result.split("\n") if l.strip() or l == ""]
        # Should not have trailing blank lines after content
        stripped = result.rstrip("\n")
        assert not stripped.endswith("\n\n")

    def test_full_report_output_contains_sections(self):
        """End-to-end: simulate a report with all major sections."""
        md = MdReportBuilder(lang="en")
        md.h1("Test Report")
        md.meta("Generated", "now")
        md.h2("Table Section")
        md.table(["A", "B"], [["1", "2"]])
        md.h2("Chart Section")
        md.mermaid_bar("Chart", ["X"], [3.0])
        md.mermaid_pie("Pie", {"Yes": 5})
        result = md.build()
        assert "# Test Report" in result
        assert "## Table Section" in result
        assert "## Chart Section" in result
        assert "```mermaid" in result

    def test_language_affects_generated_report(self):
        en = MdReportBuilder(lang="en")
        en.h1(en.t("report_title"))
        zh = MdReportBuilder(lang="zh_CN")
        zh.h1(zh.t("report_title"))
        assert "Paper Evaluation Report" in en.build()
        assert "论文评估报告" in zh.build()


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_build(self):
        md = MdReportBuilder()
        result = md.build()
        assert result == "\n"

    def test_special_chars_in_table(self):
        md = MdReportBuilder()
        md.table(["Name", "Value"], [["foo_bar", "a | b"]])
        result = md.build()
        assert "foo_bar" in result
        assert "a | b" in result

    def test_large_float_in_bar(self):
        md = MdReportBuilder()
        md.mermaid_bar("Big", ["A"], [3.1415926535])
        result = md.build()
        assert "3.14" in result

    def test_chaining(self):
        """All builder methods return self for fluid chaining."""
        md = (
            MdReportBuilder()
            .h1("A")
            .h2("B")
            .p("C")
            .table(["X"], [["Y"]])
            .code_block("z")
            .mermaid("flowchart LR\nA-->B")
            .hr()
        )
        assert isinstance(md, MdReportBuilder)
        assert "# A" in str(md)
