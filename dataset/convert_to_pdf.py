#!/usr/bin/env python3
"""
Convert WeChat article markdown files to PDF using fpdf2 + Chinese fonts.

Usage:
    python dataset/convert_to_pdf.py              # Convert all articles
    python dataset/convert_to_pdf.py --test        # Convert first article only

Output:
    dataset/自适应光学 Research/*.pdf
"""

import logging
import os
import re
import sys
from pathlib import Path

from fpdf import FPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent / "自适应光学 Research"

# ── Font Configuration ────────────────────────────────────────────────────

FONT_PATHS = [
    ("C:/Windows/Fonts/NotoSansSC-VF.ttf", "NotoSansSC"),
    ("C:/Windows/Fonts/msyh.ttc", "YaHei"),
    ("C:/Windows/Fonts/simhei.ttf", "SimHei"),
]

CHINESE_FONT = None
FONT_NAME = ""
for path, name in FONT_PATHS:
    if os.path.exists(path):
        CHINESE_FONT = path
        FONT_NAME = name
        break

if CHINESE_FONT is None:
    log.error("No Chinese font found!")
    sys.exit(1)

log.info("Using Chinese font: %s (%s)", FONT_NAME, CHINESE_FONT)


# ── Custom PDF Class ──────────────────────────────────────────────────────


class ArticlePDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.add_font(FONT_NAME, "", CHINESE_FONT)
        self.add_font(FONT_NAME, "B", CHINESE_FONT)
        self.set_auto_page_break(auto=True, margin=25)

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT_NAME, "", 9)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"— {self.page_no()} —", align="C")

    def add_title(self, title: str):
        self.set_font(FONT_NAME, "B", 18)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 10, title, align="L")
        self.ln(3)

    def add_meta(self, author: str, date: str, source: str):
        self.set_font(FONT_NAME, "", 9)
        self.set_text_color(130, 130, 130)
        parts = []
        if author:
            parts.append(f"作者：{author}")
        if date:
            parts.append(f"日期：{date}")
        if source:
            parts.append(f"来源：{source}")
        meta_text = "  |  ".join(parts)
        self.multi_cell(0, 5, meta_text, align="L")
        self.set_draw_color(220, 220, 220)
        self.line(self.l_margin, self.get_y() + 2, self.w - self.r_margin, self.get_y() + 2)
        self.ln(6)

    def add_heading(self, text: str, level: int):
        sizes = {1: 16, 2: 14, 3: 12.5, 4: 11.5, 5: 11, 6: 10.5}
        size = sizes.get(level, 11)
        if level <= 2 and self.get_y() > 40:
            self.add_page()
        self.set_font(FONT_NAME, "B", size)
        self.set_text_color(30, 30, 30)
        self.ln(2)
        self.multi_cell(0, 8 if level == 1 else 7.5, text, align="L")
        if level == 2:
            self.set_draw_color(200, 200, 200)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def add_paragraph(self, text: str):
        if not text.strip():
            self.ln(3)
            return
        self.set_font(FONT_NAME, "", 10.5)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6.5, text, align="L")
        self.ln(1)

    def add_image(self, alt: str, src: str):
        self.set_font(FONT_NAME, "", 9)
        self.set_text_color(100, 100, 100)
        label = f"[图片: {alt}]" if alt else "[图片]"
        self.multi_cell(0, 5, label, align="C")
        self.set_font(FONT_NAME, "", 7)
        self.set_text_color(170, 170, 170)
        # Truncate very long URLs to avoid PDF rendering issues
        src_display = src if len(src) < 200 else src[:197] + "..."
        try:
            self.multi_cell(0, 4, src_display, align="C")
        except Exception:
            pass  # Skip URL if it still causes issues
        self.set_text_color(50, 50, 50)
        self.ln(2)

    def add_list_item(self, text: str):
        self.set_font(FONT_NAME, "", 10.5)
        self.set_text_color(50, 50, 50)
        x = self.get_x()
        self.cell(5, 6.5, chr(8226))
        self.multi_cell(0, 6.5, text, align="L")

    def add_blockquote(self, text: str):
        self.set_font(FONT_NAME, "", 10)
        self.set_text_color(100, 100, 100)
        quote_x = self.l_margin + 5
        self.set_x(quote_x)
        self.multi_cell(self.w - self.r_margin - quote_x, 6, text, align="L")
        self.ln(2)

    def add_separator(self):
        self.set_draw_color(200, 200, 200)
        y = self.get_y() + 3
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(6)


# ── Markdown Parser ──────────────────────────────────────────────────────


def parse_frontmatter(text: str) -> tuple[dict, str]:
    meta = {"title": "", "author": "", "date": "", "source": ""}
    body = text
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1)
        body = text[fm_match.end():]
        for line in fm_text.split("\n"):
            m = re.match(r'^(\w+):\s*"?(.*?)"?\s*$', line)
            if m:
                key = m.group(1).lower()
                val = m.group(2).strip('" ')
                if key in meta:
                    meta[key] = val
    return meta, body


def parse_markdown_lines(body: str) -> list:
    elements = []
    in_list = False
    in_code = False
    code_lines = []
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                elements.append(("code", "\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(stripped)
            i += 1
            continue
        if not stripped:
            if in_list:
                elements.append(("list_end", ""))
                in_list = False
            i += 1
            continue
        h_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if h_match:
            if in_list:
                elements.append(("list_end", ""))
                in_list = False
            level = len(h_match.group(1))
            text = h_match.group(2).strip().replace("**", "")
            elements.append(("heading", text, level))
            i += 1
            continue
        if stripped.startswith(">"):
            if in_list:
                elements.append(("list_end", ""))
                in_list = False
            text = re.sub(r"^>\s*", "", stripped).replace("**", "").replace("*", "")
            elements.append(("blockquote", text))
            i += 1
            continue
        li_match = re.match(r"^[\-\*]\s+(.+)$", stripped)
        if li_match:
            if not in_list:
                in_list = True
            text = li_match.group(1).strip().replace("**", "").replace("*", "")
            elements.append(("list_item", text))
            i += 1
            continue
        img_match = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", stripped)
        if img_match:
            if in_list:
                elements.append(("list_end", ""))
                in_list = False
            elements.append(("image", img_match.group(1), img_match.group(2)))
            i += 1
            continue
        para_lines = []
        while i < len(lines):
            s = lines[i].strip()
            if not s or s.startswith("#") or s.startswith("```") or s.startswith(">") or \
               re.match(r"^[\-\*]\s+", s) or re.match(r"^---+\s*$", s) or \
               re.match(r"^!\[.*?\]\(.*?\)\s*$", s):
                break
            cleaned = s.replace("**", "").replace("*", "")
            cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
            cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
            para_lines.append(cleaned)
            i += 1
        if para_lines:
            if in_list:
                elements.append(("list_end", ""))
                in_list = False
            elements.append(("paragraph", "\n".join(para_lines)))
        if not para_lines:
            i += 1
    if in_list:
        elements.append(("list_end", ""))
    if in_code:
        elements.append(("code", "\n".join(code_lines)))
    return elements


# ── PDF Generation ────────────────────────────────────────────────────────


def markdown_to_pdf(md_filepath: Path, pdf_filepath: Path) -> bool:
    try:
        with open(md_filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        log.error("  Error reading %s: %s", md_filepath.name, e)
        return False

    meta, body = parse_frontmatter(text)
    elements = parse_markdown_lines(body)

    pdf = ArticlePDF()
    pdf.add_page()

    title = meta.get("title", "")
    if title:
        pdf.add_title(title)
    else:
        for elem in elements:
            if elem[0] == "heading" and elem[2] == 1:
                pdf.add_title(elem[1])
                break

    pdf.add_meta(
        author=meta.get("author", ""),
        date=meta.get("date", ""),
        source=meta.get("source", ""),
    )

    for elem in elements:
        type_ = elem[0]
        if type_ == "heading":
            pdf.add_heading(elem[1], elem[2])
        elif type_ == "paragraph":
            pdf.add_paragraph(elem[1])
        elif type_ == "image":
            pdf.add_image(elem[1], elem[2])
        elif type_ == "list_item":
            pdf.add_list_item(elem[1])
        elif type_ == "list_end":
            pdf.ln(2)
        elif type_ == "blockquote":
            pdf.add_blockquote(elem[1])
        elif type_ == "code":
            pdf.add_code(elem[1])
        elif type_ == "separator":
            pdf.add_separator()

    try:
        pdf.output(str(pdf_filepath))
        return True
    except Exception as e:
        log.error("  Error writing PDF %s: %s", pdf_filepath.name, e)
        return False


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    test_mode = "--test" in sys.argv
    md_files = sorted([f for f in DATASET_DIR.glob("*.md") if f.name[0].isdigit()])
    if not md_files:
        log.error("No article markdown files found in %s", DATASET_DIR)
        sys.exit(1)
    if test_mode:
        log.info("TEST MODE: Converting first article only")
        md_files = [md_files[0]]

    total = len(md_files)
    success = 0
    failed = 0
    log.info("Converting %d articles to PDF...", total)

    for i, md_file in enumerate(md_files, 1):
        pdf_file = md_file.with_suffix(".pdf")
        if pdf_file.exists() and test_mode:
            log.info("[%d/%d] %s -> exists, skipping", i, total, md_file.name)
            continue
        log.info("[%d/%d] %s -> PDF", i, total, md_file.name)
        if markdown_to_pdf(md_file, pdf_file):
            pdf_size = pdf_file.stat().st_size
            log.info("  Saved: %s (%d KB)", pdf_file.name, pdf_size // 1024)
            success += 1
        else:
            failed += 1

    log.info("Conversion complete: %d success, %d failed out of %d", success, failed, total)
    print(f"PDFs saved to: {DATASET_DIR}")


if __name__ == "__main__":
    main()
