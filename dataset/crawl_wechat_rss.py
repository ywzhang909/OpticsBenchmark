#!/usr/bin/env python3
"""
Crawl WeChat public account articles from RSS feed and save as dataset.

Usage:
    python dataset/crawl_wechat_rss.py

Output:
    dataset/自适应光学 Research/*.md        - Article files in markdown
    dataset/自适应光学 Research/_metadata.json  - Index of all articles
"""

import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import httpx
from lxml import html as lxml_html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

RSS_URL = "https://werss.impecunious909.asia/feed/MP_WXS_3988686387.rss"
DATASET_DIR = Path(__file__).parent / "自适应光学 Research"
REQUEST_DELAY = 1.0  # seconds between requests to be polite
REQUEST_TIMEOUT = 60

# Browser-like headers to bypass WeChat anti-bot
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://mp.weixin.qq.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
}


# ── RSS Feed Fetching & Parsing ────────────────────────────────────────────


def fetch_rss(url: str) -> str:
    """Fetch the RSS feed XML."""
    log.info("Fetching RSS feed: %s", url)
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=REQUEST_TIMEOUT) as client:
        resp = client.get(url)
        resp.raise_for_status()
    log.info("RSS feed fetched (%d bytes)", len(resp.text))
    return resp.text


def parse_rss_items(xml_content: str) -> list[dict]:
    """Extract article items from RSS XML."""
    root = ET.fromstring(xml_content)
    # RSS 2.0: rss/channel/item
    channel = root.find("channel")
    if channel is None:
        # Try direct items under rss
        channel = root

    items = []
    for i, elem in enumerate(channel.findall("item")):
        def _t(tag: str) -> str:
            el = elem.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        def _a(attr: str) -> str:
            el = elem.find(attr)
            return el.text.strip() if el is not None and el.text else ""

        title = _t("title")
        link = _t("link")
        pub_date = _t("pubDate")
        description = _t("description")
        guid = _t("guid") or _t("id") or link

        items.append({
            "index": i + 1,
            "id": guid,
            "title": title,
            "link": link,
            "pub_date": pub_date,
            "description": description,
        })

    log.info("Parsed %d items from RSS feed", len(items))
    return items


# ── Article Page Fetching ──────────────────────────────────────────────────


def fetch_article(url: str, retries: int = 3) -> Optional[str]:
    """Fetch a WeChat article page. Retries on failure."""
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(
                headers=HEADERS, follow_redirects=True, timeout=REQUEST_TIMEOUT
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()

            html_text = resp.text

            # Verify we got real content, not a captcha page
            if "环境异常" in html_text and "rich_media_content" not in html_text:
                log.warning("  Blocked by CAPTCHA (attempt %d/%d)", attempt, retries)
                if attempt < retries:
                    time.sleep(REQUEST_DELAY * 2)
                    continue
                return None

            return html_text

        except Exception as e:
            log.warning("  Fetch error (attempt %d/%d): %s", attempt, retries, e)
            if attempt < retries:
                time.sleep(REQUEST_DELAY)
                continue
            return None
    return None


# ── Content Extraction ─────────────────────────────────────────────────────


def extract_title(html_text: str) -> str:
    """Extract article title from WeChat page."""
    m = re.search(
        r'<h1[^>]*id=["\']activity-name["\'][^>]*>.*?<span[^>]*class=["\']js_title_inner["\'][^>]*>(.*?)</span>',
        html_text, re.DOTALL
    )
    if m:
        return clean_text(m.group(1))
    # Fallback: og:title
    m = re.search(r'<meta\s+property=["\']og:title["\'][^>]*content=["\']([^"\']+)', html_text)
    if m:
        return m.group(1)
    # Fallback: <title> tag
    m = re.search(r'<title>(.*?)</title>', html_text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return "Untitled"


def extract_author(html_text: str) -> str:
    """Extract author/nickname from WeChat page."""
    # Try og:article:author first
    m = re.search(r'<meta\s+property=["\']og:article:author["\'][^>]*content=["\']([^"\']+)', html_text)
    if m:
        return m.group(1)
    # Try profile nickname
    m = re.search(
        r'rich_media_meta_nickname[^>]*>.*?<a[^>]*>.*?<strong[^>]*>(.*?)</strong>',
        html_text, re.DOTALL
    )
    if m:
        return clean_text(m.group(1))
    # Try js_profile_name
    m = re.search(r'id=["\']js_profile_name["\'][^>]*>(.*?)</', html_text, re.DOTALL)
    if m:
        return clean_text(m.group(1))
    return ""


def extract_content_html(html_text: str) -> Optional[str]:
    """Extract the raw HTML of the #js_content div."""
    # Find the start of js_content div
    m = re.search(
        r'<div[^>]*id=["\']js_content["\'][^>]*>(.*?)</div>\s*</div>\s*</div>\s*</div>\s*$',
        html_text, re.DOTALL
    )
    if m:
        return m.group(1)

    # More flexible: find rich_media_content closing
    m = re.search(
        r'<div[^>]*class=["\'][^"\']*rich_media_content[^"\']*["\'][^>]*id=["\']js_content["\'][^>]*>(.*?)</div>\s*</div>\s*</div>\s*</div>',
        html_text, re.DOTALL
    )
    if m:
        return m.group(1)

    # Even more flexible
    m = re.search(
        r'<div[^>]*id=["\']js_content["\'][^>]*>(.*?)</div>\s*</div>',
        html_text, re.DOTALL
    )
    if m:
        # Verify we didn't get something too short
        if len(m.group(1)) > 100:
            return m.group(1)

    return None


def clean_text(text: str) -> str:
    """Clean up whitespace and HTML entities from text."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    # Decode common HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    return text


def extract_publish_date(html_text: str) -> str:
    """Extract publish date from WeChat page."""
    # Try og:article:published_time
    m = re.search(r'<meta\s+property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)', html_text)
    if m:
        return m.group(1)
    # Try em (publish time) pattern
    m = re.search(r'id=["\']publish_time["\'][^>]*>(.*?)</em>', html_text)
    if m:
        return clean_text(m.group(1))
    return ""


def _process_element(el, indent=0) -> str:
    """Recursively process an lxml element and return markdown text."""
    tag = el.tag
    if tag is None or isinstance(tag, str) and tag.startswith("_"):
        return ""

    # Skip invisible elements
    style = (el.get("style") or "").lower()
    if "display:none" in style.replace(" ", ""):
        return ""

    # Block-level elements
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        text = clean_text(el.text_content())
        if text:
            prefix = "#" * int(tag[1])
            return f"\n{prefix} {text}\n"
        return ""

    if tag == "p":
        text = _process_inline_children(el).strip()
        if text:
            return f"\n{text}\n"
        return ""

    if tag in ("div", "section", "article", "main", "header", "footer"):
        # Container - process children
        result = ""
        for child in el:
            result += _process_element(child, indent)
        return result

    if tag in ("ul", "ol"):
        items = []
        for li in el.xpath("./li"):
            text = _process_inline_children(li).strip()
            if text:
                items.append(text)
        if items:
            return "\n" + "\n".join(f"- {item}" for item in items) + "\n"
        return ""

    if tag == "blockquote":
        text = _process_inline_children(el).strip()
        if text:
            return f"\n> {text}\n"
        return ""

    if tag == "hr":
        return "\n---\n"

    if tag == "br":
        return "\n"

    if tag == "img":
        src = el.get("data-src") or el.get("src") or ""
        alt = el.get("alt", "")
        if src:
            if alt:
                return f"\n![{alt}]({src})\n"
            else:
                return f"\n![]({src})\n"
        return ""

    if tag == "figure":
        # process img and figcaption inside
        result = ""
        for child in el:
            result += _process_element(child)
        return result

    # Inline elements get processed by _process_inline_children
    return _process_inline_children(el)


def _process_inline_children(el) -> str:
    """Process inline children of an element, returning plain text with markdown formatting."""
    parts = []
    # Process text before first child
    if el.text:
        text = clean_text(el.text)
        if text:
            parts.append(text)
    # Process children
    for child in el:
        tag = child.tag
        if tag is None:
            continue
        if tag in ("strong", "b"):
            text = clean_text(child.text_content())
            if text:
                parts.append(f"**{text}**")
        elif tag in ("em", "i"):
            text = clean_text(child.text_content())
            if text:
                parts.append(f"*{text}*")
        elif tag in ("a",):
            href = child.get("href", "")
            text = clean_text(child.text_content())
            if text and href:
                parts.append(f"[{text}]({href})")
            elif text:
                parts.append(text)
        elif tag == "span":
            text = clean_text(child.text_content())
            if text:
                parts.append(text)
        elif tag == "br":
            parts.append("\n")
        elif tag == "img":
            src = child.get("data-src") or child.get("src") or ""
            alt = child.get("alt", "")
            if src:
                if alt:
                    parts.append(f"![{alt}]({src})")
                else:
                    parts.append(f"![]({src})")
        elif tag == "code":
            text = clean_text(child.text_content())
            if text:
                parts.append(f"`{text}`")
        else:
            # Recurse for unknown inline elements
            inner = _process_inline_children(child)
            if inner:
                parts.append(inner)
        # Tail text
        if child.tail:
            tail = clean_text(child.tail)
            if tail:
                parts.append(tail)
    return "".join(parts)


def html_to_markdown(html_content: str, title: str) -> str:
    """
    Convert WeChat article HTML content to clean markdown.
    Strips inline styles, keeps structure (headings, paragraphs, images, etc.)
    """
    if not html_content or not html_content.strip():
        return ""

    try:
        doc = lxml_html.fromstring(html_content)
    except Exception:
        # Fallback: regex-based cleaning
        return html_to_markdown_fallback(html_content, title)

    # Remove invisible elements
    for el in doc.xpath("//*[contains(@style, 'display:none') or contains(@style, 'display: none')]"):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)

    # Remove script and style elements
    for tag in ("script", "style", "svg", "iframe", "noscript"):
        for el in doc.xpath(f"//{tag}"):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

    # Process top-level children only
    result = ""
    for child in doc:
        result += _process_element(child)

    # Clean up excessive blank lines
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    result = result.strip()

    # If lxml extraction was too sparse, fallback
    if len(result) < 50:
        return html_to_markdown_fallback(html_content, title)

    return result


def html_to_markdown_fallback(html_content: str, title: str) -> str:
    """
    Fallback regex-based HTML to text conversion.
    Handles the WeChat article HTML format with inline styles.
    """
    text = html_content

    # Remove script and style blocks
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)

    # Replace headings
    for i in range(6, 0, -1):
        # <h1 ...>text</h1> -> ## text
        text = re.sub(
            rf'<h{i}[^>]*>(.*?)</h{i}>',
            lambda m: '\n' + '#' * i + ' ' + clean_text(
                re.sub(r'<[^>]+>', '', m.group(1))
            ) + '\n',
            text, flags=re.DOTALL
        )

    # Replace images: <img ... data-src="url" ...> -> ![alt](url)
    def _replace_img(m):
        attrs = m.group(1)
        src = ''
        alt = ''
        ms = re.search(r'data-src=["\']([^"\']+)', attrs)
        if ms:
            src = ms.group(1)
        else:
            ms = re.search(r'src=["\']([^"\']+)', attrs)
            if ms:
                src = ms.group(1)
        ms = re.search(r'alt=["\']([^"\']*)', attrs)
        if ms:
            alt = ms.group(1)
        if src:
            return f'\n![{alt}]({src})\n'
        return ''

    text = re.sub(r'<img\s+([^>]*?)\s*/?>', _replace_img, text)

    # Replace paragraphs
    text = re.sub(r'<p[^>]*>(.*?)</p>', lambda m: '\n' + clean_text(
        re.sub(r'<[^>]+>', '', m.group(1))
    ) + '\n', text, flags=re.DOTALL)

    # Replace <br> with newline
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

    # Replace <a> with [text](url)
    text = re.sub(r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                  r'[\2](\1)', text, flags=re.DOTALL)

    # Replace <strong>/<b> with **text**
    text = re.sub(r'<(strong|b)[^>]*>(.*?)</\1>', r'**\2**', text, flags=re.DOTALL)

    # Replace <em>/<i> with *text*
    text = re.sub(r'<(em|i)[^>]*>(.*?)</\1>', r'*\2*', text, flags=re.DOTALL)

    # Replace <li> with - item
    text = re.sub(r'<li[^>]*>(.*?)</li>', lambda m: '- ' + clean_text(
        re.sub(r'<[^>]+>', '', m.group(1))
    ) + '\n', text, flags=re.DOTALL)

    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    text = text.replace('&ldquo;', '"').replace('&rdquo;', '"')
    text = text.replace('&lsquo;', "'").replace('&rsquo;', "'")
    text = re.sub(r'&#\d+;', '', text)

    # Clean up excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text.strip()


def sanitize_filename(title: str, index: int) -> str:
    """Create a safe filename from article title."""
    # Remove or replace invalid filename characters
    name = re.sub(r'[\\/:*?"<>|]', '', title)
    name = re.sub(r'\s+', '_', name)
    name = name.strip('._')
    # Limit length
    if len(name) > 120:
        # Try to keep meaning by cutting at a natural boundary
        name = name[:117]
    if not name:
        name = f"article_{index:03d}"
    return name


def build_markdown_file(title: str, author: str, pub_date: str,
                        source_url: str, description: str,
                        content_md: str) -> str:
    """Build the complete markdown file content."""
    lines = [
        "---",
        f"title: \"{title}\"",
    ]
    if author:
        lines.append(f"author: \"{author}\"")
    if pub_date:
        lines.append(f"date: {pub_date}")
    lines.append(f"source: {source_url}")
    lines.extend(["", f"# {title}", ""])
    if description:
        lines.append(f"> {description}")
        lines.append("")

    if content_md:
        lines.append(content_md)
    else:
        lines.append("*（文章内容获取失败）*")

    return "\n".join(lines)


# ── Main Pipeline ──────────────────────────────────────────────────────────


def crawl_all_articles(items: list[dict], output_dir: Path) -> list[dict]:
    """Crawl all articles and save to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = []
    total = len(items)
    success = 0
    failed = 0

    for item in items:
        idx = item["index"]
        title = item["title"]
        url = item["link"]
        pub_date = item["pub_date"]
        desc = item["description"]

        log.info("[%d/%d] %s", idx, total, title[:60])

        # Fetch article page
        html_text = fetch_article(url)
        if html_text is None:
            log.warning("  FAILED to fetch article")
            # Save minimal file
            content_md = f"*（无法访问文章页面）*\n\n原文链接：{url}"
            failed += 1
        else:
            # Extract metadata
            page_title = extract_title(html_text)
            author = extract_author(html_text)
            page_pub_date = extract_publish_date(html_text) or pub_date

            # Extract content
            content_html = extract_content_html(html_text)
            if content_html:
                content_md = html_to_markdown(content_html, page_title)
                log.info("  Content extracted: %d chars → %d chars md",
                         len(content_html), len(content_md))
            else:
                log.warning("  No content found in page")
                content_md = ""

            # If we got a better title from the page, use it
            effective_title = page_title if page_title and page_title != "Untitled" else title
            effective_date = page_pub_date or pub_date

            # Build markdown
            md_content = build_markdown_file(
                title=effective_title,
                author=author,
                pub_date=effective_date,
                source_url=url,
                description=desc,
                content_md=content_md,
            )

            # Save file
            filename = f"{idx:03d}_{sanitize_filename(effective_title, idx)}.md"
            filepath = output_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_content)

            metadata_list.append({
                "id": item["id"],
                "index": idx,
                "title": effective_title,
                "author": author,
                "pub_date": effective_date,
                "source_url": url,
                "description": desc[:200] if desc else "",
                "filename": filename,
                "filepath": str(filepath.relative_to(output_dir.parent)),
            })

            success += 1
            log.info("  Saved: %s", filename)

        # Polite delay between requests
        time.sleep(REQUEST_DELAY)

    log.info("=" * 50)
    log.info("Crawl complete: %d success, %d failed out of %d", success, failed, total)
    return metadata_list


def save_metadata(metadata_list: list[dict], output_dir: Path):
    """Save metadata index JSON."""
    metadata = {
        "account_name": "自适应光学 Research",
        "account_description": "自适应光学 Research - WeChat public account covering Adaptive Optics research",
        "rss_source": RSS_URL,
        "total_articles": len(metadata_list),
        "crawl_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "articles": metadata_list,
    }

    filepath = output_dir / "_metadata.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log.info("Metadata saved: %s", filepath)


def main():
    print("=" * 60)
    print("  WeChat RSS Crawler - 自适应光学 Research")
    print("=" * 60)
    print()

    # Step 1: Fetch RSS feed
    log.info("Step 1: Fetching RSS feed...")
    rss_xml = fetch_rss(RSS_URL)

    # Step 2: Parse items
    log.info("Step 2: Parsing RSS items...")
    items = parse_rss_items(rss_xml)
    print(f"\n  Found {len(items)} articles in RSS feed\n")

    # Step 3: Crawl each article
    log.info("Step 3: Crawling articles...")
    metadata_list = crawl_all_articles(items, DATASET_DIR)

    # Step 4: Save metadata
    log.info("Step 4: Saving metadata index...")
    save_metadata(metadata_list, DATASET_DIR)

    # Summary
    print()
    print("=" * 60)
    print("  CRAWL COMPLETE")
    print(f"  Account: 自适应光学 Research")
    print(f"  Directory: {DATASET_DIR}")
    print(f"  Articles: {len(metadata_list)}")
    print(f"  Metadata: {DATASET_DIR / '_metadata.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
