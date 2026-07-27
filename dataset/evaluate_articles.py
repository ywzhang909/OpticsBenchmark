"""
定性评测脚本：使用在线 LLM (Qwen vLLM) 对全部25篇自适应光学文章进行逐一评测。
支持断点续评，中间结果保存在 JSON 缓存文件中。
每次运行只评测未缓存的文章，生成完整报告。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

# ── 配置 ──────────────────────────────────────────────────────────────
BASE_URL = "http://impecunious909.asia/vllm/v1/chat/completions"
API_KEY = "sk-11235813"
MODEL = "qwen"
DATASET_DIR = Path(__file__).parent / "自适应光学 Research"
OUTPUT_PATH = DATASET_DIR / "_evaluation.md"
CACHE_PATH = DATASET_DIR / "_eval_cache.json"

client = httpx.Client(proxy=None, timeout=httpx.Timeout(300.0))
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

DIMS = ["研究创新性", "技术深度", "应用价值", "表达清晰度", "领域重要性"]

SYSTEM_PROMPT = """你是一位自适应光学领域的高级专家评审。请从以下5个维度对文章进行定性评测，每项给出0-1之间的分数和简要理由。

评测维度：
1. 研究创新性 - 原创性和新颖程度
2. 技术深度 - 技术复杂度、理论严谨性和实验验证充分程度
3. 应用价值 - 在自适应光学领域及相关应用中的实际价值和推广潜力
4. 表达清晰度 - 文章结构和表达的清晰程度
5. 领域重要性 - 所研究课题在自适应光学领域中的重要性和关注度

请严格按照以下JSON格式回复，不要包含额外内容：
{
  "scores": {
    "研究创新性": 0.xx,
    "技术深度": 0.xx,
    "应用价值": 0.xx,
    "表达清晰度": 0.xx,
    "领域重要性": 0.xx
  },
  "justifications": {
    "研究创新性": "理由...",
    "技术深度": "理由...",
    "应用价值": "理由...",
    "表达清晰度": "理由...",
    "领域重要性": "理由..."
  },
  "summary": "总体评价（50-100字）",
  "strengths": ["优势1", "优势2", "优势3"],
  "weaknesses": ["不足1", "不足2"]
}"""


def read_article(path: Path) -> dict:
    """读取文章 .md 文件，解析 frontmatter 和正文。"""
    content = path.read_text(encoding="utf-8")
    frontmatter: dict[str, str] = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2].strip()
            for line in fm_text.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, val = line.split(":", 1)
                    frontmatter[key.strip()] = val.strip()
    return {
        "filename": path.name,
        "title": frontmatter.get("title", path.stem),
        "author": frontmatter.get("author", ""),
        "link": frontmatter.get("link", ""),
        "body": body,
    }


def extract_json(text: str) -> dict | None:
    """从文本中提取第一个完整 JSON 对象。"""
    if not text:
        return None
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
    return None


def call_llm(messages: list[dict]) -> dict | None:
    """调用 vLLM 端点，重试5次应对 Cloudflare 524 超时。"""
    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    max_retries = 5
    for attempt in range(max_retries):
        try:
            r = client.post(BASE_URL, json=data, headers=HEADERS)

            if r.status_code == 524:
                wait = 20 * (attempt + 1)
                print(f"    ⚠ 524 Cloudflare超时 (重试 {attempt+1}/{max_retries}, 等待{wait}s)")
                time.sleep(wait)
                continue

            if r.status_code != 200:
                print(f"    ⚠ HTTP {r.status_code} (重试 {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(15 * (attempt + 1))
                    continue
                return None

            resp = r.json()
            msg = resp["choices"][0]["message"]
            content = msg.get("content")
            reasoning = msg.get("reasoning", "")

            # 优先从 content 字段提取
            if content and content.strip():
                result = extract_json(content)
                if result:
                    return result

            # 从 reasoning 字段提取
            if reasoning:
                result = extract_json(reasoning)
                if result:
                    return result
                if len(reasoning) > 500:
                    result = extract_json(reasoning[-1500:])
                    if result:
                        return result

            print(f"    ⚠ 无法提取JSON (attempt {attempt+1})")
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            return None

        except httpx.ReadTimeout:
            print(f"    ⚠ ReadTimeout (重试 {attempt+1}/{max_retries})")
            time.sleep(20 * (attempt + 1))
            continue
        except Exception as e:
            print(f"    ⚠ 异常: {e}")
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            return None
    return None


def evaluate_article(article: dict) -> dict | None:
    """对单篇文章进行评测。"""
    body = article["body"]
    if len(body) > 6000:
        body = body[:6000] + "\n\n...（原文过长已截断）"

    user_prompt = f"""## 文章信息
标题：{article['title']}
作者：{article['author']}

## 正文内容
{body}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = call_llm(messages)
    if result is None:
        return None

    # 补全字段
    for d in DIMS:
        if d not in result.get("scores", {}):
            result.setdefault("scores", {})[d] = 0.0
        if d not in result.get("justifications", {}):
            result.setdefault("justifications", {})[d] = "未评分"

    result.setdefault("summary", "")
    result.setdefault("strengths", [])
    result.setdefault("weaknesses", [])
    result["title"] = article["title"]
    result["filename"] = article["filename"]

    return result


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_report(all_results: list) -> str:
    valid = [r for r in all_results if r is not None]

    scores_by_dim = {d: [] for d in DIMS}
    for r in valid:
        for d in DIMS:
            scores_by_dim[d].append(r["scores"].get(d, 0))

    avg_scores = {d: (sum(v) / len(v)) if v else 0 for d, v in scores_by_dim.items()}
    overall_avg = sum(avg_scores.values()) / len(DIMS)
    ranked = sorted(valid, key=lambda r: sum(r["scores"].values()), reverse=True)

    lines = [
        "# 自适应光学 Research 公众号文章定性评测报告\n",
        f"**评测模型**: {MODEL} (vLLM: cyankiwi/Qwen3.6-27B-AWQ-INT4)\n",
        f"**评测时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**文章总数**: {len(all_results)} | **评测成功**: {len(valid)} | **评测失败**: {len(all_results) - len(valid)}\n",
        f"**综合均分**: {overall_avg:.3f}\n",
    ]

    # 一、整体维度分析
    lines.extend([
        "---\n",
        "## 一、整体维度分析\n",
        "| 评测维度 | 平均分 | 最高分 | 最低分 | 解读 |\n",
        "|---|---|---|---|---|\n",
    ])
    interp = {
        "研究创新性": lambda s: "创新突出" if s >= 0.7 else ("创新一般" if s >= 0.5 else "创新不足"),
        "技术深度": lambda s: "技术扎实" if s >= 0.7 else ("深度中等" if s >= 0.5 else "深度不足"),
        "应用价值": lambda s: "应用前景好" if s >= 0.7 else ("价值一般" if s >= 0.5 else "价值有限"),
        "表达清晰度": lambda s: "表达清晰" if s >= 0.7 else ("表达一般" if s >= 0.5 else "表达欠佳"),
        "领域重要性": lambda s: "核心方向" if s >= 0.7 else ("方向一般" if s >= 0.5 else "边缘方向"),
    }
    for d in DIMS:
        vals = scores_by_dim[d]
        if vals:
            lines.append(f"| {d} | {sum(vals)/len(vals):.3f} | {max(vals):.2f} | {min(vals):.2f} | {interp[d](sum(vals)/len(vals))} |\n")

    # 二、分数分布
    lines.extend(["\n## 二、各维度分数分布\n\n", "| 维度 | 0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |\n", "|---|---|---|---|---|---|\n"])
    for d in DIMS:
        buckets = [0, 0, 0, 0, 0]
        for s in scores_by_dim[d]:
            if s <= 0.2: buckets[0] += 1
            elif s <= 0.4: buckets[1] += 1
            elif s <= 0.6: buckets[2] += 1
            elif s <= 0.8: buckets[3] += 1
            else: buckets[4] += 1
        total = len(scores_by_dim[d])
        if total > 0:
            lines.append(f"| {d} | {'/'.join(str(b) for b in buckets)} |\n")

    # 三、综合排名
    lines.extend(["\n---\n", "## 三、文章综合排名\n\n", "| 排名 | 文章标题 | 均分 | 创新性 | 技术深度 | 应用价值 | 表达清晰 | 领域重要 | 主要亮点 |\n", "|---|---|---|---|---|---|---|---|---|\n"])
    for i, r in enumerate(ranked, 1):
        s = r["scores"]
        avg = sum(s.values()) / len(DIMS)
        title_short = r["title"] if len(r["title"]) <= 42 else r["title"][:39] + "..."
        strengths = "; ".join(r.get("strengths", [])[:2]) or "-"
        lines.append(f"| {i} | {title_short} | {avg:.3f} | {s[DIMS[0]]:.2f} | {s[DIMS[1]]:.2f} | {s[DIMS[2]]:.2f} | {s[DIMS[3]]:.2f} | {s[DIMS[4]]:.2f} | {strengths} |\n")

    # 四、单篇详细评价
    lines.extend(["\n---\n", "## 四、单篇详细评价\n\n"])
    for i, r in enumerate(ranked, 1):
        lines.extend([
            f"### {i}. {r['title']}\n\n",
            f"**综合均分**: {sum(r['scores'].values()) / len(DIMS):.3f}\n\n",
            "| 维度 | 分数 | 理由 |\n",
            "|---|---|---|\n",
        ])
        for d in DIMS:
            lines.append(f"| {d} | {r['scores'].get(d, 0):.2f} | {r['justifications'].get(d, '')} |\n")
        lines.append(f"\n**总体评价**: {r.get('summary', '')}\n\n")
        lines.append("**主要优势**:\n")
        for s in r.get("strengths", []):
            lines.append(f"- {s}\n")
        lines.append("**主要不足**:\n")
        for w in r.get("weaknesses", []):
            lines.append(f"- {w}\n")
        lines.append("\n")

    return "".join(lines)


def main():
    md_files = sorted(DATASET_DIR.glob("*.md"))
    md_files = [f for f in md_files if f.name not in ("_metadata.md", "_metadata.json")]
    print(f"共发现 {len(md_files)} 篇文章")

    articles = []
    for f in md_files:
        article = read_article(f)
        articles.append(article)

    cache = load_cache()
    all_results = []

    print(f"\n开始评测（模型: {MODEL}, 缓存命中: {len(cache)} 篇）\n" + "=" * 60)

    for i, article in enumerate(articles, 1):
        fname = article["filename"]

        if fname in cache:
            print(f"\n[{i}/{len(articles)}] {article['title'][:60]}")
            print("    [缓存] 跳过")
            all_results.append(cache[fname])
            continue

        print(f"\n[{i}/{len(articles)}] {article['title'][:60]}")
        t0 = time.time()
        result = evaluate_article(article)
        elapsed = time.time() - t0

        if result:
            cache[fname] = result
            save_cache(cache)
            scores = result.get("scores", {})
            avg = sum(scores.values()) / len(scores) if scores else 0
            print(f"    OK {elapsed:.0f}s | 均分:{avg:.3f} | 创新:{scores.get(DIMS[0], 0):.2f} 深度:{scores.get(DIMS[1], 0):.2f} 价值:{scores.get(DIMS[2], 0):.2f}")
        else:
            print(f"    FAIL {elapsed:.0f}s")

        all_results.append(result)

    print(f"\n生成评测报告...")
    report = generate_report(all_results)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"报告已保存: {OUTPUT_PATH}")

    valid = [r for r in all_results if r is not None]
    print(f"\n完成: {len(valid)}/{len(all_results)} 成功")
    if valid:
        for d in DIMS:
            vals = [r["scores"].get(d, 0) for r in valid]
            print(f"  {d}: {sum(vals)/len(vals):.3f}")
        all_avgs = [sum(r["scores"].values()) / len(DIMS) for r in valid]
        print(f"  综合均分: {sum(all_avgs)/len(all_avgs):.3f}")


if __name__ == "__main__":
    main()
