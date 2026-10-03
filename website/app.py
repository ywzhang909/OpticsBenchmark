"""
OptiS 文献处理工作台 — Streamlit 应用入口

功能：
- 侧边栏配置 LLM 模型（预设 / 自定义，支持保存与连接测试）
- Tab 1 文献信息提取：上传 PDF/TXT → 提取 13 个结构化字段（JSON）
- Tab 2 论文评审：上传 PDF/TXT → 生成结构化评审意见（Markdown）
- Tab 3 评估报告：上传 PDF/TXT → Rubric LLM-as-Judge 评估报告（Markdown）
- Tab 4 自由问答：基于论文文本的对话问答

运行（在仓库根目录执行）::

    .venv/bin/python -m streamlit run website/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# 确保 src 可导入（无论从哪个目录启动）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from website.llm_client import (  # noqa: E402
    MODEL_TYPES,
    PROVIDER_PRESETS,
    PROVIDER_TYPES,
    LLMClient,
    LLMConfig,
    discover_preset_configs,
    load_llm_config_from_yaml,
    save_llm_config_to_yaml,
)
from website.services import (  # noqa: E402
    MAX_DOC_CHARS,
    eval_paper_report,
    extract_paper_info,
    load_document,
    review_paper,
)

_CONFIG_DIR = _PROJECT_ROOT / "website" / "configs"
_CUSTOM_CONFIG = _CONFIG_DIR / "custom_llm.yaml"

#: 侧边栏配置组件的默认值（仅首次运行写入 session_state）
_CONFIG_DEFAULTS: dict[str, object] = {
    "cfg_provider_type": "openai",
    "cfg_model_type": "qwen",
    "cfg_model_name": "gpt-4o",
    "cfg_api_key": "",
    "cfg_base_url": "https://api.openai.com/v1",
    "cfg_temperature": 0.0,
    "cfg_max_tokens": 4096,
}

st.set_page_config(
    page_title="OptiS 文献处理工作台",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 OptiS 文献处理工作台")
st.caption("调用 `src/` 工具函数 + 可配置 LLM 模型，处理上传的 PDF 文献与评审材料")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _client_key(config: LLMConfig) -> str:
    """session_state 中客户端缓存的 key（按配置签名区分）。"""
    return f"llm_client::{config.signature()}"


def get_client(config: LLMConfig) -> LLMClient:
    """获取（或创建）与配置对应的 :class:`LLMClient`。"""
    key = _client_key(config)
    if key not in st.session_state:
        st.session_state[key] = LLMClient(config)
    return st.session_state[key]


def reset_clients() -> None:
    """关闭并清理所有缓存的 LLM 客户端。"""
    for key in list(st.session_state.keys()):
        if key.startswith("llm_client::"):
            client = st.session_state.pop(key)
            try:
                client.close()
            except Exception:
                pass


def show_usage(result) -> None:
    """展示 token / 成本 / 延迟信息。"""
    usage = getattr(result, "usage", {}) or {}
    total = usage.get("total_tokens") or usage.get("completion_tokens", 0)
    st.caption(f"Tokens: {total} | Cost: ${result.cost:.4f} | Latency: {result.latency:.1f}s")


# ---------------------------------------------------------------------------
# 侧边栏：LLM 模型配置
# ---------------------------------------------------------------------------


def render_sidebar() -> LLMConfig:
    """渲染侧边栏配置，返回当前生效的 :class:`LLMConfig`。"""
    for key, value in _CONFIG_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.sidebar.header("⚙️ LLM 模型配置")

    # 预设来源：仓库 configs/llm/*.yaml + 内置预设
    presets = discover_preset_configs(_PROJECT_ROOT / "configs" / "llm")
    preset_labels = ["— 不使用预设 —"] + list(presets.keys()) + list(PROVIDER_PRESETS.keys())

    def _apply_preset(cfg: LLMConfig) -> None:
        st.session_state["cfg_provider_type"] = cfg.provider_type
        st.session_state["cfg_model_type"] = cfg.model_type
        st.session_state["cfg_model_name"] = cfg.model_name
        st.session_state["cfg_api_key"] = cfg.api_key
        st.session_state["cfg_base_url"] = cfg.base_url
        st.session_state["cfg_temperature"] = cfg.temperature
        st.session_state["cfg_max_tokens"] = cfg.max_tokens
        reset_clients()

    preset = st.sidebar.selectbox("预设配置", preset_labels, key="preset_select")
    if preset.startswith("—"):
        pass
    elif preset in presets:
        _apply_preset(presets[preset])
    elif preset in PROVIDER_PRESETS:
        _apply_preset(LLMConfig(**PROVIDER_PRESETS[preset]))

    with st.sidebar.form("llm_config_form", clear_on_submit=False):
        provider_type = st.selectbox("Provider", PROVIDER_TYPES, key="cfg_provider_type")
        model_type = st.selectbox("模型类型", MODEL_TYPES, key="cfg_model_type")
        model_name = st.text_input(
            "模型名称",
            key="cfg_model_name",
            help="例如 gpt-4o / deepseek-chat / qwen-plus / llama3.1",
        )
        api_key = st.text_input(
            "API Key",
            type="password",
            key="cfg_api_key",
            help="云端服务必填；本地 Ollama 可留空。支持 ${ENV_VAR} 引用",
        )
        base_url = st.text_input(
            "Base URL / Host",
            key="cfg_base_url",
            help="OpenAI 兼容端点；Ollama 填 http://localhost:11434",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, step=0.1, key="cfg_temperature")
        max_tokens = st.number_input("Max Tokens", 256, 32768, step=256, key="cfg_max_tokens")
        submitted = st.form_submit_button("✅ 应用配置")

    if submitted:
        reset_clients()
        st.sidebar.success("配置已应用")

    config = LLMConfig(
        provider_type=provider_type,
        model_type=model_type,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("💾 保存配置", use_container_width=True):
            save_llm_config_to_yaml(_CUSTOM_CONFIG, config)
            st.sidebar.success(f"已保存到 {_CUSTOM_CONFIG.relative_to(_PROJECT_ROOT)}")
    with col_b:
        if st.button("📂 加载", use_container_width=True):
            if _CUSTOM_CONFIG.is_file():
                try:
                    _apply_preset(load_llm_config_from_yaml(_CUSTOM_CONFIG))
                    st.sidebar.success("已加载自定义配置")
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"加载失败: {exc}")
            else:
                st.sidebar.warning("暂无已保存的自定义配置")

    if st.sidebar.button("🔌 连接测试", use_container_width=True):
        problems = config.validate()
        if problems:
            st.sidebar.error("；".join(problems))
        else:
            with st.sidebar:
                with st.spinner("正在测试连接..."):
                    result = get_client(config).chat(
                        [{"role": "user", "content": "请回复：连接成功"}]
                    )
            if result.ok:
                st.sidebar.success(f"✅ 连接成功（{result.content[:40]}）")
            else:
                st.sidebar.error(f"❌ 连接失败: {result.error}")

    return config


# ---------------------------------------------------------------------------
# Tab 通用：文件上传 + 文本提取
# ---------------------------------------------------------------------------


def document_uploader(tab_key: str, label: str) -> tuple[str | None, bool | None]:
    """渲染文件上传组件，返回 ``(文本, 是否截断)``（未上传时为 ``None``）。"""
    uploaded = st.file_uploader(
        label,
        type=["pdf", "txt", "md"],
        key=f"uploader_{tab_key}",
        help="支持 PDF（PyPDF2 提取）与纯文本文件",
    )
    if uploaded is None:
        return None, None
    try:
        text, truncated = load_document(uploaded.getvalue(), uploaded.name)
    except Exception as exc:
        st.error(f"❌ 文件解析失败: {exc}")
        return None, None
    st.success(f"✅ 已读取文件 `{uploaded.name}`（{len(text):,} 字符）")
    if truncated:
        st.warning(
            f"⚠️ 文档超过 {MAX_DOC_CHARS:,} 字符，已截断（仅前 {MAX_DOC_CHARS:,} 字符送入模型）"
        )
    return text, truncated


def download_button(key: str, content: str, filename: str, mime: str, label: str) -> None:
    """下载按钮。"""
    st.download_button(
        label,
        data=content.encode("utf-8"),
        file_name=filename,
        mime=mime,
        key=f"download_{key}",
    )


# ---------------------------------------------------------------------------
# Tab 1：文献信息提取
# ---------------------------------------------------------------------------


def render_tab_extract(config: LLMConfig) -> None:
    st.subheader("📄 文献信息提取")
    st.caption(
        "从上传的论文 PDF/TXT 中提取 13 个结构化字段"
        "（标题、年份、DOI、期刊、关键词、作者、摘要、目标、创新点、方法、性能指标等）"
    )
    text, truncated = document_uploader("extract", "上传论文文件（PDF / TXT）")
    if text is None:
        return

    if st.button("🚀 开始提取", type="primary", key="btn_extract"):
        problems = config.validate()
        if problems:
            st.error("模型配置不完整：" + "；".join(problems))
            return
        with st.spinner("正在调用 LLM 提取结构化字段..."):
            data, result = extract_paper_info(text, get_client(config))
        if result.error:
            st.error(f"❌ 调用失败: {result.error}")
            return
        show_usage(result)
        if data is None:
            st.warning(
                "⚠️ 模型未返回有效的 JSON，已展示原始输出。可尝试降低 temperature 或更换模型。"
            )
            st.text(result.content)
            download_button(
                "extract_raw",
                result.content,
                "extraction_raw.txt",
                "text/plain",
                "⬇️ 下载原始输出",
            )
            return
        st.json(data)
        download_button(
            "extract_json",
            json.dumps(data, ensure_ascii=False, indent=2),
            "paper_info_extraction.json",
            "application/json",
            "⬇️ 下载 JSON 结果",
        )

    with st.expander("📖 查看提取的论文文本"):
        st.text_area("论文文本", text, height=300, key="extract_text_view")


# ---------------------------------------------------------------------------
# Tab 2：论文评审
# ---------------------------------------------------------------------------


def render_tab_review(config: LLMConfig) -> None:
    st.subheader("📝 论文评审")
    st.caption(
        "按 `prompts/templates/paper_review.txt` 模板对论文进行多维度评审"
        "（摘要、方法论、实验、创新性、写作），输出带评分的 Markdown 评审意见"
    )
    text, _ = document_uploader("review", "上传评审材料（PDF / TXT）")
    if text is None:
        return

    if st.button("🚀 开始评审", type="primary", key="btn_review"):
        problems = config.validate()
        if problems:
            st.error("模型配置不完整：" + "；".join(problems))
            return
        with st.spinner("正在调用 LLM 生成评审意见..."):
            review_md, result = review_paper(text, get_client(config))
        if result.error:
            st.error(f"❌ 调用失败: {result.error}")
            return
        show_usage(result)
        st.markdown(review_md)
        download_button(
            "review_md",
            review_md,
            "paper_review.md",
            "text/markdown",
            "⬇️ 下载评审意见（Markdown）",
        )


# ---------------------------------------------------------------------------
# Tab 3：评估报告
# ---------------------------------------------------------------------------


def render_tab_eval(config: LLMConfig) -> None:
    st.subheader("📊 评估报告")
    st.caption(
        "解析文档字段后用 Rubric LLM-as-Judge（`src/evaluators/rubric_based_evaluator.py`）"
        "按准确性 / 完整性 / 可读性三个维度评分并检测幻觉，生成含图表的 Markdown 报告"
    )
    text, _ = document_uploader("eval", "上传评估材料（PDF / TXT）")
    if text is None:
        return

    offline = st.checkbox(
        "离线模式（不调用 LLM 评委，得分为 0）",
        value=False,
        key="eval_offline",
        help="仅演示报告格式；在线模式将使用侧边栏配置的模型作为评委",
    )
    if st.button("🚀 生成评估报告", type="primary", key="btn_eval"):
        problems = config.validate()
        if problems and not offline:
            st.error("模型配置不完整：" + "；".join(problems))
            return
        judge_config = None if offline else config.to_judge_config()
        with st.spinner("正在解析字段并执行 Rubric 评估..."):
            parsed, metrics, report_md = eval_paper_report(
                text,
                judge_config=judge_config,
                lang="zh_CN",
                source_file="uploaded-document",
            )
        st.success("✅ 评估完成")
        cols = st.columns(4)
        cols[0].metric("准确性 Accuracy", f"{metrics.get('accuracy', 0):.2f}")
        cols[1].metric("完整性 Completeness", f"{metrics.get('completeness', 0):.2f}")
        cols[2].metric("可读性 Readability", f"{metrics.get('readability', 0):.2f}")
        cols[3].metric("幻觉率 Hallucination", f"{metrics.get('hallucination_rate', 0):.2f}")
        st.divider()
        st.markdown(report_md)
        download_button(
            "eval_md",
            report_md,
            "paper_eval_report.md",
            "text/markdown",
            "⬇️ 下载评估报告（Markdown）",
        )


# ---------------------------------------------------------------------------
# Tab 4：自由问答
# ---------------------------------------------------------------------------


def render_tab_qa(config: LLMConfig) -> None:
    st.subheader("💬 自由问答")
    st.caption("基于论文文本（或任意粘贴内容）与配置的 LLM 对话")
    text, _ = document_uploader("qa", "上传论文文件（可选，用于提供上下文）")
    default_doc = text or ""
    doc_text = st.text_area(
        "论文文本（上下文）",
        value=default_doc,
        height=200,
        key="qa_doc_text",
        help="可手动粘贴任意文本",
    )
    question = st.text_input("你的问题", key="qa_question")
    if st.button("🚀 提问", type="primary", key="btn_qa"):
        if not question.strip():
            st.warning("请输入问题")
            return
        problems = config.validate()
        if problems:
            st.error("模型配置不完整：" + "；".join(problems))
            return
        messages: list[dict[str, str]] = []
        if doc_text.strip():
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "你是学术文献分析助手。请基于用户提供的论文文本回答以下问题。"
                        "如果问题无法从文本中回答，请明确说明。"
                    ),
                }
            )
            messages.append({"role": "user", "content": f"论文文本：\n{doc_text[:MAX_DOC_CHARS]}"})
            messages.append({"role": "user", "content": f"问题：{question}"})
        else:
            messages.append({"role": "user", "content": question})
        with st.spinner("正在思考..."):
            result = get_client(config).chat(messages)
        if result.error:
            st.error(f"❌ 调用失败: {result.error}")
            return
        show_usage(result)
        st.markdown(result.content)
        download_button(
            "qa_answer",
            result.content,
            "qa_answer.md",
            "text/markdown",
            "⬇️ 下载回答",
        )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    config = render_sidebar()
    tab_extract, tab_review, tab_eval, tab_qa = st.tabs(
        ["📄 文献信息提取", "📝 论文评审", "📊 评估报告", "💬 自由问答"]
    )
    with tab_extract:
        render_tab_extract(config)
    with tab_review:
        render_tab_review(config)
    with tab_eval:
        render_tab_eval(config)
    with tab_qa:
        render_tab_qa(config)

    st.divider()
    st.caption(
        f"数据来源：`src/` 工具函数 | 项目根目录：`{_PROJECT_ROOT}` | "
        "修改配置后请重新点击「应用配置」"
    )


if __name__ == "__main__":
    main()
