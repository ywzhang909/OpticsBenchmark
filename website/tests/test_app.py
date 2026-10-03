"""AppTest — Streamlit 应用验证测试（离线，不发起网络请求）。

使用 `streamlit.testing.v1.AppTest` 验证 app.py 的 UI 渲染、
侧边栏配置、Tab 切换、文件上传、按钮交互等。
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

#: 默认的 AppTest 运行超时（秒）
TIMEOUT = 30

#: 上传用的样例文档（AO 分析格式，可被 parse_ao_analysis 解析）
SAMPLE_DOC = """====== 文献调研辅助工具 - 分析报告 ======

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


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _get_tab(at: AppTest, keyword: str):
    """根据 subheader 关键词定位 Tab。"""
    for tab in at.tabs:
        for h in tab.subheader:
            if keyword in h.value:
                return tab
    return None


def _upload_to_tab(at: AppTest, keyword: str, name: str = "sample.txt"):
    """在匹配 ``keyword`` 的 Tab 中上传文件并重跑，返回刷新后的 AppTest。

    注意：``run()`` 会重建整个元素树，上传前捕获的 ``tab`` 引用随即失效，
    调用方必须重新经 ``at`` 取元素，否则读到的仍是上一次的旧快照。
    """
    tab = _get_tab(at, keyword)
    assert tab is not None, f"Tab '{keyword}' not found"
    tab.file_uploader[0].set_value([(name, SAMPLE_DOC.encode("utf-8"), "text/plain")]).run()
    return at


# ---------------------------------------------------------------------------
# 基础渲染
# ---------------------------------------------------------------------------


class TestAppRender:
    """应用启动时的静态 UI 验证。"""

    def test_no_exception(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        assert not at.exception

    def test_title(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        assert len(at.title) == 1
        assert "OptiS 文献处理工作台" in at.title[0].value

    def test_tabs_count(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        assert len(at.tabs) == 4

    def test_sidebar_has_selectbox(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        assert len(at.sidebar.selectbox) >= 1

    def test_has_caption(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        found = any("调用" in c.value or "处理" in c.value for c in at.caption)
        assert found, "App should render a caption description"


# ---------------------------------------------------------------------------
# 侧边栏配置
# ---------------------------------------------------------------------------


class TestSidebarConfig:
    """侧边栏 LLM 配置交互验证。"""

    def test_form_submit_applies_config(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        apply_btn = next((b for b in at.sidebar.button if "应用" in b.label), None)
        assert apply_btn is not None
        apply_btn.click().run()
        assert not at.exception

    def test_save_config_button_exists(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        labels = [b.label for b in at.sidebar.button]
        assert any("保存" in label for label in labels), f"Buttons: {labels}"

    def test_load_config_button_exists(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        labels = [b.label for b in at.sidebar.button]
        assert any("加载" in label for label in labels), f"Buttons: {labels}"

    def test_connection_test_button_exists(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        labels = [b.label for b in at.sidebar.button]
        assert any("连接" in label for label in labels), f"Buttons: {labels}"

    def test_temperature_slider_default(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        assert len(at.sidebar.slider) >= 1
        assert at.sidebar.slider[0].value == 0.0

    def test_temperature_slider_change(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        at.sidebar.slider[0].set_value(0.5).run()
        assert at.sidebar.slider[0].value == 0.5
        assert not at.exception

    def test_max_tokens_number_input(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        assert len(at.sidebar.number_input) >= 1
        assert at.sidebar.number_input[0].value == 4096

    def test_provider_selectbox_has_options(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        selectboxes = at.sidebar.selectbox
        # 第一个是"预设配置"，第二个是"Provider"
        assert len(selectboxes) >= 2
        assert len(selectboxes[0].options) >= 2

    def test_text_input_api_key_exists(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        text_inputs = at.sidebar.text_input
        assert len(text_inputs) >= 2  # API Key + Base URL
        api_key_input = next((t for t in text_inputs if "API Key" in t.label), None)
        assert api_key_input is not None


# ---------------------------------------------------------------------------
# Tab 1 - 文献信息提取
# ---------------------------------------------------------------------------


class TestTabExtract:
    """Tab 1 文献信息提取的 UI 验证。"""

    def test_extract_tab_has_subheader(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        tab = _get_tab(at, "提取")
        assert tab is not None
        assert any("提取" in h.value for h in tab.subheader)

    def test_extract_tab_has_caption(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        tab = _get_tab(at, "提取")
        assert len(tab.caption) >= 1

    def test_extract_tab_has_file_uploader(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        tab = _get_tab(at, "提取")
        assert len(tab.file_uploader) >= 1

    def test_extract_button_visible_after_upload(self):
        at = _upload_to_tab(AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT), "提取")
        assert not at.exception
        labels = [b.label for b in at.button]
        assert any("提取" in label for label in labels), f"Buttons: {labels}"

    def test_extract_shows_read_success_message(self):
        at = _upload_to_tab(AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT), "提取")
        assert any("已读取文件" in s.value for s in at.success)


# ---------------------------------------------------------------------------
# Tab 2 - 论文评审
# ---------------------------------------------------------------------------


class TestTabReview:
    """Tab 2 论文评审的 UI 验证。"""

    def test_review_tab_has_subheader(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        tab = _get_tab(at, "评审")
        assert tab is not None
        assert any("评审" in h.value for h in tab.subheader)

    def test_review_tab_has_file_uploader(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        tab = _get_tab(at, "评审")
        assert len(tab.file_uploader) >= 1

    def test_review_button_visible_after_upload(self):
        at = _upload_to_tab(AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT), "评审")
        assert not at.exception
        labels = [b.label for b in at.button]
        assert any("评审" in label for label in labels), f"Buttons: {labels}"


# ---------------------------------------------------------------------------
# Tab 3 - 评估报告
# ---------------------------------------------------------------------------


class TestTabEval:
    """Tab 3 评估报告的 UI 验证。"""

    def test_eval_tab_has_subheader(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        tab = _get_tab(at, "评估")
        assert tab is not None
        assert any("评估" in h.value for h in tab.subheader)

    def test_eval_tab_has_file_uploader(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        tab = _get_tab(at, "评估")
        assert len(tab.file_uploader) >= 1

    def test_eval_offline_checkbox_and_report(self):
        at = _upload_to_tab(AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT), "评估")
        assert not at.exception

        offline = next((c for c in at.checkbox if "离线" in c.label), None)
        assert offline is not None, f"Checkboxes: {[c.label for c in at.checkbox]}"
        assert offline.value is False
        offline.check().run()

        btn = next(b for b in at.button if "评估" in b.label)
        btn.click().run(timeout=TIMEOUT)

        assert not at.exception
        assert any("评估完成" in s.value for s in at.success)
        metrics = [m.label for m in at.metric]
        assert any("Accuracy" in label for label in metrics), f"Metrics: {metrics}"


# ---------------------------------------------------------------------------
# Tab 4 - 自由问答
# ---------------------------------------------------------------------------


class TestTabQA:
    """Tab 4 自由问答的 UI 验证。"""

    def test_qa_tab_has_subheader(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        tab = _get_tab(at, "问答")
        assert tab is not None
        assert any("问答" in h.value for h in tab.subheader)

    def test_qa_tab_has_text_area(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        tab = _get_tab(at, "问答")
        assert len(tab.text_area) >= 1

    def test_qa_button_exists(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        tab = _get_tab(at, "问答")
        labels = [b.label for b in tab.button]
        assert any("提问" in label for label in labels), f"Buttons: {labels}"

    def test_qa_text_input_question(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        tab = _get_tab(at, "问答")
        text_inputs = tab.text_input
        question_input = next((t for t in text_inputs if "问题" in t.label), None)
        assert question_input is not None


# ---------------------------------------------------------------------------
# 跨 Tab 完整性
# ---------------------------------------------------------------------------


class TestCrossTab:
    """多个 Tab 间的通用性验证。"""

    def test_all_tabs_have_subheader(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        for tab in at.tabs:
            assert len(tab.subheader) >= 1, "Tab appears empty"

    def test_all_tabs_render_cleanly(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        assert not at.exception

    def test_app_has_divider(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        assert len(at.divider) >= 1

    def test_session_state_populated(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        state = at.session_state
        assert "cfg_provider_type" in state
        assert "cfg_model_name" in state
        assert state["cfg_provider_type"] == "openai"
        assert state["cfg_model_name"] == "gpt-4o"

    def test_session_state_defaults(self):
        at = AppTest.from_file(str(APP_PATH)).run(timeout=TIMEOUT)
        state = at.session_state
        assert state["cfg_base_url"] == "https://api.openai.com/v1"
        assert state["cfg_temperature"] == 0.0
        assert state["cfg_max_tokens"] == 4096
