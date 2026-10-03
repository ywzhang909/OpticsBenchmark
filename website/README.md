# OptiS 文献处理工作台

基于 [Streamlit](https://streamlit.io/) 的文献处理 Web 应用，复用仓库 `src/` 下的工具函数，
通过可配置的 LLM 模型处理上传的 PDF 文献与评审材料。

## 功能

| Tab | 功能 | 底层工具 |
|-----|------|----------|
| 📄 文献信息提取 | 提取论文 13 个结构化字段（标题/年份/DOI/期刊/关键词/作者/摘要/目标/创新点/方法/性能指标等），输出 JSON | `src/tools/paper_reader.read_file` + `prompts/paper_info_extract/zero-shot_v1.0.txt` |
| 📝 论文评审 | 按评审模板生成多维度评分评审意见（Markdown） | `prompts/templates/paper_review.txt` + `prompts/system/research_agent.txt` |
| 📊 评估报告 | Rubric LLM-as-Judge 三维度评分 + 幻觉检测，生成含图表的 Markdown 报告 | `src/tools/paper_eval_report`（`RubricBasedEvaluator` + `generate_report`） |
| 💬 自由问答 | 基于论文文本的对话问答 | `src/llm`（`create_provider` / `create_llm`） |

## 侧边栏 LLM 配置

- **预设配置**：一键加载 `configs/llm/*.yaml` 或内置预设（OpenAI / Qwen / DeepSeek / GLM / Claude / Gemini / Groq / Ollama / Together）
- **自定义**：Provider、模型类型、模型名称、API Key、Base URL、Temperature、Max Tokens
- **保存/加载**：自定义配置持久化到 `website/configs/custom_llm.yaml`
- **连接测试**：不发起文档处理即可验证模型连通性
- API Key 支持 `${ENV_VAR}` 引用，从环境变量读取

## 安装与运行

```bash
# 1. 安装依赖（复用仓库 .venv 或新建虚拟环境）
python -m pip install streamlit PyPDF2 PyYAML

# 2. 在仓库根目录启动（src 才能被导入）
streamlit run website/app.py
# 或
.venv/bin/python -m streamlit run website/app.py
```

浏览器访问 `http://localhost:8501`。

## 目录结构

```
website/
├── app.py            # Streamlit 入口（侧边栏配置 + 4 个 Tab）
├── llm_client.py     # LLM 客户端封装：LLMConfig / ChatResult / LLMClient
├── services.py       # 文档处理服务：加载、信息提取、评审、评估报告
├── requirements.txt  # 额外依赖
├── configs/          # 保存的自定义 LLM 配置
└── tests/            # 离线单元测试（无需网络 / API Key）
```

## 测试

```bash
python -m pytest website/tests/ -v
```

测试使用假 LLM 客户端，不发起任何网络请求。

## 说明

- 上传的 PDF 通过 PyPDF2 提取文本；无法解析（扫描件/无文本层）会给出友好错误提示
- 超长文档（> 30,000 字符）自动截断并提示
- 评估报告的「离线模式」不调用 LLM 评委，得分全为 0，仅用于预览报告格式
