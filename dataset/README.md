# OptiS Benchmark — Dataset Directory

本目录包含光学设计智能体评测所需的全部数据集。当前涵盖 4 类任务数据，包括就绪和待完善两个状态。

---

## 目录结构

```
dataset/
├── paper_info_extract/           # [就绪] 论文信息提取任务
│   ├── dataset_json/
│   │   ├── dataset_v1.json      # 15 条评测样本（含论文标题 + PDF 路径）
│   │   └── gold_answer_v1.json  # 13 字段结构化标准答案（1325 行，含每篇论文的完整标注）
│   └── data_v1/                 # 15 篇原始 PDF 论文（2015–2026 年，以 metasurface/metamaterial 为主）
├── info_extraction/              # [就绪] 信息抽取扩展数据
│   └── AO/                      # 自适应光学（Adaptive Optics）领域
│       └── *_analysis.txt       # 274 篇论文的 AI 生成分析（中文 + 英文混合）
├── paper_review/                 # [待完善] 论文审稿任务
│   ├── dataset_json/            # （空，待填充 JSON 标注）
│   └── data_v1/                 # （空，待放置原始 PDF）
├── optics_question_answer/       # [待完善] 光学问答任务
│   └── (空目录)                 # 待生成
└── README.md                     # 本文件
```

---

## 数据集详情

### 1. `paper_info_extract` — 论文信息提取（✅ 就绪）

从光学科学论文中提取 13 个结构化字段，用于评估 LLM 的信息抽取能力。

| 项目 | 说明 |
|------|------|
| **样本量** | 15 篇论文 |
| **论文领域** | Metasurface、metamaterial、nanophotonics（2015–2026 年，主要来自 *Advanced Functional Materials*, *Scientific Reports*, *Nanoscale* 等期刊） |
| **抽取字段** | title, Publication Year, DOI, Journal, Ten Keywords, Authors, Contact Authors, Affiliations, Abstract, Objective, Novelty, Method, Performance Metrics |
| **输出格式** | 结构化 JSON |
| **原始文件** | 15 个 PDF（位于 `data_v1/`） |
| **标准答案** | `gold_answer_v1.json` — 每篇论文含 13 字段的手工精细标注 |

**标注示例**（节选自 `gold_answer_v1.json`）：
```json
{
  "id": 2,
  "data": {
    "title": "Synthetic waves sensor enables turbulence-resistant imaging",
    "Publication Year": "2026",
    "DOI": "10.1117/12.3092313",
    "Journal": "Proceedings of SPIE Vol. 13992 (AOMATT 2025)",
    "Ten Keywords": [
      "Atmospheric Turbulence",
      "Synthetic Wave",
      "Turbulence-Resistant Imaging",
      "3D Phase Measurement",
      "Structural Similarity Index (SSIM)",
      ...
    ],
    "Authors": ["Yuting Xiao", "Yu Luo", "Lianwei Chen", "Mingbo Pu", "Xiangang Luo"],
    "Abstract": "Atmospheric turbulence severely distorts optical wavefronts...",
    "Objective": ["Core objective: Realize real-time 3D phase measurement..."],
    "Novelty": ["Realize depth-resolved 3D phase detection along the optical axis..."],
    "Method": ["Core scheme: Split the beam to generate two coherent sub-waves..."],
    "Performance Metrics": ["Phase detection error <±0.006..."]
  }
}
```

**数据集文件**：
```json
// dataset_v1.json — 每项含 title + PDF location path
[
  {
    "title": "Synthetic waves sensor enables turbulence -resistant imaging",
    "location": ".../data_v1/13992 2R Synthetic waves sensor...pdf"
  },
  ...
]
```

**关联配置文件**：
| 文件 | 路径 |
|------|------|
| 任务配置 | `configs/tasks/paper_info_extract.yaml` |
| 评测配置 | `configs/evaluations/paper_info_extract.yaml` |
| 评测维度 | Exact Match（title, DOI, authors 等 8 字段）、ROUGE（objective, method, metrics）、BERTScore（objective, novelty, method, metrics） |

---

### 2. `info_extraction/AO` — 自适应光学分析文本（✅ 就绪）

自适应光学（Adaptive Optics）领域论文的 AI 生成分析文本集合，可用于信息抽取、文档摘要、RAG 检索等任务。

| 项目 | 说明 |
|------|------|
| **样本量** | 274 篇分析文件 |
| **领域** | 自适应光学（波前传感、波前校正、大气湍流补偿、热晕效应、变形镜、深度学习在 AO 中的应用等） |
| **语言** | 中文 + 英文混合 |
| **格式** | 纯文本（UTF-8，CRLF） |
| **命名规范** | `{作者} - {年份} - {标题}_analysis.txt` |
| **内容结构** | 每篇 txt 文件包含论文标题、作者、年份、核心方法、性能指标等关键信息的自然语言描述 |
| **来源论文跨度** | ~1977–2026 年，覆盖经典文献和最新研究 |

**文件示例**：
| 文件名 | 主题 |
|--------|------|
| `Nishizaki 等 - 2019 - Deep learning wavefront sensing_analysis.txt` | 深度学习波前传感 |
| `Wang 等 - 2021 - Deep learning wavefront sensing and aberration correction...txt` | 深度学习大气湍流校正 |
| `Vorontsov 等 - 2024 - Analysis of speckle-beacon size impact...txt` | 深湍流波前传感 |
| `Zhang 等 - 2024 - Shack-hartmann wavefront sensor based on...txt` | 衍射透镜阵列 SHWFS |
| `Zhao 等 - 2025 - Intensity adaptive optics_analysis.txt` | 强度自适应光学 |
| `吴书云 - 激光大气传输热晕效应及自适应光学校正的数值仿真和实验研究_analysis.txt` | 热晕效应（中文） |

---

### 3. `paper_review` — 论文审稿（⏳ 待完善）

对光学论文进行学术审稿并给出评分与建议。

| 项目 | 说明 |
|------|------|
| **状态** | `dataset_json/` 和 `data_v1/` 目录已创建，内容为空 |
| **计划评测方法** | ROUGE-L、内容覆盖度、连贯性、事实性 |

---

### 4. `optics_question_answer` — 光学问答（⏳ 待完善）

光学领域知识问答评测。

| 项目 | 说明 |
|------|------|
| **状态** | 目录已创建，内容为空 |
| **计划任务类型** | 光学原理、设计规范、材料特性等知识问答 |

---

## 论文领域覆盖

| 领域 | 论文数 | 来源 |
|------|--------|------|
| Metasurface / Metamaterial | 15 | `paper_info_extract/data_v1/`（PDF） |
| Adaptive Optics | 274 | `info_extraction/AO/`（分析文本） |
| **合计** | **289** | |

AO 数据集覆盖的关键主题：
- 波前传感（Shack-Hartmann、金字塔传感器、相位恢复、深度学习波前传感）
- 波前校正（SPGD、PID、强化学习、模型预测控制）
- 大气湍流补偿（热晕效应、深湍流、信标辅助）
- 变形镜（MEMS、压电、液晶空间光调制器）
- 激光通信与相干合束
- 自适应光学显微镜与生物成像
- 无波前传感器自适应光学（sensorless AO）

---

## 数据格式规范

### paper_info_extract JSON（标注数据）

```json
{
  "id": 1,
  "data": {
    "title": "论文标题",
    "Publication Year": "年份",
    "DOI": "完整 DOI 号",
    "Journal": "期刊名称",
    "Ten Keywords": ["关键词列表"],
    "Authors": ["作者列表"],
    "Contact Authors": ["通讯作者列表"],
    "Affiliations": ["机构列表"],
    "Abstract": "摘要文本",
    "Objective": ["目标列表"],
    "Novelty": ["创新点列表"],
    "Method": ["方法描述列表"],
    "Performance Metrics": ["性能指标列表"]
  }
}
```

### paper_info_extract JSON（评测样本）

```json
{
  "title": "论文标题（用于匹配）",
  "location": "PDF 文件绝对路径"
}
```

### info_extraction/AO TXT（分析文本）

纯文本文件，每篇约 500–3000 字，包含：
- 论文元信息（标题、作者、年份、期刊）
- 核心方法与创新点
- 实验设计与关键结果
- 性能指标数值

---

## 数据使用

```bash
# 查看数据集样本
uv run python -c "
import json
with open('dataset/paper_info_extract/dataset_json/dataset_v1.json') as f:
    data = json.load(f)
print(f'Samples: {len(data)}')
print(f'Papers: {[d[\"title\"][:50] for d in data]}')
"

# 查看标准答案字段
uv run python -c "
import json
with open('dataset/paper_info_extract/dataset_json/gold_answer_v1.json') as f:
    gold = json.load(f)
print(f'Gold entries: {len(gold)}')
print(f'Fields: {list(gold[0][\"data\"].keys())}')
"
```

---

## 数据准备

```bash
# 自动下载（如数据集托管在远程仓库）
bash scripts/download_data.sh

# 或手动将数据放置到对应子目录
```

---

## 贡献新数据集

1. 在 `dataset/` 下创建任务命名的子目录
2. 提供 JSON 标注文件 + 原始输入文件（如 PDF）
3. 确保数据符合任务 config 中定义的格式规范
4. 配置 `configs/tasks/<task_name>.yaml` 中的数据集路径
5. 配置 `configs/evaluations/<task_name>.yaml` 中的评测方法
6. 提交 Pull Request

---

## 许可证

数据集采用 CC BY-SA 4.0 许可证。
