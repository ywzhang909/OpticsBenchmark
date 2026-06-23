# OptiS Benchmark — 数据集目录

本目录包含光学设计智能体评测所需的全部数据集。

## 目录结构

```
dataset/
├── paper_info_extract/           # 论文信息提取任务数据集
│   ├── dataset_json/             # JSON 标注数据
│   │   ├── dataset_v1.json      # 15 篇光学论文的评测样本
│   │   └── gold_answer_v1.json  # 13 字段结构化 gold answer
│   └── data_v1/                 # 原始 PDF 论文文件（15 篇）
├── info_extraction/              # 信息抽取附加数据
│   └── AO/                      # 自适应光学论文分析文本（160+ 篇）
├── paper_review/                 # 论文审稿任务（待完善）
│   ├── dataset_json/            # （空）
│   └── data_v1/                 # （空）
├── optics_question_answer/       # 光学问答任务（待完善）
└── README.md                     # 本文件
```

## 数据集说明

### paper_info_extract — 论文信息提取（就绪）

- **任务**: 从光学科学论文中提取 13 个结构化字段（标题、作者、摘要、方法、指标等）
- **数据量**: 15 篇论文 / 15 个评测样本
- **格式**: JSON（标注数据）+ PDF（原始论文）
- **状态**: ✅ 可用

### info_extraction/AO — 自适应光学分析（就绪）

- **任务**: 自适应光学（Adaptive Optics）领域论文的 AI 生成分析文本
- **数据量**: 160+ 篇分析文件（中英混合）
- **格式**: TXT
- **状态**: ✅ 可用

### paper_review — 论文审稿（待完善）

- **任务**: 对光学论文进行学术审稿并给出评分与建议
- **状态**: ⏳ 数据集待生成

### optics_question_answer — 光学问答（待完善）

- **任务**: 光学领域知识问答评测
- **状态**: ⏳ 数据集待生成

## 数据格式

### paper_info_extract JSON 格式

```json
{
  "paper_id": "paper_001",
  "title": "Achromatic flat optical components via compensation...",
  "fields": {
    "authors": ["Zhang, Y.", "Li, X."],
    "abstract": "We demonstrate...",
    "method": "Compensation between structure and material dispersions",
    "metrics": {
      "efficiency": 0.85,
      "bandwidth": "400-700nm"
    }
  }
}
```

## 数据准备

```bash
# 自动下载（如数据集托管在远程仓库）
bash scripts/download_data.sh

# 或手动将数据放置到对应子目录
```

## 贡献新数据集

1. 在 `dataset/` 下创建任务命名的子目录
2. 确保数据符合任务 config 中定义的格式规范
3. 提供 JSON 标注文件 + 原始输入文件
4. 配置 `configs/tasks/<task_name>.yaml` 中的数据集路径
5. 提交 Pull Request

## 许可证

数据集采用 CC BY-SA 4.0 许可证。
