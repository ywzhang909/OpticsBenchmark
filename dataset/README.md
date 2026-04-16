# OptiS Benchmark - Dataset Directory

This directory contains the评测数据集 for optical design agent evaluation.

## 目录结构

```
dataset/
├── raw/                    # 原始数据（通过脚本下载）
├── processed/              # 处理后的标准数据（JSONL 格式）
│   ├── lens_design.jsonl
│   ├── system_analysis.jsonl
│   └── ...
├── README.md               # 本文件
└── download.sh            # 数据下载脚本
```

## 数据格式

所有数据集使用 JSONL（JSON Lines）格式，每行是一个有效的 JSON 对象。

### 标准字段

```json
{
  "task_id": "lens_001",
  "instruction": "设计一个焦距为50mm的消色差双合透镜...",
  "expected_output": {
    "lens_type": "achromat_doublet",
    "focal_length": 50.0,
    "materials": ["N-BK7", "F2"],
    "mtf_target": 0.7
  },
  "metadata": {
    "difficulty": 3,
    "category": "lens_design",
    "estimated_time": 15
  }
}
```

## 数据集说明

### lens_design.jsonl
- **任务类型**: 镜头设计与优化
- **评测指标**: MTF 性能、像差、公差分析
- **数据量**: 50 个评测样本

### system_analysis.jsonl
- **任务类型**: 光学系统性能分析
- **评测指标**: 分析完整性、指标准确性
- **数据量**: 75 个评测样本

## 数据准备

### 自动下载

```bash
chmod +x download_data.sh
./scripts/download_data.sh
```

### 手动下载

1. 访问数据集仓库
2. 下载处理后的 JSONL 文件
3. 放置到 `dataset/processed/` 目录

## 贡献数据集

欢迎贡献新的评测数据集！

1. 确保数据格式符合上述标准
2. 包含足够的元信息（难度、类别等）
3. 提供预期输出的ground truth
4. 提交 Pull Request

## 许可证

数据集采用 CC BY-SA 4.0 许可证。
