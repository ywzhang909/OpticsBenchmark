# Fine-tune Dataset

**Path:** `dataset/fine_tune/` — 微调数据集目录。

包含用于微调任务的数据集文件，格式为 OpenAI 微调 JSONL 格式。

## 目录结构

```
dataset/fine_tune/
├── paper_info_extract/           # 论文信息提取任务微调数据
│   ├── gold_answer_v1.jsonl      # 标准答案数据（JSONL 格式）
│   └── test.jsonl                # 测试数据（200 条样本）
└── README.md                     # 本文件
```

## 数据格式

每个 JSONL 文件包含多行 JSON 对象，每行格式如下：

```json
{
  "messages": [
    {"role": "system", "content": "You are an optics assistant. Extract structured metadata from the given paper."},
    {"role": "user", "content": "paper title: <论文标题>"},
    {"role": "assistant", "content": "{\"title\": \"<论文标题>\", \"publication_year\": \"<年份>\", \"doi\": \"<DOI>\"}"}
  ]
}
```

## 使用示例

```bash
# 查看数据集样本
head -n 5 dataset/fine_tune/paper_info_extract/test.jsonl

# 统计数据条数
wc -l dataset/fine_tune/paper_info_extract/test.jsonl
```

## 生成数据

使用 `utils/build_finetune_dataset.py` 脚本生成微调数据：

```bash
python utils/build_finetune_dataset.py \
  -g dataset/paper_info_extract/dataset_json/gold_answer_v1.json \
  -p prompts/paper_info_extract/zero-shot_v1.0.txt \
  -d dataset/paper_info_extract/dataset_json/dataset_v1.json \
  -o results/finetune/train.jsonl --val-ratio 0.2
```
