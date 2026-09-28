# Paper Info Extract Fine-tune Dataset

**Path:** `dataset/fine_tune/paper_info_extract/` — 论文信息提取任务微调数据集。

包含用于微调模型从光学论文中提取结构化元数据的数据。

## 文件说明

| 文件 | 描述 | 大小 |
|------|------|------|
| `gold_answer_v1.jsonl` | 标准答案数据（JSONL 格式） | ~100KB |
| `test.jsonl` | 测试数据（200 条样本） | ~87KB |

## 数据格式

每个 JSONL 文件包含多行 JSON 对象，每行格式如下：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an optics assistant. Extract structured metadata from the given paper."
    },
    {
      "role": "user",
      "content": "paper title: <论文标题>"
    },
    {
      "role": "assistant",
      "content": "{\"title\": \"<论文标题>\", \"publication_year\": \"<年份>\", \"doi\": \"<DOI>\"}"
    }
  ]
}
```

## 字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| `title` | string | 论文标题 |
| `publication_year` | string | 发表年份 |
| `doi` | string | 数字对象标识符（DOI） |

## 使用示例

```bash
# 查看数据集样本
head -n 5 dataset/fine_tune/paper_info_extract/test.jsonl

# 统计数据条数
wc -l dataset/fine_tune/paper_info_extract/test.jsonl

# 验证数据格式
python -c "
import json
with open('dataset/fine_tune/paper_info_extract/test.jsonl') as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        assert 'messages' in data, f'Line {i}: missing messages'
        assert len(data['messages']) == 3, f'Line {i}: expected 3 messages'
print('Data format is valid')
"
```

## 关联配置

| 配置文件 | 路径 |
|----------|------|
| OpenAI 微调配置 | `configs/fine_tuning/GPT_OpenAI_finetune.yaml` |
| DashScope 微调配置 | `configs/fine_tuning/qwen_dashscope.yaml` |
| 测试微调配置 | `configs/finetune/gpt4o_mini_test.yaml` |
