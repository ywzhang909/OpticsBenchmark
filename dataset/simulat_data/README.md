# dataset/simulat_data — 模拟评分数据

用于 `src/utils/` 下一致性评估与偏见评估脚本的模拟数据，评分均为 0–5 量纲（同一量纲，无需归一化）。

## 两种数据形态

**一致性评估**：一份文件一位裁判，12 个样本各一条，共 12 条，`id` 唯一。

```json
{"id": 1, "output_model": "gpt-4o", "context": "薄透镜近轴成像公式……", "judge_model": "gpt-4o", "score": 1.4}
```

**偏见评估**：一份文件一位裁判对整个 `n × m` 网格的打分，共 `m * n` 条，`id` 重复，每个模型对同一样本各一条。

```json
{"id": 1, "output_model": "deepseek-v3", "context": "薄透镜近轴成像公式……", "judge_model": "gpt-4o", "score": 2.37}
```

两条形态的记录字段一致：

- `id`:样本编号；评估按 `id` 配对，偏见评估再按 `output_model` 区分同一样本的多个模型输出
- `output_model` / `context` / `judge_model`:元数据。两个评估脚本都用它们做交叉校验（同一裁判出现在多份文件、同一 `id` 的 `output_model` 或 `context` 不一致时告警）
- `score`:标量

`eval_bias.py` 的裁判名默认取自记录中唯一的 `judge_model`（可用 `--judge-model` 覆盖），因此裁判必须同时是被评的 m 个模型之一，否则无法区分自身输出与他人输出。

- **样本数**:每个文件 12 个样本（id = 1–12），≥ 10 条
- **生成方式**:`score` 由固定随机种子 `numpy.random.default_rng(42)` 生成，含小幅噪声、无重复 `(id, output_model)`、无无定义样本（偏见评估 `num_undefined = 0`）；`output_model` 与 `context` 为手工指定，同一 `id` 在所有文件中完全一致（所有裁判评的是同一条输出）

## 文件说明

| 文件 | 形态 | 用途 | 裁判 / 被评模型 | 对应脚本 |
|------|------|------|------------------|----------|
| `consistency_scores_a.jsonl` | 12 条 | 裁判 A 打分，用于一致性（皮尔逊）对比 | 裁判 `gpt-4o`；被评 5 个模型 | `src/utils/eval_consistency.py` |
| `consistency_scores_b.jsonl` | 12 条 | 裁判 B 打分，与 A 中等正相关 | 裁判 `claude-3-5-sonnet`；同上 | 同上 |
| `bias_judge_gpt-4o.jsonl` | 12 × 4 = 48 条 | 单一裁判的自我偏好（`y_self` 为自身输出，`y_other` 为其余 3 个模型的均值） | 裁判 `gpt-4o`；被评 `gpt-4o` / `qwen2.5-72b-instruct` / `llama-3.3-70b-instruct` / `deepseek-v3` | `src/utils/eval_bias.py` |

两份一致性数据的预期结果为 `Pearson r = 0.9658, p = 3.4781e-07`（n = 12）。

偏见数据的裁判 `gpt-4o` 对自身输出额外抬分约 +0.5 分，预期 `mean ErrorRate_SE = 0.1494`（中位数 0.1428，12 个样本全部有定义）。

## 使用示例

```bash
# 一致性评估：两份列表的皮尔逊相关系数（路径有默认值，可省略）
python -m src.utils.eval_consistency
python -m src.utils.eval_consistency \
    --scores_a dataset/simulat_data/consistency_scores_a.jsonl \
    --scores_b dataset/simulat_data/consistency_scores_b.jsonl \
    -o results/consistency.json

# 偏见评估：单文件 48 条网格（路径有默认值，可省略）
python -m src.utils.eval_bias
python -m src.utils.eval_bias \
    --scores dataset/simulat_data/bias_judge_gpt-4o.jsonl \
    -o results/bias.json
# 裁判名与记录不一致时覆盖（会有告警）
python -m src.utils.eval_bias \
    --scores dataset/simulat_data/bias_judge_gpt-4o.jsonl \
    --judge-model deepseek-v3
```
