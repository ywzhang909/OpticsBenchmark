# Optis Benchmark - TODO & Roadmap

> 项目待办事项和开发路线图

---

## 项目状态

| 状态 | 说明 |
|------|------|
| `v0.1.0-alpha` | 基础框架搭建完成 |
| `v0.2.0-alpha` | 功能完善中 (当前) |
| `v1.0.0` | 正式发布 (目标) |

---

## 版本路线图

### v0.1.0-alpha (已完成)

- [x] 项目结构设计
- [x] 核心模块实现 (Agent, Evaluator, Runner)
- [x] 配置文件系统
- [x] 基础 CLI 接口
- [x] Local 环境
- [x] ZOS-API 环境 (基础)
- [x] 基础文档

### v0.2.0-alpha (当前版本)

- [x] 支持 OpenAI GPT (Chat Completions + Responses API)
- [x] 支持 Anthropic Claude (官方 SDK)
- [x] 支持 Google Gemini (Interactions API)
- [x] 支持 Moonshot Kimi (OpenAI 兼容)
- [x] 支持 Zhipu GLM (OpenAI SDK)
- [x] 支持 DeepSeek (OpenAI 兼容)
- [x] 支持 Alibaba Qwen (OpenAI 兼容)
- [x] 支持 Meta Llama (Together AI)
- [x] 支持 Mistral (官方 SDK)
- [x] 支持 Ollama (本地推理)
- [x] LLM Judge 评测实现
- [x] 结构化输出支持 (OpenAI Responses API)
- [x] 报告生成器 (HTML/Markdown)
- [x] Fine-tuning 支持 (OpenAI API) — 数据转换 / 任务管理 / 状态监控
- [ ] 完善 ZOS-API 环境
  - [ ] 镜头加载/保存
  - [ ] MTF 分析
  - [ ] 光线追迹
  - [ ] 优化循环
- [ ] 单元测试覆盖完善

### v0.3.0-alpha

- [ ] 支持更多光学软件
  - [ ] CODE V 集成
  - [ ] ASAP 集成
- [ ] 多 Agent 协作评测
- [ ] 动态难度调整

### v0.5.0-beta

- [ ] 发布测试数据集 v1
- [ ] 性能优化
- [ ] 错误处理完善
- [ ] 完整文档

### v1.0.0 (目标)

- [ ] 稳定 API
- [ ] 完整测试
- [ ] 社区数据集
- [ ] 官方网站
- [ ] 公开排行榜

---

## 已知问题

| Issue | 描述 | 优先级 | 状态 |
|-------|------|--------|------|
| pyproject.toml entry point | `pyproject.toml` 引用 `src.main:main` 但实际入口是 `src/llm_pred.py` | 中 | 待修复 |
| ZOS-API stub | `ZOSAPIEnvironment` 高层方法返回占位数据 | 低 | 待实现 |
| CI/CD 缺失 | 无 GitHub Actions 工作流 | 中 | 待创建 |

---

## 功能待办

### 核心功能

- [ ] **Evaluator 扩展**
  - [ ] 多指标权重配置
  - [ ] 置信区间计算
  - [ ] 统计显著性检验

- [ ] **Runner 优化**
  - [ ] 断点续传
  - [ ] 增量评测
  - [ ] 分布式执行
  - [ ] 资源限制

### 环境集成

- [ ] **ZOS-API 完善**
  - [ ] 公差分析
  - [ ] 多波长 MTF
  - [ ] 非序列追迹
  - [ ] CAD 导出

- [ ] **新环境**
  - [ ] CODE V 环境
  - [ ] ASAP 环境
  - [ ] Python 光学库环境 (POPPY, raytracing)
  - [ ] Docker 隔离环境

### 数据集

- [ ] **核心数据集**
  - [ ] 镜头设计数据集 v1 (100+ 样本)
  - [ ] 系统分析数据集 v1 (150+ 样本)
  - [ ] 公差分析数据集 v1 (50+ 样本)

- [ ] **扩展数据集**
  - [ ] 多谱段数据集 (SWIR, MWIR, LWIR)
  - [ ] 显微镜物镜数据集
  - [ ] 摄影镜头数据集
  - [ ] 激光光学数据集

---

## 文档待办

- [x] README (中英双语)
- [x] 贡献指南
- [x] 评测理论
- [ ] 光学基础 (docs/foundation/optical-basics.md)
- [ ] 智能体理论 (docs/foundation/agent-theory.md)
- [ ] 评测方法论 (docs/foundation/evaluation.md)
- [ ] API 参考
- [ ] 部署指南
- [ ] FAQ

---

## 技术待办

### 代码质量

- [ ] 单元测试覆盖 > 80%
- [ ] 类型提示完整
- [ ] 文档字符串完整
- [ ] Ruff 检查通过

### CI/CD

- [ ] GitHub Actions 工作流
  - [ ] 单元测试
  - [ ] 代码质量检查
  - [ ] 自动部署文档

### 性能

- [ ] 评测吞吐量 > 10 tasks/min
- [ ] 内存占用优化
- [ ] 并发模型优化

---

## 长期愿景

### 生态系统

- [ ] Optis Hub - 插件市场
- [ ] Optis Studio - Web 界面
- [ ] Optis Leaderboard - 公开排行榜

### 研究方向

- [ ] 多模态光学理解
- [ ] 光学设计知识图谱
- [ ] 主动学习数据标注
- [ ] 跨域知识迁移

---

## 开发计划 (按优先级)

### P0 - 必须完成

1. [ ] 完善 ZOS-API 集成
2. [ ] 发布核心数据集
3. [ ] 完整测试覆盖

### P1 - 重要

4. [ ] CODE V 环境
5. [ ] 报告生成优化
6. [ ] 错误处理完善
7. [ ] 文档完善

### P2 - 增强

8. [ ] Docker 支持
9. [ ] 分布式执行
10. [ ] 结果可视化

### P3 - 未来

11. [ ] Optis Studio
12. [ ] 主动学习
13. [ ] 多模态支持

---

## 需要帮助

如果您想贡献，请查看以下标签的 Issue：

| 标签 | 说明 | 难度 |
|------|------|------|
| `good first issue` | 适合新手的入门任务 | 低 |
| `help wanted` | 需要帮助 | 中 |
| `documentation` | 文档相关 | 低 |
| `testing` | 测试相关 | 中 |

---

*最后更新: 2026-08-19*
