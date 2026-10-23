# Agentica 使用文档

**一个人，一支 agent 团队。**
CLI 终端、本机 Web、Desktop App 与 Python SDK 共用同一套记忆、模型配置与工具，跑在你自己的机器上。

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/logo.png" height="30" alt="Logo">
</div>

[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![Downloads](https://static.pepy.tech/badge/agentica)](https://pepy.tech/project/agentica)
[![GitHub stars](https://img.shields.io/github/stars/shibing624/agentica?style=social)](https://github.com/shibing624/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/shibing624/agentica/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://github.com/shibing624/agentica)

---

## 你想怎么用 Agentica？

| 你的目标 | 从哪里开始 |
|---|---|
| **在终端里写代码、查问题、跑长任务** | 1. [安装与配置](getting-started/installation.md)（`uv tool install agentica`）<br>2. [CLI 终端指南](getting-started/terminal.md)（`@文件`、`/model`、`/resume`、`/goal`、`/worktree`） |
| **用网页 / 桌面端，或在手机微信里遥控电脑上的会话** | [Web、桌面端与微信/IM](advanced/gateway.md)（`uv tool install "agentica[gateway]"` → `agentica-gateway`） |
| **在自己的 Python 项目里嵌入 Agent** | 1. [5 分钟上手 (SDK)](getting-started/quickstart.md)（`uv add agentica`）<br>2. [Agent 与 DeepAgent](concepts/agent.md) · [最佳实践配方](guides/best_practices.md) |
| **了解内部架构与设计原因（开发者选读）** | [架构与内部实现](introduction/architecture.md) · [内置工具设计判据](concepts/builtin-tool-bar.md) |

---

## 30 秒体验

### 方式一：交互式 CLI

```bash
uv tool install agentica
export ZAI_API_KEY="your-api-key"   # 或 OPENAI_API_KEY / DEEPSEEK_API_KEY / ANTHROPIC_API_KEY
agentica
```

### 方式二：Python SDK

```bash
uv add agentica
```

```python
from agentica import Agent, ZhipuAI

agent = Agent(model=ZhipuAI())
result = agent.run_sync("一句话介绍北京")
print(result.content)
```

---

## 文档导航

### 快速上手

| 文档 | 内容 |
|------|------|
| [Agentica 是什么](introduction/what-is-agentica.md) | 产品形态（CLI / Web / Desktop / SDK）与核心能力 |
| [安装与配置](getting-started/installation.md) | 安装 CLI / Web / SDK，配置各家模型 API Key |
| [5 分钟上手 (SDK)](getting-started/quickstart.md) | 同步/异步调用、流式输出、绑定工具、结构化输出 |
| [CLI 终端指南](getting-started/terminal.md) | 命令行参数、`@文件` 引用、斜杠命令、多任务协作、系统沙箱 |
| [Web、桌面端与微信/IM](advanced/gateway.md) | 本地网页、Desktop App、微信扫码接入、手机遥控本机 CLI 会话 |

### 核心用法 (Python SDK)

| 文档 | 内容 |
|------|------|
| [Agent 与 DeepAgent](concepts/agent.md) | 轻量 `Agent` 与全功能预设 `DeepAgent`、多轮对话、会话恢复 |
| [模型与提供商](concepts/model.md) | 切换模型、设置 `auxiliary_model` 副模型与 `fallback_models` 兜底链 |
| [工具与自定义工具](concepts/tools.md) | 内置工具一览、用普通 Python 函数或 `@tool` 编写自定义工具、切换搜索引擎 |
| [记忆与常驻规则](concepts/memory.md) | 会话历史、跨会话长期记忆（`MEMORY.md`）与常驻偏好规则（`AGENTS.md`） |
| [知识库 (RAG)](concepts/rag.md) | 本地文档解析、向量检索与重排序 |
| [存储与数据库](concepts/storage.md) | SQLite / Postgres 会话存储与媒体管理 |

### 多智能体与协作

| 文档 | 内容 |
|------|------|
| [如何选择协作方式](multi-agent/choosing.md) | 一张图选对 `task`、`delegate`、peer、`as_tool`、Workflow、Swarm |
| [Subagent (子任务)](multi-agent/subagent.md) | 内置只读子智能体与 `.agentica/agents/*.md` 自定义子智能体 |
| [Workflow (确定性流水线)](multi-agent/workflow.md) | 固定步骤编排，混合 Python 函数与 Agent 调用 |
| [Swarm (多智能体自治)](multi-agent/swarm.md) | 多智能体并行分工与自主协作 |
| [多会话共用仓库 (Worktree)](multi-agent/worktrees.md) | 多个会话同时改同一个仓库时的分支隔离与状态可见 |

### 进阶功能

| 文档 | 内容 |
|------|------|
| [长任务循环 (/goal)](advanced/goals.md) | 让 Agent 自动持续推进直到目标完成，支持 Token 与时间预算 |
| [上下文压缩 (/compact)](advanced/compression.md) | 长会话自动清理旧工具输出、满窗自动交接笔记、用 `search_session` 查回历史 |
| [技能系统 (Skills)](advanced/skills.md) | 用 `SKILL.md` 扩展领域工作流，安装外部技能包 |
| [定时任务 (/cron)](guides/cron_scheduler.md) | 按计划自动运行 Agent 任务（CLI 内嵌 / 独立后台 / Web 管理） |
| [运行预算与控制 (RunConfig)](advanced/run-config.md) | 单次调用超时、最大费用预算、工具与技能白名单 |
| [安全护栏 (Guardrails)](advanced/guardrails.md) | 输入、输出与工具参数拦截校验 |
| [生命周期钩子 (Hooks)](advanced/hooks.md) | 在 Agent 开始/结束、工具调用前后、压缩前后插入自定义逻辑 |
| [MCP 外部工具集成](advanced/mcp.md) | 接入标准 MCP 工具服务（stdio / SSE / HTTP） |
| [ACP (IDE 集成)](advanced/acp.md) | 接入 Zed、JetBrains 等支持 ACP 协议的编辑器 |

### 配置与参考

| 文档 | 内容 |
|------|------|
| [统一配置文件 (config.yaml)](guides/config.md) | `~/.agentica/config.yaml` 完整示例与字段速查 |
| [模型提供商速查](guides/models.md) | OpenAI、Claude、DeepSeek、智谱、通义、Kimi、豆包、Ollama 配置示例 |
| [OpenAI Responses API](guides/openai-responses.md) | `OpenAIResponses` 与 `wire_api: responses` 配置说明 |
| [最佳实践与常见配方](guides/best_practices.md) | 5 个即拷即用的 Agent 配方与常见坑排查 |
| [Agent API 参数表](api/agent.md) | `Agent`、`PromptConfig`、`ToolConfig`、`RunResponse` 完整字段表 |
| [评测对比 (Benchmark)](guides/benchmark.md) | 与 Codex CLI 在相同模型下的公开题库对比与复现命令 |

### 开发者选读（架构与协议）

这部分面向想要了解内部实现机制、参与框架开发或对接外部桌面组件的开发者，日常使用无需阅读：

- [架构与内部实现](introduction/architecture.md) — 执行循环、Prompt Cache 快照机制、压缩底层不变量、配置与审批内部实现
- [内置工具设计判据](concepts/builtin-tool-bar.md) — 什么能力该放进内置工具、CLI 与 SDK 的工具边界
- [外部进程协议：Shell Hooks](getting-started/shell-hooks.md) / [Notify Sink](getting-started/notify-sink.md) / [Attach Socket](getting-started/attach.md)

---

## 社区与支持

- **GitHub Issues** -- [提交 issue](https://github.com/shibing624/agentica/issues)
- **微信交流群** -- 添加微信号 `xuming624`，备注 "llm"

## 许可证

[Apache License 2.0](https://github.com/shibing624/agentica/blob/main/LICENSE)
