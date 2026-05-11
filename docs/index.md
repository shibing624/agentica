# Agentica

**轻量级、功能强大的 Python 框架，用于构建自主 AI 智能体。**

<div align="center">
  <img src="https://github.com/shibing624/agentica/raw/main/docs/assets/logo.png" height="30" alt="Logo">
</div>

[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![Downloads](https://static.pepy.tech/badge/agentica)](https://pepy.tech/project/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/shibing624/agentica/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-green.svg)](https://github.com/shibing624/agentica)

---

## 为什么选择 Agentica?

- **Async-First 架构** -- 所有核心方法原生 async，同步适配器无缝兼容。
- **开发者友好** -- 简洁直观的面向对象 API，极低学习曲线。
- **模块化可扩展** -- 模型、记忆后端、向量存储均可自由替换。
- **功能完备** -- 内置 40+ 工具、RAG、多智能体团队、工作流编排、安全守卫。
- **生产就绪** -- CLI / Web UI / API 服务多种部署方式，支持 MCP 与 ACP 协议。

## 系统架构

<div align="center">
  <img src="https://github.com/shibing624/agentica/raw/main/docs/assets/architecturev2.jpg" alt="Architecture" width="600" />
</div>

## 快速导航

| 文档 | 内容 |
|------|------|
| **入门** | |
| [What is Agentica](introduction/what-is-agentica.md) | 项目定位与核心特性 |
| [架构总览](introduction/architecture.md) | 五层架构与数据流 |
| [安装](getting-started/installation.md) | 安装与环境配置 |
| [快速入门](getting-started/quickstart.md) | 5 分钟上手 |
| [CLI 终端](getting-started/terminal.md) | 命令行交互模式 |
| **核心概念** | |
| [Agent](concepts/agent.md) | Agent 核心概念、Model、Memory、Tools |
| [Model](concepts/model.md) | 20+ 模型提供商 |
| [Tools](concepts/tools.md) | 内置工具与自定义工具 |
| [Memory & Workspace](concepts/memory.md) | 记忆系统与工作空间 |
| [Knowledge (RAG)](concepts/rag.md) | 知识库、向量检索 |
| **多智能体** | |
| [选择编排模式](multi-agent/choosing.md) | as_tool、Workflow、Subagent、Swarm 决策树 |
| [Workflow](multi-agent/workflow.md) | 确定性工作流编排 |
| [Swarm](multi-agent/swarm.md) | 自主多智能体协作 |
| [Subagent](multi-agent/subagent.md) | 子任务委派 |
| **高级功能** | |
| [Hooks](advanced/hooks.md) | 生命周期钩子 |
| [RunConfig](advanced/run-config.md) | 运行时配置（超时、成本、白名单） |
| [Guardrails](advanced/guardrails.md) | 4 层安全守卫 |
| [Context Compression](advanced/compression.md) | 上下文压缩 |
| [Skills](advanced/skills.md) | Markdown Skill 系统 |
| [Daily Tasks](advanced/daily-tasks.md) | 定时任务、失败可见性与运行历史 |
| [MCP](advanced/mcp.md) | Model Context Protocol |
| [ACP](advanced/acp.md) | Agent Client Protocol |
| **参考** | |
| [模型提供商](guides/models.md) | 全部模型配置指南 |
| [最佳实践](guides/best_practices.md) | 设计原则与生产部署 |
| [Agent API](api/agent.md) | 完整 API 参考 |
| [依赖分层 RFC](rfcs/dependency-layering.md) | `agentica-core` / extras 的延后方案 |

## 30 秒上手

```bash
pip install -U agentica
export ZAI_API_KEY="your-api-key"
```

```python
import asyncio
from agentica import Agent, ZhipuAI

async def main():
    agent = Agent(model=ZhipuAI())
    result = await agent.run("一句话介绍北京")
    print(result.content)

asyncio.run(main())
```

```
北京是中国的首都，是一座拥有三千多年历史的文化名城，也是全国的政治、文化和国际交流中心。
```

[查看完整快速入门 ->](getting-started/quickstart.md)

## 社区与支持

- **GitHub Issues** -- [提交 issue](https://github.com/shibing624/agentica/issues)
- **微信交流群** -- 添加微信号 `xuming624`，备注 "llm"

## 许可证

[Apache License 2.0](https://github.com/shibing624/agentica/blob/main/LICENSE)
