# What is Agentica

**Agentica** 是一个轻量级、功能强大的 Python 框架，用于构建自主 AI 智能体。

## 一句话定义

> Async-first Python Agent Framework -- 用最少的代码构建能思考、决策和执行动作的 AI 智能体。

## 定位

| 维度 | 说明 |
|------|------|
| **类型** | Python Agent Framework |
| **架构** | Async-first, 同步适配器无缝兼容 |
| **设计哲学** | 开发者友好，面向对象 API，极低学习曲线 |
| **模型支持** | 20+ LLM 提供商，本地/云端模型 |
| **核心能力** | Agent, Tools, RAG, Workflow, Swarm, Subagent, MCP/ACP |

## 核心特性

- **Async-First 架构** -- 所有核心方法原生 async，同步适配器无缝兼容
- **开发者友好** -- 简洁直观的面向对象 API，5 行代码创建一个 Agent
- **模块化可扩展** -- 模型、记忆后端、向量存储均可自由替换
- **功能完备** -- 内置 40+ 工具、RAG、多智能体协作（Swarm/Subagent/Workflow；CLI 另有 `task`/`delegate`/peer）、工作流编排、安全守卫
- **生产就绪** -- CLI / Web UI / API 服务多种部署方式
- **协议支持** -- MCP (Model Context Protocol) + ACP (Agent Client Protocol)
- **安全机制** -- Death Spiral 检测、Cost Budget、Guardrails 三层守卫

## 快速上手

```python
import asyncio
from agentica import Agent, ZhipuAI

async def main():
    agent = Agent(model=ZhipuAI())
    result = await agent.run("Hello!")
    print(result.content)

asyncio.run(main())
```

## 与其他框架的对比

| 特性 | Agentica | LangChain | CrewAI | OpenAI Agents SDK |
|------|----------|-----------|--------|-------------------|
| **架构** | Async-first | Sync-first | Sync | Async |
| **学习曲线** | 低 | 高 | 中 | 低 |
| **模型支持** | 20+ 提供商 | 多 | 少 | 仅 OpenAI |
| **Tool 系统** | 函数/类/MCP | Chain/Tool | Tool | Function |
| **多智能体** | as_tool/Swarm/Subagent/Workflow | Agent Executor | Crew | Handoff |
| **RAG** | 内置完整 | 需组合 | 无 | 无 |
| **安全守卫** | 4 层 Guardrails | 无 | 无 | 无 |
| **IDE 集成** | ACP 协议 | 无 | 无 | 无 |

## Agentica 不是什么

- **不是 LLM 本身** -- Agentica 是框架，不提供模型推理能力
- **不是 LangChain 替代品** -- 虽然功能有重叠，但设计哲学不同（简洁 vs 抽象）
- **不是低代码平台** -- 面向开发者的 Python 框架，不是拖拽式 UI
- **不是 RAG 专用工具** -- RAG 是众多能力之一，不是核心定位

## 下一步

- [架构总览](architecture.md) -- 理解 Agentica 的技术架构
- [安装指南](../getting-started/installation.md) -- 开始安装和配置
- [快速入门](../getting-started/quickstart.md) -- 5 分钟上手
