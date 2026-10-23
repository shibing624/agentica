# Agentica 是什么

**Agentica** 既是一套开箱即用的本地 AI 智能体工作台（CLI 终端、本地 Web、桌面 App、手机微信/IM 遥控），也是一个轻量、模块化的 Python Agent SDK。

## 两种使用方式，共享同一套状态

你可以直接把它当现成工具用，也可以把它嵌进自己的 Python 代码里：

| 使用方式 | 适合谁 | 怎么启动 |
|---|---|---|
| **CLI 终端 (`agentica`)** | 日常写代码、查 Bug、重构、跑测试 | `uv tool install agentica` → `agentica` |
| **本地 Web / 桌面端 (`agentica-gateway`)** | 喜欢图形界面、查看可视化执行轨迹、多账号管理、接入微信/飞书/企微/Telegram | `uv tool install "agentica[gateway]"` → `agentica-gateway`（或直接装 Desktop App） |
| **Python SDK (`from agentica import Agent`)** | 在自己的业务服务、数据处理脚本或研究项目里构建智能体 | `uv add agentica` |

在默认配置下，CLI、Web 和桌面端共用同一个 `~/.agentica` 目录：你在终端里配好的模型（`config.yaml`）、积累的长期记忆（`MEMORY.md` / `AGENTS.md`）、安装的技能（`skills/`）和聊过的会话历史，打开网页或桌面端都在。

## 为什么选择 Agentica？

- **不绑定单一模型厂商** — 支持 OpenAI、Claude、DeepSeek、智谱（含免费模型 `glm-4.7-flash`）、通义千问、Kimi、火山引擎豆包、Ollama 本地模型等，终端里敲 `/model` 随时切换，还能给子任务单配便宜的 `auxiliary_model`，给线上配 `fallback_models` 自动容灾。
- **长对话不失忆、不爆窗口** — 超大命令输出自动存文件，窗口吃紧时自动清理早期工具结果，满窗时自动把目标和进度整理进 `<session>.notes.md` 并开启新窗口，原始对话随时可通过 `search_session` 查回。
- **多会话像团队一样配合** — 进程内可并行派发只读子任务（`task`），大任务可交给独立后台进程（`delegate`），多个开着的终端可以互相看状态、发消息（`/list-agents`、`/send-message`），共用同一个仓库时还能用 `/worktree` 自动隔离分支。
- **人离开电脑也能管** — 开启 Gateway 后，在微信、飞书、企微或 Telegram 里发 `@list` 就能看到家里或开发机上跑着的终端会话，发 `@会话名 指令` 直接把话敲进那个终端。
- **代码简洁、异步优先** — Python SDK 5 行代码即可跑起一个 Agent，所有接口同时提供 `async`（`await agent.run()`）与 `sync`（`agent.run_sync()`），只读工具自动并发执行。

## 5 行代码跑起第一个 Agent

```python
from agentica import Agent, ZhipuAI

agent = Agent(model=ZhipuAI())
result = agent.run_sync("Hello!")
print(result.content)
```

需要全套文件读写、命令执行、网页搜索、长期记忆和自动压缩时，直接用 `DeepAgent`：

```python
from agentica import DeepAgent, OpenAIChat

agent = DeepAgent(model=OpenAIChat(id="gpt-4o"), work_dir="./my-project")
result = agent.run_sync("检查这个项目的测试为什么失败，并给出修复方案")
print(result.content)
```

## 与其他工具/框架的对比

| 特性 | Agentica | Claude Code / Codex CLI | LangChain / CrewAI |
|------|----------|-------------------------|--------------------|
| **模型选择** | 多厂商自由切换 + 本地 Ollama | 绑定单一厂商 | 多厂商 |
| **开箱即用 CLI + Web + Desktop** | ✅ 三端共用本地状态 | 仅 CLI / 专属客户端 | ❌ 需自行开发前端 |
| **微信 / 飞书 / 企微手机直连** | ✅ 内置 Gateway，无需公网 IP | ❌ | ❌ |
| **跨终端多会话协作** | ✅ `task` + `delegate` + peer 消息 | 仅会话内子任务 | 仅代码内编排 |
| **Python SDK 可嵌入** | ✅ 原生 async/sync，极简面向对象 API | 部分或无 | ✅（但抽象层较重） |
| **开源协议** | Apache 2.0 | 部分开源 | MIT / Apache |

## 下一步

- [安装与配置](../getting-started/installation.md) — 安装 CLI、Web 或 SDK，配好第一个 API Key
- [CLI 终端指南](../getting-started/terminal.md) — 终端交互、斜杠命令与多任务用法
- [5 分钟上手 (SDK)](../getting-started/quickstart.md) — 用 Python 编写带工具和结构化输出的 Agent
- [架构与内部实现（开发者选读）](architecture.md) — 深入了解执行循环与上下文压缩的底层机制
