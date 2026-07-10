[**🇨🇳中文**](https://github.com/shibing624/agentica/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/agentica/blob/main/README_EN.md) | [**🇯🇵日本語**](https://github.com/shibing624/agentica/blob/main/README_JP.md)

<div align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# Agentica: Build AI Agents
[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![Downloads](https://static.pepy.tech/badge/agentica)](https://pepy.tech/project/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.12%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#社区与支持)

**Agentica** 不是套一层 LLM API 的聊天壳，而是一个 Async-First 的 agent harness——让 Agent 真正跑起来：调工具、跑长任务、多智能体协作、跨会话记忆，并持续自我进化。

| 它凭什么不一样 | |
|------|------|
| **跑得久，不跑飞** | 专门 Agentic loop 驱动的 LLM ↔ Tool 长循环，内置上下文压缩、成本预算、死循环防护，长任务不断链 |
| **能干活，不只聊天** | 文件、执行、搜索、浏览器、MCP、多智能体、Workflow——真实动手，不绑定单一 IDE |
| **记得住，会遗忘** | 记忆按条目存储、相关性召回、drift 防御，确认过的偏好同步到全局 `~/.agentica/AGENTS.md` |
| **越用越强** | 工具失败 / 用户纠正 / 成功序列沉淀为经验卡片，自动编译成可复用的 `SKILL.md`，跨会话生效 |
| **全可换，不锁死** | 模型、工具、记忆、Skill、Guardrails、MCP 都是可替换部件，而非封闭 SaaS 黑盒 |

## 🔥 News

- [2026/07/05] **v1.4.7**：CLI 新增 cron 运行时（`/cron` 命令 + daemon）、自管理（`/upgrade`、`/config set|env`）；统一配置到 `~/.agentica/config.yaml`。详见 [Release-v1.4.7](https://github.com/shibing624/agentica/releases/tag/v1.4.7)
- [2026/06/03] **v1.4.6**：支持fallback模型可配置，支持多个fallback模型；支持 LSP， CLI 开启 LSP 开关（`--enable-diagnostics`/`--diagnostics-server`）；支持 `agentica doctor`；支持 `/goal` 长程任务。详见 [Release-v1.4.6](https://github.com/shibing624/agentica/releases/tag/v1.4.6)
- [2026/05/11] **v1.4.4**：MemoryExtractHooks 优化，新增 `auto_extract_memory_background` 后台抽取（不再阻塞 `on_agent_end`），memory 抽取优先走更快更便宜的 `auxiliary_model`。详见 [Release-v1.4.4](https://github.com/shibing624/agentica/releases/tag/v1.4.4)
- [2026/05/10] **v1.4.3**：Skill 生命周期重构 + VaG 解耦，新增 `SkillLifecycleHooks` 统一扩展点。详见 [Release-v1.4.3](https://github.com/shibing624/agentica/releases/tag/v1.4.3)

## 架构

Agentica 提供了从底层模型路由到顶层多智能体协作的完整抽象：

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/architecturev2.jpg" width="800" alt="Agentica Architecture" />
</div>

### 核心执行引擎 (Agentic Loop)

Agentica 的单体 Agent 运行在一个纯粹的基于控制流的 `while(true)` 引擎中，严格依据工具调用来驱动，并内置了防死循环、成本追踪、上下文微压缩（Compaction）和四层安全护栏：

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agent_loop.png" width="800" alt="Agentica Loop Architecture" />
</div>

## 安装

```bash
pip install -U agentica
```

## 快速开始

```python
from agentica import Agent, OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
result = agent.run_sync("一句话介绍北京")
print(result.content)
```

## 功能特性

- **Async-First** — 原生 async API，`asyncio.gather()` 并行工具执行，同步适配器兼容
- **20+ 模型** — OpenAI / DeepSeek / Claude / 智谱 / Qwen / Moonshot / Ollama / LiteLLM 等
- **40+ 内置工具** — 搜索、代码执行、文件操作、浏览器、OCR、图像生成
- **RAG** — 知识库管理、混合检索、Rerank，集成 LangChain / LlamaIndex
- **多智能体** — `Agent.as_tool()`（轻量组合）、Swarm（并行/自治）和 Workflow（确定性编排）
- **Actor-Critic 精炼** — `refine()` + 多 Critic 并行评审，`SchemaCritic` 程序级零成本验证 / `AgentCritic` 异构强模型把关，循环检测自动早停
- **`/goal` 长任务循环** — `await agent.run_goal("xxx")` 持续推进，自动判断完成、续跑、暂停；支持 token / wall-clock / turn 三种 hard cap；CLI `/goal /subgoal` 即开即用，详见 [文档](https://shibing624.github.io/agentica/advanced/goals)
- **安全守卫** — 输入/输出/工具级 Guardrails，流式实时检测
- **MCP / ACP** — Model Context Protocol 和 Agent Communication Protocol 支持
- **Skill 系统** — 基于 Markdown 的技能注入，支持项目级、用户级和外部托管 skill 目录
- **持久化记忆** — 索引/内容分离、相关性召回、四类型分类、drift 防御，可同步长期偏好到全局 `AGENTS.md`
- **多模态** — 文本、图像、音频、视频理解
- **自进化** — 经验卡片自动编译为可跨会话复用的 `SKILL.md`（流程见下图）

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/evo_pipeline.png" width="900" alt="Agentica Self-Evolution Pipeline" />
</div>

## Agent 用例

### 自定义工具组合

```python
from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFileTool, BuiltinExecuteTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[BuiltinWebSearchTool(), BuiltinFileTool(work_dir="./workspace"), BuiltinExecuteTool(work_dir="./workspace")],
)
agent.run_sync("帮我搜 Python 3.13 新特性，写到 features.md")
```


### 完全体（CLI / Gateway / 长任务）

```python
from agentica import DeepAgent
agent = DeepAgent()  # 40+ 内置工具 + 压缩 + 长期记忆 + skills + MCP，开箱即用
```

## CLI

```bash
agentica 
```

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/cli_snap.png" width="800" />

### 长任务：`/goal`

让 Agent 持续向一个目标推进，每轮结束自动判断是否完成，没完成就续跑——直到 judge 判 done、预算耗尽、或用户主动停下。

CLI：

```text
/goal 实现 xxx 功能并跑通 pytest    # 设置目标 + 自动开跑
/goal status                       # 显示状态、预算、subgoals
/goal pause | resume | clear
/subgoal 必须补单测                  # 给目标加验收条件
```

完整说明：[Standing Goal Loop 文档](https://shibing624.github.io/agentica/advanced/goals)。

## Web UI / Gateway

**Gateway 现在已经集成到 `agentica` 主库中**。

安装 Gateway 运行时：

```bash
pip install -U "agentica[gateway]"
```

启动：

```bash
agentica-gateway
```
<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/agentica-web.png" width="800" />

Web网页会启动在 `http://127.0.0.1:8789/chat`。

除Web网页，还支持手机端接入 QQ / 飞书 / 微信 / 企微 / Telegram / Discord / Slack 等。内置调度定时任务。

## 示例

查看 [examples/](https://github.com/shibing624/agentica/tree/main/examples) 获取完整示例，涵盖：

| 类别 | 内容 |
|------|------|
| **基础用法** | Hello World、流式输出、结构化输出、多轮对话、多模态、**Agentic Loop 对比** |
| **工具** | 自定义工具、Async 工具、搜索、代码执行、并行工具、并发安全、成本追踪、沙箱隔离、压缩 |
| **Agent 模式** | Agent 作为工具、并行执行、团队协作、辩论、路由分发、Swarm、子 Agent、模型层钩子、会话恢复 |
| **安全护栏** | 输入/输出/工具级 Guardrails、流式护栏 |
| **记忆** | 会话历史、WorkingMemory、上下文压缩、Workspace 记忆、LLM 自动记忆 |
| **RAG** | PDF 问答、高级 RAG、LangChain / LlamaIndex 集成 |
| **工作流** | 数据管道、投资研究、新闻报道、代码审查 |
| **MCP** | Stdio / SSE / HTTP 传输、JSON 配置 |
| **可观测性** | Langfuse、Token 追踪、Usage 聚合 |
| **应用** | LLM OS、深度研究、客服系统、**金融研究（6-Agent 流水线）** |

[→ 查看完整示例目录](https://github.com/shibing624/agentica/blob/main/examples/README.md)

## 文档

完整使用文档：**https://shibing624.github.io/agentica**

## 社区与支持

- **GitHub Issues** — [提交 issue](https://github.com/shibing624/agentica/issues)
- **微信群** — 添加微信号 `xuming624`，备注 "llm"，加入技术交流群

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" />

## 引用

如果您在研究中使用了 Agentica，请引用：

> Xu, M. (2026). Agentica: A Human-Centric Framework for Large Language Model Agent Workflows. GitHub. https://github.com/shibing624/agentica

## 许可证

[Apache License 2.0](LICENSE)

## 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 致谢

- [phidatahq/phidata](https://github.com/phidatahq/phidata)
- [openai/openai-agents-python](https://github.com/openai/openai-agents-python)
