[**🇨🇳中文**](https://github.com/shibing624/agentica/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/agentica/blob/main/README_EN.md) | [**🇯🇵日本語**](https://github.com/shibing624/agentica/blob/main/README_JP.md)

<div align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/logo.png" height="150" alt="Agentica Logo">
  </a>
</div>

-----------------

# Agentica

**一个人，一支 agent 团队。**

CLI 终端、本机 Web、Desktop App 是同一套产品：多个会话并行干活、互相通信；人走开了也能从微信/企微把本机 agent 喊回来。

[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![GitHub stars](https://img.shields.io/github/stars/shibing624/agentica?style=social)](https://github.com/shibing624/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/shibing624/agentica/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://github.com/shibing624/agentica/blob/main/requirements.txt)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#社区与支持)

**Agentica** 是给开发者使用的本机 Agent：**CLI** + **Web** + **Desktop App** + Python SDK。一个终端会话就是一个可协作的 agent，多个会话组成一支队伍。

|  | |
|------|------|
| **CLI** | `agentica` 进交互终端；`delegate` / `task` / peer 消息让多个会话一起干活 |
| **Web** | `agentica-gateway` 起本机网页（聊天、轨迹、设置），微信/企微等 IM 也可直连 |
| **Desktop App** | Electron 壳，窗口里就是同一套 Web；同一 `~/.agentica` 上已有 gateway 就直接连过去 |
| **人可以离开现场** | Web Gateway 与 `PEER_BRIDGE` 让微信/企微等 IM 直连本机 CLI：`@会话名` 自己寻址，或者只说一句人话让网关 agent 去群发 |

## 评测

同一 `deepseek-v4-flash-official`、同一 Responses 接口、思考 `high`，对 Codex CLI。完整说明与复现命令见 [评测页](https://shibing624.github.io/agentica/guides/benchmark)。

**Coding** — [Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark) Python 全量 34 题，两边全对；Agentica 更快、更少工具、更少 token。

![Agentica_pk_Codex](https://github.com/shibing624/agentica/blob/main/docs/assets/polyglot-agentica-vs-codex.png)

| 指标 | Agentica | Codex CLI |
|---|---|---|
| 准确率 | **34/34（100%）** | **34/34（100%）** |
| 总墙钟 | **1486s** | 2332s（1.57×） |
| 平均墙钟 / 题 | **43.7s** | 68.6s |
| tool calls | **139** | 168（1.21×） |
| 输入 token | **1.24M** | 2.85M（2.30×） |
| 输出 token | **173k** | 201k（1.17×） |

**Data analysis** — [InfiAgent-DABench](https://github.com/InfiAgent/InfiAgent) DAEval validation 全量 257 题。准确率 Agentica 略高，墙钟和输入 token 差一截。

![Agentica_pk_Codex_DABench](https://github.com/shibing624/agentica/blob/main/docs/assets/dabench-agentica-vs-codex.png)

| 指标 | Agentica | Codex CLI |
|---|---|---|
| 准确率 | **220/257（85.6%）** | 215/257（83.66%） |
| 总墙钟 | **3241s** | 6609s（2.04×） |
| 平均墙钟 / 题 | **12.6s** | 25.7s |
| tool calls | **499** | 861（1.73×） |
| 输入 token | **3.87M** | 13.02M（3.36×） |
| 输出 token | **277k** | 420k（1.51×） |

原始结果在 [`evaluation/code_benchmark/results/`](https://github.com/shibing624/agentica/tree/main/evaluation/code_benchmark/results)。

## 安装

```bash
pip install -U agentica
```

## 配置

三选一，配上任一模型厂商的 API Key 即可（优先级：shell 环境变量 > `.env` > `config.yaml`）：

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-xxx"
# 或用可免费起步的智谱：export ZAI_API_KEY="your-api-key"
```

也可以写进 `~/.agentica/.env`，或运行 `agentica setup` 生成 `~/.agentica/config.yaml`（CLI 内随时用 `/model` 切换模型）。完整配置说明见 [安装文档](https://shibing624.github.io/agentica/getting-started/installation)。

## 快速开始

### CLI（推荐先玩这个）

```bash
agentica
```

进入交互终端后直接说话即可，例如「帮我看下这个仓库的单测为什么挂了」。多会话协作用 `delegate` / peer 消息，详见下文 [CLI](#cli) 一节。

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/cli_snap.png" width="800" alt="Agentica CLI 截图" />

### Web

```bash
pip install -U "agentica[gateway]"
agentica-gateway
```

本机网页在 `http://127.0.0.1:8881/chat`（聊天、轨迹、设置）。微信 / 企微 / 飞书 / Telegram 等 IM 也可直连，详见 [Gateway 文档](https://github.com/shibing624/agentica/blob/main/docs/advanced/gateway.md)。

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agentica-web.png" width="800" alt="Agentica Web UI 截图" />

### Desktop App

窗口里就是同一套 Web UI。同一 `~/.agentica` 上已经有 gateway 就直接连过去，没有才拉起一个。

```bash
pip install -e ".[gateway]"   # 仓库根目录，一次即可
cd desktop && npm install && npm start
```

详见 [`desktop/README.md`](https://github.com/shibing624/agentica/blob/main/desktop/README.md)。

### Python SDK

让 Agent 直接搜资料 + 写文件，一行 `run_sync` 开始干活：

```python
from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFileTool, BuiltinExecuteTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[BuiltinWebSearchTool(), BuiltinFileTool(work_dir="./workspace"), BuiltinExecuteTool(work_dir="./workspace")],
)
agent.run_sync("帮我搜 Python 3.13 新特性，写到 features.md")
```

开箱即用的完全体（内置工具 + 压缩 + 长期记忆 + skills + MCP）：

```python
from agentica import DeepAgent
agent = DeepAgent()
```

## 功能特性

**核心引擎**

- **Async-First** — 原生 async API，`asyncio.gather()` 并行工具执行，同步适配器兼容
- **内置工具** — 搜索、代码执行、文件操作、浏览器、OCR、图像生成
- **多模型** — OpenAI Chat Completions / [Responses API](https://shibing624.github.io/agentica/guides/openai-responses)、DeepSeek、Claude、ZhipuAI、Qwen、Moonshot、Ollama、LiteLLM 等
- **安全守卫** — 输入/输出/工具级 Guardrails，流式实时检测
- **多模态** — 文本、图像、音频、视频理解

**产品入口**

- **CLI** — `agentica` 交互终端；进程内 `task`、进程级 `delegate`、跨终端 peer 消息
- **Web** — `agentica-gateway` 本机 SPA（聊天 / 轨迹 / 设置）+ IM 渠道
- **Desktop App** — Electron 薄壳，同一套 Web，先 attach 再 spawn

**协作**

- **多智能体** — SDK：`Agent.as_tool()`、Workflow、Swarm、[Markdown Subagent](https://shibing624.github.io/agentica/multi-agent/subagent)；CLI：进程内 `task`、进程级 `delegate`、跨终端 peer 消息（见 [终端文档](https://shibing624.github.io/agentica/getting-started/terminal)）
- **Actor-Critic 精炼** — `refine()` + 多 Critic 并行评审，`SchemaCritic` 程序级零成本验证 / `AgentCritic` 异构强模型把关，循环检测自动早停

**记忆与进化**

- **持久化记忆** — 索引/内容分离、相关性召回、四类型分类、drift 防御；常驻规则写在 `users/{user_id}/AGENTS.md`（CLI default 也可写 `~/.agentica/AGENTS.md` symlink）
- **Skill 系统** — 基于 Markdown 的技能注入，支持项目级、用户级和外部托管 skill 目录
- **自进化** — 经验卡片自动编译为可跨会话复用的 `SKILL.md`，流程见 [Skills 文档](https://shibing624.github.io/agentica/advanced/skills)

**集成**

- **MCP / ACP** — Model Context Protocol 和 Agent Communication Protocol 支持
- **RAG** — 知识库管理、混合检索、Rerank，集成 LangChain / LlamaIndex

## 架构

Agentica 提供从底层模型路由到顶层多智能体协作的完整抽象：

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/architecturev2.jpg" width="800" alt="Agentica 架构图" />
</div>

### 核心执行引擎 (Agentic Loop)

Agentica 的单体 Agent 运行在一个纯粹的基于控制流的 `while(true)` 引擎中，严格依据工具调用来驱动，并内置防死循环、成本追踪、[两层上下文压缩](https://github.com/shibing624/agentica/blob/main/docs/advanced/compression.md)（免费淘汰 → LLM 摘要）和四层安全护栏：

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agent_loop.png" width="800" alt="Agentica Agent Loop 架构图" />
</div>

## CLI

```bash
agentica
```

### 协作：`task` / `delegate` / peer

| 机制 | 做什么 | 何时用 |
|------|--------|--------|
| `task` | 同进程 subagent（默认 auxiliary 模型，只读） | 搜代码、查资料等短活 |
| `delegate` | 另起一个完整 `agentica --query --print` 进程 | 需要独立 context / 另一目录的大活；`/ps`、`wait`、`/stop` 可管 |
| peer | 两个交互终端互发纯文本（`list_agents` / `send_message`） | 告诉另一会话「这边改了什么」，不是雇工 |

选型细节：[Choosing](https://shibing624.github.io/agentica/multi-agent/choosing) · [终端文档](https://shibing624.github.io/agentica/getting-started/terminal)。

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

## 和其它 Agent CLI 对比

| | Agentica | Claude Code | Codex CLI |
|---|---|---|---|
| 模型选择 | ✅ 多厂商自由切换 | 仅 Claude 模型 | 仅 OpenAI 模型 |
| 跨终端多会话协作 | ✅ peer + `delegate` / `task` | ❌ | ❌ |
| Web + Desktop + IM | ✅ 本机网页 / Desktop App / 微信企微飞书 Telegram | ❌ | ❌ |
| 自进化 Skill | ✅ 经验自动编译 `SKILL.md` | ❌ | ❌ |
| Python SDK | ✅ 完整 SDK，可嵌入任意代码 | 部分（绑定 Claude） | ❌ |
| 开源 | ✅ Apache 2.0 | ❌ | ✅ |

## 🔥 News

- [2026/08/10] **v1.4.12**：上下文压缩升级：三层上下文压缩收敛为两层（截断旧 tool result → LLM/native 摘要）；新增跨终端 peer 消息（`list_agents` / `send_message`）与进程级 `delegate`（独立 `agentica --query --print`，经 `/ps` `/stop` `wait` 托管），与进程内 `task` 分工明确。详见 [Release-v1.4.12](https://github.com/shibing624/agentica/releases/tag/v1.4.12)
- [2026/08/04] **v1.4.11**：新增 OpenAI Responses API（含原生 compaction）、Markdown 可配置 subagent、`apply_patch` 多文件；CLI resume/状态栏/压缩提示增强；裁减 prompt 与 grep/glob schema；修复 Learned Experiences 污染与 `write_todos` 全量回显。详见 [Release-v1.4.11](https://github.com/shibing624/agentica/releases/tag/v1.4.11)
- [2026/07/24] **v1.4.10**：支持视觉模型原生图片输入与模型能力 catalog 路由；新增 `/rename` 和按名称 `/resume`。详见 [Release-v1.4.10](https://github.com/shibing624/agentica/releases/tag/v1.4.10)

<details>
<summary>更多版本</summary>

- [2026/07/21] **v1.4.9**：内置 subagent 全部改为只读；`edit_file` 改为 tip 提示而非硬拒；修复 `ask_user_question` CLI 卡死。详见 [Release-v1.4.9](https://github.com/shibing624/agentica/releases/tag/v1.4.9)
- [2026/07/05] **v1.4.7**：CLI 新增 cron 运行时（`/cron` 命令 + daemon）、自管理（`/upgrade`、`/config set|env`）；统一配置到 `~/.agentica/config.yaml`。详见 [Release-v1.4.7](https://github.com/shibing624/agentica/releases/tag/v1.4.7)
- [2026/06/03] **v1.4.6**：支持fallback模型可配置，支持多个fallback模型；支持 LSP， CLI 开启 LSP 开关（`--enable-diagnostics`/`--diagnostics-server`）；支持 `agentica doctor`；支持 `/goal` 长程任务。详见 [Release-v1.4.6](https://github.com/shibing624/agentica/releases/tag/v1.4.6)
- [2026/05/11] **v1.4.4**：MemoryExtractHooks 优化，新增 `auto_extract_memory_background` 后台抽取（不再阻塞 `on_agent_end`），memory 抽取优先走更快更便宜的 `auxiliary_model`。详见 [Release-v1.4.4](https://github.com/shibing624/agentica/releases/tag/v1.4.4)
- [2026/05/10] **v1.4.3**：Skill 生命周期重构 + VaG 解耦，新增 `SkillLifecycleHooks` 统一扩展点。详见 [Release-v1.4.3](https://github.com/shibing624/agentica/releases/tag/v1.4.3)

</details>

## 文档

完整使用文档：**https://shibing624.github.io/agentica**

## 社区与支持

> 如果 Agentica 帮到了你，欢迎点个 ⭐ Star，让更多人看到！

- **GitHub Issues** — [提交 issue](https://github.com/shibing624/agentica/issues)
- **微信群** — 添加微信号 `xuming624`，备注 "llm"，加入技术交流群

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/wechat.jpeg" width="200" alt="微信群二维码" />

## 引用

如果您在研究中使用了 Agentica，请引用：

> Xu, M. (2026). Agentica: A Human-Centric Framework for Large Language Model Agent Workflows. GitHub. https://github.com/shibing624/agentica

BibTeX：

```bibtex
@misc{xu2026agentica,
  author    = {Xu, Ming},
  title     = {Agentica: A Human-Centric Framework for Large Language Model Agent Workflows},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/shibing624/agentica}
}
```

仓库根目录也提供了 [CITATION.cff](https://github.com/shibing624/agentica/blob/main/CITATION.cff)。

## 许可证

[Apache License 2.0](https://github.com/shibing624/agentica/blob/main/LICENSE)

## 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](https://github.com/shibing624/agentica/blob/main/CONTRIBUTING.md)。

## 致谢

- [phidatahq/phidata](https://github.com/phidatahq/phidata)
- [openai/openai-agents-python](https://github.com/openai/openai-agents-python)
