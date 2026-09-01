<p align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/logo.png" height="150" alt="Agentica Logo">
  </a>
</p>

<h1 align="center">Agentica</h1>

<p align="center"><b>一个人，一支 agent 团队。</b><br />CLI 终端、本机 Web、Desktop App 是同一套产品，跑在你自己的机器上。</p>

<h3 align="center"><a href="#桌面版">⬇️ Download Desktop App</a></h3>

<p align="center">macOS · Windows · Linux</p>

<p align="center">
  <a href="https://badge.fury.io/py/agentica"><img src="https://badge.fury.io/py/agentica.svg" alt="PyPI version" /></a>
  <a href="https://github.com/shibing624/agentica"><img src="https://img.shields.io/github/stars/shibing624/agentica?style=social" alt="GitHub stars" /></a>
  <a href="https://github.com/shibing624/agentica/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License Apache 2.0" /></a>
  <a href="https://github.com/shibing624/agentica/blob/main/requirements.txt"><img src="https://img.shields.io/badge/Python-3.10%2B-green.svg" alt="Python 3.10+" /></a>
  <a href="#社区与支持"><img src="https://img.shields.io/badge/wechat-group-green.svg?logo=wechat" alt="Wechat Group" /></a>
</p>

<p align="center">简体中文 | <a href="https://github.com/shibing624/agentica/blob/main/README_EN.md">English</a></p>

## 为什么选 Agentica

### 1. 🏆 同一个模型，更准、更快、更省钱

工具面刻意收窄、接口保持底层，对 DeepSeek 等开放模型深度适配。和 OpenAI Codex 各跑同一个模型、同一批公开题目，正面对比：

<p align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/benchmark-agentica-vs-codex.png" width="920" alt="Benchmark: Agentica 在 coding 与 data analysis 两个公开题库上准确率不低于 Codex，墙钟和输入 token 都少一截" />
</p>

**编程题全对、数据分析题准确率更高——而且快一倍、输入 token 少三分之二。** 复现命令、逐项指标与原始 `predictions.jsonl` 见 [评测页](https://shibing624.github.io/agentica/guides/benchmark)。

### 2. 🤝 一个会话是一个 agent，多个会话是一支队伍

一个终端会话就是一个可协作的 agent。进程内 `task` 拉临时 subagent，进程级 `delegate` 另起一整个 agent 去干独立的大活，跨终端 peer 消息让两个会话互相说话——都不需要额外部署任何东西。

### 3. 🧬 自进化，越用越强

跑完的经验自动编译成可跨会话复用的 `SKILL.md`；下次遇到同类任务，agent 读的是自己上次的结论，不是从零开始。流程见 [Skills 文档](https://shibing624.github.io/agentica/advanced/skills)。

### 4. 📱 人可以离开现场

微信 / 企微 / 飞书 / Telegram 直连本机 agent：`@会话名` 自己寻址，或者只说一句人话，让网关 agent 去指挥这台机器上的所有会话。

## 安装

```bash
pip install -U agentica
```

### 桌面版

窗口里就是同一套 Web UI，同一 `~/.agentica`，和 CLI / 浏览器混用没有区别。

> [!IMPORTANT]
> 当前构建**未签名**，系统可能拦第一次启动，按对应系统操作一次即可。机器上还没有 `agentica-gateway` 时，桌面版会在**第一次打开**时用 uv 安装一份托管 runtime（Python 3.12 + `agentica[gateway]`），放在 Application Support，不进 `~/.agentica`。已经 `pip install` 过的继续用你原来的。

<details>
<summary><b>🍎 macOS 提示「Agentica 已损坏，无法打开」</b></summary>

macOS 给从网络下载的文件加了隔离标记，应用未签名时会被误报成「已损坏」。删掉标记即可：

1. 打开 dmg，把 `Agentica.app` 拖进「应用程序」。
2. 打开「终端」，粘贴这条命令回车（要输开机密码，输入时不显示字符）：

   ```bash
   sudo xattr -rd com.apple.quarantine /Applications/Agentica.app
   ```

</details>

<details>
<summary><b>🪟 Windows 提示「Windows 已保护你的电脑」</b></summary>

SmartScreen 会拦未签名的安装程序：点「更多信息」→「仍要运行」。只有第一次需要。

</details>

<details>
<summary><b>🐧 Linux 双击 AppImage 没反应</b></summary>

浏览器下载的 AppImage 默认没有执行权限（deb 走包管理器，没这个问题）：

```bash
chmod +x agentica-desktop-linux-x86_64.AppImage
```

</details>

也可以直接从源码跑：`cd desktop && npm install && npm start`，详见 [`desktop/README.md`](https://github.com/shibing624/agentica/blob/main/desktop/README.md)。

## 配置

配上任一模型厂商的 API Key 即可（优先级：shell 环境变量 > `.env` > `config.yaml`）：

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-xxx"
# 或用可免费起步的智谱：export ZAI_API_KEY="your-api-key"
```

也可以写进 `~/.agentica/.env`，或运行 `agentica setup` 生成 `~/.agentica/config.yaml`（CLI 内随时 `/model` 切换）。完整说明见 [安装文档](https://shibing624.github.io/agentica/getting-started/installation)。

## 快速开始

### CLI（推荐先玩这个）

终端输入：

```bash
agentica
```

进交互终端后直接说话，例如「帮我看下这个仓库的单测为什么挂了」。

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/cli_snap.png" width="800" alt="Agentica CLI 截图" />

### Web

```bash
pip install -U "agentica[gateway]"
agentica-gateway
```

本机网页在 `http://127.0.0.1:8881/chat`（聊天、轨迹、设置）。首次启动会建一个 `default` 账号并把随机初始密码打在终端里；管理员可以在「用户管理」页加账号，每个账号有自己独立的会话和记忆。微信 / 企微 / 飞书 / Telegram 直连见 [Gateway 文档](https://github.com/shibing624/agentica/blob/main/docs/advanced/gateway.md)。

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agentica-web.png" width="800" alt="Agentica Web UI 截图" />

自托管（Docker，镜像里编好 UI，运行时不需要 Node）：

```bash
cp .env.docker.example .env   # 填 OPENAI_API_KEY
docker compose up -d --build
```

浏览器打开 `http://127.0.0.1:8881/chat`。数据在 named volume，当前目录挂到容器 `/workspace`。

### TypeScript SDK

给**已经在跑的 gateway** 写 Node 脚本时用，不是启动 Web 的步骤：

```bash
npm install @agentica/sdk
```

包名必须写全 **`@agentica/sdk`**（`registry.npmjs.org`，不要写成 `agentica-sdk`）。

```ts
import { Agentica } from "@agentica/sdk";
const agentica = new Agentica({
  baseURL: "http://127.0.0.1:8881",
  apiKey: process.env.AGENTICA_GATEWAY_TOKEN, // ~/.agentica/cache/gateway/runtime.json
});
for await (const event of agentica.chat.stream({ message: "ping", session_id: "demo" })) {
  if (event.event === "content") process.stdout.write(String(event.data));
}
```

源码在 [`sdk-ts/`](https://github.com/shibing624/agentica/tree/main/sdk-ts)。

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
- **内置工具** — `read_file` / `write_file` / `apply_patch` / `grep` / `glob`、`execute`、网页搜索；长报告用 `write_file` 写 HTML
- **多模型** — OpenAI Chat Completions / [Responses API](https://shibing624.github.io/agentica/guides/openai-responses)、DeepSeek、Claude、ZhipuAI、Qwen、Moonshot、Ollama、LiteLLM 等
- **安全守卫** — 输入/输出/工具级 Guardrails，流式实时检测
- **多模态** — 文本、图像、音频、视频理解

**产品入口**

- **CLI** — `agentica` 交互终端；进程内 `task`、进程级 `delegate`、跨终端 peer 消息
- **Web** — `agentica-gateway` 本机 SPA（聊天 / 轨迹 / 设置）+ IM 渠道
- **Desktop App** — App 跟 Web、CLI 共用同一套历史记录、工作目录、模型、配置等状态

**协作**

- **多智能体** — SDK：`Agent.as_tool()`、Workflow、Swarm、[Markdown Subagent](https://shibing624.github.io/agentica/multi-agent/subagent)；CLI：`task` / `delegate` / peer 消息（见 [终端文档](https://shibing624.github.io/agentica/getting-started/terminal)）
- **Actor-Critic 精炼** — `refine()` + 多 Critic 并行评审，`SchemaCritic` 程序级零成本验证 / `AgentCritic` 异构强模型把关，循环检测自动早停

**记忆与进化**

- **持久化记忆** — 索引/内容分离、相关性召回、四类型分类、drift 防御；常驻规则写在 `AGENTS.md`
- **Skill 系统** — 基于 Markdown 的技能注入，支持项目级、用户级和外部托管 skill 目录
- **自进化** — 经验卡片自动编译为可跨会话复用的 `SKILL.md`

**集成**

- **MCP / ACP** — Model Context Protocol 和 Agent Communication Protocol 支持
- **RAG** — 知识库管理、混合检索、Rerank，集成 LangChain / LlamaIndex

架构与执行引擎（Agentic Loop、两层上下文压缩、四层护栏）见 [架构文档](https://shibing624.github.io/agentica/introduction/architecture)。

## 多会话协作：`task` / `delegate` / peer

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

- [2026/08/25] **v1.4.14**：权限档更新（ask 不再藏写工具，增加「拒绝类似」）；Web/桌面真正多账号；文件工具收成 `apply_patch` + `write_file`（`read_file` 支持 tail）；Worktree 默认进仓库内；系统 skill 包内加载；桌面版首次打开自动装 Python runtime。详见 [Release-v1.4.14](https://github.com/shibing624/agentica/releases/tag/v1.4.14)
- [2026/08/20] **v1.4.13**：Web 换成 Vite + React SPA 并新增轨迹页；网页界面默认英文、设置里可切简体中文；新增 **Desktop App 安装包**（macOS dmg / Windows NSIS / Linux AppImage·deb）。详见 [Release-v1.4.13](https://github.com/shibing624/agentica/releases/tag/v1.4.13)
- [2026/08/10] **v1.4.12**：上下文压缩升级：三层上下文压缩收敛为两层（截断旧 tool result → LLM/native 摘要）；新增跨终端 peer 消息（`list_agents` / `send_message`）与进程级 `delegate`（独立 `agentica --query --print`，经 `/ps` `/stop` `wait` 托管），与进程内 `task` 分工明确。详见 [Release-v1.4.12](https://github.com/shibing624/agentica/releases/tag/v1.4.12)

<details>
<summary>更多版本</summary>

- [2026/08/04] **v1.4.11**：新增 OpenAI Responses API（含原生 compaction）、Markdown 可配置 subagent、`apply_patch` 多文件；CLI resume/状态栏/压缩提示增强；裁减 prompt 与 grep/glob schema；修复 Learned Experiences 污染与 `write_todos` 全量回显。详见 [Release-v1.4.11](https://github.com/shibing624/agentica/releases/tag/v1.4.11)
- [2026/07/24] **v1.4.10**：支持视觉模型原生图片输入与模型能力 catalog 路由；新增 `/rename` 和按名称 `/resume`。详见 [Release-v1.4.10](https://github.com/shibing624/agentica/releases/tag/v1.4.10)
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
