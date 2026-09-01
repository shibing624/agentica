<p align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/logo.png" height="150" alt="Agentica Logo">
  </a>
</p>

<h1 align="center">Agentica</h1>

<p align="center"><b>One person, a team of agents.</b><br />A CLI, a local web app and a desktop app — one product, running on your own machine.</p>

<h3 align="center"><a href="#desktop-app">⬇️ Download the desktop app</a></h3>

<p align="center">macOS · Windows · Linux</p>

<p align="center">
  <a href="https://badge.fury.io/py/agentica"><img src="https://badge.fury.io/py/agentica.svg" alt="PyPI version" /></a>
  <a href="https://github.com/shibing624/agentica"><img src="https://img.shields.io/github/stars/shibing624/agentica?style=social" alt="GitHub stars" /></a>
  <a href="https://github.com/shibing624/agentica/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License Apache 2.0" /></a>
  <a href="https://github.com/shibing624/agentica/blob/main/requirements.txt"><img src="https://img.shields.io/badge/Python-3.10%2B-green.svg" alt="Python 3.10+" /></a>
  <a href="#community--support"><img src="https://img.shields.io/badge/wechat-group-green.svg?logo=wechat" alt="Wechat Group" /></a>
</p>

<p align="center"><a href="https://github.com/shibing624/agentica/blob/main/README.md">简体中文</a> | English</p>

## Why Agentica

### 1. 🏆 Same model, less time and fewer tokens

A deliberately narrow toolset over clean low-level interfaces, tuned for open models like DeepSeek. Head-to-head with OpenAI Codex on the same model and the same public suites:

<p align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/benchmark-agentica-vs-codex.png" width="920" alt="Benchmark: Agentica matches or beats Codex on accuracy across coding and data analysis while using far less wall-clock time and input tokens" />
</p>

**A clean sweep on coding, higher accuracy on data analysis — at roughly half the wall-clock and a third of the input tokens.** Reproduction commands, per-metric tables and the raw `predictions.jsonl` are on the [benchmark page](https://shibing624.github.io/agentica/guides/benchmark).

### 2. 🤝 One session is an agent; several are a team

A terminal session *is* a collaborator. In-process `task` spawns a cheap read-only subagent, process-level `delegate` starts a whole other agent for work that deserves its own context window, and cross-terminal peer messaging lets two sessions talk. Nothing extra to deploy.

### 3. 🧬 Self-evolving

What a session learned is compiled into a reusable `SKILL.md`, so the next run of a similar task starts from last time's conclusion instead of from scratch. See the [Skills docs](https://shibing624.github.io/agentica/advanced/skills).

### 4. 📱 You can leave the desk

WeChat / WeCom / Feishu / Telegram reach the agents on this machine: address one with `@session-name`, or just say what you want and let the gateway agent direct every session on the box.

## Installation

```bash
pip install -U agentica
```

### Desktop app

The window is the same web UI, on the same `~/.agentica` — mixing it with the CLI or a browser makes no difference.

> [!IMPORTANT]
> These builds are **unsigned**, so the OS may block the first launch — one step, once.
> If this machine has no `agentica-gateway` yet, the first launch installs a managed
> runtime (uv + Python 3.12 + `agentica[gateway]`) under Application Support, not
> inside `~/.agentica`. An existing `pip install` is used as-is.

| OS | Installer |
|---|---|
| **macOS** 11+ | [Apple silicon (arm64)](https://github.com/shibing624/agentica/releases/latest/download/agentica-desktop-darwin-arm64.dmg) · [Intel (x64)](https://github.com/shibing624/agentica/releases/latest/download/agentica-desktop-darwin-x64.dmg) |
| **Windows** 10+ | [x64 installer (NSIS)](https://github.com/shibing624/agentica/releases/latest/download/agentica-desktop-win32-x64.exe) |
| **Linux** x64 | [AppImage](https://github.com/shibing624/agentica/releases/latest/download/agentica-desktop-linux-x86_64.AppImage) · [deb](https://github.com/shibing624/agentica/releases/latest/download/agentica-desktop-linux-amd64.deb) |

They are attached to every [GitHub Release](https://github.com/shibing624/agentica/releases) too.

<details>
<summary><b>🍎 macOS says "Agentica is damaged and can't be opened"</b></summary>

macOS quarantines anything downloaded from the web, and reports unsigned apps as damaged. Remove the flag:

1. Open the dmg and drag `Agentica.app` into Applications.
2. Open Terminal and run this (it asks for your login password; nothing is echoed while you type):

   ```bash
   sudo xattr -rd com.apple.quarantine /Applications/Agentica.app
   ```

</details>

<details>
<summary><b>🪟 Windows says "Windows protected your PC"</b></summary>

SmartScreen blocks unsigned installers: click "More info" → "Run anyway". First launch only.

</details>

<details>
<summary><b>🐧 Double-clicking the AppImage does nothing</b></summary>

Browsers download AppImages without the execute bit (the deb goes through your package manager and is unaffected):

```bash
chmod +x agentica-desktop-linux-x86_64.AppImage
```

</details>

You can also run it from source: `cd desktop && npm install && npm start`. See [`desktop/README.md`](https://github.com/shibing624/agentica/blob/main/desktop/README.md).

## Configuration

Provide an API key for any model provider (precedence: shell env > `.env` > `config.yaml`):

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-xxx"
# or start free with ZhipuAI: export ZAI_API_KEY="your-api-key"
```

You can also write it into `~/.agentica/.env`, or run `agentica setup` to generate `~/.agentica/config.yaml` (switch models anytime with `/model`). Full details: [Installation docs](https://shibing624.github.io/agentica/getting-started/installation).

## Quick Start

### CLI (try this first)

```bash
agentica
```

Once the interactive terminal is up, just talk — e.g. "find out why the tests in this repo are failing".

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/cli_snap.png" width="800" alt="Agentica CLI screenshot" />

### Web

```bash
pip install -U "agentica[gateway]"
agentica-gateway
```

Serves at `http://127.0.0.1:8881/chat` (chat, traces, settings). The first start creates a `default` account and prints a random initial password in the terminal; an administrator can add more under User management, and each account gets its own conversations and memory. For WeChat / WeCom / Feishu / Telegram, see the [Gateway docs](https://github.com/shibing624/agentica/blob/main/docs/advanced/gateway.md).

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agentica-web.png" width="800" alt="Agentica Web UI screenshot" />

Self-host with Docker (the image compiles the UI; runtime still has no Node):

```bash
cp .env.docker.example .env   # fill OPENAI_API_KEY
docker compose up -d --build
```

Open `http://127.0.0.1:8881/chat`. State lives in a named volume; the current directory is mounted at `/workspace`.

### TypeScript SDK

For Node programs talking to a **running** gateway. This is not how you start the web UI:

```bash
npm install @agentica-ai/sdk
```

Always the full name **`@agentica-ai/sdk`** (from `registry.npmjs.org`; not `agentica-sdk`).

```ts
import { Agentica } from "@agentica-ai/sdk";
const agentica = new Agentica({
  baseURL: "http://127.0.0.1:8881",
  apiKey: process.env.AGENTICA_GATEWAY_TOKEN, // ~/.agentica/cache/gateway/runtime.json
});
for await (const event of agentica.chat.stream({ message: "ping", session_id: "demo" })) {
  if (event.event === "content") process.stdout.write(String(event.data));
}
```

Source: [`sdk-ts/`](https://github.com/shibing624/agentica/tree/main/sdk-ts).

### Python SDK

Give the Agent search + files and start working with one `run_sync`:

```python
from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFileTool, BuiltinExecuteTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[BuiltinWebSearchTool(), BuiltinFileTool(work_dir="./workspace"), BuiltinExecuteTool(work_dir="./workspace")],
)
agent.run_sync("Search Python 3.13 new features and write them to features.md")
```

Or grab the batteries-included preset (built-in tools + compression + long-term memory + skills + MCP):

```python
from agentica import DeepAgent
agent = DeepAgent()
```

## Features

**Core engine**

- **Async-First** — Native async API, `asyncio.gather()` parallel tool execution, sync adapter included
- **Built-in tools** — `read_file` / `write_file` / `apply_patch` / `grep` / `glob`, `execute`, web search; long reports are HTML via `write_file`
- **Many models** — OpenAI Chat Completions / [Responses API](https://shibing624.github.io/agentica/guides/openai-responses), DeepSeek, Claude, ZhipuAI, Qwen, Moonshot, Ollama, LiteLLM and more
- **Guardrails** — Input / output / tool-level guardrails, streaming real-time detection
- **Multi-Modal** — Text, image, audio, video understanding

**Product surfaces**

- **CLI** — `agentica` interactive terminal; in-process `task`, process-level `delegate`, cross-terminal peer messaging
- **Web** — `agentica-gateway` local SPA (chat / traces / settings) plus IM channels
- **Desktop app** — Thin Electron shell over the same web UI; attaches to a running gateway before starting one

**Collaboration**

- **Multi-Agent** — SDK: `Agent.as_tool()`, Workflow, Swarm, [Markdown Subagents](https://shibing624.github.io/agentica/multi-agent/subagent); CLI: `task` / `delegate` / peer messaging (see [Terminal docs](https://shibing624.github.io/agentica/getting-started/terminal))
- **Actor-Critic Refinement** — `refine()` with parallel multi-critic review, `SchemaCritic` for zero-cost program-level validation, `AgentCritic` for heterogeneous strong-model gating, automatic loop-detection early-stop

**Memory & evolution**

- **Persistent Memory** — Index/content separation, relevance-based recall, four-type classification, drift defense; standing rules live in `AGENTS.md`
- **Skill System** — Markdown-based skill injection with project, user, and managed external skill directories
- **Self-Evolution** — Experience cards auto-compile into reusable `SKILL.md` across sessions

**Integrations**

- **MCP / ACP** — Model Context Protocol and Agent Communication Protocol support
- **RAG** — Knowledge base management, hybrid retrieval, Rerank, LangChain / LlamaIndex integration

For the architecture and the execution engine (agentic loop, two-layer context compression, four-layer guardrails), see the [architecture docs](https://shibing624.github.io/agentica/introduction/architecture).

## Multi-session collaboration: `task` / `delegate` / peer

| Mechanism | What it does | When to use |
|-----------|--------------|-------------|
| `task` | In-process subagent (aux model by default, read-only) | Short lookups: search code, gather facts |
| `delegate` | Spawns a full `agentica --query --print` process | Large work needing its own context / cwd; managed via `/ps`, `wait`, `/stop` |
| peer | Plain text between two interactive terminals (`list_agents` / `send_message`) | Inform another session — not hire a worker |

Details: [Choosing](https://shibing624.github.io/agentica/multi-agent/choosing) · [Terminal docs](https://shibing624.github.io/agentica/getting-started/terminal).

## Examples

See [examples/](https://github.com/shibing624/agentica/tree/main/examples) for full examples, covering:

| Category | Content |
|----------|---------|
| **Basics** | Hello World, streaming, structured output, multi-turn, multi-modal, **Agentic Loop comparison** |
| **Tools** | Custom tools, async tools, search, code execution, parallel tools, concurrency safety, cost tracking, sandbox isolation, compression |
| **Agent Patterns** | Agent-as-tool, parallel execution, multi-agent collaboration, debate, routing, Swarm, sub-agent, model-layer hooks, session resume |
| **Guardrails** | Input / output / tool-level guardrails, streaming guardrails |
| **Memory** | Session history, WorkingMemory, context compression, Workspace memory, LLM auto-memory |
| **RAG** | PDF Q&A, advanced RAG, LangChain / LlamaIndex integration |
| **Workflows** | Data pipeline, investment research, news reporting, code review |
| **MCP** | Stdio / SSE / HTTP transport, JSON config |
| **Observability** | Langfuse, token tracking, usage aggregation |
| **Applications** | LLM OS, deep research, customer service, **financial research (6-Agent pipeline)** |

[→ View full examples directory](https://github.com/shibing624/agentica/blob/main/examples/README.md)

## How it compares

| | Agentica | Claude Code | Codex CLI |
|---|---|---|---|
| Model choice | ✅ Many providers, freely swappable | Claude models only | OpenAI models only |
| Cross-terminal multi-session collab | ✅ peer + `delegate` / `task` | ❌ | ❌ |
| Web + Desktop + IM | ✅ local web app / desktop app / WeChat, WeCom, Feishu, Telegram | ❌ | ❌ |
| Self-evolving skills | ✅ experience auto-compiles into `SKILL.md` | ❌ | ❌ |
| Python SDK | ✅ full SDK, embed anywhere | partial (Claude-bound) | ❌ |
| Open source | ✅ Apache 2.0 | ❌ | ✅ |

## 🔥 News

- [2026/09/01] **v1.4.15**: Adds the `@agentica-ai/sdk` TypeScript client and a Gateway Docker image; `/export` now writes the session JSONL; `apply_patch` matches context exactly; CLI no longer treats `[/path]` in tool output as Rich markup. See [Release-v1.4.15](https://github.com/shibing624/agentica/releases/tag/v1.4.15)
- [2026/08/25] **v1.4.14**: Permissions match Codex (ask keeps write tools visible; deny-similar); true multi-account Web/Desktop; file tools narrowed to `apply_patch` + `write_file` with `read_file` tail; worktrees live inside the repo; bundled skills load in-place; desktop first-launch bootstraps a Python runtime. See [Release-v1.4.14](https://github.com/shibing624/agentica/releases/tag/v1.4.14)
- [2026/08/20] **v1.4.13**: The web UI is a Vite + React SPA with a new traces page; the UI ships in English with Simplified Chinese in settings; and there are now **desktop installers** (macOS dmg / Windows NSIS / Linux AppImage and deb). See [Release-v1.4.13](https://github.com/shibing624/agentica/releases/tag/v1.4.13)
- [2026/08/10] **v1.4.12**: Two-layer context compression (tool-result eviction → LLM/native summarise); fixes the read-and-reread loop and Anthropic paths where eviction never ran; compaction counts on `RunResponse`. Adds cross-terminal peer messaging (`list_agents` / `send_message`) and process-level `delegate` (full `agentica --query --print`, managed via `/ps` `/stop` `wait`) vs cheap in-process `task`. See [Release-v1.4.12](https://github.com/shibing624/agentica/releases/tag/v1.4.12)

<details>
<summary>Older releases</summary>

- [2026/08/04] **v1.4.11**: Adds OpenAI Responses API (with provider-native compaction), Markdown-configurable subagents, and multi-file `apply_patch`; improves CLI resume/status/compaction feedback; trims prompt and grep/glob schema cost; fixes Learned Experiences corruption and `write_todos` full-list echo. See [Release-v1.4.11](https://github.com/shibing624/agentica/releases/tag/v1.4.11)
- [2026/07/24] **v1.4.10**: Adds native image input with catalog-driven model capability routing; introduces `/rename` and name-based `/resume`; fixes Pillow core dependency metadata. See [Release-v1.4.10](https://github.com/shibing624/agentica/releases/tag/v1.4.10)
- [2026/07/21] **v1.4.9**: Unified 3-tier permission across SDK/CLI/Web (`ask`/`auto`/`allow-all`); built-in subagents are read-only; `edit_file` gives advisory tips instead of hard-rejecting; fixes `ask_user_question` CLI freeze. See [Release-v1.4.9](https://github.com/shibing624/agentica/releases/tag/v1.4.9)
- [2026/07/05] **v1.4.7**: Adds a cron runtime (`/cron` command + daemon) and self-management (`/upgrade`, `/config set|env`); unifies config into `~/.agentica/config.yaml`; `/resume` accepts full/prefix/ellipsis session ids. Also fixes stream-upload OOM and `/api/upload` path traversal (CWE-22). See [Release-v1.4.7](https://github.com/shibing624/agentica/releases/tag/v1.4.7)
- [2026/06/03] **v1.4.6**: Cross-provider fallback supports tool-calling turns; adds edit-time LSP diagnostics flags (`--enable-diagnostics`/`--diagnostics-server`), an enhanced `agentica doctor`, and `/goal` budget flags. See [Release-v1.4.6](https://github.com/shibing624/agentica/releases/tag/v1.4.6)
- [2026/05/11] **v1.4.4**: MemoryExtractHooks optimization — `auto_extract_memory_background` runs extraction in the background, preferring the cheaper `auxiliary_model`. See [Release-v1.4.4](https://github.com/shibing624/agentica/releases/tag/v1.4.4)
- [2026/05/10] **v1.4.3**: Skill lifecycle refactor + VaG decoupling, with a unified `SkillLifecycleHooks` extension point. See [Release-v1.4.3](https://github.com/shibing624/agentica/releases/tag/v1.4.3)

</details>

## Documentation

Full documentation: **https://shibing624.github.io/agentica**

## Community & Support

> If Agentica helps you, please give it a ⭐ Star so more people find it!

- **GitHub Issues** — [Open an issue](https://github.com/shibing624/agentica/issues)
- **WeChat Group** — Add `xuming624` on WeChat, mention "llm" to join the developer group

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/wechat.jpeg" width="200" alt="WeChat group QR code" />

## Citation

If you use Agentica in your research, please cite:

> Xu, M. (2026). Agentica: A Human-Centric Framework for Large Language Model Agent Workflows. GitHub. https://github.com/shibing624/agentica

BibTeX:

```bibtex
@misc{xu2026agentica,
  author    = {Xu, Ming},
  title     = {Agentica: A Human-Centric Framework for Large Language Model Agent Workflows},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/shibing624/agentica}
}
```

A [CITATION.cff](https://github.com/shibing624/agentica/blob/main/CITATION.cff) is also available in the repo root.

## License

[Apache License 2.0](https://github.com/shibing624/agentica/blob/main/LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](https://github.com/shibing624/agentica/blob/main/CONTRIBUTING.md).

## Acknowledgements

- [phidatahq/phidata](https://github.com/phidatahq/phidata)
- [openai/openai-agents-python](https://github.com/openai/openai-agents-python)
