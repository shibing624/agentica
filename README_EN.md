[**🇨🇳中文**](https://github.com/shibing624/agentica/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/agentica/blob/main/README_EN.md) | [**🇯🇵日本語**](https://github.com/shibing624/agentica/blob/main/README_JP.md)

<div align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/logo.png" height="150" alt="Agentica Logo">
  </a>
</div>

-----------------

# Agentica

**Make agents that run for hours — not seconds. Stay on track, do real work, get better with use.**
Async-first Python agent harness · 40+ tools · 20+ models · MCP · CLI + Web Gateway

[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![GitHub stars](https://img.shields.io/github/stars/shibing624/agentica?style=social)](https://github.com/shibing624/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/shibing624/agentica/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://github.com/shibing624/agentica/blob/main/requirements.txt)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#community--support)

**Agentica** is not a chat wrapper around an LLM API. It is an Async-First agent harness that makes agents actually *run*:
tool calling, long-running tasks, multi-agent orchestration, cross-session memory, and continuous self-evolution.

|  | |
|------------|---------------|
| **Runs long, doesn't run away** | `Runner`-driven LLM ↔ tool loop with context compaction, cost budgets, and loop safety — long tasks stay on track |
| **Does work, not just chat** | Files, execution, search, browser, MCP, multi-agent, Workflow — real actions, not tied to a single IDE |
| **Multi-session collaboration** | Cross-terminal peer messaging; `delegate` spawns a full process (own context/cwd); `task` stays the cheap in-process subagent |
| **Remembers and forgets** | Memory stored as indexed entries with relevance recall and drift defense; standing rules live in `users/{user_id}/AGENTS.md` (`~/.agentica/AGENTS.md` is a default-CLI symlink) |
| **Gets better with use** | Tool failures, user corrections, and success sequences become experience cards that auto-compile into reusable `SKILL.md` across sessions |
| **Fully swappable, not locked in** | Models, tools, memory, skills, guardrails, and MCP are replaceable parts — not a closed hosted platform |

## Installation

```bash
pip install -U agentica
```

## Configuration

Three ways to provide a model API key (precedence: shell env > `.env` > `config.yaml`):

```bash
export OPENAI_API_KEY="sk-xxx"
# or start free with ZhipuAI: export ZAI_API_KEY="your-api-key"
```

You can also write it into `~/.agentica/.env`, or run `agentica setup` to generate `~/.agentica/config.yaml` (switch models anytime with `/model` inside the CLI). Full details: [Installation docs](https://shibing624.github.io/agentica/getting-started/installation).

## Quick Start

### CLI (try this first)

```bash
agentica
```

Once the interactive terminal is up, just talk — e.g. "find out why the tests in this repo are failing". Use `/goal` for long-running tasks and `delegate` / peer messaging for multi-session collaboration; details in the [CLI](#cli) section below.

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/cli_snap.png" width="800" alt="Agentica CLI screenshot" />

### Python SDK

No need to learn `asyncio`. `run_sync` runs the full agentic loop internally
(parallel tools, streaming, compression, retries) — from the outside it's just
a normal sync function:

```python
from agentica import Agent, OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
result = agent.run_sync("Describe Beijing in one sentence")
print(result.content)
```

```
Beijing is the capital of China, a historic city with over 3,000 years of history, and the nation's political, cultural, and international exchange center.
```

Make an agent that actually *works* — search the web and write a file, one `run_sync`:

```python
from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFileTool, BuiltinExecuteTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[BuiltinWebSearchTool(), BuiltinFileTool(work_dir="./workspace"), BuiltinExecuteTool(work_dir="./workspace")],
)
agent.run_sync("Search Python 3.13 new features and write them to features.md")
```

Or grab the batteries-included full power (40+ built-in tools + compression + long-term memory + skills + MCP):

```python
from agentica import DeepAgent
agent = DeepAgent()
```

## Features

**Core engine**

- **Async-First** — Native async API, `asyncio.gather()` parallel tool execution, sync adapter included
- **40+ Built-in Tools** — Search, code execution, file operations, browser, OCR, image generation
- **20+ Models** — OpenAI Chat Completions / [Responses API](https://shibing624.github.io/agentica/guides/openai-responses), DeepSeek, Claude, ZhipuAI, Qwen, Moonshot, Ollama, LiteLLM and more
- **Guardrails** — Input / output / tool-level guardrails, streaming real-time detection
- **Multi-Modal** — Text, image, audio, video understanding

**Long tasks & collaboration**

- **`/goal` Long-running Tasks** — `await agent.run_goal("xxx")` keeps pushing toward a goal, auto-judging completion, resuming, or pausing; supports token / wall-clock / turn hard caps; CLI `/goal /subgoal` ready out of the box, see [docs](https://shibing624.github.io/agentica/advanced/goals)
- **Multi-Agent** — SDK: `Agent.as_tool()`, Workflow, Swarm, [Markdown Subagents](https://shibing624.github.io/agentica/multi-agent/subagent); CLI: in-process `task`, process-level `delegate`, cross-terminal peer messaging (see [Terminal docs](https://shibing624.github.io/agentica/getting-started/terminal))
- **Actor-Critic Refinement** — `refine()` with parallel multi-critic review, `SchemaCritic` for zero-cost program-level validation, `AgentCritic` for heterogeneous strong-model gating, and automatic loop-detection early-stop

**Memory & evolution**

- **Persistent Memory** — Index/content separation, relevance-based recall, four-type classification, drift defense; standing rules in `users/{user_id}/AGENTS.md` (`~/.agentica/AGENTS.md` is a default-CLI symlink)
- **Skill System** — Markdown-based skill injection with project, user, and managed external skill directories
- **Self-Evolution** — Experience cards auto-compile into reusable `SKILL.md` across sessions (pipeline below)

**Integrations**

- **MCP / ACP** — Model Context Protocol and Agent Communication Protocol support
- **RAG** — Knowledge base management, hybrid retrieval, Rerank, LangChain / LlamaIndex integration

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/evo_pipeline.png" width="900" alt="Agentica Self-Evolution Pipeline" />
</div>

## Architecture

Agentica provides a complete abstraction stack, from low-level model routing to high-level multi-agent orchestration:

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/architecturev2.jpg" width="800" alt="Agentica Architecture" />
</div>

### Core Execution Engine (Agentic Loop)

At its core, a single Agent runs inside a pure deterministic `while(true)` engine driven strictly by tool calls, featuring built-in infinite-loop prevention, cost tracking, [two-layer context compression](https://github.com/shibing624/agentica/blob/main/docs/advanced/compression.md) (free eviction, then LLM summarisation), and a 4-layer guardrail system:

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agent_loop.png" width="800" alt="Agentica Loop Architecture" />
</div>

## CLI

```bash
agentica
```

### Long-running tasks: `/goal`

Let the Agent keep pushing toward a goal; at the end of each round it automatically decides whether the goal is met, and if not, continues — until the judge says done, the budget is exhausted, or the user stops manually.

CLI:

```text
/goal implement xxx and pass pytest    # set goal + auto-run
/goal status                          # show status, budget, subgoals
/goal pause | resume | clear
/subgoal add unit tests               # append an acceptance condition
```

Full guide: [Standing Goal Loop docs](https://shibing624.github.io/agentica/advanced/goals).

### Collaboration: `task` / `delegate` / peer

| Mechanism | What it does | When to use |
|-----------|--------------|-------------|
| `task` | In-process subagent (aux model by default, read-only) | Short lookups: search code, gather facts |
| `delegate` | Spawns a full `agentica --query --print` process | Large work needing its own context / cwd; managed via `/ps`, `wait`, `/stop` |
| peer | Plain-text between two interactive terminals (`list_agents` / `send_message`) | Inform another session — not hire a worker |

Details: [Choosing](https://shibing624.github.io/agentica/multi-agent/choosing) · [Terminal docs](https://shibing624.github.io/agentica/getting-started/terminal).

## Web UI / IM Integration

```bash
pip install -U "agentica[gateway]"
```

Launch:

```bash
agentica-gateway
```

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agentica-web.png" width="800" alt="Agentica Web UI screenshot" />

By default it serves at `http://127.0.0.1:8881/chat`.

Supports IM (QQ / Feishu / WeChat / WeCom / Telegram / Discord / Slack) on mobile, with built-in scheduled tasks.

IM integration details (scan code binding, channel configuration, environment variables): [Gateway docs](https://github.com/shibing624/agentica/blob/main/docs/advanced/gateway.md).

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

| | Agentica | Claude Code | Codex CLI | Gemini CLI |
|---|---|---|---|---|
| Model choice | ✅ 20+ providers, freely swappable | Claude models only | OpenAI models only | Gemini models only |
| Cross-terminal multi-session collab | ✅ peer + `delegate` / `task` | ❌ | ❌ | ❌ |
| `/goal` long-task loop | ✅ budgets + auto-judged completion + resume | ❌ | ❌ | ❌ |
| Web UI + IM Gateway | ✅ WeChat / WeCom / Feishu / Telegram etc. straight to your machine | ❌ | ❌ | ❌ |
| Self-evolving skills | ✅ experience auto-compiles into `SKILL.md` | ❌ | ❌ | ❌ |
| Python SDK | ✅ full SDK, embed anywhere | partial (Claude-bound) | ❌ | ❌ |
| Open source | ✅ Apache 2.0 | ❌ | ✅ | ✅ |

## 🔥 News

- [2026/08/10] **v1.4.12**: Two-layer context compression (~70%→50% tool-result eviction → LLM/native summarise); fixes the read-and-reread loop and Anthropic paths where eviction never ran; Layer 0 spills or truncates based on recoverability; compaction counts on `RunResponse`; CLI-only assumptions no longer dump on SDK. Adds cross-terminal peer messaging (`list_agents` / `send_message`) and process-level `delegate` (full `agentica --query --print`, managed via `/ps` `/stop` `wait`) vs cheap in-process `task`. See [Release-v1.4.12](https://github.com/shibing624/agentica/releases/tag/v1.4.12)
- [2026/08/04] **v1.4.11**: Adds OpenAI Responses API (with provider-native compaction), Markdown-configurable subagents, and multi-file `apply_patch`; improves CLI resume/status/compaction feedback; trims prompt and grep/glob schema cost; fixes Learned Experiences corruption and `write_todos` full-list echo. See [Release-v1.4.11](https://github.com/shibing624/agentica/releases/tag/v1.4.11)
- [2026/07/24] **v1.4.10**: Adds native image input with catalog-driven model capability routing; introduces `/rename` and name-based `/resume`; fixes Pillow core dependency metadata. See [Release-v1.4.10](https://github.com/shibing624/agentica/releases/tag/v1.4.10)
- [2026/07/21] **v1.4.9**: Unified 3-tier permission across SDK/CLI/Web (`ask`/`auto`/`allow-all`; drops yolo/full/strict); built-in subagents are read-only (`task` defaults to `explore`, edit/execute denied — fixes aux-model garbage code); `OpenAIChat` parses Claude `<invoke>` text tool calls leaked by OpenAI-compatible proxies; `edit_file` gives advisory tips instead of hard-rejecting; fixes `ask_user_question` CLI freeze. See [Release-v1.4.9](https://github.com/shibing624/agentica/releases/tag/v1.4.9)

<details>
<summary>Older releases</summary>

- [2026/07/05] **v1.4.7**: CLI gets a unified braille spinner (turns across the thinking/tool/answering phases, so a live process is visually distinct from a hang); fixes the `ask_user_input` input-hang and `/btw` corrupting the main model instance; adds a cron runtime (`/cron` command + daemon), self-management (`/upgrade`, `/config set|env`); unifies config into `~/.agentica/config.yaml` (main + aux model; drops `cli_config.json`/`task_model`, preserves user comments); `/resume` accepts full/prefix/ellipsis session ids. Also fixes stream-upload OOM and `/api/upload` path traversal (CWE-22). See [Release-v1.4.7](https://github.com/shibing624/agentica/releases/tag/v1.4.7)
- [2026/06/03] **v1.4.6**: Cross-provider fallback now supports tool-calling turns — a fallback model can invoke tools and produce the final answer, while its provider-specific transcript is compacted to keep replay to the recovered primary clean; fallback models are cloned per run for concurrency safety. Adds edit-time LSP diagnostics CLI flags (`--enable-diagnostics`/`--diagnostics-server`), an enhanced `agentica doctor`, `/checkpoint restore --yes` confirmation, and `/goal` budget flags. See [Release-v1.4.6](https://github.com/shibing624/agentica/releases/tag/v1.4.6)
- [2026/05/11] **v1.4.4**: MemoryExtractHooks optimization — new `auto_extract_memory_background` runs memory extraction in the background (no longer blocking `on_agent_end`), and extraction prefers the cheaper/faster `auxiliary_model`. See [Release-v1.4.4](https://github.com/shibing624/agentica/releases/tag/v1.4.4)
- [2026/05/10] **v1.4.3**: Skill lifecycle refactor + VaG decoupling — VaG experimental code moved to the `evaluation/vag/` research module, with a unified `SkillLifecycleHooks` extension point. See [Release-v1.4.3](https://github.com/shibing624/agentica/releases/tag/v1.4.3)

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
