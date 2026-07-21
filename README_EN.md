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
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#community--support)

**Agentica** is not a chat wrapper around an LLM API. It is an Async-First agent harness that makes agents actually *run*:
tool calling, long-running tasks, multi-agent orchestration, cross-session memory, and continuous self-evolution.

|  | |
|------------|---------------|
| **Runs long, doesn't run away** | `Runner`-driven LLM ↔ tool loop with context compaction, cost budgets, and loop safety — long tasks stay on track |
| **Does work, not just chat** | Files, execution, search, browser, MCP, multi-agent, Workflow — real actions, not tied to a single IDE |
| **Remembers and forgets** | Memory stored as indexed entries with relevance recall and drift defense; confirmed preferences sync into global `~/.agentica/AGENTS.md` |
| **Gets better with use** | Tool failures, user corrections, and success sequences become experience cards that auto-compile into reusable `SKILL.md` across sessions |
| **Fully swappable, not locked in** | Models, tools, memory, skills, guardrails, and MCP are replaceable parts — not a closed hosted platform |

## 🔥 News

- [2026/07/21] **v1.4.9**: Unified 3-tier permission across SDK/CLI/Web (`ask`/`auto`/`allow-all`; drops yolo/full/strict); built-in subagents are read-only (`task` defaults to `explore`, edit/execute denied — fixes aux-model garbage code); `OpenAIChat` parses Claude `<invoke>` text tool calls leaked by OpenAI-compatible proxies; `edit_file` gives advisory tips instead of hard-rejecting; fixes `ask_user_question` CLI freeze. See [Release-v1.4.9](https://github.com/shibing624/agentica/releases/tag/v1.4.9)
- [2026/07/05] **v1.4.7**: CLI gets a unified braille spinner (turns across the thinking/tool/answering phases, so a live process is visually distinct from a hang); fixes the `ask_user_input` input-hang and `/btw` corrupting the main model instance; adds a cron runtime (`/cron` command + daemon), self-management (`/upgrade`, `/config set|env`); unifies config into `~/.agentica/config.yaml` (main + aux model; drops `cli_config.json`/`task_model`, preserves user comments); `/resume` accepts full/prefix/ellipsis session ids. Also fixes stream-upload OOM and `/api/upload` path traversal (CWE-22). See [Release-v1.4.7](https://github.com/shibing624/agentica/releases/tag/v1.4.7)
- [2026/06/03] **v1.4.6**: Cross-provider fallback now supports tool-calling turns — a fallback model can invoke tools and produce the final answer, while its provider-specific transcript is compacted to keep replay to the recovered primary clean; fallback models are cloned per run for concurrency safety. Adds edit-time LSP diagnostics CLI flags (`--enable-diagnostics`/`--diagnostics-server`), an enhanced `agentica doctor`, `/checkpoint restore --yes` confirmation, and `/goal` budget flags. See [Release-v1.4.6](https://github.com/shibing624/agentica/releases/tag/v1.4.6)
- [2026/05/11] **v1.4.4**: MemoryExtractHooks optimization — new `auto_extract_memory_background` runs memory extraction in the background (no longer blocking `on_agent_end`), and extraction prefers the cheaper/faster `auxiliary_model`. See [Release-v1.4.4](https://github.com/shibing624/agentica/releases/tag/v1.4.4)
- [2026/05/10] **v1.4.3**: Skill lifecycle refactor + VaG decoupling — VaG experimental code moved to the `evaluation/vag/` research module, with a unified `SkillLifecycleHooks` extension point. See [Release-v1.4.3](https://github.com/shibing624/agentica/releases/tag/v1.4.3)

## Architecture

Agentica provides a complete abstraction stack, from low-level model routing to high-level multi-agent orchestration:

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/architecturev2.jpg" width="800" alt="Agentica Architecture" />
</div>

### Core Execution Engine (Agentic Loop)

At its core, a single Agent runs inside a pure deterministic `while(true)` engine driven strictly by tool calls, featuring built-in infinite-loop prevention, cost tracking, context micro-compression (Compaction), and a 4-layer guardrail system:

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agent_loop.png" width="800" alt="Agentica Loop Architecture" />
</div>

## Installation

```bash
pip install -U agentica
```

## Quick Start

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

## Features

- **Async-First** — Native async API, `asyncio.gather()` parallel tool execution, sync adapter included
- **20+ Models** — OpenAI / DeepSeek / Claude / ZhipuAI / Qwen / Moonshot / Ollama / LiteLLM and more
- **40+ Built-in Tools** — Search, code execution, file operations, browser, OCR, image generation
- **RAG** — Knowledge base management, hybrid retrieval, Rerank, LangChain / LlamaIndex integration
- **Multi-Agent** — `Agent.as_tool()` (lightweight composition), Swarm (parallel / autonomous), and Workflow (deterministic orchestration)
- **Actor-Critic Refinement** — `refine()` with parallel multi-critic review, `SchemaCritic` for zero-cost program-level validation, `AgentCritic` for heterogeneous strong-model gating, and automatic loop-detection early-stop
- **`/goal` Long-running Tasks** — `await agent.run_goal("xxx")` keeps pushing toward a goal, auto-judging completion, resuming, or pausing; supports token / wall-clock / turn hard caps; CLI `/goal /subgoal` ready out of the box, see [docs](https://shibing624.github.io/agentica/advanced/goals)
- **Guardrails** — Input / output / tool-level guardrails, streaming real-time detection
- **MCP / ACP** — Model Context Protocol and Agent Communication Protocol support
- **Skill System** — Markdown-based skill injection with project, user, and managed external skill directories
- **Persistent Memory** — Index/content separation, relevance-based recall, four-type classification, drift defense, optional sync into global `AGENTS.md`
- **Multi-Modal** — Text, image, audio, video understanding
- **Self-Evolution** — Experience cards auto-compile into reusable `SKILL.md` across sessions (pipeline below)

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/evo_pipeline.png" width="900" alt="Agentica Self-Evolution Pipeline" />
</div>

## Agent Recipes

### Custom tool composition

```python
from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFileTool, BuiltinExecuteTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[BuiltinWebSearchTool(), BuiltinFileTool(work_dir="./workspace"), BuiltinExecuteTool(work_dir="./workspace")],
)
agent.run_sync("Search Python 3.13 new features and write them to features.md")
```

### Full power (CLI / Gateway / long-running tasks)

```python
from agentica import DeepAgent
agent = DeepAgent()  # 40+ builtin tools + compression + long-term memory + skills + MCP, batteries-included
```

## CLI

```bash
agentica 
```

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/cli_snap.png" width="800" />

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

## Web UI / IM Integration

```bash
pip install -U "agentica[gateway]"
```

Launch:

```bash
agentica-gateway
```

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/agentica-web.png" width="800" />

By default it serves at `http://127.0.0.1:8881/chat`.

Supports IM (QQ / Feishu / WeChat / WeCom / Telegram / Discord / Slack). Supports scheduled tasks. 

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

## Documentation

Full documentation: **https://shibing624.github.io/agentica**

## Community & Support

- **GitHub Issues** — [Open an issue](https://github.com/shibing624/agentica/issues)
- **WeChat Group** — Add `xuming624` on WeChat, mention "llm" to join the developer group

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" />

## Citation

If you use Agentica in your research, please cite:

> Xu, M. (2026). Agentica: A Human-Centric Framework for Large Language Model Agent Workflows. GitHub. https://github.com/shibing624/agentica

## License

[Apache License 2.0](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- [phidatahq/phidata](https://github.com/phidatahq/phidata)
- [openai/openai-agents-python](https://github.com/openai/openai-agents-python)
