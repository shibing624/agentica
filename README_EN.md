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

**Agentica** is not just a chat wrapper around an LLM API. It is an Async-First agent harness for running real agents:
tool calling, long-running task loops, multi-agent orchestration, cross-session memory, and skill-driven self-learning workflows.

| Capability | What it means |
|------------|---------------|
| **Long-running Agent Loop** | `Runner` manages the LLM ↔ tool loop with compression, retries, cost budgets, and loop safety |
| **Works Beyond Chat** | Files, execution, search, browser, MCP, multi-agent collaboration, and workflows instead of a single chat endpoint |
| **Memory That Survives Sessions** | Workspace memory is stored as indexed entries with relevance recall, and confirmed preferences can sync into `~/.agentica/AGENTS.md` |
| **Skill-Based Self-Learn** | SkillTool can load external skills, built-in agent self-learning strategy |
| **Self-Evolution** | Tool failures, user corrections, and success sequences become experience cards that auto-compile into reusable `SKILL.md` files across sessions |
| **Open Composable Harness** | Models, tools, memory, skills, guardrails, and MCP are replaceable building blocks instead of a closed hosted platform |

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

Set up your API keys first:

```bash
export OPENAI_API_KEY="sk-xxx"              # OpenAI
export ZHIPUAI_API_KEY="your-api-key"       # ZhipuAI (glm-4.7-flash is free)
export DEEPSEEK_API_KEY="your-api-key"      # DeepSeek
```

### Sync vs Async

| Your code style | Recommended API |
|---|---|
| Plain script / Jupyter / FastAPI route (default) | `agent.run_sync(...)`, `agent.print_response_sync(...)`, `for chunk in agent.run_stream_sync(...)` |
| Already inside an asyncio loop / want to `gather` N agents in parallel | `await agent.run(...)`, `async for chunk in agent.run_stream(...)` |

`run_sync` is just `asyncio.run(self.run(...))` under the hood, and tool calls
still run concurrently via `asyncio.gather`. **The sync API does not sacrifice
performance** — it just hides the event loop.

```python
import asyncio
from agentica import Agent, OpenAIChat

async def main():
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
    result = await agent.run("Describe Shanghai in one sentence")
    print(result.content)

asyncio.run(main())
```

### Recommended imports

Core SDK + builtin tools are exported at the top level — no long paths to remember:

```python
from agentica import (
    Agent, DeepAgent, Workspace, tool,
    OpenAIChat,                                       # openai is a hard dep
    BuiltinFileTool, BuiltinExecuteTool,              # files / shell
    BuiltinFetchUrlTool, BuiltinWebSearchTool,        # web
    BuiltinTodoTool, BuiltinTaskTool,                 # task list / sub-agent
    HistoryConfig, WorkspaceMemoryConfig, RunConfig,  # configs
)

# Other models / heavy tools live in submodules (avoid pulling heavy deps at startup)
from agentica.model.anthropic.claude import Claude   # pip install anthropic
from agentica.model.ollama.chat import Ollama
from agentica.tools.shell_tool import ShellTool
```

## Features

- **Async-First** — Native async API, `asyncio.gather()` parallel tool execution, sync adapter included
- **Runner Agentic Loop** — LLM ↔ tool-call auto-loop, multi-turn chain-of-thought, infinite-loop detection, cost budgeting, compression pipeline, API retry
- **20+ Models** — OpenAI / DeepSeek / Claude / ZhipuAI / Qwen / Moonshot / Ollama / LiteLLM and more
- **40+ Built-in Tools** — Search, code execution, file operations, browser, OCR, image generation
- **RAG** — Knowledge base management, hybrid retrieval, Rerank, LangChain / LlamaIndex integration
- **Multi-Agent** — `Agent.as_tool()` (lightweight composition), Swarm (parallel / autonomous), and Workflow (deterministic orchestration)
- **Actor-Critic Refinement** — `refine()` with parallel multi-critic review, `SchemaCritic` for zero-cost program-level validation, `AgentCritic` for heterogeneous strong-model gating, and automatic loop-detection early-stop
- **Guardrails** — Input / output / tool-level guardrails, streaming real-time detection
- **MCP / ACP** — Model Context Protocol and Agent Communication Protocol support
- **Skill System** — Markdown-based skill injection with project, user, and managed external skill directories
- **Multi-Modal** — Text, image, audio, video understanding
- **Persistent Memory** — Index/content separation, relevance-based recall, four-type classification, drift defense, and optional sync into global `AGENTS.md`

## Workspace Memory

Workspace provides persistent cross-session memory with index/recall design. When needed, confirmed user and feedback memories can also be compiled into global `~/.agentica/AGENTS.md` so new sessions inherit them automatically:

```python
from agentica import Workspace

workspace = Workspace("./workspace")
workspace.initialize()

# Write a typed memory entry (each entry is an individual file, index auto-updated)
await workspace.write_memory_entry(
    title="Python Style",
    content="User prefers concise, typed Python.",
    memory_type="feedback",              # user|feedback|project|reference
    description="python coding style",   # keywords for relevance scoring
    sync_to_global_agent_md=True,        # sync into ~/.agentica/AGENTS.md
)

# Relevance-based recall (returns top-k most relevant entries for the query)
memory = await workspace.get_relevant_memories(query="how to write python")
```

Agents automatically recall the most relevant memories for the current query, rather than dumping all memory:

```python
from agentica import DeepAgent, Workspace
from agentica.agent.config import WorkspaceMemoryConfig

agent = DeepAgent(
    workspace=Workspace("./workspace"),
    long_term_memory_config=WorkspaceMemoryConfig(
        max_memory_entries=5,  # inject at most 5 relevant memories
        sync_memories_to_global_agent_md=True,
    ),
)
```

`DeepAgent` enables `SkillTool(auto_load=True)` by default, so it automatically discovers skills from `~/.agentica/skills/` and `.agentica/skills/`; it also turns on `tool_config.auto_load_mcp=True`, which auto-loads `mcp_config.json/yaml/yml` from the working directory when present. In practice, `DeepAgent` now boots as a one-command runtime with skills + MCP + memory already wired in.

## Self-Evolution

Agentica doesn't just *remember facts* — it remembers *how to do things*. While the agent runs, every signal it produces (tool failures, user corrections, successful tool sequences) is captured as an **Experience event**, compiled into **experience cards**, and once the same rule is confirmed N times the framework **auto-generates a `SKILL.md`** under `<workspace>/generated_skills/`. The next session bootstraps a brand-new agent with `SkillTool` automatically discovering and injecting that skill — so newly-spawned agents inherit hard-earned procedural know-how without any manual glue.

The whole pipeline is local, auditable, and dependency-free. Deterministic capture (tool error / success) costs zero LLM tokens; only "user-correction classification" and "should we spawn a skill" use the `auxiliary_model`.

### Pipeline

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/evo_pipeline.png" width="900" alt="Agentica Self-Evolution Pipeline" />
</div>

Event capture (amber) → experience compilation & lifecycle (blue stores + dashed grey frame) → skill-spawn gate + LLM judgement (pink) → automatic reuse in the next session — fully auditable, zero external dependency.

### How to enable

Minimal change: attach `ExperienceCaptureHooks` to your agent and set `ExperienceConfig.skill_upgrade=SkillUpgradeConfig(mode="shadow")` to enable the full self-evolution loop end-to-end.

```python
from agentica import Agent, Workspace, OpenAIChat
from agentica.agent.config import ExperienceConfig, SkillUpgradeConfig
from agentica.hooks import (
    ConversationArchiveHooks,
    ExperienceCaptureHooks,
    MemoryExtractHooks,
    _CompositeRunHooks,
)

workspace = Workspace("./workspace", user_id="alice")
workspace.initialize()

model = OpenAIChat(id="gpt-4o-mini")

hooks = _CompositeRunHooks([
    ConversationArchiveHooks(),                # archive every conversation
    MemoryExtractHooks(),                      # LLM extracts long-term memories
    ExperienceCaptureHooks(
        ExperienceConfig(
            capture_tool_errors=True,          # deterministic, zero LLM cost
            capture_success_patterns=True,     # deterministic, zero LLM cost
            capture_user_corrections=True,     # classified by auxiliary_model
            feedback_confidence_threshold=0.6,
            promotion_count=3,                 # rule confirmed 3x → tier=hot
            skill_upgrade=SkillUpgradeConfig(
                mode="shadow",                 # off | draft | shadow
                min_repeat_count=3,            # need ≥3 repeats to consider spawn
                min_tier="warm",
                min_success_applications=1,    # set 0 for cold-start demos
            ),
        )
    ),
])

agent = Agent(
    model=model,
    auxiliary_model=model,                     # used by correction classify / skill judge
    workspace=workspace,
)
agent._default_run_hooks = hooks

# Run your normal workload — failures / corrections / successes accumulate in workspace
agent.run_sync("Read ./docs/agent.md for me")
```

After a few sessions the workspace looks like:

```
workspace/users/alice/
├── experiences/
│   ├── events.jsonl                        # all raw events (append-only)
│   ├── EXPERIENCE.md                       # card index
│   └── <hash>__list_dir_before_read.md     # compiled experience card
├── generated_skills/
│   ├── INDEX.md                            # L1 keyword router
│   └── list-dir-before-read/
│       ├── SKILL.md                        # auto-generated reusable skill
│       └── meta.json                       # status: shadow / draft / promoted
└── reports/learning/                       # per-run learning reports
```

Full e2e demo (Session 1 evolves a skill → Session 2 uses a fresh agent that consumes it across sessions): [`examples/self_evolution/01_self_evolution_e2e.py`](examples/self_evolution/01_self_evolution_e2e.py).

> **Trade-offs**: `mode="shadow"` installs locally to the workspace without affecting other users; `mode="draft"` only writes a draft for human review; `mode="off"` disables auto-skill generation but still captures experience cards. `min_success_applications` is the *"need ≥N tool_recovery events first"* safety gate — it prevents the loop from generating skills for tasks the agent never actually solved. Set to `0` only for cold-start demos.

## Actor-Critic Refinement (refine)

Agentica ships a **protocol-level Actor-Critic pattern**: an Actor produces a draft, multiple Critics review it in parallel, rejected drafts are revised against feedback, and the loop terminates either on unanimous approval or via early-stop. This is the same architecture validated by the [CarePilot paper (arXiv:2603.24157)](https://arxiv.org/abs/2603.24157) — a 7B fine-tuned model + Actor-Critic framework hit 48.9% task accuracy on a medical-GUI benchmark, ~13 points above zero-shot GPT-5; the ablation showed dropping the critic loop collapses task accuracy from 48.9% to 12.5%.

agentica's design principle here is **"the SDK provides protocol, not capability"**. Generic self-critique inside a strong base model belongs to the LLM; the SDK owns the three things no LLM can replace —

- **Business-constraint injection** — `SchemaCritic` enforces Pydantic schema validation as a deterministic, zero-LLM-cost verifier that beats any LLM critic on schema conformance
- **Auditable reflection trail** — `RefineResult.history` records every round's draft + per-critic verdict; the full process is observable and replayable
- **Heterogeneous composition** — cheap Actor + strong Critic, parallel multi-critic review, mixing deterministic and LLM-based verifiers — only the SDK layer can express this

```python
import asyncio
from typing import Literal
from pydantic import BaseModel, Field

from agentica import Agent, OpenAIChat
from agentica.critic import SchemaCritic, AgentCritic, CritiqueStyle, refine


class IntentReply(BaseModel):
    intent: Literal["question", "command", "complaint", "smalltalk"]
    confidence: float = Field(ge=0.0, le=1.0)


async def main():
    actor = Agent(
        name="classifier",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=(
            "Classify the user message as one of: question / command / complaint / smalltalk. "
            'Output JSON only: {"intent": "<one_of_the_four>", "confidence": <float in 0..1>}. '
            "No prose, no markdown code fences."
        ),
    )
    reviewer = Agent(
        name="reviewer",
        model=OpenAIChat(id="gpt-4o"),
        instructions="Check whether the classification is reasonable. Reply APPROVED if so, otherwise list issues.",
    )

    result = await refine(
        actor,
        task='Classify the following user message:\n"My phone keeps freezing, please refund me!"',
        critics=[
            SchemaCritic(IntentReply),                              # program-level (zero LLM cost)
            AgentCritic(reviewer, style=CritiqueStyle.STRICT),      # LLM-level (style-aware)
        ],
        max_iter=3,
    )

    print(result.final_draft)        # e.g. {"intent": "complaint", "confidence": 0.95}
    print(result.approved, result.stopped_reason, result.iterations)
    for r in result.history:         # per-round draft + each critic's verdict (auditable)
        print(r.draft, [(v.critic_name, v.approved) for v in r.verdicts])


asyncio.run(main())
```

**Key features**:

- `Critic` Protocol — duck-typed; a custom critic (regex / API call / any business rule) is < 20 lines
- Multi-critic parallel execution (`asyncio.gather`); freely mix LLM critics with zero-cost program critics
- `CritiqueStyle.STRICT/NEUTRAL/LENIENT` controls the LLM critic's reviewing temperament (paper recommends NEUTRAL as default)
- **Loop detection early-stop** — when two consecutive rounds produce identical verdicts, the loop terminates automatically, avoiding wasted tokens
- `RefineResult.history` exposes the full reflection trail and pairs with the CHAT log (`logger.chat`) for first-class multi-agent observability

**When NOT to use**: if the base model is GPT-5+ class and the task has no business-specific constraints, `refine()` will burn tokens for marginal gain. Use it when (1) output must satisfy a programmatic schema, (2) you want a heterogeneous actor + critic (cheap actor, strong critic), or (3) you have business red-lines that generic self-critique cannot enforce.

Full examples:

- [`examples/agent_patterns/13_actor_critic_refine.py`](examples/agent_patterns/13_actor_critic_refine.py) — canonical `refine()` usage with parallel `SchemaCritic` + `AgentCritic` and a per-round audit trail
- [`examples/agent_patterns/04_debate.py`](examples/agent_patterns/04_debate.py) — multi-agent debate using `AgentCritic` to extract structured rebuttals

## Agent Recipes

`Agent` has many parameters, but most production code uses one of these 5 templates. Copy-paste and adapt:

### One-shot script (minimal)

```python
agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
print(agent.run_sync("Describe Beijing in one sentence").content)
```

### Multi-turn conversation

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_context=True,
    num_history_turns=5,
)
agent.run_sync("My name is Alice, I'm an ML engineer.")
agent.run_sync("What is my name?")  # the model remembers
```

### Tool-based Agent (custom tool set)

```python
from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFileTool, BuiltinExecuteTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[BuiltinWebSearchTool(), BuiltinFileTool(work_dir="./workspace"), BuiltinExecuteTool(work_dir="./workspace")],
)
agent.run_sync("Search Python 3.13 new features and write them to features.md")
```

### Multi-user + long-term memory + session archive

One Agent instance per user. `session_id` is usually just the `user_id`:

```python
from agentica import Agent, OpenAIChat, Workspace, WorkspaceMemoryConfig

def create_agent(user_id: str) -> Agent:
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        workspace=Workspace("~/.agentica/workspace", user_id=user_id),
        session_id=user_id,                      # session log goes to ~/.agentica/projects/.../{user_id}.jsonl
        enable_long_term_memory=True,            # ← REQUIRED — opt-in switch
        long_term_memory_config=WorkspaceMemoryConfig(
            auto_archive=True,                   # archive each conversation after run()
            auto_extract_memory=True,            # LLM extracts memory entries
        ),
        add_history_to_context=True,
        num_history_turns=5,
    )
```

> **Common pitfall**: setting `long_term_memory_config` but forgetting `enable_long_term_memory=True` — all memory/archive features get silently ignored. Since v1.4.1, `Agent.__init__` now warns about this misconfiguration.

### Long-session token saving: customize history

Search-tool results are typically huge and rarely needed in later turns. You can drop them from history and truncate AI replies:

```python
from agentica import Agent, OpenAIChat, HistoryConfig

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_context=True,
    num_history_turns=10,
    history_config=HistoryConfig(
        excluded_tools=["search_*", "web_search"],   # drop matching tool results, paired tool_calls auto-stripped
        assistant_max_chars=200,                      # truncate AI replies to 200 chars
    ),
)
```

For more advanced filtering (strip user-prompt prefixes, drop messages by metadata, etc.), use the `history_filter` callback. See `examples/memory/03_history_filter.py`.

### Full power (CLI / Gateway / long-running tasks)

```python
from agentica import DeepAgent
agent = DeepAgent()  # 40+ builtin tools + compression + long-term memory + skills + MCP, batteries-included
```

## CLI

```bash
agentica --model_provider zhipuai --model_name glm-4.7-flash
```

Install an external skill pack:

```bash
agentica extensions install https://github.com/obra/superpowers
```

If you are already inside the interactive CLI, you can install and refresh skills in-place:

```text
> /extensions install https://github.com/obra/superpowers
> /extensions list
> /extensions remove learn-from-experience
> /extensions reload
```

Local directories and custom targets are also supported:

```bash
agentica extensions install /path/to/skill-repo --target-dir ~/.agentica/skills
```

If you install into a custom directory instead of a standard search path, add that directory to `AGENTICA_EXTRA_SKILL_PATH` so `DeepAgent` and the CLI can auto-discover it.

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/cli_snap.png" width="800" />

## Web UI

[agentica-gateway](https://github.com/shibing624/agentica-gateway) provides a web interface, and also supports Feishu App and WeCom direct integration with Agentica.

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
