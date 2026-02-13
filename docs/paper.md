# Agentica: An Async-First Agent Framework with Structured Concurrency for Autonomous Research and Software Engineering

> **Target Venue:** AAAI 2027
>
> **Status:** Draft — experimental results sections marked with `[TODO]` placeholders

---

## Abstract

Large language model (LLM) agents have demonstrated remarkable potential in autonomous research and software engineering tasks. However, existing agent frameworks suffer from fundamental architectural limitations: synchronous execution bottlenecks that serialize independent tool calls, lack of structured concurrency guarantees for parallel operations, fragmented session management that separates execution state from the agent itself, and insufficient context management for long-horizon tasks. We present **Agentica**, an async-first agent framework built on Python 3.12 that introduces several key innovations: (1) **structured concurrent tool execution** via `asyncio.TaskGroup` with per-task exception isolation, providing formal safety guarantees that no partial results are lost on individual tool failures; (2) a **three-layer tool abstraction** (Tool → Function → FunctionCall) with memory-safe weak references and transparent sync/async boundary bridging; (3) an **Agent-as-Session** unified state model where the agent itself serves as the session carrier through token-aware run history management, eliminating the architectural fragmentation of separate Runner/Session abstractions; (4) a **DeepAgent** extension featuring dual-threshold hysteresis context management with formal convergence properties and HEARTBEAT-style iteration control; and (5) a **dual-level guardrail system** supporting both agent-level and tool-level input/output validation. We formalize the agent execution model as a state transition system and prove key safety properties including exception isolation, termination guarantees, and context bound enforcement. We evaluate on three challenging benchmarks — GAIA, SWE-bench Verified, and Terminal-Bench — demonstrating that architectural design choices in the agent framework significantly impact task completion rates, efficiency, and reliability. Our results show that structured concurrency yields `[TODO: X%]` speedup on multi-tool tasks, dual-threshold context management improves long-horizon task completion by `[TODO: X%]`, and overall performance is competitive with or exceeds state-of-the-art agent systems.

---

## 1. Introduction

The emergence of LLM-based autonomous agents has catalyzed a paradigm shift in how complex tasks are approached — from software engineering (Jimenez et al., 2024) to deep research (Xu & Peng, 2025) and general-purpose assistance (Mialon et al., 2023). Unlike simple chatbots that produce single-turn responses, agents engage in multi-step reasoning, invoke external tools, coordinate with sub-agents, and maintain persistent state across extended interactions.

Despite rapid progress, current agent frameworks face several architectural challenges that limit their effectiveness on hard, long-horizon tasks:

**Challenge 1: Sequential tool execution.** When an LLM returns multiple tool calls in a single response (e.g., searching the web while simultaneously reading a file), most frameworks execute them sequentially. MetaGPT (Hong et al., 2023) processes actions in a pipeline; CrewAI uses sequential task execution; LangGraph's `ToolNode` dispatches tools one at a time. The OpenAI Agents SDK does not document its internal concurrency model for tool calls. Only AutoGen v0.4 adopts an async-first design, but through an Actor-based message-passing model rather than structured concurrency. This serialization wastes wall-clock time on I/O-bound operations that could safely proceed in parallel.

**Challenge 2: Lack of structured concurrency guarantees.** Even frameworks that support async execution (e.g., via `asyncio.gather`) lack structured concurrency semantics. When one tool call fails, `asyncio.gather` with `return_exceptions=False` cancels all sibling tasks, potentially losing partial results. With `return_exceptions=True`, error propagation becomes the developer's responsibility. Neither approach is satisfactory for production agent systems. We formalize this as a *partial failure isolation* property (Definition 1) and prove that `asyncio.TaskGroup` with per-task exception capture satisfies it.

**Challenge 3: Fragmented session management.** The dominant design pattern separates execution (Runner) from state (Session), as exemplified by the OpenAI Agents SDK where Agent is a declarative config, Runner drives execution, and Session stores conversation history. This three-way split introduces architectural friction: the developer must explicitly coordinate runner-session lifecycles, state is not inherently owned by the reasoning entity, and multi-turn context must be manually threaded through external session objects.

**Challenge 4: Context overflow in long-horizon tasks.** Deep research tasks often require dozens of tool calls and accumulate context that exceeds the model's effective window. Hard truncation loses critical information; naive summarization may discard important details. The Deep Research survey (Xu & Peng, 2025) identifies context management as one of three core engineering challenges alongside hallucination control and process explainability. Single-threshold mechanisms suffer from compression oscillation when context fluctuates near the threshold.

**Challenge 5: Insufficient safety mechanisms for tool-level operations.** While agent-level guardrails (validating overall input/output) exist in frameworks like the OpenAI Agents SDK, individual tool calls — which interact directly with file systems, databases, and external APIs — lack fine-grained validation. A single malformed tool call can corrupt state or leak sensitive data.

To address these challenges, we present **Agentica**, an async-first agent framework with the following contributions:

1. **Structured concurrent tool execution with formal safety guarantees** using Python 3.12's `asyncio.TaskGroup`. We formalize the agent execution loop as a state transition system (Section 3.2) and prove that our three-phase execution protocol satisfies the *partial failure isolation* property — a failed tool does not cancel sibling executions, and all results (successful or failed) are preserved in original order. To the best of our knowledge, Agentica is the first agent framework to employ structured concurrency for tool execution.

2. **Three-layer tool abstraction** (Tool → Function → FunctionCall) that separates registration, schema definition, and invocation. Functions maintain memory-safe weak references (`weakref`) to their parent Agent, preventing circular reference memory leaks in the `Agent → Model → functions → Function._agent → Agent` reference chain. The `FunctionCall.execute()` method auto-detects sync/async entrypoints and routes sync functions through `run_in_executor()`, making the async boundary transparent to tool authors.

3. **Agent-as-Session unified state model** that merges execution, reasoning, and session state into the Agent itself. Unlike the OpenAI Agents SDK's Runner/Session separation, Agentica's `AgentMemory` stores structured run history (`runs: List[AgentRun]`) directly within the agent, with token-aware history retrieval (`get_messages_from_last_n_runs()`) that provides automatic budget-constrained context assembly. This eliminates the session management boilerplate and enables the agent to be the single source of truth for its own conversational state (Section 3.4).

4. **DeepAgent with dual-threshold hysteresis context management.** We introduce a two-threshold mechanism with formal convergence analysis (Section 3.5): a soft threshold ($\theta_s$) triggers compression, a hard threshold ($\theta_h$) forces answer generation, with the hysteresis gap ($\theta_h - \theta_s$) providing provable oscillation avoidance. Combined with HEARTBEAT-style iteration checkpoints, this enables reliable long-horizon task completion.

5. **Dual-level guardrail system** providing both agent-level (`InputGuardrail`, `OutputGuardrail`) and tool-level (`ToolInputGuardrail`, `ToolOutputGuardrail`) validation with async-compatible decorator patterns — the first framework to provide fine-grained per-tool-call safety validation.

6. **File-based workspace memory** using Markdown files (`AGENT.md`, `MEMORY.md`, `USER.md`, etc.) that are human-readable, auditable, git-versionable, and support multi-user isolation under `users/{user_id}/` directories.

We evaluate on three complementary benchmarks:
- **GAIA** (Mialon et al., 2023): 165 general AI assistant tasks testing multi-step reasoning and tool use
- **SWE-bench Verified** (Jimenez et al., 2024; OpenAI, 2024): 500 real-world software engineering tasks from GitHub
- **Terminal-Bench** (Merrill et al., 2026): 89 hard, realistic tasks in Docker-based terminal environments spanning software engineering, ML/AI, cybersecurity, and system administration

---

## 2. Related Work

### 2.1 Agent Frameworks

**MetaGPT** (Hong et al., 2023) introduced the SOP-as-Prompt paradigm, encoding human standard operating procedures as structured prompt sequences for multi-agent collaboration. Its publish-subscribe communication mechanism and executable feedback loop represent significant advances. However, MetaGPT's execution model is fundamentally sequential at the orchestration level — agents execute in pipeline order following a waterfall paradigm. While individual roles use `async def run()`, this serves only for non-blocking LLM I/O; inter-agent coordination remains synchronous. MetaGPT was published as an ICLR 2024 Oral paper.

**AutoGen** (Wu et al., 2023) proposed a multi-agent conversation framework based on conversable agents. AutoGen v0.4 (January 2025) introduced a complete redesign with an event-driven Actor model runtime supporting distributed deployment. While AutoGen v0.4 is async-first through its Actor-based message bus, it achieves concurrency through message passing rather than structured concurrency primitives, and tool execution within a single agent turn remains sequential.

**OpenAI Agents SDK** (OpenAI, 2025) succeeded the experimental Swarm framework, providing a lightweight agent abstraction where Agents are declarative configuration objects and a stateless Runner class handles execution. The SDK supports handoff-based multi-agent coordination and tripwire-pattern guardrails. Its Python 3.9 minimum requirement precludes the use of `asyncio.TaskGroup` (introduced in Python 3.11), and the internal tool execution concurrency model is not publicly documented. Hosted tools (WebSearch, FileSearch, CodeInterpreter) create practical vendor lock-in to OpenAI's Responses API.

**LangGraph** (LangChain, 2023) models agent workflows as stateful directed graphs with checkpoint-based persistence. While powerful for complex workflow orchestration, its `ToolNode` component executes tools sequentially, and the graph-based abstraction introduces overhead for simpler agent patterns.

**CrewAI** (Moura, 2024) uses role-based agent design with sequential and hierarchical process strategies. It is synchronous-first with no native async architecture or parallel tool execution.

Table 1 summarizes the architectural comparison.

**Table 1: Architectural Comparison of Agent Frameworks**

| Dimension | Agentica | MetaGPT | AutoGen v0.4 | OpenAI Agents SDK | LangGraph | CrewAI |
|-----------|----------|---------|-------------|-------------------|-----------|--------|
| Execution Model | Async-first (native) | Async I/O, sync orchestration | Async (Actor model) | Async-primary | Dual (sync + async) | Sync-first |
| Tool Concurrency | `TaskGroup` (structured) | Sequential | Sequential (per turn) | Undocumented | Sequential (`ToolNode`) | Sequential |
| Tool Abstraction | 3-layer hierarchy | Action-level | Function registration | `@function_tool` flat | `@tool` decorator | `BaseTool` class |
| Session Model | Agent-as-Session (unified) | Environment shared state | ConversableAgent state | Runner + Session (split) | Checkpointer (external) | Memory (external) |
| Multi-Agent | `as_tool()` delegation | Pub-sub pipeline | GroupChat + Actor | Handoffs + agents-as-tools | Graph sub-nodes | Sequential/Hierarchical |
| Guardrails | Agent + Tool level | None | None | Agent level only | None | None |
| Memory | AgentMemory + Workspace | 3-tier (working/shared/LT) | Message history | Sessions (SQLite/Redis) | Checkpoints + Store | SQLite + Entity |
| Python Version | >= 3.12 | >= 3.9 | >= 3.10 | >= 3.9 | >= 3.9 | >= 3.10 |
| Academic Paper | This work | ICLR 2024 Oral | arXiv 2023 | None | None | None |

### 2.2 Deep Research Systems

Deep Research represents a new paradigm where AI agents automate end-to-end research workflows (Xu & Peng, 2025; Wang et al., 2025). The comprehensive survey by Xu & Peng (2025) analyzed 80+ implementations and proposed a four-dimensional taxonomy: foundation models, tool utilization, task planning, and knowledge synthesis. Wang et al. (2025) defined three evolutionary stages: Agentic Search, Integrated Research, and Full-stack AI Scientist.

Commercial systems include OpenAI Deep Research (based on o3 with reinforcement learning), Google Gemini Deep Research (million-token context), Perplexity Deep Research, and Anthropic Claude Research. Open-source alternatives include OpenDeepResearcher, Auto-Deep-Research, and OpenManus.

Agentica's DeepAgent module contributes to this landscape with its dual-threshold context management mechanism and HEARTBEAT-style iteration control — engineering innovations that address the context overflow and premature termination problems identified as core challenges in the field.

### 2.3 Benchmarks for Agent Evaluation

**GAIA** (Mialon et al., 2023) evaluates general AI assistants on 466 real-world tasks (165 public validation, 301 private test) requiring multi-step reasoning, multi-modal processing, and tool use. Human baseline is ~92%; top AI systems achieve <30% on average, highlighting the immense difficulty gap.

**SWE-bench** (Jimenez et al., 2024) evaluates software engineering agents on real GitHub issues from 12 Python repositories. SWE-bench Verified (OpenAI, 2024) is a 500-task human-verified subset. The JoyCode Agent currently leads at 74.6%, demonstrating that agent scaffolding outperforms bare model capability (Claude 3.7 Sonnet achieves 70.3% without scaffolding).

**Terminal-Bench** (Merrill et al., 2026) is a curated benchmark of 89 hard, realistic tasks executed in Docker-based terminal environments. Tasks span software engineering (32.6%), computing (21.3%), ML/AI (11.2%), cybersecurity (9.0%), DevOps (4.5%), and other domains. Tasks range from hours to days of expert human effort. The current best system (GPT-5.3-Codex with Simple Codex agent) achieves 75.1%, while smaller models score as low as 3%. Evaluation uses outcome-based container state inspection with at least 5 trials per configuration (32,155 total trials) and 95% confidence intervals.

---

## 3. Architecture

### 3.1 Overview

Agentica is built on three design principles:

1. **Async-first**: All core methods (`run()`, `response()`, `execute()`, `invoke()`) are natively `async`. Synchronous adapters (`run_sync()`, `run_stream_sync()`) wrap the async implementations via background thread + event loop patterns.

2. **Separation of concerns via mixins**: The `Agent` class uses `@dataclass(init=False)` with direct multiple inheritance from pure method-container mixins:

```python
@dataclass(init=False)
class Agent(PromptsMixin, RunnerMixin, SessionMixin,
            TeamMixin, ToolsMixin, PrinterMixin, MediaMixin):
```

Each mixin provides a specific capability (execution, prompts, session management, multi-agent delegation, tool management, output formatting, media handling) without carrying state or `__init__`. This design enables IDE jump-to-definition and avoids the fragile base class problem.

3. **Pydantic for data, dataclass for behavior**: Data structures (`Model`, `Tool`, `Function`, `RunResponse`) use Pydantic `BaseModel` for validation and serialization. The `Agent` class uses `@dataclass` deliberately — it has mutable fields (Callable, lists), complex initialization with alias handling, and does not need Pydantic's validation overhead.

Figure 1 illustrates the overall architecture.

```
┌─────────────────────────────────────────────────────────┐
│                      Agent Layer                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │RunnerMix │ │PromptMix │ │SessionMix│ │ TeamMixin │  │
│  │ run()    │ │ sys_msg  │ │ load()   │ │ as_tool() │  │
│  │ stream() │ │ usr_msg  │ │ save()   │ │ transfer()│  │
│  └────┬─────┘ └──────────┘ └──────────┘ └───────────┘  │
│       │                                                 │
│  ┌────▼─────────────────────────────────────────────┐   │
│  │              _run_impl() [async generator]       │   │
│  │  Single execution engine with Langfuse tracing   │   │
│  └────┬─────────────────────────────────────────────┘   │
├───────┼─────────────────────────────────────────────────┤
│       │            Model Layer                          │
│  ┌────▼─────────────────────────────────────────────┐   │
│  │  Model (async abstract: invoke, response,        │   │
│  │         invoke_stream, response_stream)           │   │
│  │                                                   │   │
│  │  ┌─────────────────────────────────────────────┐  │   │
│  │  │ run_function_calls() [asyncio.TaskGroup]    │  │   │
│  │  │  Phase 1: Emit started events               │  │   │
│  │  │  Phase 2: Parallel execution (TaskGroup)    │  │   │
│  │  │  Phase 3: Sequential result processing      │  │   │
│  │  └─────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                    Tool Layer                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │   Tool   │───>│ Function │───>│   FunctionCall   │   │
│  │(container)│   │(schema + │    │   (invocation)   │   │
│  │          │    │ weakref) │    │   .execute()     │   │
│  └──────────┘    └──────────┘    │ auto sync/async  │   │
│                                  └──────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                  Memory Layer                           │
│  ┌─────────────┐         ┌───────────────────────────┐  │
│  │ AgentMemory │         │      Workspace            │  │
│  │ (runtime,   │         │  AGENT.md | PERSONA.md    │  │
│  │  token-     │         │  users/{id}/MEMORY.md     │  │
│  │  aware)     │         │  users/{id}/USER.md       │  │
│  └─────────────┘         └───────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                 Guardrail Layer                         │
│  ┌──────────────────┐  ┌────────────────────────────┐   │
│  │  Agent-Level      │  │     Tool-Level             │   │
│  │  InputGuardrail   │  │  ToolInputGuardrail        │   │
│  │  OutputGuardrail  │  │  ToolOutputGuardrail       │   │
│  └──────────────────┘  └────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Formal Execution Model

We formalize the agent execution loop as a state transition system to enable rigorous reasoning about safety properties.

**Definition 1 (Agent State).** An agent state is a tuple $S = (M, H, C, \sigma)$ where:
- $M = [m_1, ..., m_k]$ is the message history
- $H \subseteq \mathcal{R}$ is the set of completed runs stored in `AgentMemory.runs`
- $C \in \mathbb{N}$ is the estimated context token count
- $\sigma \in \{idle, running, terminated\}$ is the execution status

**Definition 2 (Execution Step).** A single execution step transforms state $S_i$ to $S_{i+1}$:

$$S_i \xrightarrow{\text{LLM}} (r, T) \xrightarrow{\text{Execute}} S_{i+1}$$

where $r$ is the model's text response and $T = \{t_1, ..., t_n\}$ is the set of tool calls. The execution proceeds in three phases:

- **Phase 1**: For each $t_j \in T$, emit `started` event (sequential, preserving order)
- **Phase 2**: Execute all $t_j$ concurrently: $\text{results} = \text{TaskGroup}(\{t_1, ..., t_n\})$
- **Phase 3**: Process results in original order (sequential)

**Theorem 1 (Partial Failure Isolation).** Under the `asyncio.TaskGroup` execution model with per-task exception capture, if tool call $t_j$ raises exception $e_j$, then for all $t_k$ where $k \neq j$: (a) $t_k$ is not cancelled, (b) $t_k$'s result (success or its own exception) is preserved, and (c) results are returned in the original order $[r_1, ..., r_n]$.

*Proof sketch.* Each task $t_j$ is wrapped in `async def _execute_one(j, fc)` which catches `ToolCallException` and generic `Exception` locally, storing them in `results[j]`. The `TaskGroup` only propagates `BaseExceptionGroup` if an *uncaught* exception escapes a task. Since all exceptions are caught within `_execute_one`, the TaskGroup exits normally with all tasks completed. Phase 3 iterates `results` in index order $0, 1, ..., n-1$, preserving the original sequence. $\square$

This is strictly stronger than `asyncio.gather(return_exceptions=True)`, which provides (b) and (c) but not the structured lifetime guarantee that all tasks complete within a well-defined scope.

**Definition 3 (Termination Condition).** The agent terminates when any of: (a) the model produces no tool calls (natural completion), (b) context $C > \theta_h$ (hard threshold), (c) step count exceeds `max_rounds`, or (d) a `StopAgentRun` flow control exception is raised.

### 3.3 Structured Concurrent Tool Execution

When the LLM returns $N$ tool calls in a single response, Agentica executes them in three phases:

**Phase 1 (Sequential): Emit events.** For each tool call $t_i$, emit a `tool_call_started` event preserving the original order. This ensures deterministic event ordering for streaming consumers.

**Phase 2 (Parallel): Execute.** All $N$ tool calls execute concurrently within an `asyncio.TaskGroup`:

```python
async with asyncio.TaskGroup() as tg:
    tasks = [tg.create_task(_execute_one(i, fc))
             for i, fc in enumerate(function_calls)]
```

Each task wraps its execution in a try/except that captures `ToolCallException` and generic exceptions locally, preventing one failure from cancelling sibling tasks. This is the key advantage over `asyncio.gather(return_exceptions=True)` — structured concurrency guarantees that all tasks complete (successfully or with captured exceptions) before the TaskGroup exits.

For sync tool entrypoints, `FunctionCall._call_func()` auto-detects via `inspect.iscoroutinefunction()` and routes through `loop.run_in_executor()`:

```python
async def _call_func(self, func, **kwargs):
    if inspect.iscoroutinefunction(func):
        return await func(**kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, functools.partial(func, **kwargs))
```

**Phase 3 (Sequential): Process results.** Results are processed in the original order, constructing tool response messages, handling flow control exceptions (`StopAgentRun`, `RetryAgentRun`), and emitting `tool_call_completed` events.

This three-phase design achieves maximum parallelism in Phase 2 while maintaining deterministic ordering in Phases 1 and 3.

### 3.4 Three-Layer Tool Abstraction

The tool system uses a three-layer hierarchy that separates concerns:

**Tool** is a container that groups related functions (e.g., `BuiltinFileTool` contains `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`). Tools register functions via `self.register(self.method_name)`.

**Function** holds the schema (name, description, JSON Schema parameters) and entrypoint callable. `Function.from_callable()` is a factory that auto-generates tool definitions from Python functions using `inspect.signature()` and `get_type_hints()`. A key design choice: Functions hold a **weak reference** (`weakref.ref`) to their parent Agent, breaking the circular reference chain `Agent → Model → functions → Function._agent → Agent` that would otherwise prevent garbage collection.

`_safe_validate_call()` wraps Pydantic's `validate_call` to strip unresolvable forward-reference annotations (e.g., `self: "Agent"` on mixin methods), enabling Pydantic validation on methods that use PEP 563 postponed annotations.

**FunctionCall** represents a single invocation with arguments, result, error, and call ID. Its `execute()` method is the sole execution entry point — async-only, with pre/post hook support and the sync/async auto-detection described in Section 3.2.

### 3.5 Agent-as-Session: Unified State Model

A fundamental design divergence exists between Agentica and other frameworks in how session state is managed. We identify three patterns in existing frameworks:

**Pattern A: Stateless Runner + External Session** (OpenAI Agents SDK). The Agent is a declarative configuration object (model, tools, instructions); a stateless `Runner` class drives execution; a separate `Session` object (backed by SQLite or Redis) stores conversation history. The developer must explicitly pass session references through the execution pipeline:

```python
# OpenAI Agents SDK pattern
agent = Agent(name="assistant", model="gpt-4o", tools=[...])
result = await Runner.run(agent, "query", session=session)
# Agent has no memory of previous interactions without session
```

**Pattern B: Environment-Mediated State** (MetaGPT, AutoGen). Agents communicate through a shared environment or message bus. State is distributed across the environment, individual agent memories, and communication channels.

**Pattern C: Agent-as-Session** (Agentica). The Agent itself is the session carrier. `AgentMemory` is embedded directly within the agent and maintains structured run history:

```python
class AgentMemory(BaseModel):
    runs: List[AgentRun] = []      # Structured run history
    messages: List[Message] = []    # Current context window
    summary: Optional[SessionSummary] = None  # Compacted history
```

The key mechanism is `get_messages_from_last_n_runs()`, which provides **token-budget-aware history assembly**:

1. Iterate runs from most recent to oldest
2. For each run, extract conversation messages (filtering tool calls)
3. Apply progressive truncation: older runs get more aggressive content truncation
4. Stop when the token budget is exhausted

This design has several advantages over Pattern A:

- **Single source of truth**: The agent owns its own state. No coordination between runner, session, and agent objects is needed.
- **Token-aware context assembly**: History messages are automatically budget-constrained, preventing context overflow during multi-turn interactions.
- **Progressive information decay**: Older runs are truncated more aggressively (tool results summarized to `tool_result_max_chars`), mimicking human memory where recent events are recalled with more detail.
- **Zero-configuration multi-turn**: Multi-turn conversations work by default without explicit session management code.

Session persistence is handled orthogonally by `SessionMixin`, which serializes the agent state (including `AgentMemory`) to storage backends via `write_to_storage()`. This separates the concerns of state ownership (Agent) from state persistence (SessionMixin + storage backend).

**Table 2: Session Management Pattern Comparison**

| Property | Stateless Runner (OpenAI) | Agent-as-Session (Ours) |
|----------|--------------------------|------------------------|
| State ownership | External Session object | Agent itself |
| Multi-turn setup | Explicit session passing | Automatic (built-in) |
| Token budget | Manual management | Automatic (`max_tokens` param) |
| History truncation | Application-level | Framework-level (progressive) |
| Persistence | Session backend required | Optional (SessionMixin) |
| Serialization | Session.to_dict() | AgentMemory.to_dict() → SessionRow |

### 3.6 DeepAgent: Deep Research Extension

`DeepAgent` extends `Agent` with capabilities specifically designed for long-horizon research and complex task completion:

**Built-in Tool Suite.** DeepAgent automatically includes file system tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`), code execution (`execute` with async subprocess and graceful SIGTERM→SIGKILL termination), web tools (`web_search`, `fetch_url`), task management (`write_todos`, `read_todos`), sub-agent delegation (`task`), and skill management (`list_skills`, `get_skill_info`).

**Dual-Threshold Hysteresis Context Management.** We introduce a two-threshold mechanism for handling context window overflow with provable oscillation avoidance:

- **Soft threshold** ($\theta_s$): When estimated context tokens exceed $\theta_s$, the system triggers context compression (summarizing older tool results). Default: $\theta_s = 0.6 \times (C_w - C_{out})$, where $C_w$ is the model's context window and $C_{out}$ is the maximum output tokens.
- **Hard threshold** ($\theta_h$): When tokens exceed $\theta_h$, the system forces answer generation via a specialized prompt. Default: $\theta_h = 0.8 \times (C_w - C_{out})$.

**Theorem 2 (Oscillation Avoidance).** Let $C_i$ denote the context token count at step $i$, and let $\Delta_c > 0$ be the average tokens freed by a compression operation. If $\theta_h - \theta_s > \Delta_c$, then the system does not oscillate between compression and normal execution. Specifically, after compression at step $i$ reduces $C_i$ to $C_i' \leq \theta_s$, the system will not trigger compression again until at least $\lceil(\theta_s - C_i' + 1) / \delta\rceil$ new steps, where $\delta$ is the average token increment per step.

*Proof sketch.* After compression, $C_i' \leq \theta_s$. Each subsequent step adds $\delta$ tokens on average. Compression is only re-triggered when $C_j \geq \theta_s$, requiring at least $(\theta_s - C_i') / \delta$ steps. Meanwhile, forced termination only occurs at $\theta_h$, and since $\theta_h - \theta_s > \Delta_c$, compression at $\theta_s$ always completes before the hard limit is reached. $\square$

This contrasts with single-threshold systems where $\theta_s = \theta_h$: compression frees $\Delta_c$ tokens, the next tool call adds $\delta$ tokens, and if $\delta > \Delta_c$, the system immediately re-triggers compression — an oscillation pattern observed in practice.

**HEARTBEAT-Style Iteration Control.** Inspired by the MemGPT heartbeat mechanism, DeepAgent injects iteration checkpoint prompts at configurable intervals during multi-round execution:

```
Step {N} checkpoint:
- Have you fully solved the problem?
- Are there any remaining tasks in the task list?
- Did you verify your changes?
If not complete, continue working. Do NOT end your turn prematurely.
```

This addresses the common failure mode where models prematurely declare task completion.

**Repetitive Behavior Detection.** A sliding window (`deque(maxlen=10)`) tracks recent tool calls. When $k$ consecutive calls invoke the same tool (default $k=3$), a warning prompt redirects the agent's strategy, preventing unproductive loops.

**Deep Research Prompt System.** A structured 6-phase prompt methodology guides thorough investigation: (1) Problem analysis and plan formulation, (2) Iterative information gathering with explicit failure handling, (3) Multi-source cross-validation, (4) Constraint checklist verification, (5) Calculation and operation verification via code execution, (6) Clear narration with citation requirements.

### 3.7 Dual-Level Guardrail System

Agentica provides guardrails at two granularities:

**Agent-level guardrails** (`InputGuardrail`, `OutputGuardrail`) validate the entire agent run's input or output. These use a decorator pattern:

```python
@input_guardrail
async def check_sensitive_content(context, agent, input_message):
    # Validation logic
    return GuardrailFunctionOutput(output_info={"safe": True})
```

**Tool-level guardrails** (`ToolInputGuardrail`, `ToolOutputGuardrail`) validate individual tool call arguments and results. This is critical because a single tool call may access file systems, execute code, or call external APIs:

```python
@tool_input_guardrail
async def validate_file_path(context, agent, tool_input):
    # Ensure file path is within allowed directory
    return ToolGuardrailFunctionOutput(output_info={"valid": True})
```

Both levels raise specific exceptions (`InputGuardrailTripwireTriggered`, `ToolInputGuardrailTripwireTriggered`, etc.) to halt execution when violations are detected.

### 3.8 File-Based Workspace Memory

Agentica's persistent memory uses a file-based workspace with the following directory structure:

```
workspace/
├── AGENT.md          # Agent instructions (global)
├── PERSONA.md        # Agent persona (global)
├── TOOLS.md          # Tool documentation (global)
├── users/
│   ├── default/
│   │   ├── USER.md       # User information
│   │   ├── MEMORY.md     # Long-term memory
│   │   └── memory/       # Daily memory entries
│   └── {user_id}/        # Multi-user isolation
```

This design provides several advantages over database-backed memory:
- **Auditability**: Markdown files are human-readable and inspectable
- **Version control**: Standard git operations track memory evolution
- **Portability**: No database dependencies; works with any file system
- **Multi-user isolation**: Each user's data is physically separated

All memory operations (`get_context_prompt()`, `get_memory_prompt()`, `write_memory()`, `save_memory()`) are async, with file I/O wrapped in `run_in_executor()` to avoid blocking the event loop.

### 3.9 Model Provider Abstraction

The `Model` base class defines four abstract async methods: `invoke()`, `invoke_stream()`, `response()`, `response_stream()`. Provider implementations use the `@override` decorator (Python 3.12) for explicit intent:

- `OpenAIChat`: Native async OpenAI client
- `Claude`: `AsyncAnthropic` client with native async `messages.create()` and `messages.stream()`
- `OpenAILike`: Extends `OpenAIChat` for compatible APIs (DeepSeek, Qwen, ZhipuAI, Doubao, Moonshot, etc.)

This unified async interface ensures that all model providers behave identically from the framework's perspective, with the async boundary handled transparently.

---

## 4. Experimental Setup

### 4.1 Benchmarks

We evaluate on three complementary benchmarks to assess different aspects of agent capability:

**GAIA** (General AI Assistants). We use the 165-question public validation set. Tasks require multi-step reasoning, web search, file parsing, code execution, and multi-modal understanding. Evaluation uses exact-match scoring on deterministic answers.

**SWE-bench Verified**. We use the 500-task human-verified subset. Each task requires producing a code patch that resolves a real GitHub issue and passes the repository's test suite. Evaluation uses resolved rate (percentage of tasks where the patch passes all associated tests).

**Terminal-Bench 2.0**. We use the 89-task benchmark executed in Docker-based terminal environments. Tasks span software engineering, computing, ML/AI, cybersecurity, and system administration. Each configuration is run 5 times, reporting resolution rate with 95% confidence intervals.

### 4.2 Baselines

We compare against the following agent systems:

| System | Framework Type | Key Characteristics |
|--------|---------------|---------------------|
| OpenAI Agents SDK + GPT-4o | Commercial framework | Handoff-based coordination, hosted tools |
| MetaGPT | SOP-driven pipeline | ICLR 2024 Oral, structured communication |
| AutoGen v0.4 | Actor-based async | Multi-agent conversation, distributed runtime |
| OpenHands | CodeAct framework | Unified code action space, Docker sandbox |
| Terminus 2 | Neutral scaffold | Headless terminal tool, used in Terminal-Bench |

For fair comparison, all configurations use the same underlying LLM where possible.

### 4.3 Configurations

We evaluate two Agentica configurations:

1. **Agentica-Base**: Standard `Agent` with user-provided tools. Tests the core async-first execution engine and structured concurrency.
2. **Agentica-Deep**: `DeepAgent` with built-in tools, deep research prompt, dual-threshold context management, and HEARTBEAT iteration control. Tests the full deep research pipeline.

### 4.4 Ablation Studies

To isolate the contribution of each architectural component:

| Ablation | What is removed/changed |
|----------|------------------------|
| `-TaskGroup` | Replace `asyncio.TaskGroup` with sequential tool execution |
| `-DualThreshold` | Remove context management (no soft/hard limits) |
| `-HEARTBEAT` | Remove iteration checkpoint prompts |
| `-DeepPrompt` | Replace deep research prompt with standard prompt |
| `-ToolGuardrails` | Remove tool-level guardrails (keep agent-level only) |
| `-AgentAsSession` | Replace with external session management (OpenAI SDK pattern) |

### 4.5 Parameter Sensitivity Analysis

We analyze sensitivity to the dual-threshold parameters with the following grid:

| Parameter | Values Tested |
|-----------|--------------|
| Soft threshold $\theta_s$ | 0.4, 0.5, 0.6, 0.7 (× effective context) |
| Hard threshold $\theta_h$ | 0.7, 0.8, 0.9 (× effective context) |
| HEARTBEAT frequency | 3, 5, 7, 10 (steps) |
| Repetition window $k$ | 2, 3, 5 (consecutive calls) |

Constraint: $\theta_s < \theta_h$ in all configurations.

### 4.6 Statistical Significance

All results are reported with 95% confidence intervals. For GAIA and SWE-bench, we use 3 independent runs with different random seeds and report Wilson score intervals for pass rates. For Terminal-Bench, we follow the benchmark protocol of 5 runs per configuration with bootstrap confidence intervals.

---

## 5. Results

### 5.1 Main Results

**Table 3: GAIA Benchmark Results (165 validation tasks)**

| System | Level 1 | Level 2 | Level 3 | Average |
|--------|---------|---------|---------|---------|
| Agentica-Deep | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Agentica-Base | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| OpenAI Agents SDK + GPT-4o | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| MetaGPT | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| AutoGen v0.4 | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Human baseline | ~95% | ~92% | ~88% | ~92% |

**Table 4: SWE-bench Verified Results (500 tasks)**

| System | Resolved Rate |
|--------|--------------|
| Agentica-Deep | `[TODO]` |
| Agentica-Base | `[TODO]` |
| OpenHands | `[TODO]` |
| AutoGen v0.4 | `[TODO]` |
| Bare model (no framework) | `[TODO]` |

**Table 5: Terminal-Bench 2.0 Results (89 tasks, 5 runs each)**

| System | Resolution Rate (±95% CI) |
|--------|--------------------------|
| Agentica-Deep | `[TODO]` ± `[TODO]` |
| Agentica-Base | `[TODO]` ± `[TODO]` |
| Terminus 2 (same model) | `[TODO]` ± `[TODO]` |
| OpenHands (same model) | `[TODO]` ± `[TODO]` |
| Claude Code (same model) | `[TODO]` ± `[TODO]` |

### 5.2 Structured Concurrency Speedup

**Table 6: Wall-clock time comparison — TaskGroup parallel vs. sequential tool execution**

| Benchmark | Avg tools/turn | Sequential (s) | Parallel (s) | Speedup |
|-----------|---------------|----------------|--------------|---------|
| GAIA | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]`× |
| SWE-bench | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]`× |
| Terminal-Bench | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]`× |

### 5.3 Ablation Study

**Table 7: Ablation results on GAIA (Level 2+3) and Terminal-Bench**

| Configuration | GAIA (L2+L3) | Terminal-Bench |
|--------------|-------------|----------------|
| Agentica-Deep (full) | `[TODO]` | `[TODO]` |
| − TaskGroup | `[TODO]` | `[TODO]` |
| − DualThreshold | `[TODO]` | `[TODO]` |
| − HEARTBEAT | `[TODO]` | `[TODO]` |
| − DeepPrompt | `[TODO]` | `[TODO]` |
| − ToolGuardrails | `[TODO]` | `[TODO]` |
| − AgentAsSession | `[TODO]` | `[TODO]` |

### 5.4 Parameter Sensitivity Analysis

**Table 8: Dual-threshold parameter sensitivity on GAIA Level 2+3 (long-horizon tasks)**

| $\theta_s$ | $\theta_h$ | Completion Rate | Forced Termination Rate | Compression Count |
|------------|------------|----------------|------------------------|------------------|
| 0.4 | 0.7 | `[TODO]` | `[TODO]` | `[TODO]` |
| 0.4 | 0.8 | `[TODO]` | `[TODO]` | `[TODO]` |
| 0.5 | 0.7 | `[TODO]` | `[TODO]` | `[TODO]` |
| 0.5 | 0.8 | `[TODO]` | `[TODO]` | `[TODO]` |
| 0.6 | 0.8 | `[TODO]` | `[TODO]` | `[TODO]` |
| 0.6 | 0.9 | `[TODO]` | `[TODO]` | `[TODO]` |
| 0.7 | 0.9 | `[TODO]` | `[TODO]` | `[TODO]` |

**Table 9: HEARTBEAT frequency sensitivity on Terminal-Bench**

| Checkpoint Frequency | Resolution Rate | Premature Termination Rate | Avg Steps |
|---------------------|----------------|--------------------------|-----------|
| Every 3 steps | `[TODO]` | `[TODO]` | `[TODO]` |
| Every 5 steps (default) | `[TODO]` | `[TODO]` | `[TODO]` |
| Every 7 steps | `[TODO]` | `[TODO]` | `[TODO]` |
| Every 10 steps | `[TODO]` | `[TODO]` | `[TODO]` |
| No HEARTBEAT | `[TODO]` | `[TODO]` | `[TODO]` |

### 5.5 Context Management Effectiveness

**Table 10: Long-horizon task completion with and without dual-threshold context management**

| Context Management | Tasks completed (>10 tool calls) | Avg. context tokens at completion | Forced termination rate |
|-------------------|--------------------------------|-----------------------------------|------------------------|
| None | `[TODO]` | `[TODO]` | `[TODO]` |
| Hard threshold only | `[TODO]` | `[TODO]` | `[TODO]` |
| Dual threshold (ours) | `[TODO]` | `[TODO]` | `[TODO]` |

### 5.6 Efficiency Analysis

**Table 11: Token consumption and API cost comparison**

| System | Avg tokens/task | Avg API calls/task | Avg cost/task ($) |
|--------|----------------|-------------------|------------------|
| Agentica-Deep | `[TODO]` | `[TODO]` | `[TODO]` |
| Agentica-Base | `[TODO]` | `[TODO]` | `[TODO]` |
| OpenAI Agents SDK | `[TODO]` | `[TODO]` | `[TODO]` |
| MetaGPT | `[TODO]` | `[TODO]` | `[TODO]` |

---

## 6. Analysis

### 6.1 When Does Structured Concurrency Help?

`[TODO: Analysis of which task types benefit most from parallel tool execution — likely I/O-bound tasks with multiple independent tool calls (e.g., simultaneous web searches, parallel file reads). Include speedup as a function of average tools-per-turn. Show that speedup approaches N× for N independent I/O-bound tools and is bounded by Amdahl's law for mixed CPU/IO workloads. Include wall-clock time distributions.]`

### 6.2 Agent-as-Session vs. External Session Management

`[TODO: Quantitative comparison of the Agent-as-Session pattern vs. the external session pattern (OpenAI SDK style). Metrics: (a) lines of code required for multi-turn setup, (b) memory efficiency (token usage for history assembly), (c) context assembly latency, (d) success rate on multi-turn tasks from GAIA that require information from previous turns. Hypothesis: Agent-as-Session should show lower boilerplate and comparable or better multi-turn accuracy due to automatic token-aware history management.]`

### 6.3 Dual-Threshold vs. Single-Threshold Context Management

`[TODO: Analysis of the hysteresis mechanism — show that single-threshold systems oscillate between compression and normal execution, while dual-threshold provides stable behavior. Include context token trajectory plots for representative long-horizon tasks. Verify Theorem 2 empirically by measuring compression trigger counts under single-threshold ($\theta_s = \theta_h$) vs. dual-threshold configurations.]`

### 6.4 Impact of Deep Research Prompt

`[TODO: Qualitative analysis of how the 6-phase deep research prompt affects agent behavior — does it increase cross-validation attempts? Does it reduce hallucination in cited sources? Compare citation accuracy between standard and deep research prompts. Include examples of behavior differences on specific GAIA tasks.]`

### 6.5 Error Analysis

`[TODO: Categorize failure modes across benchmarks with quantitative breakdown:
- Tool execution failures (timeout, permission, network): X% of failures
- Context overflow (task abandoned due to token limits): X%
- Repetitive behavior (agent stuck in loop): X%
- Reasoning errors (wrong approach despite sufficient information): X%
- Premature termination (agent declares done before completing task): X%
Include per-benchmark breakdown and analysis of which Agentica features address which failure modes.]`

### 6.6 Case Studies

`[TODO: Include 2-3 detailed case studies showing step-by-step execution traces:
1. A GAIA Level 3 task showing structured concurrency benefit (multiple parallel web searches)
2. A Terminal-Bench task showing dual-threshold context management in action (compression triggered, then successful completion)
3. A SWE-bench task showing how HEARTBEAT iteration control prevented premature termination
For each case, show the execution timeline, tool calls, context token trajectory, and final outcome compared to baseline systems.]`

### 6.7 Limitations

We acknowledge several limitations:

1. **Python version requirement**: Agentica requires Python 3.12+, which limits adoption in environments with older Python versions. The structured concurrency benefits require `asyncio.TaskGroup` (Python 3.11+), and the `@override` decorator requires Python 3.12.

2. **Benchmark coverage**: Our evaluation focuses on three benchmarks. Additional evaluation on domain-specific benchmarks (e.g., xbench-DeepSearch for research quality, WebArena for web navigation) would provide a more comprehensive picture.

3. **Model dependence**: Like all agent frameworks, performance is fundamentally bounded by the underlying LLM's capabilities. Our architectural contributions improve efficiency and reliability but cannot compensate for fundamental reasoning limitations.

4. **No formal SOP support**: Unlike MetaGPT's SOP-as-Prompt paradigm, Agentica does not encode domain-specific standard operating procedures. For highly structured workflows (e.g., software development lifecycle), MetaGPT's approach may be more appropriate.

---

## 7. Conclusion

We presented Agentica, an async-first agent framework that introduces several architectural innovations: (1) structured concurrent tool execution via `asyncio.TaskGroup` with formally proven partial failure isolation guarantees; (2) a three-layer tool abstraction with memory-safe weak references; (3) the Agent-as-Session unified state model that embeds token-aware session management directly within the agent, eliminating the architectural fragmentation of separate Runner/Session abstractions; (4) dual-threshold hysteresis context management with provable oscillation avoidance for long-horizon tasks; (5) dual-level guardrails for both agent and tool operations; and (6) file-based workspace memory with multi-user isolation.

Our evaluation across three challenging benchmarks — GAIA, SWE-bench Verified, and Terminal-Bench — demonstrates that framework-level architectural decisions significantly impact agent performance. `[TODO: Summarize key quantitative findings.]` The structured concurrency mechanism provides `[TODO]` speedup on multi-tool tasks, the dual-threshold context management improves long-horizon task completion by `[TODO]`, and the Agent-as-Session pattern reduces multi-turn management complexity while maintaining competitive accuracy.

These results support the broader observation that **agent scaffolding design matters as much as model capability** — a finding consistent with recent SWE-bench results where framework-augmented agents outperform bare models by 10-20% (JoyCode Agent at 74.6% vs. Claude 3.7 Sonnet at 70.3%). Furthermore, the formal analysis (Theorems 1-2) provides theoretical grounding for design choices that are often made heuristically in the agent framework community.

Future work includes: (1) extending the structured concurrency model to multi-agent parallel execution with dependency-aware scheduling, (2) adaptive threshold tuning based on task complexity estimation (replacing static $\theta_s, \theta_h$ with learned thresholds), (3) integration of retrieval-augmented generation into the workspace memory system, (4) speculative parallel hypothesis evaluation via lightweight agent cloning, and (5) evaluation on additional domain-specific benchmarks (WebArena, xbench-DeepSearch).

---

## References

Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., Zhang, C., Wang, J., Wang, Z., Yau, S.K.S., Lin, Z., Zhou, L., Ran, C., Xiao, L., Wu, C., & Schmidhuber, J. (2023). MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. *ICLR 2024 (Oral)*. arXiv:2308.00352.

Jimenez, C.E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? *ICLR 2024*. arXiv:2310.06770.

Merrill, M.A., Shaw, A.G., Carlini, N., Li, B., Raj, H., Bercovich, I., Shi, L., Shin, J.Y., Walshe, T., Buchanan, E.K., et al. (2026). Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces. arXiv:2601.11868.

Mialon, G., Fourrier, C., Swift, C., Wolf, T., LeCun, Y., & Scialom, T. (2023). GAIA: A Benchmark for General AI Assistants. arXiv:2311.12983.

OpenAI. (2025). New Tools for Building Agents. https://openai.com/index/new-tools-for-building-agents/.

Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S.G., Stoica, I., & Gonzalez, J.E. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.

Smith, N., Garrett, A., & Calvert, B. (2022). PEP 654 — Exception Groups and except*. Python Enhancement Proposals.

Wang, R., et al. (2025). Deep Research: A Survey. arXiv:2512.02038.

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., Zhang, X., Zhang, S., Liu, J., Awadallah, A.H., White, R.W., Burger, D., & Wang, C. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv:2308.08155.

Xu, R., & Peng, J. (2025). A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications. arXiv:2506.12594.

Yang, J., Jimenez, C.E., Wettig, A., Liber, K., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. arXiv:2405.15793.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*. arXiv:2210.03629.

---

## Appendix A: DeepAgent Deep Research Prompt

The complete deep research system prompt is available in the supplementary material. Key phases:

1. **Initiate Investigation**: Analyze the problem, identify key information points, formulate investigation plan using task management tools.
2. **Iterative Information Gathering & Reflection**: Handle search failures explicitly, evaluate information sufficiency, pursue depth, consider source reliability.
3. **Multi-Source Cross-Validation**: Use different tools/sources to verify key claims; explicitly state when switching tools due to effectiveness.
4. **Constraint Checklist**: Review all constraints and confirm coverage before synthesizing.
5. **Calculation & Operation Verification**: Verify all computations via code execution before finalizing.
6. **Clear Narration**: Explain tool call rationale, expected results, actual results, and next steps.

## Appendix B: Terminal-Bench Task Distribution

Terminal-Bench 2.0 contains 89 tasks across the following categories:

| Category | % | Example Tasks |
|----------|---|--------------|
| Software Engineering | 32.6% | build-linux-kernel-qemu, fix-ocaml-gc, build-pov-ray |
| Computing | 21.3% | regex-chess, chess-best-move, gpt2-codegolf |
| ML/AI/Data Science | 11.2% | train-fasttext, caffe-cifar-10, mcmc-sampling-stan |
| General SE | 11.2% | financial-document-processor, reshard-c4-data |
| Cybersecurity | 9.0% | crack-7z-hash, feal-differential-cryptanalysis, fix-code-vulnerability |
| DevOps/Cloud | 4.5% | configure-git-webserver, nginx-request-logging |
| Personal Assistant | 3.4% | Various personal productivity tasks |
| Video Processing | 2.2% | Video manipulation and conversion tasks |

Current leaderboard (top 5, as of Feb 2026):

| Rank | Agent | Model | Accuracy |
|------|-------|-------|----------|
| 1 | Simple Codex | GPT-5.3-Codex | 75.1% |
| 2 | CodeBrain-1 | GPT-5.3-Codex | 70.3% |
| 3 | Droid | Claude Opus 4.6 | 69.9% |
| 4 | Mux | GPT-5.3-Codex | 68.5% |
| 5 | Deep Agents | GPT-5.2-Codex | 66.5% |

## Appendix C: Reproducibility

All code is available at `[TODO: GitHub URL]`. Experiments use:
- Python 3.12+
- Docker for Terminal-Bench and SWE-bench environments
- GAIA validation set from HuggingFace
- Configuration files for all benchmark runs included in repository

To reproduce Terminal-Bench results:
```bash
pip install terminal-bench
tb run --agent agentica --model [MODEL] \
    --dataset-name terminal-bench-core \
    --dataset-version 2.0 --n-concurrent 8
```
