# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentica is a Python AI agent framework for building, managing, and deploying autonomous AI agents. It supports multi-agent teams, workflows, RAG, MCP tools, and a file-based workspace memory system. The project uses an **async-first** architecture where all core methods are natively async, with `_sync()` adapters for synchronous callers.

**Python >= 3.12 required.**

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run all tests
python -m pytest tests/ -v --tb=short

# Run a single test file
python -m pytest tests/test_agent.py -v

# Run a single test case
python -m pytest tests/test_agent.py::TestAgentInitialization::test_default_initialization -v

# CLI entry point
agentica
```

No Makefile, linter config, or type-checker is configured in this repo. CI runs `pytest` on Python 3.12.

## Architecture

### Async-First Convention

All core methods (`run`, `response`, `execute`, `invoke`) are **async by default**. The naming convention:

| Method | Type | Purpose |
|--------|------|---------|
| `run()` | async | Non-streaming run |
| `run_stream()` | async generator | Streaming run |
| `run_sync()` | sync adapter | Wraps `run()` via `run_sync()` utility |
| `run_stream_sync()` | sync iterator | Background thread + queue pattern |

There are no `a`-prefixed methods (`arun`, `aresponse`, etc.) — those were removed. Sync tools are executed via `loop.run_in_executor()` inside async context. Sync DB calls (session read/write) and file I/O (workspace) are wrapped in `loop.run_in_executor()` to avoid blocking the event loop.

### Agent (`agentica/agent/`)

The `Agent` class uses `@dataclass(init=False)` with explicit `__init__` and **direct multiple inheritance** from mixin classes. Mixins are pure method containers (no state, no `__init__`). IDE can jump directly to implementations.

```python
@dataclass(init=False)
class Agent(PromptsMixin, RunnerMixin, SessionMixin, TeamMixin, ToolsMixin, PrinterMixin, MediaMixin):
```

- `runner.py` — `RunnerMixin`: Core `_run_impl()` (the single execution engine), `run()`, `run_stream()`, `run_sync()`, `run_stream_sync()`
- `prompts.py` — `PromptsMixin`: System/user message construction. `get_system_message()`, `get_messages_for_run()`, and `_build_enhanced_system_message()` are **async** (they await workspace context/memory loading).
- `session.py` — `SessionMixin`: Session persistence. `read_from_storage()`, `write_to_storage()`, `load_session()`, `rename()`, `rename_session()`, `delete_session()`, `generate_session_name()`, `auto_rename_session()` are all **async** with DB calls wrapped in `run_in_executor`.
- `team.py` — `TeamMixin`: Multi-agent delegation. `as_tool()` creates **async** entrypoints that `await self.run()`. `get_transfer_function()` also uses async entrypoints.
- `tools.py` — `ToolsMixin`: Tool registration and management. `update_memory()` is **async**.
- `printer.py` — `PrinterMixin`: `print_response()` (async) + `print_response_sync()`
- `media.py` — `MediaMixin`: Image/video handling

Note: Agent uses `@dataclass` (not BaseModel) deliberately — it has mutable fields (Callable, lists), complex `__init__` with alias handling, and doesn't need Pydantic validation/serialization. Data models (RunResponse, Tool, Function) use BaseModel.

### Model (`agentica/model/`)

`Model` base class (`base.py`) defines four abstract async methods: `invoke()`, `invoke_stream()`, `response()`, `response_stream()`. Tool calls are executed in parallel via `asyncio.TaskGroup` in `run_function_calls()` (structured concurrency, each task catches its own exceptions to prevent sibling cancellation).

Providers are in subdirectories (e.g., `openai/`, `anthropic/`, `deepseek/`, `zhipuai/`). All extend the same async-only interface. Provider overrides use `@override` decorator (Python 3.12+) for `invoke()`, `invoke_stream()`, `response()`, `response_stream()`.

Key provider notes:
- `openai/chat.py` (`OpenAIChat`) — Uses native async OpenAI client
- `anthropic/claude.py` (`Claude`) — Uses `AsyncAnthropic` client with native async `messages.create()` and `messages.stream()`. `add_image()` and `format_messages()` are async.
- `openai/like.py` (`OpenAILike`) — Extends `OpenAIChat` for OpenAI-compatible APIs. Subclassed by DeepSeek, Doubao, Moonshot, Nvidia, Qwen, Yi, ZhipuAI, etc.

### Tools (`agentica/tools/base.py`)

Three-layer hierarchy: `Tool` (container) → `Function` (schema + entrypoint) → `FunctionCall` (invocation). `FunctionCall.execute()` is async-only and auto-detects sync/async entrypoints via `inspect.iscoroutinefunction()`. Functions hold a **weakref** to their parent Agent.

`_safe_validate_call()` wraps pydantic's `validate_call` to strip unresolvable forward-reference annotations (e.g., `self: "Agent"` on mixin methods) before validation.

**Builtin Tools** (`agentica/deep_tools.py`) — all I/O-bound tools are **async**:
- `edit_file(file_path, old_string, new_string, replace_all=False)` — flat params for LLM-friendly schema (no nested dict/list)
- `read_file` — async with `aiofiles` streaming read
- `write_file` — async with atomic write (`tempfile` + `os.replace`)
- `ls`, `glob` — async via `run_in_executor`
- `grep` — async ripgrep (`rg`) subprocess with pure-Python fallback
- `execute` — async subprocess with graceful SIGTERM→SIGKILL termination
- `web_search`, `fetch_url` — async wrappers around async backends
- `task` — async `run_stream()` to subagent
- `write_todos`, `read_todos` — sync (pure CPU, no I/O)

Flow control exceptions: `StopAgentRun`, `RetryAgentRun`.

### Guardrails (`agentica/guardrails/`)

Unified package for input/output validation at two levels:
- **Agent-level**: `InputGuardrail`, `OutputGuardrail` — validate entire agent runs
- **Tool-level**: `ToolInputGuardrail`, `ToolOutputGuardrail` — validate individual tool calls

Both use decorator pattern (`@input_guardrail`, `@tool_input_guardrail`) and support async functions.

### Memory

Two-tier system:
- **Runtime**: `AgentMemory` (`memory/agent_memory.py`) — in-memory conversation history with token-aware truncation
- **Persistent**: `Workspace` (`workspace.py`) — file-based storage using Markdown files (`AGENT.md`, `PERSONA.md`, `TOOLS.md`, `USER.md`, `MEMORY.md`) with multi-user isolation under `users/{user_id}/`. `get_context_prompt()`, `get_memory_prompt()`, `write_memory()`, `save_memory()` are **async** (file I/O via `run_in_executor`). `initialize()`, `read_file()`, `write_file()`, `append_file()` remain **sync** for init-time use.

### Prompts (`agentica/prompts/`)

Modular system prompt assembly via `PromptBuilder` (`builder.py`). Components:
- `base/` — Core prompt modules (soul, tools, heartbeat, task_management, self_verification, deep_agent)
- Each module has a `.py` loader and an `.md` template in `base/md/`
- Model-agnostic design — removed model-specific prompt optimizations
- No compact modes — single streamlined prompt per module
- Identity handled directly in builder (no separate identity module)
- Shared `load_prompt()` utility in `base/utils.py` (DRY)

### Workflow (`agentica/workflow.py`)

Deterministic multi-agent pipeline. `run()` is async; subclasses override it for step orchestration. `run_sync()` provided as adapter. Session storage (`read_from_storage`, `write_to_storage`, `load_session`) are **async** with DB calls wrapped in `run_in_executor`.

### Public API (`agentica/__init__.py`)

Uses **lazy loading** with thread-safe double-checked locking (`threading.Lock`) for optional/heavy modules (database backends, knowledge, vector DBs, embeddings, MCP, ACP, guardrails). Core modules (Agent, Model, Memory, Tools, Workspace) are imported eagerly.

## Key Patterns

- `@dataclass(init=False)` with explicit `__init__` for Agent, direct multiple inheritance from mixins
- Pydantic `BaseModel` for data structures (Model, Tool, Function, RunResponse, ToolCallInfo, Workflow)
- `Function.from_callable()` factory to auto-generate tool definitions from Python functions
- Token-aware message history via `AgentMemory.get_messages_from_last_n_runs()`
- `RunResponse` with `RunEvent` enum for structured event streaming
- `RunResponse.tool_calls` → `List[ToolCallInfo]` for flat attribute access (no nested `.get()` chains)
- `RunResponse.tool_call_times` → `Dict[str, float]` one-liner per-tool timing
- Langfuse integration for tracing (context manager pattern in runner)
- `@override` decorator (Python 3.12) on model provider method overrides
- `asyncio.TaskGroup` for structured concurrent tool execution (not `asyncio.gather`)
- All blocking I/O (DB, file system) wrapped in `loop.run_in_executor()` within async methods
- Tests for async methods use `asyncio.run()` and `unittest.mock.AsyncMock`
