# AGENT.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

Agentica is a Python framework for building AI agents with support for multi-model LLMs, tools, multi-turn conversations, RAG, workflows, MCP integration, and a Skills system.

## Common Commands

### Installation
```bash
pip install -U agentica          # From PyPI
pip install .                     # From source (development)
```

### Running Tests
```bash
python -m pytest tests/                        # Run all tests
python -m pytest tests/test_agent.py           # Run single test file
python -m pytest tests/test_agent.py::TestAgentInitialization  # Run specific test class
python -m pytest tests/test_agent.py -k "test_default"         # Run tests matching pattern
```

### CLI Usage
```bash
agentica                                      # Interactive mode
agentica --query "Your question"              # Single query
agentica --model_provider zhipuai --model_name glm-4.7-flash  # Specify model
agentica --tools calculator shell wikipedia   # Enable specific tools
```

## Architecture

### Core Components (`agentica/`)

| File | Purpose |
|------|---------|
| `agent.py` | Core `Agent` class - main entry point for building agents |
| `deep_agent.py` | `DeepAgent` - Agent with built-in file/execute/search tools |
| `tools/buildin_tools.py` | Built-in tools for DeepAgent (ls, read_file, execute, web_search, etc.) |
| `workspace.py` | `Workspace` - Agent workspace with AGENT.md, PERSONA.md, memory management |
| `memory/` | `AgentMemory`, `MemoryManager`, `WorkspaceMemorySearch` - session and long-term memory |
| `guardrails.py` | Input/output validation and filtering for agents |
| `workflow.py` | Workflow engine for multi-step task orchestration |
| `cli/` | Interactive CLI with file references (@filename) and commands (/help) |
| `run_response.py` | `RunResponse`, `RunEvent` - agent execution responses |
| `prompts/` | Centralized prompt templates (base prompts, model-specific optimizations) |

### Model Layer (`agentica/model/`)

| File | Purpose |
|------|---------|
| `base.py` | Abstract `Model` base class |
| `message.py` | `Message` class for conversation messages |
| `response.py` | `ModelResponse` for LLM responses |
| `openai/` | OpenAI API integration (GPT-4, GPT-3.5, etc.) |
| `anthropic/` | Anthropic API integration (Claude models) |
| `zhipuai/` | ZhipuAI API integration (GLM models) |
| `deepseek/` | DeepSeek API integration |
| `ollama/` | Ollama local model integration |

### Tools System (`agentica/tools/`)

| File | Purpose |
|------|---------|
| `base.py` | `Tool` base class and tool registration |
| `registry.py` | Global tool registry and discovery |
| `calculator.py` | Math calculations |
| `shell.py` | Shell command execution |
| `file_tools.py` | File system operations |
| `web_search.py` | Web search capabilities |
| `wikipedia.py` | Wikipedia search |
| `python_repl.py` | Python code execution |

### ACP (Agent Client Protocol) (`agentica/acp/`)

**NEW**: ACP support for IDE integration (Zed, JetBrains, etc.)

| File | Purpose |
|------|---------|
| `server.py` | `ACPServer` - Main ACP server for IDE integration |
| `protocol.py` | JSON-RPC protocol handler |
| `handlers.py` | ACP method handlers (initialize, tools/list, tools/call, etc.) |
| `types.py` | ACP data models (ACPRequest, ACPResponse, ACPTool, etc.) |

**Usage:**
```bash
agentica acp  # Start ACP server mode
```

**IDE Configuration:**
```json
{
  "agent_servers": {
    "Agentica": {
      "command": "agentica",
      "args": ["acp"],
      "env": {"OPENAI_API_KEY": "..."}
    }
  }
}
```

---

# 开发历史记录

## 最近更新 (2026-02-18)

### 中间注入消息的 role 和顺序修复

#### 1. tool_call_hook warnings 延迟追加（消息顺序修复）
- **问题**：Phase 1 中 `tool_call_hook` 产生的 repetitive behavior warning 直接 append 到 `function_call_results`，排在 tool results 之前，破坏了 OpenAI API 要求的 `assistant(tool_calls) → tool` 必需配对序列
- **解决方案**：引入 `deferred_warnings` 列表，Phase 1 中收集 warnings，Phase 3 所有 tool results 处理完后再 extend 到 `function_call_results`
- **修改文件**：`agentica/model/base.py`（`run_function_calls` 方法）

#### 2. 所有注入的功能性 prompt 统一改用 `role: "user"`
- **问题**：`role: "system"` 的中间消息在部分 LLM API 中不兼容（某些模型不支持 system 出现在非首位，或不允许 system 紧接 assistant）
- **解决方案**：将以下 5 处 `Message(role="system", ...)` 统一改为 `Message(role="user", ...)`：
  - `model/base.py` Phase 0：`force_answer` 消息
  - `model/base.py` Phase 1：`repetitive_behavior` / `force_strategy_change` warnings
  - `deep_agent.py` `post_tool_hook`：`step_reflection` prompt
  - `deep_agent.py` `post_tool_hook`：`iteration_checkpoint` prompt
- **修改文件**：`agentica/model/base.py`、`agentica/deep_agent.py`

#### 修改后的消息序列
```
[assistant] (tool_calls: [call_1, call_2])
[tool]      tool result for call_1          ← tool results 紧跟 assistant
[tool]      tool result for call_2
[user]      Repetitive Behavior Warning     ← deferred，排在 tool results 后
[user]      Step Reflection                 ← post_tool_hook 注入
[user]      Iteration Checkpoint            ← post_tool_hook 注入
```

### tool_call_limit 的 break 位置修复（前次提交）

- **问题**：`tool_call_limit` 检查在 `for` 循环内部，一个 assistant message 含多个 tool calls 时，达到 limit 就 break，后续 tool call 没有对应 result message → OpenAI API 400 错误
- **解决方案**：将 `tool_call_limit` 检查移到循环外，确保当前批次所有 tool calls 都处理完后再判断
- **修改文件**：`agentica/model/base.py`

### subagent model 浅拷贝 + 运行时状态重置（前次提交）

- **问题**：parent 和 subagent 共享同一个 `Model` 实例，导致多种共享可变状态问题（hooks 闭包引用 parent DeepAgent、HTTP client 含 RLock 不可深拷贝、function_call_stack 累计等）
- **解决方案**：`tools/buildin_tools.py` subagent 创建时用 `model_copy()` 浅拷贝并逐个重置运行时字段（tools, functions, function_call_stack, tool_choice, metrics, client, http_client, _pre_tool_hook, _tool_call_hook, _post_tool_hook, _current_messages）
- **修改文件**：`agentica/tools/buildin_tools.py`

**涉及文件**：

| 文件路径 | 修改类型 |
|---------|---------|
| `agentica/model/base.py` | 修改（deferred warnings + role 改 user） |
| `agentica/deep_agent.py` | 修改（post_tool_hook role 改 user） |
| `agentica/tools/buildin_tools.py` | 修改（subagent model 浅拷贝 + 状态重置） |

---

## 历史记录 v3 (2026-02-10)

### Workflow 生命周期重构与代码清理

#### 1. Workflow run 包装机制重构
- **变更**：删除 `__init_subclass__` + `functools` 类级方法替换，改为 `__init__` 中实例级 run 包装
- **实现**：用 `object.__setattr__` 绕过 Pydantic `__setattr__`，在实例上设置 `_user_run` 和包装后的 `run`
- **拆分方法**：`_prepare_run`、`_process_result`、`_annotate_response`、`_finalize_run`，更清晰
- **修改文件**：`agentica/workflow.py`、`tests/test_workflow.py`

#### 2. 删除 run_workflow() 向后兼容
- 直接删除 `run_workflow()` 方法，不保留旧代码兼容
- **修改文件**：`agentica/workflow.py`、`tests/test_workflow.py`

#### 3. 删除 reasoning.py
- 删除 `agentica/reasoning.py`（`ReasoningStep`/`ReasoningSteps`/`NextAction` 均未使用）
- 移除 `run_response.py` 中 `from agentica.reasoning import ReasoningStep` 导入
- 移除 `RunResponseExtraData` 中 `reasoning_steps` 和 `reasoning_messages` 字段
- **保留** 所有 `reasoning_content` 相关代码（现代推理模型在用）
- **修改文件**：`agentica/run_response.py`，删除 `agentica/reasoning.py`

#### 4. 模块拆分重构
- `agentica/memory.py` → `agentica/memory/` 包（`agent_memory.py`、`manager.py`、`models.py`、`search.py`、`summarizer.py`、`workflow.py`）
- `agentica/cli.py` → `agentica/cli/` 包（`main.py`、`config.py`、`display.py`、`interactive.py`）

#### 5. Workflow 示例更新
- 删除 `01_simple_workflow.py`、`03_news_article.py`、`04_novel_writing.py`
- 新增 `01_data_pipeline.py`、`03_news_report.py`、`04_code_review.py`

#### 6. 并行工具调用方案规划
- 分析了 `run_function_calls` / `arun_function_calls` 串行执行问题
- 推荐两步走方案：先 ThreadPool 并行（1 文件 60 行），后补齐 Async 体系
- 方案写入 `update_tech_v3.md` (v3.2)

**涉及文件**：

| 文件路径 | 修改类型 |
|---------|---------|
| `agentica/workflow.py` | 修改（重构 run 包装机制） |
| `agentica/run_response.py` | 修改（移除 reasoning 导入和字段） |
| `agentica/reasoning.py` | 删除 |
| `agentica/memory.py` → `agentica/memory/` | 拆分为包 |
| `agentica/cli.py` → `agentica/cli/` | 拆分为包 |
| `tests/test_workflow.py` | 修改 |
| `tests/test_memory.py` | 修改 |
| `examples/workflow/` | 新增/删除多个示例 |
| `update_tech_v3.md` | 追加并行工具调用方案 |

---

## 历史记录 v2 (2026-02-09)

### Subagent 进度感知与 Prompt 修复

#### 1. Subagent（task 工具）进度展示

- **问题**：subagent 执行时，主 agent 的 CLI 端完全无法感知子代理内部进度，只显示一个阻塞的 `task` 工具调用
- **解决方案**：
  - `tools/buildin_tools.py` 的 `task()` 方法改为流式调用 `subagent.run(stream=True, stream_intermediate_steps=True)`，遍历子代理事件收集工具使用信息
  - 返回 JSON 新增 `tool_calls_summary`（工具名+简要信息列表）、`execution_time`（耗时秒）、`tool_count`
  - 新增 `_format_tool_brief()` 静态方法：针对不同工具类型生成可读简要信息
  - `cli.py` 新增 `_display_task_result()` 方法，对 `task` 工具做特殊展示：内部工具调用列表 + `Execution Summary: N tool uses, cost: X.Xs`
- **修改文件**：`agentica/tools/buildin_tools.py`、`agentica/cli.py`

#### 2. 修复 Prompt 负面示例导致模型错误调用工具格式

- **问题**：`BuiltinTaskTool` 的 system prompt 中包含负面示例 `DO NOT use XML-style tags like <tool_call>task<arg_key>...</arg_key></tool_call>`，弱模型（如 GLM-4-flash）反而会模仿这个 XML 格式来调用工具
- **解决方案**：删除负面示例，替换为简洁的正面指引 `Use your standard function calling mechanism to invoke task(...)`
- **修改文件**：`agentica/tools/buildin_tools.py`（`TASK_SYSTEM_PROMPT_TEMPLATE`）

---

### CLI 流式显示优化与 Ctrl+C 真中止

#### 1. 最终回答与工具调用/思考之间的换行分隔

- **问题**：多轮工具调用场景中，工具调用结果、思考过程与最终回答之间没有换行分隔，内容挤在一起
- **解决方案**：在 `StreamDisplayManager` 的 `end_tool_section()` 和 `end_thinking()` 结束时重置 `response_started = False`，使下一段 content 输出时重新触发 `start_response()` 加空行分隔
- **修改文件**：`agentica/cli.py`

#### 2. Ctrl+C 真中止（agent.cancel）

- **问题**：Ctrl+C 只中断了流迭代，agent 内部仍在运行
- **解决方案**：`KeyboardInterrupt` 捕获时调用 `current_agent.cancel()`，通过协作式取消机制通知 agent 在下一个检查点抛出 `AgentCancelledError`；同时捕获 `AgentCancelledError` 防止异常向上传播
- **修改文件**：`agentica/cli.py`（导入 `AgentCancelledError`，修改异常处理逻辑）

#### 3. 工具结果预览显示（Claude Code 风格）

- **问题**：工具执行完成后 `ToolCallCompleted` 事件被直接 `continue` 跳过，不显示任何结果
- **解决方案**：新增 `StreamDisplayManager.display_tool_result()` 方法，在每个工具调用下方显示执行结果预览
  - 使用 `⎿` 连接符（Claude Code 风格），暗色 dim 显示
  - 最多显示 4 行，每行最多 120 字符，超出部分显示 `... (N more lines)`
  - 错误结果用 `dim red` + `⚠` 标记
- **修改文件**：`agentica/cli.py`（新增 `display_tool_result` 方法，修改 `ToolCallCompleted` 事件处理）

### Model 层异步工具调用（解除事件循环阻塞）

#### 4. 异步工具执行链路

- **问题**：`DeepAgent` 默认 `enable_multi_round=False` 时，工具执行走 Model 层内置递归循环。`aresponse`/`aresponse_stream` 内部调用同步的 `handle_tool_calls` → `run_function_calls` → `function_call.execute()` → `subprocess.run()`，直接阻塞 asyncio 事件循环，导致 uvicorn 所有请求被阻塞
- **解决方案**：新增完整异步工具调用链路
  - `model/base.py`：新增 `arun_function_calls` async 方法，用 `await function_call.aexecute()` 代替同步 `execute()`
  - `model/openai/chat.py`：新增 `ahandle_tool_calls` 和 `ahandle_stream_tool_calls` async 方法
  - `aresponse` 改用 `await self.ahandle_tool_calls()`
  - `aresponse_stream` 改用 `async for ... in self.ahandle_stream_tool_calls()`

**调用链对照**：

| 层级 | 同步路径 | Async 路径（新增） |
|------|---------|------------------|
| Model base | `run_function_calls` → `execute()` | `arun_function_calls` → `await aexecute()` |
| OpenAI chat | `handle_tool_calls` | `ahandle_tool_calls` |
| OpenAI chat | `handle_stream_tool_calls` | `ahandle_stream_tool_calls` |
| OpenAI chat | `response` → `handle_tool_calls` | `aresponse` → `await ahandle_tool_calls` |
| OpenAI chat | `response_stream` → `handle_stream_tool_calls` | `aresponse_stream` → `async for ahandle_stream_tool_calls` |

**修改文件**：
- `agentica/model/base.py`：新增 `arun_function_calls`
- `agentica/model/openai/chat.py`：新增 `ahandle_tool_calls`、`ahandle_stream_tool_calls`，修改 `aresponse`、`aresponse_stream`

---


## 历史记录 v1

### Agent Framework 基础架构 (2024-12-15)

建立了Agentica框架的基础架构，包括Agent基类、模型集成、工具系统和CLI界面。

#### 核心功能
- Agent基类设计和实现
- 多模型LLM支持(OpenAI, Anthropic, ZhipuAI等)
- 工具系统和注册机制
- 内存管理和会话持久化
- CLI交互界面

#### 工具生态
- 文件操作工具
- Shell命令执行
- Web搜索集成
- Python REPL
- 计算器功能

#### 测试覆盖
- 单元测试框架
- 集成测试用例
- 性能基准测试

# AGENT.md --- 2026-02-12

This file provides guidance to AI coding agents when working with code in this repository.

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

### Agent (`agentica/agent/`) + Runner (`agentica/runner.py`)

The `Agent` class uses `@dataclass(init=False)` with explicit `__init__` and **direct multiple inheritance** from mixin classes. Execution is delegated to an independent `Runner` class.

```python
@dataclass(init=False)
class Agent(PromptsMixin, TeamMixin, ToolsMixin, PrinterMixin):
    def __init__(self, ...):
        ...
        self._runner = Runner(self)

    async def run(self, message, **kw) -> RunResponse:
        return await self._runner.run(message, **kw)
```

- `runner.py` — `Runner`: Independent execution engine (holds `self.agent` reference). Core `_run_impl()`, `run()`, `run_stream()`, `run_sync()`, `run_stream_sync()`
- `prompts.py` — `PromptsMixin`: System/user message construction
- `team.py` — `TeamMixin`: Multi-agent delegation
- `tools.py` — `ToolsMixin`: Tool registration and management
- `printer.py` — `PrinterMixin`: Response printing utilities

Note: Agent uses `@dataclass` (not BaseModel). The entire Model hierarchy also uses `@dataclass` (converted from Pydantic in v3).

### Model (`agentica/model/`)

`Model` base class (`base.py`) is a `@dataclass` with `ABC`, defining abstract async methods: `invoke()`, `invoke_stream()`, `response()`, `response_stream()`. All Model subclasses use `@dataclass` (not Pydantic BaseModel).

**Core providers** (5 directories): `openai/`, `anthropic/`, `ollama/`, `litellm/`, `azure/`

**OpenAI-compatible providers** via registry factory (`model/providers.py`):
```python
from agentica.model.providers import create_provider
model = create_provider("deepseek", api_key="...")  # Returns OpenAILike with correct config
```
Registered providers: deepseek, qwen, zhipuai, moonshot, doubao, together, xai, yi, nvidia, sambanova, groq, cerebras, mistral

**Structured output**: Each provider implements native structured output:
- OpenAI: `beta.chat.completions.parse` with `response_format`
- Claude: synthetic tool_use mode
- LiteLLM: `response_format={"type": "json_schema", ...}`
- Ollama: `format=schema`

### Tools (`agentica/tools/`)

Three-layer hierarchy: `Tool` (container) → `Function` (schema + entrypoint) → `FunctionCall` (invocation). `FunctionCall.execute()` is async-only.

**`@tool` decorator** (`tools/decorators.py`): Attach metadata to functions for auto-registration:
```python
from agentica.tools import tool
@tool(name="add", description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b
```

**Global tool registry** (`tools/registry.py`): `register_tool()`, `get_tool()`, `list_tools()`, `unregister_tool()`, `clear_registry()`

`Function.from_callable()` auto-detects `_tool_metadata` from `@tool` decorator.

**Builtin Tools** (`agentica/tools/buildin_tools.py`) — all I/O-bound tools are **async**:
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

Three-layer unified architecture:
- **`core.py`**: `GuardrailTriggered` (base exception), `GuardrailOutput` (allow/block), `BaseGuardrail` (base class with `_invoke()`), `run_guardrails_seq()` (execution engine)
- **`agent.py`**: `InputGuardrail`, `OutputGuardrail` — validate entire agent runs. `@input_guardrail`, `@output_guardrail` decorators
- **`tool.py`**: `ToolInputGuardrail`, `ToolOutputGuardrail` — validate individual tool calls with three-way behavior (allow/reject_content/raise_exception)

All guardrail functions support both sync and async. `GuardrailFunctionOutput` is a backward-compatible alias for `GuardrailOutput`.

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

## Recent Changes (2026-02-25)

### V3 Architecture Refactoring

Eight-phase refactoring to simplify and modernize the codebase:

| Phase | Description | Key Changes |
|-------|-------------|-------------|
| 1 | Model layer simplification | Remove 19 provider dirs → `providers.py` registry/factory |
| 2 | Model @dataclass conversion | Pydantic BaseModel → stdlib @dataclass for all Model classes |
| 3 | Async consistency + structured output | ABC/@abstractmethod, unified structured output for all providers |
| 4 | Tool registration mechanism | `@tool` decorator + global tool registry |
| 5 | Runner extraction | RunnerMixin → independent `Runner` class, Agent delegates via `_runner` |
| 6 | Guardrails unification | New `core.py` abstraction layer, base.py → agent.py |
| 7 | `__init__.py` simplification | 594 → 399 lines, streamlined lazy loading |
| 8 | Tests + cleanup | 35 new tests covering all phases, CLAUDE.md update |

**Test result**: 622 tests pass (587 original + 35 new v3 tests)

**Testing convention**: All tests MUST mock LLM API keys (use `api_key="fake_openai_key"` or mock `agent._runner.run`). No real API calls in tests.

## Previous Changes (2026-02-18)

Fixed 6 example files to align with V2 async-first API:

- **Agent no longer accepts `db=`** — pass `db` to `AgentMemory(db=...)` instead, then set `agent.memory`
- **Agent no longer accepts `load_workspace_context=` / `load_workspace_memory=`** — these moved to `MemoryConfig` (enabled by default)
- **Agent no longer accepts `workspace_path=` / `user_id=`** — use `Workspace(path, user_id=...)` object and pass via `workspace=`
- **Agent no longer has `compression_manager` attribute** — access via `agent.tool_config.compress_tool_results`; pass `CompressionManager` through `ToolConfig(compression_manager=...)`
- **Agent no longer has `get_user_memories()` / `clear_user_memories()`** — use `agent.memory.load_user_memories()` / `agent.memory.clear()`
- **Agent no longer has `load_session()` method** — session management removed from Agent direct API
- **DeepAgent no longer accepts `db=`** — same pattern, use `AgentMemory`
