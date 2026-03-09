# AGENT.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

Agentica is a Python AI agent framework (Python >= 3.10) for building autonomous AI agents. Async-first architecture with support for multi-model LLMs, tools, RAG, multi-agent teams, workflows, MCP/ACP protocols, and a Skills system.

## Common Commands

```bash
# Install
pip install -U agentica          # From PyPI
pip install -e .                  # From source (development)

# Run tests
python -m pytest tests/ -v --tb=short                              # All tests
python -m pytest tests/test_agent.py -v                            # Single file
python -m pytest tests/test_agent.py::TestAgentInitialization -v   # Single class
python -m pytest tests/test_agent.py -k "test_default" -v          # Pattern match

# CLI
agentica                                      # Interactive mode
agentica --query "Your question"              # Single query
agentica --model_provider zhipuai --model_name glm-4.7-flash  # Specify model
agentica --tools calculator shell wikipedia   # Enable specific tools
agentica acp                                  # Start ACP server for IDE integration
```

No Makefile, linter config, or type-checker is configured. CI runs `pytest` on Python 3.10+.

## Architecture

### Async-First Convention

All core methods are **async by default**:

| Method | Type | Purpose |
|--------|------|---------|
| `run()` | async | Non-streaming run |
| `run_stream()` | async generator | Streaming run |
| `run_sync()` | sync adapter | Wraps `run()` |
| `run_stream_sync()` | sync iterator | Background thread + queue pattern |

No `a`-prefixed methods (`arun`, `aresponse`). Sync tools use `loop.run_in_executor()`. All blocking I/O (DB, file system) wrapped in `run_in_executor()`.

### Core Components (`agentica/`)

| File/Dir | Purpose |
|----------|---------|
| `agent/` | `Agent` class (package: `base.py`, `config.py`, `prompts.py`, `team.py`, `tools.py`, `printer.py`) |
| `runner.py` | `Runner` - independent execution engine, delegated from Agent |
| `deep_agent.py` | `DeepAgent` - Agent with built-in file/execute/search tools |
| `subagent.py` | Subagent system for spawning isolated ephemeral agent tasks |
| `workspace.py` | `Workspace` - file-based storage (AGENT.md, PERSONA.md, MEMORY.md) |
| `memory/` | `WorkingMemory` (`working.py`), `MemorySummarizer`, `WorkspaceMemorySearch` |
| `guardrails/` | Three-layer: `core.py` (base), `agent.py` (input/output), `tool.py` (tool-level) |
| `workflow.py` | Deterministic multi-agent pipeline |
| `search/` | Search enhancement: `orchestrator.py`, `query_decomposer.py`, `evidence_store.py`, `answer_verifier.py` |
| `prompts/` | Modular prompt assembly via `PromptBuilder` (`builder.py`) + `base/md/` templates |
| `cli/` | Interactive CLI (`main.py`, `config.py`, `display.py`, `interactive.py`) |
| `acp/` | Agent Client Protocol for IDE integration (`server.py`, `protocol.py`, `handlers.py`, `types.py`, `session.py`) |
| `mcp/` | Model Context Protocol integration |
| `skills/` | Markdown-based skill injection system |
| `run_response.py` | `RunResponse`, `RunEvent` - agent execution responses |
| `run_config.py` | `RunConfig` - runtime configuration |
| `hooks.py` | `AgentHooks`, `RunHooks` - lifecycle hooks |
| `compression/` | Tool result compression |

### Agent + Runner Pattern

`Agent` uses `@dataclass(init=False)` with explicit `__init__` and direct multiple inheritance from mixins. Execution delegated to `Runner`.

```python
@dataclass(init=False)
class Agent(PromptsMixin, TeamMixin, ToolsMixin, PrinterMixin):
    def __init__(self, ...):
        self._runner = Runner(self)

    async def run(self, message, **kw) -> RunResponse:
        return await self._runner.run(message, **kw)
```

Note: Agent and all Model classes use `@dataclass` (not Pydantic BaseModel).

### Model Layer (`agentica/model/`)

`Model` base class (`base.py`) is a `@dataclass` with `ABC`.

**Core provider directories** (6):
| Dir | Purpose |
|-----|---------|
| `openai/` | OpenAI API integration (GPT-4, etc.) |
| `anthropic/` | Anthropic API integration (Claude models) |
| `kimi/` | Kimi/Moonshot chat integration |
| `ollama/` | Ollama local model integration |
| `litellm/` | LiteLLM universal provider |
| `azure/` | Azure OpenAI integration |

**OpenAI-compatible providers** via registry factory (`model/providers.py`):
```python
from agentica.model.providers import create_provider
model = create_provider("deepseek", api_key="...")  # Returns OpenAILike with correct config
```
Registered providers: deepseek, qwen, zhipuai, moonshot, doubao, together, xai, yi, nvidia, sambanova, groq, cerebras, mistral

**Other model files**: `base.py` (ABC), `message.py` (Message), `response.py` (ModelResponse), `content.py`, `metrics.py`, `usage.py`, `providers.py` (registry)

### Tools System (`agentica/tools/`)

Three-layer hierarchy: `Tool` (container) → `Function` (schema + entrypoint) → `FunctionCall` (invocation).

**Core files**:
| File | Purpose |
|------|---------|
| `base.py` | `Tool`, `Function`, `FunctionCall`, `ModelTool` base classes |
| `decorators.py` | `@tool` decorator for auto-registration |
| `registry.py` | Global tool registry (`register_tool`, `get_tool`, `list_tools`) |
| `buildin_tools.py` | Built-in tools for DeepAgent (async: edit_file, read_file, write_file, ls, glob, grep, execute, web_search, fetch_url, task) |

**Tool plugins** (40+ files, naming convention `*_tool.py`):
`shell_tool.py`, `code_tool.py`, `wikipedia_tool.py`, `browser_tool.py`, `ocr_tool.py`, `arxiv_tool.py`, `dalle_tool.py`, `cogview_tool.py`, `yfinance_tool.py`, `weather_tool.py`, `sql_tool.py`, `search_bocha_tool.py`, `search_exa_tool.py`, `search_serper_tool.py`, `duckduckgo_tool.py`, `jina_tool.py`, `url_crawler_tool.py`, `mcp_tool.py`, `skill_tool.py`, `lsp_tool.py`, etc.

Flow control exceptions: `StopAgentRun`, `RetryAgentRun`.

### ACP (Agent Client Protocol)

IDE integration (Zed, JetBrains, etc.):
```bash
agentica acp  # Start ACP server mode
```

### Public API (`agentica/__init__.py`)

Uses **lazy loading** with thread-safe double-checked locking for optional/heavy modules. Core modules (Agent, Model, Memory, Tools, Workspace) are imported eagerly.

## Key Patterns

- `@dataclass(init=False)` with explicit `__init__` for Agent, direct multiple inheritance from mixins
- `@dataclass` for all Model classes (not Pydantic)
- Pydantic `BaseModel` for data structures (Tool, Function, RunResponse, Workflow)
- `Function.from_callable()` factory to auto-generate tool definitions from Python functions
- `RunResponse` with `RunEvent` enum for structured event streaming
- `asyncio.gather` with `return_exceptions=True` for concurrent tool execution
- All blocking I/O wrapped in `loop.run_in_executor()` within async methods
- Message ordering: tool results must immediately follow assistant tool_calls (OpenAI API requirement)
- Injected prompts (warnings, reflections) use `role="user"` (not "system") for broad LLM compatibility
- Langfuse integration for tracing (context manager pattern in runner)

## Testing Convention

- All tests MUST mock LLM API keys (`api_key="fake_openai_key"` or mock `agent._runner.run`). No real API calls in tests.
- Tests for async methods use `asyncio.run()` and `unittest.mock.AsyncMock`
