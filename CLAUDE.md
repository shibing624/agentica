# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
agentica --model_provider zhipuai --model_name glm-4.6v-flash  # Specify model
agentica --tools calculator shell wikipedia   # Enable specific tools
```

## Architecture

### Core Components (`agentica/`)

| File | Purpose |
|------|---------|
| `agent.py` | Core `Agent` class - main entry point for building agents |
| `deep_agent.py` | `DeepAgent` - Agent with built-in file/execute/search tools |
| `deep_tools.py` | Built-in tools for DeepAgent (ls, read_file, execute, web_search, etc.) |
| `memory.py` | `AgentMemory`, `MemoryManager` - session and long-term memory |
| `guardrails.py` | Input/output validation and filtering for agents |
| `workflow.py` | Workflow engine for multi-step task orchestration |
| `cli.py` | Interactive CLI with file references (@filename) and commands (/help) |
| `run_response.py` | `RunResponse`, `RunEvent` - agent execution responses |

### Model Layer (`agentica/model/`)

Supports 20+ LLM providers through a unified `Model` base class:
- `openai/` - OpenAI, OpenAI-compatible endpoints
- `anthropic/` - Claude models
- `deepseek/`, `moonshot/`, `zhipuai/`, `qwen/`, `doubao/`, `yi/` - Chinese providers
- `ollama/` - Local models
- `litellm/` - Unified interface for 100+ providers

Key classes: `Model` (base), `Message`, `ModelResponse`

### Tools (`agentica/tools/`)

40+ built-in tools. Key pattern:
```python
class MyTool(Tool):
    def __init__(self):
        super().__init__(name="my_tool")
        self.register(self.my_function)

    def my_function(self, param: str) -> str:
        """Function description for LLM."""
        return result
```

### Memory System

- `AgentMemory` - Per-agent session memory (messages, runs)
- `MemoryManager` - CRUD operations for user memories with search methods (last_n, keyword, agentic)
- Database backends: `SqliteDb`, `PostgresDb`, `InMemoryDb`, `JsonDb`

### Knowledge/RAG (`agentica/knowledge/`)

- `Knowledge` class with `VectorDb` backends (Qdrant, ChromaDB, LanceDB, PGVector, Pinecone)
- Integrations: `LangChainKnowledge`, `LlamaIndexKnowledge`

### Skills System (`agentica/skills/`)

Prompt-based capability injection via SKILL.md files:
```markdown
---
name: My Skill
description: Skill description
allowed-tools:
  - shell
  - python
---
# Instructions for LLM
```

Search paths: `.claude/skills/`, `.agentica/skills/`, `~/.claude/skills/`, `~/.agentica/skills/`

### MCP Protocol (`agentica/mcp/`)

Model Context Protocol integration:
- `MCPServer` base class with `MCPServerStdio`, `MCPServerSse`, `MCPServerStreamableHttp`
- `MCPClient` for connecting to MCP servers

## Key Patterns

### Creating an Agent
```python
from agentica import Agent, ZhipuAI

agent = Agent(
    model=ZhipuAI(),
    tools=[MyTool()],
    instructions="Be helpful",
    add_datetime_to_instructions=True,
)
response = agent.run("Your message")
# or async: response = await agent.arun("Your message")
```

### Multi-round Conversations
```python
agent = Agent(
    enable_multi_round=True,
    max_rounds=100,
    max_tokens=128000,
)
```

### Guardrails
```python
from agentica.guardrails import input_guardrail, output_guardrail, GuardrailFunctionOutput

@input_guardrail
def check_input(ctx, agent, input_data):
    if invalid:
        return GuardrailFunctionOutput.block({"reason": "..."})
    return GuardrailFunctionOutput.allow()

agent = Agent(input_guardrails=[check_input])
```

### Tool Guardrails
```python
from agentica.tools.guardrails import tool_input_guardrail, ToolGuardrailFunctionOutput

@tool_input_guardrail
def check_tool_input(data):
    if forbidden:
        return ToolGuardrailFunctionOutput.reject_content("Forbidden operation")
    return ToolGuardrailFunctionOutput.allow()
```

## Environment Variables

Set in `~/.agentica/.env` or as environment variables:
```bash
ZHIPUAI_API_KEY="..."       # ZhipuAI (glm-4.6v-flash is free)
OPENAI_API_KEY="..."        # OpenAI
DEEPSEEK_API_KEY="..."      # DeepSeek
MOONSHOT_API_KEY="..."      # Moonshot
```

## Directory Structure

```
agentica/
├── model/          # LLM providers
├── tools/          # 40+ built-in tools
├── db/             # Database backends (sqlite, postgres, json, memory)
├── vectordb/       # Vector stores for RAG
├── knowledge/      # RAG knowledge base
├── emb/            # Embedding models
├── mcp/            # MCP protocol support
├── compression/    # Context compression
├── temporal/       # Distributed workflows
├── skills/         # Skill system
├── reranker/       # Re-ranking for RAG
└── utils/          # Utilities (tokens, etc.)
```

## Adding New Components

### New Model Provider
1. Create `agentica/model/<provider>/` directory
2. Inherit from `Model` base class
3. Implement `response()` and `aresponse()` methods
4. Export in `agentica/model/__init__.py`

### New Tool
1. Create `agentica/tools/<tool_name>_tool.py`
2. Inherit from `Tool` or use function with type hints
3. Register functions with descriptive docstrings

### New Vector Database
1. Create `agentica/vectordb/<db_name>db.py`
2. Inherit from `VectorDb`
3. Implement `insert()`, `search()`, `upsert()`, `delete()` methods
