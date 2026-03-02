# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tools module exports

Available tools:
- Base: Tool base class
- Builtin tools: File, Execute, WebSearch, FetchUrl, Todo, Task, Memory (used by DeepAgent)
- CodeTool: Code analysis, formatting, and linting
- LspTool: LSP-based code navigation (goto definition, find references)
- PatchTool: Apply diff/patch files (V4A and unified diff formats)
- @tool decorator: Attach metadata to tool functions
- Tool Registry: Global tool name → callable registry
"""
from agentica.tools.base import Tool, Function, FunctionCall
from agentica.tools.decorators import tool
from agentica.tools.registry import register_tool, get_tool, list_tools, unregister_tool, clear_registry
from agentica.tools.buildin_tools import (
    BuiltinFileTool,
    BuiltinExecuteTool,
    BuiltinWebSearchTool,
    BuiltinFetchUrlTool,
    BuiltinTodoTool,
    BuiltinTaskTool,
    BuiltinMemoryTool,
)
from agentica.tools.code_tool import CodeTool
from agentica.tools.lsp_tool import LspTool
from agentica.tools.patch_tool import PatchTool

__all__ = [
    # Base classes
    "Tool",
    "Function",
    "FunctionCall",
    # Decorator
    "tool",
    # Registry
    "register_tool",
    "get_tool",
    "list_tools",
    "unregister_tool",
    "clear_registry",
    # Builtin tools (DeepAgent)
    "BuiltinFileTool",       # File read/write/search/list
    "BuiltinExecuteTool",    # Shell command execution
    "BuiltinWebSearchTool",  # Web search
    "BuiltinFetchUrlTool",   # URL content fetching
    "BuiltinTodoTool",       # Task list management
    "BuiltinTaskTool",       # Sub-agent task delegation
    "BuiltinMemoryTool",     # Persistent memory
    # Extended tools
    "CodeTool",      # Code analysis, formatting, linting
    "LspTool",       # LSP-based code navigation
    "PatchTool",     # Apply diff/patch files
]
