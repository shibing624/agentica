"""Tools module exports.

Available tools:
- Base: Tool base class
- Builtin tools: File, Execute, WebSearch, FetchUrl, Todo, Task (used by Agent)
- CodeTool: Code analysis, formatting, and linting
- LspTool: LSP-based code navigation (goto definition, find references)
- PatchTool: Apply diff/patch files (V4A and unified diff formats)
- @tool decorator: Attach metadata to tool functions
- Tool Registry: Global tool name -> callable registry
"""
from agentica.tools.base import Tool, Function, FunctionCall
from agentica.tools.builtin.task_state_tools import BuiltinTodoTool
from agentica.tools.builtin.web_tools import BuiltinFetchUrlTool, BuiltinWebSearchTool
from agentica.tools.builtin_task_tool import BuiltinTaskTool
from agentica.tools.decorators import tool
from agentica.tools.registry import register_tool, get_tool, list_tools, unregister_tool, clear_registry
from agentica.tools.builtin import (
    BuiltinFileTool,
    BuiltinExecuteTool,
)
from agentica.tools.code_tool import CodeTool
from agentica.tools.e2b_tool import E2BExecuteTool
from agentica.tools.lsp_tool import LspTool
from agentica.tools.patch_tool import PatchTool
from agentica.tools.cron_tool import CronTool
from agentica.tools.peer_tool import PeerMessagingTool
from agentica.tools.use_capability_tool import UseCapabilityTool
from agentica.tools.helpers import tool_error, tool_result
from agentica.tools.safety import check_command_safety, redact_sensitive_text
from agentica.tools.interrupt import set_interrupt, is_interrupted

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
    # Helpers
    "tool_error",
    "tool_result",
    # Safety
    "check_command_safety",
    "redact_sensitive_text",
    # Interrupt
    "set_interrupt",
    "is_interrupted",
    # Builtin tools
    "BuiltinFileTool",       # File read/write/search/list
    "BuiltinExecuteTool",    # Shell command execution
    "BuiltinWebSearchTool",  # Web search
    "BuiltinFetchUrlTool",   # URL content fetching
    "BuiltinTodoTool",       # Task list management
    "BuiltinTaskTool",       # Sub-agent task delegation
    # Extended tools
    "CodeTool",        # Code analysis, formatting, linting
    "E2BExecuteTool",  # Remote sandboxed Python / shell execution (E2B cloud)
    "LspTool",         # LSP-based code navigation
    "PatchTool",       # Apply diff/patch files
    "CronTool",        # Cron job management
    "PeerMessagingTool",  # Message the user's other live CLI sessions
    "UseCapabilityTool",  # Stable proxy for discovering/calling deferred (optional) tools
]
