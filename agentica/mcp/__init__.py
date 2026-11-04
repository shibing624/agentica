# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: MCP: Model Context Protocol client/server.

Requires the [mcp] extras:
    pip install agentica[mcp]

The server/client classes are loaded on first access, not at import: the mcp
SDK costs ~0.2 s to import, and ``agentica.mcp.config`` (read on every agent
build that auto-loads MCP) must not pay it when the tool list comes from cache.
"""
import importlib
import importlib.util

if importlib.util.find_spec("mcp") is None:
    raise ImportError(
        "agentica.mcp requires the [mcp] extras. Install with:\n\n"
        "    pip install agentica[mcp]\n"
    )

_LAZY = {
    "MCPServer": ".server",
    "MCPServerSse": ".server",
    "MCPServerSseParams": ".server",
    "MCPServerStdio": ".server",
    "MCPServerStdioParams": ".server",
    "MCPServerStreamableHttp": ".server",
    "MCPServerStreamableHttpParams": ".server",
    "MCPClient": ".client",
}

__all__ = list(_LAZY)


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
