# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Agent Client Protocol (ACP) support for Agentica.

Enables Agentica to work as an ACP-compatible agent, allowing integration
with IDEs like Zed, JetBrains, etc.

Requires the [acp] extras:
    pip install agentica[acp]

Usage:
    agentica acp          # Start ACP server mode

Or configure in IDE:
    {
      "agent_servers": {
        "Agentica": {
          "command": "agentica",
          "args": ["acp"],
          "env": {}
        }
      }
    }
"""
try:
    import websockets  # noqa: F401
except ImportError as e:
    raise ImportError(
        "agentica.acp requires the [acp] extras. Install with:\n\n"
        "    pip install agentica[acp]\n"
    ) from e

from agentica.acp.server import ACPServer
from agentica.acp.types import (
    ACPMessage,
    ACPRequest,
    ACPResponse,
    ACPTool,
    ACPToolCall,
    ACPToolResult,
    ACPErrorCode,
    ACPMethod,
)
from agentica.acp.session import (
    SessionManager,
    ACPSession,
    SessionStatus,
    SessionMessage,
)
from agentica.acp.permissions import (
    ToolPermissionPolicy,
    PermissionMode,
    PermissionDecision,
)

__all__ = [
    "ACPServer",
    # Types
    "ACPMessage",
    "ACPRequest",
    "ACPResponse",
    "ACPTool",
    "ACPToolCall",
    "ACPToolResult",
    "ACPErrorCode",
    "ACPMethod",
    # Session
    "SessionManager",
    "ACPSession",
    "SessionStatus",
    "SessionMessage",
    # Permissions
    "ToolPermissionPolicy",
    "PermissionMode",
    "PermissionDecision",
]
