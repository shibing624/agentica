# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent Client Protocol (ACP) support for Agentica

This module enables Agentica to work as an ACP-compatible agent,
allowing integration with IDEs like Zed, JetBrains, etc.

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
]
