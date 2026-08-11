# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent module - modular agent implementation

V2 architecture with layered configuration:
- base.py: Agent class definition, fields, initialization, and run API delegation
- config.py: PromptConfig, ToolConfig, WorkspaceMemoryConfig
- prompts.py: System and user prompt building
- as_tool.py: Agent.as_tool() composition primitive
- tools.py: Default tools (knowledge search, memory, etc.)
- printer.py: Response printing utilities

Execution engine lives in agentica/runner.py (Runner class).
"""

from agentica.agent.base import Agent, AgentCancelledError
from agentica.agent.config import (
    PromptConfig,
    ToolConfig,
    WorkspaceMemoryConfig,
)
from agentica.hooks import AgentHooks, RunHooks

__all__ = [
    "Agent",
    "AgentCancelledError",
    "PromptConfig",
    "ToolConfig",
    "WorkspaceMemoryConfig",
    "AgentHooks",
    "RunHooks",
]
