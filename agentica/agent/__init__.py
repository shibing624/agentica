# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent module - modular agent implementation

V2 architecture with layered configuration:
- base.py: Agent class definition, fields, initialization, and run API delegation
- config.py: PromptConfig, ToolConfig, WorkspaceMemoryConfig, TeamConfig
- prompts.py: System and user prompt building
- team.py: Team collaboration and task transfer
- tools.py: Default tools (knowledge search, memory, etc.)
- printer.py: Response printing utilities

Execution engine lives in agentica/runner.py (Runner class).
"""

from agentica.agent.base import Agent, AgentCancelledError
from agentica.agent.config import PromptConfig, ToolConfig, WorkspaceMemoryConfig, TeamConfig
from agentica.hooks import AgentHooks, RunHooks

__all__ = [
    "Agent",
    "AgentCancelledError",
    "PromptConfig",
    "ToolConfig",
    "WorkspaceMemoryConfig",
    "TeamConfig",
    "AgentHooks",
    "RunHooks",
]
