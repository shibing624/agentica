# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent module - modular agent implementation

V2 architecture with layered configuration:
- base.py: Agent class definition, fields, and initialization
- config.py: PromptConfig, ToolConfig, MemoryConfig, TeamConfig
- prompts.py: System and user prompt building
- runner.py: Run execution logic
- team.py: Team collaboration and task transfer
- tools.py: Default tools (knowledge search, memory, etc.)
- printer.py: Response printing utilities
"""

from agentica.agent.base import Agent, AgentCancelledError
from agentica.agent.config import PromptConfig, ToolConfig, MemoryConfig, TeamConfig

__all__ = [
    "Agent",
    "AgentCancelledError",
    "PromptConfig",
    "ToolConfig",
    "MemoryConfig",
    "TeamConfig",
]
