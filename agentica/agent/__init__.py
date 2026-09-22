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

Submodules stay importable without pulling ``base`` (``model.base`` imports
``agent.hooks`` while it is still loading). Names are resolved on first use.
"""

import importlib

_EXPORTS = {
    "Agent": ("agentica.agent.base", "Agent"),
    "AgentCancelledError": ("agentica.agent.base", "AgentCancelledError"),
    "PromptConfig": ("agentica.agent.config", "PromptConfig"),
    "ToolConfig": ("agentica.agent.config", "ToolConfig"),
    "WorkspaceMemoryConfig": ("agentica.agent.config", "WorkspaceMemoryConfig"),
    "AgentHooks": ("agentica.agent.hooks", "AgentHooks"),
    "RunHooks": ("agentica.agent.hooks", "RunHooks"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = target
    value = getattr(importlib.import_module(module_path), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
