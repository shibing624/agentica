# -*- coding: utf-8 -*-
"""Subagent system: spawn isolated ephemeral agents for bounded tasks.

Public runtime API lives here. Markdown definitions:

- package defaults: ``agentica/subagents/bundled/*.md``
- user/project overrides stay in ``.agentica/agents/`` and ``$AGENTICA_HOME/agents/``
"""

from agentica.subagents.runtime import (
    SubagentConfig,
    SubagentRegistry,
    SubagentRun,
    get_available_subagent_types,
    get_custom_subagent_configs,
    get_subagent_config,
    get_subagent_configs,
    register_custom_subagent,
    unregister_custom_subagent,
)

__all__ = [
    "SubagentConfig",
    "SubagentRegistry",
    "SubagentRun",
    "get_available_subagent_types",
    "get_custom_subagent_configs",
    "get_subagent_config",
    "get_subagent_configs",
    "register_custom_subagent",
    "unregister_custom_subagent",
]
