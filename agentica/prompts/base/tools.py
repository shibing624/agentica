# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: TOOLS module - Tool usage strategy and dynamic tool list

This module provides prompts for guiding tool usage:
1. Parallel execution strategy for independent calls
2. File operation guidelines
3. Context management guidance
4. Dynamic tool list generation from registered tool instances
"""

from typing import Optional, List, Dict
from agentica.prompts.base.utils import load_prompt as _load_prompt

# Load prompt from MD file (strategy + context management, no tool table)
TOOLS_PRIORITY_PROMPT = _load_prompt("tools.md")


def get_tools_prompt(
    active_tools: Optional[List[str]] = None,
    tool_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Get the tools prompt, optionally with a dynamic tool list.

    Args:
        active_tools: List of currently enabled tool names.
            If provided, generates a tool table from active tools.
            If None, returns only the strategy prompt.
        tool_descriptions: Optional mapping of tool_name -> description.
            Used to populate the tool table. If a tool has no entry,
            the tool name itself is used as description.

    Returns:
        The tools prompt string
    """
    if active_tools:
        desc_map = tool_descriptions or {}
        table = "# Available Tools\n\n"
        table += "| Tool | Purpose |\n|------|----------|\n"
        for tool_name in active_tools:
            desc = desc_map.get(tool_name, tool_name)
            table += f"| `{tool_name}` | {desc} |\n"
        return table + "\n---\n\n" + TOOLS_PRIORITY_PROMPT
    return TOOLS_PRIORITY_PROMPT
