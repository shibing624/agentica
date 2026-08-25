# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: TOOLS module - Tool usage strategy

This module provides prompts for guiding tool usage:
1. Dedicated tools for bounded filesystem work; execute pipelines for shaping stdout
2. Choosing between the file-editing tools
3. Batching independent calls vs sequencing dependent ones

No tool table is generated here: the tool schemas the provider already
receives are the authoritative list, and duplicating them in the system
prompt only burns tokens.
"""

from agentica.prompts.base.utils import load_prompt

TOOLS_PRIORITY_PROMPT = load_prompt("tools.md")


def get_tools_prompt() -> str:
    """Get the tool usage strategy prompt."""
    return TOOLS_PRIORITY_PROMPT
