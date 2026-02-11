# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: TOOLS module - Tool usage priority and parallel strategy

This module provides prompts for guiding tool selection and usage:
1. Prefer specialized tools over bash commands
2. Parallel execution strategy for independent calls
3. Tool selection priority based on task type
4. File operation guidelines
"""

from agentica.prompts.base.utils import load_prompt as _load_prompt

# Load prompt from MD file
TOOLS_PRIORITY_PROMPT = _load_prompt("tools.md")


def get_tools_prompt() -> str:
    """Get the tools prompt.

    Returns:
        The tools prompt string
    """
    return TOOLS_PRIORITY_PROMPT
