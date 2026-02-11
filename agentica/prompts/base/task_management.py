# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Task Management module - Forced task tracking

This module provides prompts for mandatory task tracking using write_todos tools.
Key principles:
1. Use write_todos proactively for complex/multi-step tasks
2. Mark tasks as completed immediately after finishing
3. Only have ONE task in_progress at a time
4. Skip for trivial/single-step tasks

Based on OpenCode's todowrite.txt
"""

from agentica.prompts.base.utils import load_prompt as _load_prompt

# Load prompt from MD file
TASK_MANAGEMENT_PROMPT = _load_prompt("task_management.md")


def get_task_management_prompt() -> str:
    """Get the task management prompt.

    Returns:
        The task management prompt string
    """
    return TASK_MANAGEMENT_PROMPT
