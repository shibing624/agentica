# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: TASK_MANAGEMENT module - Forced task tracking

This module provides prompts for mandatory task tracking using TodoWrite tools.
Key principles:
1. Use TodoWrite VERY frequently to track tasks
2. Mark tasks as completed immediately after finishing
3. Only have ONE task in_progress at a time
4. Break complex tasks into smaller, manageable steps

Based on OpenCode's anthropic.txt and todowrite.txt
"""

from pathlib import Path

_BASE_DIR = Path(__file__).parent / "md"


def _load_prompt(filename: str) -> str:
    """Load prompt content from MD file."""
    filepath = _BASE_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8").strip()
    return ""


# Load prompts from MD files
TASK_MANAGEMENT_PROMPT = _load_prompt("task_management.md")
TASK_MANAGEMENT_PROMPT_COMPACT = _load_prompt("task_management_compact.md")


def get_task_management_prompt(compact: bool = False) -> str:
    """Get the task management prompt.

    Args:
        compact: If True, return the compact version

    Returns:
        The appropriate task management prompt string
    """
    return TASK_MANAGEMENT_PROMPT_COMPACT if compact else TASK_MANAGEMENT_PROMPT
