# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: TOOLS module - Tool usage priority and parallel strategy

This module provides prompts for guiding tool selection and usage:
1. Prefer specialized tools over bash commands
2. Parallel execution strategy for independent calls
3. Tool selection priority based on task type
4. File operation guidelines

Based on OpenCode's anthropic.txt and gemini.txt
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
TOOLS_PRIORITY_PROMPT = _load_prompt("tools.md")
TOOLS_PRIORITY_PROMPT_COMPACT = _load_prompt("tools_compact.md")


def get_tools_prompt(compact: bool = False) -> str:
    """Get the tools prompt.

    Args:
        compact: If True, return the compact version

    Returns:
        The appropriate tools prompt string
    """
    return TOOLS_PRIORITY_PROMPT_COMPACT if compact else TOOLS_PRIORITY_PROMPT
