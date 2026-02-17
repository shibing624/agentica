# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SOUL module - Core behavioral guidelines

This module provides prompts for agent behavior:
1. Technical accuracy and objectivity
2. Tone and style - concise, direct communication
3. Think before acting - consider scope and dependencies
4. Avoid over-engineering - only change what's needed
"""

from agentica.prompts.base.utils import load_prompt as _load_prompt

# Load prompt from MD file
SOUL_PROMPT = _load_prompt("soul.md")


def get_soul_prompt() -> str:
    """Get the soul prompt.

    Returns:
        The soul prompt string
    """
    return SOUL_PROMPT
