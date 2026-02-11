# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SOUL module - Behavioral guidelines and professional objectivity

This module provides prompts for agent behavior:
1. Professional objectivity - prioritize accuracy over validation
2. Tone and style - concise, direct communication
3. No time estimates - focus on what, not how long
4. Genuine helpfulness - skip pleasantries, just help
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
