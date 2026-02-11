# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SELF_VERIFICATION module - Code validation guidance

This module provides prompts for self-verification after code changes.
Key principles:
1. Run lint/typecheck/test commands after completing code changes
2. Discover validation commands from project configuration
3. Fix errors and re-run until passing
4. Use shell tool to execute validation commands
"""

from agentica.prompts.base.utils import load_prompt as _load_prompt

# Load prompt from MD file
SELF_VERIFICATION_PROMPT = _load_prompt("self_verification.md")


def get_self_verification_prompt() -> str:
    """Get the self verification prompt.

    Returns:
        The self verification prompt string
    """
    return SELF_VERIFICATION_PROMPT
