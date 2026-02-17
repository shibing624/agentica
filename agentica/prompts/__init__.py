# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Modular prompt system for Agentica

This module provides a modular prompt system.
Key components:

- base/: Core prompt modules (heartbeat, tools, soul, self_verification)
- builder.py: PromptBuilder for assembling system prompts

Usage:
    from agentica.prompts.builder import PromptBuilder

    system_prompt = PromptBuilder.build_system_prompt(
        identity="You are a helpful coding assistant",
        enable_heartbeat=True,
        active_tools=["read_file", "edit_file", "execute"],
    )
"""

from agentica.prompts.builder import PromptBuilder

__all__ = ["PromptBuilder"]
