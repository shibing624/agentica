# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Modular prompt system for Agentica

This module provides a modular prompt system inspired by OpenCode and OpenClaw.
Key components:

- base/: Core prompt modules (heartbeat, task_management, tools, soul)
- builder.py: PromptBuilder for assembling system prompts

Usage:
    from agentica.prompts.builder import PromptBuilder

    system_prompt = PromptBuilder.build_system_prompt(
        identity="You are a helpful coding assistant",
        enable_heartbeat=True,
        enable_task_management=True,
    )
"""

from agentica.prompts.builder import PromptBuilder

__all__ = ["PromptBuilder"]
