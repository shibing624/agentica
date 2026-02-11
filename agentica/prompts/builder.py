# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: PromptBuilder - Modular system prompt assembler

This module provides the PromptBuilder class for assembling system prompts
from modular components:
1. Soul (behavioral guidelines)
2. Tools (tool usage priority)
3. Heartbeat (forced iteration)
4. Task management
5. Self verification (lint/test/typecheck)

Based on OpenCode's system.ts prompt selection logic.
"""

from typing import Optional, List

from agentica.prompts.base.heartbeat import get_heartbeat_prompt
from agentica.prompts.base.task_management import get_task_management_prompt
from agentica.prompts.base.tools import get_tools_prompt
from agentica.prompts.base.soul import get_soul_prompt
from agentica.prompts.base.self_verification import get_self_verification_prompt


class PromptBuilder:
    """System Prompt modular assembler.

    Assembles system prompts from modular components based on:
    - Agent configuration (identity, workspace context)
    - Enabled features (heartbeat, task management)

    Example:
        >>> from agentica.prompts.builder import PromptBuilder
        >>>
        >>> # Build a full system prompt
        >>> prompt = PromptBuilder.build_system_prompt(
        ...     identity="You are a helpful coding assistant",
        ...     enable_heartbeat=True,
        ...     enable_task_management=True,
        ... )
    """

    @classmethod
    def build_system_prompt(
        cls,
        identity: Optional[str] = None,
        workspace_context: Optional[str] = None,
        tools_list: Optional[List[str]] = None,
        enable_heartbeat: bool = True,
        enable_task_management: bool = True,
        enable_soul: bool = True,
        enable_tools_guide: bool = True,
        enable_self_verification: bool = True,
    ) -> str:
        """Assemble the complete system prompt from modular components.

        Args:
            identity: Custom identity description
            workspace_context: Additional context from workspace (AGENT.md, etc.)
            tools_list: List of available tools (for reference in prompt)
            enable_heartbeat: Enable forced iteration mechanism
            enable_task_management: Enable task tracking instructions
            enable_soul: Enable behavioral guidelines
            enable_tools_guide: Enable tool usage priority guide
            enable_self_verification: Enable code validation guidance (lint/test/typecheck)

        Returns:
            Complete assembled system prompt
        """
        sections = []

        # 1. Identity section
        if identity:
            sections.append(f"# Identity\n\n{identity}")

        # 2. Soul (behavioral guidelines)
        if enable_soul:
            soul_prompt = get_soul_prompt()
            sections.append(soul_prompt)

        # 3. Tools usage guide
        if enable_tools_guide:
            tools_prompt = get_tools_prompt()
            sections.append(tools_prompt)

        # 4. Heartbeat (forced iteration) - CORE for task completion
        if enable_heartbeat:
            heartbeat_prompt = get_heartbeat_prompt()
            sections.append(heartbeat_prompt)

        # 5. Task management
        if enable_task_management:
            task_prompt = get_task_management_prompt()
            sections.append(task_prompt)

        # 6. Self verification (lint/test/typecheck)
        if enable_self_verification:
            verification_prompt = get_self_verification_prompt()
            sections.append(verification_prompt)

        # 7. Workspace context (injected from workspace files)
        if workspace_context:
            sections.append(f"# Workspace Context\n\n{workspace_context}")

        # 8. Available tools reference
        if tools_list:
            tools_section = "# Available Tools\n\n"
            tools_section += "You have access to the following tools:\n"
            tools_section += "\n".join(f"- {tool}" for tool in tools_list)
            sections.append(tools_section)

        # Join all sections with separators
        return "\n\n---\n\n".join(sections)
