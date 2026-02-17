# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: PromptBuilder - Modular system prompt assembler

This module provides the PromptBuilder class for assembling system prompts
from modular components:
1. Soul (behavioral guidelines)
2. Tools (tool usage strategy + dynamic tool list)
3. Heartbeat (forced iteration)
4. Self verification (lint/test/typecheck)
"""

from typing import Optional, List, Dict

from agentica.prompts.base.heartbeat import get_heartbeat_prompt
from agentica.prompts.base.tools import get_tools_prompt
from agentica.prompts.base.soul import get_soul_prompt
from agentica.prompts.base.self_verification import get_self_verification_prompt


class PromptBuilder:
    """System Prompt modular assembler.

    Assembles system prompts from modular components based on:
    - Agent configuration (identity, workspace context)
    - Enabled features (heartbeat, tools guide, self verification)
    - Active tools (dynamic tool list generation)

    Example:
        >>> from agentica.prompts.builder import PromptBuilder
        >>>
        >>> # Build a full system prompt
        >>> prompt = PromptBuilder.build_system_prompt(
        ...     identity="You are a helpful coding assistant",
        ...     enable_heartbeat=True,
        ...     active_tools=["read_file", "edit_file", "execute"],
        ... )
    """

    @classmethod
    def build_system_prompt(
        cls,
        identity: Optional[str] = None,
        workspace_context: Optional[str] = None,
        active_tools: Optional[List[str]] = None,
        tool_descriptions: Optional[Dict[str, str]] = None,
        enable_heartbeat: bool = True,
        enable_soul: bool = True,
        enable_tools_guide: bool = True,
        enable_self_verification: bool = True,
    ) -> str:
        """Assemble the complete system prompt from modular components.

        Args:
            identity: Custom identity description
            workspace_context: Additional context from workspace (AGENT.md, etc.)
            active_tools: List of currently enabled tool names for dynamic tool table
            tool_descriptions: Optional mapping of tool_name -> description for tool table
            enable_heartbeat: Enable forced iteration mechanism
            enable_soul: Enable behavioral guidelines
            enable_tools_guide: Enable tool usage strategy guide
            enable_self_verification: Enable code validation guidance (lint/test/typecheck)

        Returns:
            Complete assembled system prompt
        """
        sections = []

        # 1. Identity section
        if identity:
            sections.append(f"# Identity\n\n{identity}")

        # 2. Soul (core behavioral guidelines)
        if enable_soul:
            sections.append(get_soul_prompt())

        # 3. Tools usage guide (with optional dynamic tool list)
        if enable_tools_guide:
            sections.append(get_tools_prompt(
                active_tools=active_tools,
                tool_descriptions=tool_descriptions,
            ))

        # 4. Heartbeat (iteration control)
        if enable_heartbeat:
            sections.append(get_heartbeat_prompt())

        # 5. Self verification (lint/test/typecheck)
        if enable_self_verification:
            sections.append(get_self_verification_prompt())

        # 6. Workspace context (injected from workspace files)
        if workspace_context:
            sections.append(f"# Workspace Context\n\n{workspace_context}")

        # Join all sections with separators
        return "\n\n---\n\n".join(sections)
