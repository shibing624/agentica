# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: PromptBuilder - Modular system prompt assembler

Section ordering:
1. Identity (intro)
2. Soul (behavioral guidelines + tone)
3. Tools (tool usage strategy + dynamic tool list)
4. Heartbeat (iteration control + verification)
5. Workspace context (dynamic)
"""

from typing import Dict, List, Optional

from agentica.prompts.base.heartbeat import get_heartbeat_prompt
from agentica.prompts.base.soul import get_soul_prompt
from agentica.prompts.base.tools import get_tools_prompt


class PromptBuilder:
    """System prompt modular assembler."""

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
    ) -> str:
        """Assemble the complete system prompt from modular components."""
        sections = []

        if identity:
            sections.append(f"# Identity\n\n{identity}")

        if enable_soul:
            sections.append(get_soul_prompt())

        if enable_tools_guide:
            sections.append(
                get_tools_prompt(
                    active_tools=active_tools,
                    tool_descriptions=tool_descriptions,
                )
            )

        if enable_heartbeat:
            sections.append(get_heartbeat_prompt())

        if workspace_context:
            sections.append(f"# Workspace Context\n\n{workspace_context}")

        return "\n\n---\n\n".join(sections)
