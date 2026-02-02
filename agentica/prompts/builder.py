# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: PromptBuilder - Modular system prompt assembler

This module provides the PromptBuilder class for assembling system prompts
from modular components:
1. Identity (who the agent is)
2. Model-specific optimizations
3. Soul (behavioral guidelines)
4. Tools (tool usage priority)
5. Heartbeat (forced iteration)
6. Task management
7. Self verification (lint/test/typecheck)

Based on OpenCode's system.ts prompt selection logic.
"""

from typing import Optional, List

from agentica.prompts.base.heartbeat import get_heartbeat_prompt
from agentica.prompts.base.task_management import get_task_management_prompt
from agentica.prompts.base.tools import get_tools_prompt
from agentica.prompts.base.soul import get_soul_prompt
from agentica.prompts.base.identity import get_identity_prompt
from agentica.prompts.base.self_verification import get_self_verification_prompt

from agentica.prompts.models.claude import get_claude_prompt
from agentica.prompts.models.openai import get_openai_prompt
from agentica.prompts.models.zhipu import get_zhipu_prompt
from agentica.prompts.models.deepseek import get_deepseek_prompt
from agentica.prompts.models.default import get_default_prompt


class PromptBuilder:
    """System Prompt modular assembler.

    Assembles system prompts from modular components based on:
    - Model type (Claude, GPT, GLM, DeepSeek, etc.)
    - Agent configuration (identity, workspace context)
    - Enabled features (heartbeat, task management)

    Example:
        >>> from agentica.prompts.builder import PromptBuilder
        >>>
        >>> # Build a full system prompt for Claude
        >>> prompt = PromptBuilder.build_system_prompt(
        ...     model_id="claude-3-opus",
        ...     identity="You are a helpful coding assistant",
        ...     enable_heartbeat=True,
        ...     enable_task_management=True,
        ... )
        >>>
        >>> # Get model-specific prompt only
        >>> model_prompt = PromptBuilder.get_model_prompt("gpt-4o")
    """

    # Model ID patterns for detection
    CLAUDE_PATTERNS = ["claude", "anthropic"]
    OPENAI_PATTERNS = ["gpt-", "gpt4", "o1", "o3", "openai"]
    ZHIPU_PATTERNS = ["glm", "zhipu", "chatglm"]
    DEEPSEEK_PATTERNS = ["deepseek"]
    QWEN_PATTERNS = ["qwen", "tongyi"]
    MOONSHOT_PATTERNS = ["moonshot", "kimi"]
    YI_PATTERNS = ["yi-"]
    DOUBAO_PATTERNS = ["doubao"]

    @classmethod
    def detect_model_type(cls, model_id: str) -> str:
        """Detect the model type from model ID.

        Args:
            model_id: The model identifier string

        Returns:
            Model type: "claude", "openai", "zhipu", "deepseek", "qwen", "default"
        """
        if not model_id:
            return "default"

        model_id_lower = model_id.lower()

        if any(p in model_id_lower for p in cls.CLAUDE_PATTERNS):
            return "claude"
        elif any(p in model_id_lower for p in cls.OPENAI_PATTERNS):
            return "openai"
        elif any(p in model_id_lower for p in cls.ZHIPU_PATTERNS):
            return "zhipu"
        elif any(p in model_id_lower for p in cls.DEEPSEEK_PATTERNS):
            return "deepseek"
        elif any(p in model_id_lower for p in cls.QWEN_PATTERNS):
            return "qwen"
        elif any(p in model_id_lower for p in cls.MOONSHOT_PATTERNS):
            return "moonshot"
        elif any(p in model_id_lower for p in cls.YI_PATTERNS):
            return "yi"
        elif any(p in model_id_lower for p in cls.DOUBAO_PATTERNS):
            return "doubao"
        else:
            return "default"

    @classmethod
    def get_model_prompt(cls, model_id: str) -> str:
        """Get the model-specific prompt based on model ID.

        Args:
            model_id: The model identifier string

        Returns:
            Model-specific prompt string
        """
        model_type = cls.detect_model_type(model_id)

        if model_type == "claude":
            return get_claude_prompt()
        elif model_type == "openai":
            return get_openai_prompt()
        elif model_type == "zhipu":
            return get_zhipu_prompt()
        elif model_type == "deepseek":
            return get_deepseek_prompt()
        else:
            # For qwen, moonshot, yi, doubao, and others, use default
            return get_default_prompt()

    @classmethod
    def build_system_prompt(
        cls,
        model_id: str = "",
        identity: Optional[str] = None,
        identity_type: str = "default",
        workspace_context: Optional[str] = None,
        tools_list: Optional[List[str]] = None,
        enable_heartbeat: bool = True,
        enable_task_management: bool = True,
        enable_soul: bool = True,
        enable_tools_guide: bool = True,
        enable_self_verification: bool = True,
        compact: bool = False,
    ) -> str:
        """Assemble the complete system prompt from modular components.

        Args:
            model_id: The model identifier string for model-specific optimizations
            identity: Custom identity description (overrides identity_type)
            identity_type: Type of identity ("cli", "api", "default")
            workspace_context: Additional context from workspace (AGENT.md, etc.)
            tools_list: List of available tools (for reference in prompt)
            enable_heartbeat: Enable forced iteration mechanism
            enable_task_management: Enable task tracking instructions
            enable_soul: Enable behavioral guidelines
            enable_tools_guide: Enable tool usage priority guide
            enable_self_verification: Enable code validation guidance (lint/test/typecheck)
            compact: Use compact versions of prompts (for context-sensitive situations)

        Returns:
            Complete assembled system prompt
        """
        sections = []

        # 1. Identity section
        if identity:
            sections.append(f"# Identity\n\n{identity}")
        else:
            identity_prompt = get_identity_prompt(identity_type)
            sections.append(f"# Identity\n\n{identity_prompt}")

        # 2. Model-specific optimizations
        model_prompt = cls.get_model_prompt(model_id)
        if model_prompt:
            sections.append(model_prompt)

        # 3. Soul (behavioral guidelines)
        if enable_soul:
            soul_prompt = get_soul_prompt(compact)
            sections.append(soul_prompt)

        # 4. Tools usage guide
        if enable_tools_guide:
            tools_prompt = get_tools_prompt(compact)
            sections.append(tools_prompt)

        # 5. Heartbeat (forced iteration) - CORE for task completion
        if enable_heartbeat:
            heartbeat_prompt = get_heartbeat_prompt(compact)
            sections.append(heartbeat_prompt)

        # 6. Task management
        if enable_task_management:
            task_prompt = get_task_management_prompt(compact)
            sections.append(task_prompt)

        # 7. Self verification (lint/test/typecheck)
        if enable_self_verification:
            verification_prompt = get_self_verification_prompt(compact)
            sections.append(verification_prompt)

        # 8. Workspace context (injected from workspace files)
        if workspace_context:
            sections.append(f"# Workspace Context\n\n{workspace_context}")

        # 9. Available tools reference
        if tools_list:
            tools_section = "# Available Tools\n\n"
            tools_section += "You have access to the following tools:\n"
            tools_section += "\n".join(f"- {tool}" for tool in tools_list)
            sections.append(tools_section)

        # Join all sections with separators
        separator = "\n\n---\n\n" if not compact else "\n\n"
        return separator.join(sections)

    @classmethod
    def build_minimal_prompt(
        cls,
        model_id: str = "",
        identity: Optional[str] = None,
    ) -> str:
        """Build a minimal system prompt for simple tasks or context-constrained situations.

        Args:
            model_id: The model identifier string
            identity: Custom identity description

        Returns:
            Minimal system prompt
        """
        return cls.build_system_prompt(
            model_id=model_id,
            identity=identity,
            enable_heartbeat=False,
            enable_task_management=False,
            enable_soul=True,
            enable_tools_guide=False,
            compact=True,
        )

    @classmethod
    def build_agent_prompt(
        cls,
        model_id: str = "",
        identity: Optional[str] = None,
        workspace_context: Optional[str] = None,
    ) -> str:
        """Build a full agent system prompt with all features enabled.

        This is the recommended method for multi-round agent tasks.

        Args:
            model_id: The model identifier string
            identity: Custom identity description
            workspace_context: Context from workspace

        Returns:
            Full agent system prompt
        """
        return cls.build_system_prompt(
            model_id=model_id,
            identity=identity,
            workspace_context=workspace_context,
            enable_heartbeat=True,
            enable_task_management=True,
            enable_soul=True,
            enable_tools_guide=True,
            compact=False,
        )
