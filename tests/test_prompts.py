# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for PromptBuilder and prompt modules.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentica.agent import Agent
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse


def _mock_resp(content="OK"):
    resp = MagicMock()
    resp.content = content
    resp.parsed = None
    resp.audio = None
    resp.reasoning_content = None
    resp.created_at = None
    return resp


# ===========================================================================
# TestPromptBuilder
# ===========================================================================


class TestPromptBuilder:
    """Tests for PromptBuilder and modular prompt assembly."""

    def test_builder_default_modules(self):
        """PromptBuilder should assemble default modules."""
        from agentica.prompts.builder import PromptBuilder
        result = PromptBuilder.build_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_builder_build_includes_soul(self):
        from agentica.prompts.builder import PromptBuilder
        result = PromptBuilder.build_system_prompt(enable_soul=True)
        # Soul module should contribute some content
        assert len(result) > 100  # Non-trivial prompt

    def test_builder_selective_modules(self):
        from agentica.prompts.builder import PromptBuilder
        result = PromptBuilder.build_system_prompt(
            enable_soul=True,
            enable_heartbeat=False,
            enable_task_management=False,
            enable_tools_guide=False,
            enable_self_verification=False,
        )
        assert isinstance(result, str)

    def test_builder_with_identity(self):
        from agentica.prompts.builder import PromptBuilder
        result = PromptBuilder.build_system_prompt(identity="You are a coding assistant")
        assert "coding assistant" in result


# ===========================================================================
# TestPromptModules
# ===========================================================================


class TestPromptModules:
    """Tests for individual prompt modules."""

    def test_load_prompt_reads_md(self):
        from agentica.prompts.base.utils import load_prompt
        # Soul prompt should exist as soul.md
        content = load_prompt("soul.md")
        assert isinstance(content, str)
        assert len(content) > 0

    def test_soul_module_content(self):
        from agentica.prompts.base.soul import get_soul_prompt
        content = get_soul_prompt()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_tools_module_content(self):
        from agentica.prompts.base.tools import get_tools_prompt
        content = get_tools_prompt()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_task_management_module_content(self):
        from agentica.prompts.base.task_management import get_task_management_prompt
        content = get_task_management_prompt()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_heartbeat_module_content(self):
        from agentica.prompts.base.heartbeat import get_heartbeat_prompt
        content = get_heartbeat_prompt()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_self_verification_module_content(self):
        from agentica.prompts.base.self_verification import get_self_verification_prompt
        content = get_self_verification_prompt()
        assert isinstance(content, str)
        assert len(content) > 0


# ===========================================================================
# TestGetSystemMessage
# ===========================================================================


class TestGetSystemMessage:
    """Tests for Agent.get_system_message() prompt construction."""

    @pytest.mark.asyncio
    async def test_system_message_includes_instructions(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(model="gpt-4o-mini"),
            instructions=["Be concise", "Be accurate"],
        )
        msg = await agent.get_system_message()
        assert msg is not None
        assert "Be concise" in msg.content
        assert "Be accurate" in msg.content

    @pytest.mark.asyncio
    async def test_system_message_includes_datetime(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(model="gpt-4o-mini"),
            add_datetime_to_instructions=True,
        )
        msg = await agent.get_system_message()
        assert msg is not None
        # Should include date/time info
        content = msg.content
        assert "UTC" in content or "20" in content  # Year prefix

    @pytest.mark.asyncio
    async def test_system_message_with_system_prompt(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(model="gpt-4o-mini"),
            system_prompt="You are a helpful assistant.",
        )
        msg = await agent.get_system_message()
        assert msg is not None
        assert "helpful assistant" in msg.content

    @pytest.mark.asyncio
    async def test_system_message_callable_system_prompt(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(model="gpt-4o-mini"),
            system_prompt=lambda agent=None: "Dynamic system prompt.",
        )
        msg = await agent.get_system_message()
        assert msg is not None
        assert "Dynamic system prompt" in msg.content

    @pytest.mark.asyncio
    async def test_system_message_with_description(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(model="gpt-4o-mini"),
            description="A test assistant",
        )
        msg = await agent.get_system_message()
        assert msg is not None
        assert "test assistant" in msg.content

    @pytest.mark.asyncio
    async def test_system_message_with_agentic_prompt(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(model="gpt-4o-mini"),
            enable_agentic_prompt=True,
        )
        msg = await agent.get_system_message()
        assert msg is not None
        # Agentic prompt adds more content from PromptBuilder
        assert len(msg.content) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
